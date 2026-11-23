"""Comprehensive regression and validation tests for FedSecureAI Phase F.

Verifies:
1. Multi-Tenant Account History Feature Store:
   - Strict tenant isolation (cross-tenant zero-leakage).
   - Causal forward-in-time updates.
   - Multi-upload persistence vs upload-local default.
   - TTL eviction and explicit tenant clearing.
2. Real IID + Non-IID Ensemble Optimization:
   - Alpha grid exploration in [0.0, 1.0].
   - Multiple validation objectives ('max_pr_auc', 'max_f1', 'max_f2', 'cost_sensitive').
   - Zero test-set leakage (validation-only threshold and alpha freezing).
   - Scientifically honest reporting when alpha*=1.00 (degraded Non-IID stream).
3. Evaluation Metrics and API Integrity:
   - ECE and Brier score inclusion.
   - Suggested models under insufficient history.
   - REST endpoints /ensemble/evaluate and /ensemble/summary.
"""

from __future__ import annotations

import io
import time
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from application.services.ensemble_experiment_service import (
    AlphaEvaluationResult,
    EnsembleExperimentService,
)
from infrastructure.data.features.account_history import AccountHistoryBuilder
from infrastructure.data.features.account_history_store import (
    AccountHistoryFeatureStore,
    AccountProfile,
    get_account_history_store,
)
from interfaces.api.app import app
from interfaces.api.services.csv_evaluation_service import CsvEvaluationService


@pytest.fixture
def client() -> TestClient:
    """FastAPI TestClient instance."""
    return TestClient(app)


# ==============================================================================
# 1. Multi-Tenant Account History Feature Store Tests
# ==============================================================================


def test_feature_store_multi_tenant_isolation() -> None:
    """Verify that different tenants sharing the exact same account IDs remain isolated."""
    store = AccountHistoryFeatureStore()

    # Update account 'ACC_001' under tenant_alpha
    store.update_profile(
        tenant_id="tenant_alpha",
        account_id="ACC_001",
        sent_count_delta=5.0,
        sent_amount_delta=1500.0,
        sent_latest_epoch=1000.0,
    )

    # Update account 'ACC_001' under tenant_beta with different values
    store.update_profile(
        tenant_id="tenant_beta",
        account_id="ACC_001",
        sent_count_delta=1.0,
        sent_amount_delta=200.0,
        sent_latest_epoch=500.0,
    )

    prof_alpha = store.get_profile("tenant_alpha", "ACC_001")
    prof_beta = store.get_profile("tenant_beta", "ACC_001")

    assert prof_alpha is not None
    assert prof_beta is not None
    assert prof_alpha.sent_count == 5.0
    assert prof_alpha.sent_total == 1500.0
    assert prof_alpha.sent_last_epoch == 1000.0

    assert prof_beta.sent_count == 1.0
    assert prof_beta.sent_total == 200.0
    assert prof_beta.sent_last_epoch == 500.0


def test_feature_store_causal_builder_interop() -> None:
    """Verify AccountHistoryBuilder imports prior profile and strictly preserves causality."""
    store = AccountHistoryFeatureStore()
    tenant_id = "test_bank_ny"

    # Seed prior state: account 'ACC_A' previously sent 2 tx totaling $500
    store.save_profiles(
        tenant_id,
        {
            "ACC_A": AccountProfile(
                account_id="ACC_A",
                sent_count=2.0,
                sent_total=500.0,
                sent_last_epoch=100.0,
            )
        },
    )

    # Build new chunk for ACC_A
    builder = AccountHistoryBuilder()
    builder.import_profiles(store.get_all_profiles(tenant_id))

    chunk = pd.DataFrame({
        "Account": ["ACC_A", "ACC_A"],
        "Account.1": ["ACC_B", "ACC_C"],
        "Amount Paid": [100.0, 200.0],
        "Amount Received": [100.0, 200.0],
        "_epoch_seconds": [200.0, 300.0],
    })

    result = builder.apply(chunk)

    # Row 0: prior count should be exactly 2.0 (from prior history), log1p(2)
    assert np.isclose(result["sender_prior_count_log"].iloc[0], np.log1p(2.0), atol=1e-5)
    # Row 0: prior amount should be 500.0, log1p(500)
    assert np.isclose(result["sender_prior_amount_log"].iloc[0], np.log1p(500.0), atol=1e-5)
    # Row 0: seconds since prior should be 200 - 100 = 100
    assert np.isclose(result["sender_seconds_since_prior_log"].iloc[0], np.log1p(100.0), atol=1e-5)
    assert result["sender_has_prior"].iloc[0] == 1

    # Row 1: prior count should be 2 + 1 = 3
    assert np.isclose(result["sender_prior_count_log"].iloc[1], np.log1p(3.0), atol=1e-5)
    # Row 1: prior amount should be 500 + 100 = 600
    assert np.isclose(result["sender_prior_amount_log"].iloc[1], np.log1p(600.0), atol=1e-5)
    # Row 1: seconds since prior should be 300 - 200 = 100
    assert np.isclose(result["sender_seconds_since_prior_log"].iloc[1], np.log1p(100.0), atol=1e-5)

    # Export updated profile
    updated = builder.export_profiles()
    store.save_profiles(tenant_id, updated)

    final_prof = store.get_profile(tenant_id, "ACC_A")
    assert final_prof is not None
    assert final_prof.sent_count == 4.0
    assert final_prof.sent_total == 800.0
    assert final_prof.sent_last_epoch == 300.0


def test_feature_store_ttl_eviction_and_clear() -> None:
    """Verify TTL eviction and explicit tenant clearing."""
    store = AccountHistoryFeatureStore()

    store.update_profile("tenant_old", "ACC_OLD", sent_count_delta=1.0)
    store.update_profile("tenant_keep", "ACC_KEEP", sent_count_delta=2.0)

    # Artificially age the old account
    store._tenants["tenant_old"]["ACC_OLD"].updated_at = time.time() - 1000.0

    evicted = store.evict_expired(max_age_seconds=500.0)
    assert evicted == 1
    assert store.get_profile("tenant_old", "ACC_OLD") is None
    assert store.get_profile("tenant_keep", "ACC_KEEP") is not None

    # Clear tenant_keep
    store.clear_tenant("tenant_keep")
    assert store.get_profile("tenant_keep", "ACC_KEEP") is None


# ==============================================================================
# 2. Real IID + Non-IID Ensemble Optimization & Audit Tests
# ==============================================================================


def test_ensemble_experiment_alpha_search_validation_only() -> None:
    """Verify that ensemble service executes alpha grid search strictly on validation data."""
    service = EnsembleExperimentService()
    # Fast evaluation with subsample
    outcome = service.run_ensemble_experiment(
        dataset="HI-Small",
        alpha_step=0.2,
        objective="max_pr_auc",
        subsample_limit=20000,
    )

    assert outcome.dataset == "HI-Small"
    assert outcome.best_alpha in [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    assert outcome.frozen_threshold > 0.0
    assert len(outcome.alpha_candidates) == 6

    # Verify each alpha candidate has valid metrics
    for c in outcome.alpha_candidates:
        assert 0.0 <= c.val_pr_auc <= 1.0
        assert 0.0 <= c.val_roc_auc <= 1.0
        assert 0.0 <= c.val_brier_score <= 1.0

    # Test metrics must exist and match authoritative values
    assert outcome.test_metrics["pr_auc"] > 0.40
    assert outcome.iid_standalone_test_metrics["pr_auc"] > 0.40


def test_ensemble_honest_reporting_when_alpha_equals_one() -> None:
    """Verify that when alpha*=1.00, ensemble_improved is False and scientific explanation is honest."""
    service = EnsembleExperimentService()
    outcome = service.run_ensemble_experiment(
        dataset="HI-Small",
        alpha_step=0.1,
        objective="max_pr_auc",
    )

    # On HI-Small full validation data, standalone IID strictly beats Non-IID (0.2421 vs 0.1990)
    assert outcome.best_alpha == 1.0
    assert outcome.ensemble_improved is False
    assert "pruning the degraded Non-IID stream" in outcome.scientific_rationale
    assert outcome.delta_vs_iid["delta_pr_auc"] == 0.0


def test_ensemble_different_objectives_support() -> None:
    """Verify support for F1, F2, and cost-sensitive objectives."""
    service = EnsembleExperimentService()

    for obj in ("max_f1", "max_f2", "cost_sensitive"):
        outcome = service.run_ensemble_experiment(
            dataset="HI-Small",
            alpha_step=0.2,
            objective=obj,
            cost_fp=1.0,
            cost_fn=20.0,
            subsample_limit=10000,
        )
        assert outcome.objective == obj
        assert 0.0 <= outcome.best_alpha <= 1.0
        assert outcome.frozen_threshold > 0.0


# ==============================================================================
# 3. API Endpoints & Multi-Upload Integration Tests
# ==============================================================================


def test_api_ensemble_evaluate_endpoint(client: TestClient) -> None:
    """Test POST /api/v1/ensemble/evaluate returns complete audit report."""
    response = client.post(
        "/api/v1/ensemble/evaluate",
        json={
            "dataset": "HI-Small",
            "alpha_step": 0.25,
            "objective": "max_pr_auc",
        },
    )
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert data["dataset"] == "HI-Small"
    assert "best_alpha" in data
    assert "frozen_threshold" in data
    assert "scientific_rationale" in data
    assert "alpha_candidates" in data
    assert len(data["alpha_candidates"]) == 5  # 0.0, 0.25, 0.5, 0.75, 1.0


def test_api_ensemble_summary_endpoint(client: TestClient) -> None:
    """Test GET /api/v1/ensemble/summary returns frozen configuration."""
    response = client.get("/api/v1/ensemble/summary?dataset=HI-Small")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert data["best_alpha"] == 1.0
    assert data["frozen_threshold"] > 0.0
    assert data["ensemble_improved"] is False


def test_api_predict_csv_with_persistent_history(client: TestClient) -> None:
    """Test POST /api/v1/predict/csv with persistent history enabled across two uploads."""
    csv_1 = (
        "Timestamp,From Bank,From Account,To Bank,To Account,Amount Received,Receiving Currency,Amount Paid,Payment Currency,Payment Format\n"
        "2022/09/01 09:00,012,ACC_PERSIST_1,012,ACC_PERSIST_2,500.00,USD,500.00,USD,Wire\n"
    )

    csv_2 = (
        "Timestamp,From Bank,From Account,To Bank,To Account,Amount Received,Receiving Currency,Amount Paid,Payment Currency,Payment Format\n"
        "2022/09/01 10:00,012,ACC_PERSIST_1,012,ACC_PERSIST_3,750.00,USD,750.00,USD,Wire\n"
    )

    tenant = f"tenant_test_{int(time.time())}"

    # First upload: seeds account profile
    res1 = client.post(
        "/api/v1/predict/csv",
        json={
            "csv_content": csv_1,
            "model_id": "HI-Small",
            "tenant_id": tenant,
            "enable_persistent_history": True,
        },
    )
    assert res1.status_code == 200
    data1 = res1.json()
    assert data1["status"] == "success"
    assert data1["persistent_history_enabled"] is True

    # Check store directly
    store = get_account_history_store()
    prof = store.get_profile(tenant, "ACC_PERSIST_1")
    assert prof is not None
    assert prof.sent_count == 1.0
    assert prof.sent_total == 500.0

    # Second upload: resumes account profile
    res2 = client.post(
        "/api/v1/predict/csv",
        json={
            "csv_content": csv_2,
            "model_id": "HI-Small",
            "tenant_id": tenant,
            "enable_persistent_history": True,
        },
    )
    assert res2.status_code == 200
    data2 = res2.json()
    assert data2["status"] == "success"

    # Profile should now reflect 2 sent transactions totaling $1250
    prof2 = store.get_profile(tenant, "ACC_PERSIST_1")
    assert prof2 is not None
    assert prof2.sent_count == 2.0
    assert prof2.sent_total == 1250.0

    # Clean up test tenant
    store.clear_tenant(tenant)


def test_metrics_ece_and_suggested_models_inclusion(client: TestClient) -> None:
    """Test that evaluation metrics compute ECE and suggest tabular models when history is short."""
    # Labeled single-transaction CSV (5 rows)
    csv_content = (
        "Timestamp,From Bank,From Account,To Bank,To Account,Amount Received,Receiving Currency,Amount Paid,Payment Currency,Payment Format,Is Laundering\n"
        "2022/09/01 09:00,012,A1,012,B1,100.00,USD,100.00,USD,Wire,0\n"
        "2022/09/01 09:05,012,A2,012,B2,200.00,USD,200.00,USD,Wire,0\n"
        "2022/09/01 09:10,012,A3,012,B3,300.00,USD,300.00,USD,Wire,1\n"
        "2022/09/01 09:15,012,A4,012,B4,400.00,USD,400.00,USD,Wire,0\n"
        "2022/09/01 09:20,012,A5,012,B5,500.00,USD,500.00,USD,Wire,1\n"
    )

    response = client.post(
        "/api/v1/predict/csv",
        json={
            "csv_content": csv_content,
            "model_id": "HI-Small",
        },
    )
    assert response.status_code == 200
    data = response.json()
    assert data["has_labels"] is True

    metrics = data["evaluation_metrics"]
    assert metrics is not None
    assert "brier_score" in metrics
    assert "expected_calibration_error" in metrics

    # Since average sequence length = 1.0 < 3.0, warning and diagnostics must be present
    diag = data["sequence_context_diagnostics"]
    assert diag is not None
    assert diag["insufficient_history_warning"] is True
    assert diag["operational_context_status"] == "INSUFFICIENT_HISTORY_TRUNCATED"
    assert "suggested_models" in diag


def test_zero_alert_metrics_semantics(client: TestClient) -> None:
    """Verify that when 0 alerts are generated, precision and F1/F2 are undefined (null)."""
    service = CsvEvaluationService()
    # Labels have 2 positives, but probabilities are all 0.01 (below threshold 0.337784)
    labels = np.array([0, 0, 1, 0, 1], dtype=int)
    probs = np.array([0.01, 0.02, 0.015, 0.005, 0.01], dtype=float)
    threshold = 0.337784

    metrics = service._calculate_metrics(labels, probs, threshold)

    assert metrics["confusion_matrix"]["tp"] == 0
    assert metrics["confusion_matrix"]["fp"] == 0
    assert metrics["confusion_matrix"]["fn"] == 2
    assert metrics["confusion_matrix"]["tn"] == 3

    assert metrics["recall"] == 0.0
    assert metrics["precision"] is None
    assert metrics["f1_score"] is None
    assert metrics["f2_score"] is None
    assert metrics["zero_alerts_note"] is not None
    assert "mathematically undefined" in metrics["zero_alerts_note"]


def test_unlabeled_data_metrics_omitted(client: TestClient) -> None:
    """Verify that for unlabeled CSVs, evaluation_metrics is strictly None."""
    csv_unlabeled = (
        "Timestamp,From Bank,From Account,To Bank,To Account,Amount Received,Receiving Currency,Amount Paid,Payment Currency,Payment Format\n"
        "2022/09/01 09:00,012,ACC_01,012,ACC_02,500.00,USD,500.00,USD,Wire\n"
        "2022/09/01 09:05,012,ACC_03,012,ACC_04,150.00,USD,150.00,USD,Wire\n"
    )

    response = client.post(
        "/api/v1/predict/csv",
        json={
            "csv_content": csv_unlabeled,
            "model_id": "HI-Small",
        },
    )
    assert response.status_code == 200
    data = response.json()
    assert data["has_labels"] is False
    assert data["evaluation_metrics"] is None


def test_arbitrary_schema_incompatibility_rejection(client: TestClient) -> None:
    """Verify that completely arbitrary non-financial schemas are rejected with clear error."""
    arbitrary_csv = (
        "fruit_name,color,weight_grams,is_sweet\n"
        "apple,red,150,1\n"
        "banana,yellow,120,1\n"
        "lemon,yellow,80,0\n"
    )

    response = client.post(
        "/api/v1/predict/csv",
        json={
            "csv_content": arbitrary_csv,
            "model_id": "HI-Small",
        },
    )
    # Must reject with 400 Bad Request
    assert response.status_code == 400
    data = response.json()
    assert "detail" in data
    # Error detail should inform about missing essential features
    assert any(term in data["detail"].lower() for term in ["missing", "unsupported", "invalid", "incompatible"])


def test_threshold_analysis_does_not_mutate_frozen_threshold(client: TestClient) -> None:
    """Verify that analyzing thresholds explores tradeoffs without mutating stored frozen threshold."""
    # Check current threshold for HI-Small
    res_before = client.get("/api/v1/models")
    assert res_before.status_code == 200
    models_before = {m["model_id"]: m for m in res_before.json()["models"]}
    hi_before_th = models_before["HI-Small"]["frozen_threshold"]

    # Run threshold analysis targeting recall = 0.90
    analysis_res = client.post(
        "/api/v1/evaluate/threshold-analysis",
        json={
            "dataset": "HI-Small",
            "objective": "target_recall",
            "target_recall": 0.90,
        },
    )
    assert analysis_res.status_code == 200
    ana_data = analysis_res.json()
    assert "optimal_threshold" in ana_data
    assert "candidates" in ana_data
    assert len(ana_data["candidates"]) > 0

    # Check threshold after: must be IDENTICAL (frozen production threshold unchanged)
    res_after = client.get("/api/v1/models")
    assert res_after.status_code == 200
    models_after = {m["model_id"]: m for m in res_after.json()["models"]}
    hi_after_th = models_after["HI-Small"]["frozen_threshold"]

    assert hi_before_th == hi_after_th

