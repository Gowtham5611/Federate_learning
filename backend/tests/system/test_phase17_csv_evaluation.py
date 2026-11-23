"""Tests for FedSecureAI Phase 17: CSV Evaluation Pipeline and Research Routes."""

from pathlib import Path
import pytest
from fastapi import HTTPException

from interfaces.api.routes.evaluation import evaluate_csv, CsvEvaluateRequest
from interfaces.api.routes.research import (
    get_research_overview,
    get_benchmark_models,
    get_convergence_curves,
    get_threshold_tradeoffs,
    get_ablation_summary,
    download_sample_csv,
)
from interfaces.api.services.csv_evaluation_service import CsvEvaluationService
from interfaces.api.services.model_service import ModelService

_BACKEND_DIR = Path(__file__).resolve().parents[2]
BASE_DIR = _BACKEND_DIR if (_BACKEND_DIR / "data").exists() else Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module", autouse=True)
def preload_models():
    """Ensure models are loaded once for test suite."""
    ms = ModelService()
    ms.load_all_models()


def test_research_overview():
    """Verify research overview metadata."""
    overview = get_research_overview()
    assert "HI-Small" in overview["verified_datasets"]
    assert "LI-Medium" in overview["verified_datasets"]
    hi = overview["verified_datasets"]["HI-Small"]
    assert hi["champion_model"] == "hi_small_lstm_fedavg_iid_seed42"
    assert hi["frozen_threshold"] == pytest.approx(0.33778378, abs=1e-5)
    assert hi["parameter_count"] == 31553

    li = overview["verified_datasets"]["LI-Medium"]
    assert li["champion_model"] == "config_5_bilstm_fedavg_iid"
    assert li["frozen_threshold"] == pytest.approx(0.23848299, abs=1e-5)
    assert li["parameter_count"] == 63105


def test_research_models():
    """Verify master benchmark model matrix from Phase 8."""
    models = get_benchmark_models()
    assert len(models) == 27  # 12 HI-Small + 12 LI-Medium + 3 Centralized
    datasets = {m["dataset"] for m in models}
    assert datasets == {"HI-Small", "LI-Medium"}

    # Check champion HI-Small model
    hi_champ = next(m for m in models if m["model_name"] == "hi_small_lstm_fedavg_iid_seed42")
    assert hi_champ["architecture"] == "LSTM"
    assert hi_champ["distribution"] == "IID"
    assert hi_champ["algorithm"] == "FedAvg"
    assert hi_champ["test_pr_auc"] > 0.47

    # Check champion LI-Medium model
    li_champ = next(m for m in models if m["config_id"] == "config_5")
    assert li_champ["architecture"] == "BiLSTM"
    assert li_champ["test_pr_auc"] > 0.25


def test_research_curves():
    """Verify 120 round-by-round convergence datapoints."""
    curves = get_convergence_curves()
    assert len(curves) == 120  # 24 configs * 5 rounds
    rounds = {c["round"] for c in curves}
    assert rounds == {1, 2, 3, 4, 5}


def test_research_ablation():
    """Verify empirical ablation findings."""
    ablation = get_ablation_summary()
    assert "fedprox_mu_0_0001_ablation" in ablation
    assert "iid_vs_noniid_degradation" in ablation
    assert "phase7_hybrid_ensembling" in ablation
    assert "phase10_probability_calibration" in ablation


def test_download_sample_csv():
    """Verify sample CSV download for HI-Small and LI-Medium."""
    resp_hi = download_sample_csv("HI-Small")
    assert resp_hi.status_code == 200
    assert b"timestamp" in resp_hi.body

    resp_li = download_sample_csv("LI-Medium")
    assert resp_li.status_code == 200
    assert b"timestamp" in resp_li.body

    with pytest.raises(HTTPException) as exc:
        download_sample_csv("InvalidDataset")
    assert exc.value.status_code == 404


def test_evaluate_csv_hi_small_processed_with_labels():
    """Verify CSV evaluation of processed HI-Small slice with ground truth labels."""
    csv_text = (BASE_DIR / "data/demo_samples/hi_small_sample_25.csv").read_text(encoding="utf-8")
    req = CsvEvaluateRequest(
        csv_content=csv_text,
        dataset="HI-Small",
        filename="hi_small_sample_25.csv",
        row_limit=50,
    )
    res = evaluate_csv(req)

    assert res["status"] == "success"
    assert res["dataset"] == "HI-Small"
    assert "HI-Small Champion" in res["model_name"] or res.get("model_id") == "hi_small_lstm_fedavg_iid_seed42"
    assert res["decision_threshold"] == pytest.approx(0.33778378, abs=1e-5)
    assert res["evaluated_sequences"] == 25
    assert res["has_labels"] is True
    assert res["evaluation_metrics"] is not None

    metrics = res["evaluation_metrics"]
    assert "precision" in metrics
    assert "recall" in metrics
    assert "f1_score" in metrics
    assert "accuracy" in metrics
    assert "confusion_matrix" in metrics
    cm = metrics["confusion_matrix"]
    assert cm["tp"] + cm["fp"] + cm["tn"] + cm["fn"] == 25

    # Check predictions format
    preds = res["predictions"]
    assert len(preds) == 25
    first = preds[0]
    assert "row_id" in first
    assert "timestamp" in first
    assert "predicted_probability" in first
    assert "decision" in first
    assert first["decision"] in ("FLAGGED_LAUNDERING", "CLEARED_LEGITIMATE", "SUSPICIOUS", "LEGITIMATE", "LOW_CONFIDENCE_NO_HISTORY")
    assert "predicted_label" in first
    assert first["predicted_label"] in ("SUSPICIOUS", "LEGITIMATE")
    assert len(first["sequence_features"]) == 10
    assert len(first["sequence_features"][0]) == 57


def test_evaluate_csv_li_medium_processed_with_labels():
    """Verify CSV evaluation of processed LI-Medium slice with ground truth labels."""
    csv_text = (BASE_DIR / "data/demo_samples/li_medium_sample_25.csv").read_text(encoding="utf-8")
    req = CsvEvaluateRequest(
        csv_content=csv_text,
        dataset="LI-Medium",
        filename="li_medium_sample_25.csv",
        row_limit=50,
    )
    res = evaluate_csv(req)

    assert res["status"] == "success"
    assert res["dataset"] == "LI-Medium"
    assert "LI-Medium Champion" in res["model_name"] or res.get("model_id") == "config_5_bilstm_fedavg_iid"
    assert res["decision_threshold"] == pytest.approx(0.23848299, abs=1e-5)
    assert res["evaluated_sequences"] == 25
    assert res["has_labels"] is True
    assert res["evaluation_metrics"] is not None


def test_evaluate_csv_raw_format():
    """Verify CSV evaluation of raw 11-column IBM AML transaction format."""
    csv_text = (BASE_DIR / "data/demo_samples/hi_small_raw_sample_25.csv").read_text(encoding="utf-8")
    req = CsvEvaluateRequest(
        csv_content=csv_text,
        dataset="HI-Small",
        filename="hi_small_raw_sample_25.csv",
        row_limit=50,
    )
    res = evaluate_csv(req)

    assert res["status"] == "success"
    assert res["dataset"] == "HI-Small"
    assert res["evaluated_sequences"] == 23
    assert len(res["predictions"]) == 23


def test_evaluate_csv_unlabeled_transactions():
    """Verify label-free prediction when 'Is Laundering' column is absent."""
    csv_text = (BASE_DIR / "data/demo_samples/hi_small_sample_25.csv").read_text(encoding="utf-8")
    # Drop the label column
    lines = csv_text.splitlines()
    header_cols = lines[0].split(",")
    label_idx = header_cols.index("is_laundering")
    unlabeled_lines = [
        ",".join(col for i, col in enumerate(line.split(",")) if i != label_idx)
        for line in lines
    ]
    unlabeled_csv = "\n".join(unlabeled_lines)

    req = CsvEvaluateRequest(
        csv_content=unlabeled_csv,
        dataset="HI-Small",
        filename="unlabeled_transactions.csv",
    )
    res = evaluate_csv(req)

    assert res["status"] == "success"
    assert res["has_labels"] is False
    assert res["evaluation_metrics"] is None
    assert len(res["predictions"]) == 25
    assert res["predictions"][0]["ground_truth_label"] is None


def test_evaluate_csv_empty_content_rejection():
    """Verify rejection of empty CSV content."""
    req = CsvEvaluateRequest(csv_content="", dataset="HI-Small")
    with pytest.raises(HTTPException) as exc:
        evaluate_csv(req)
    assert exc.value.status_code == 400
    assert "empty" in exc.value.detail.lower()


def test_evaluate_csv_missing_columns_rejection():
    """Verify rejection when required columns are missing."""
    invalid_csv = "colA,colB,colC\n1,2,3\n4,5,6"
    req = CsvEvaluateRequest(csv_content=invalid_csv, dataset="HI-Small", filename="bad.csv")
    with pytest.raises(HTTPException) as exc:
        evaluate_csv(req)
    assert exc.value.status_code == 400
    assert "missing required columns" in exc.value.detail.lower()


def test_evaluate_csv_invalid_dataset_rejection():
    """Verify rejection when dataset is not HI-Small or LI-Medium."""
    csv_text = (BASE_DIR / "data/demo_samples/hi_small_sample_25.csv").read_text(encoding="utf-8")
    req = CsvEvaluateRequest(csv_content=csv_text, dataset="Unknown-Dataset")
    with pytest.raises(HTTPException) as exc:
        evaluate_csv(req)
    assert exc.value.status_code == 400
    assert "unknown dataset" in exc.value.detail.lower()


def test_route_registration_aliases():
    """Verify that evaluate and predict routes match on root, /api, and /api/v1 prefixes."""
    from backend.interfaces.api.app import app
    from starlette.routing import Match

    paths_to_verify = [
        "/evaluate/csv",
        "/api/evaluate/csv",
        "/api/v1/evaluate/csv",
        "/predict",
        "/api/predict",
        "/api/v1/predict",
        "/health",
        "/api/health",
        "/api/v1/health",
    ]

    for p in paths_to_verify:
        method = "GET" if "health" in p else "POST"
        scope = {"type": "http", "method": method, "path": p, "headers": []}
        matched = any(r.matches(scope)[0] == Match.FULL for r in app.routes)
        assert matched is True, f"Route {p} must be registered and match in FastAPI app"


def test_evaluate_csv_no_positive_labels_edge_case():
    """Verify metric calculation when all labels are 0 (no positive laundering labels)."""
    csv_text = (BASE_DIR / "data/demo_samples/hi_small_sample_25.csv").read_text(encoding="utf-8")
    lines = csv_text.splitlines()
    header_cols = lines[0].split(",")
    label_idx = header_cols.index("is_laundering")

    all_zero_lines = [lines[0]]
    for line in lines[1:]:
        parts = line.split(",")
        parts[label_idx] = "0"
        all_zero_lines.append(",".join(parts))
    zero_csv = "\n".join(all_zero_lines)

    req = CsvEvaluateRequest(csv_content=zero_csv, dataset="HI-Small")
    res = evaluate_csv(req)

    assert res["status"] == "success"
    assert res["has_labels"] is True
    assert res["evaluation_metrics"] is not None
    metrics = res["evaluation_metrics"]
    assert metrics["positive_samples"] == 0
    assert metrics["negative_samples"] == 25
    assert metrics["precision"] in (0.0, None)
    assert metrics["recall"] == 0.0
    assert metrics["pr_auc"] is None
    assert metrics["roc_auc"] is None
    assert metrics["edge_case_note"] is not None
    assert "negative" in metrics["edge_case_note"].lower()


def test_evaluate_csv_no_negative_labels_edge_case():
    """Verify metric calculation when all labels are 1 (all positive laundering labels)."""
    csv_text = (BASE_DIR / "data/demo_samples/hi_small_sample_25.csv").read_text(encoding="utf-8")
    lines = csv_text.splitlines()
    header_cols = lines[0].split(",")
    label_idx = header_cols.index("is_laundering")

    all_one_lines = [lines[0]]
    for line in lines[1:]:
        parts = line.split(",")
        parts[label_idx] = "1"
        all_one_lines.append(",".join(parts))
    one_csv = "\n".join(all_one_lines)

    req = CsvEvaluateRequest(csv_content=one_csv, dataset="HI-Small")
    res = evaluate_csv(req)

    assert res["status"] == "success"
    assert res["has_labels"] is True
    assert res["evaluation_metrics"] is not None
    metrics = res["evaluation_metrics"]
    assert metrics["positive_samples"] == 25
    assert metrics["negative_samples"] == 0
    assert metrics["roc_auc"] is None
    assert metrics["edge_case_note"] is not None
    assert "positive" in metrics["edge_case_note"].lower()


def test_evaluate_csv_auto_pipeline_inference():
    """Verify that omitting dataset parameter automatically infers the default champion pipeline."""
    csv_text = (BASE_DIR / "data/demo_samples/hi_small_sample_25.csv").read_text(encoding="utf-8")
    req = CsvEvaluateRequest(
        csv_content=csv_text,
        filename="transactions_stream.csv",
        row_limit=50,
    )
    res = evaluate_csv(req)
    assert res["status"] == "success"
    assert res["dataset"] == "HI-Small"
    assert res["evaluated_sequences"] == 25
    assert len(res["predictions"]) == 25
    assert res["predictions"][0]["predicted_label"] in ("SUSPICIOUS", "LEGITIMATE")


def test_evaluate_csv_risk_distribution_deciles():
    """Verify that probability histogram distribution contains 10 deciles summing to total evaluated count."""
    csv_text = (BASE_DIR / "data/demo_samples/hi_small_sample_25.csv").read_text(encoding="utf-8")
    req = CsvEvaluateRequest(csv_content=csv_text, dataset="HI-Small")
    res = evaluate_csv(req)

    assert "risk_distribution" in res
    dist = res["risk_distribution"]
    assert len(dist) == 10
    total_in_bins = sum(b["count"] for b in dist)
    assert total_in_bins == res["evaluated_sequences"]
    assert dist[0]["bin_range"] == "0.0-0.1"
    assert dist[9]["bin_range"] == "0.9-1.0"


