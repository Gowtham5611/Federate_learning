"""End-to-end integration and contract tests for Phase 14B AML Analyst Dashboard."""

from __future__ import annotations

import json
from pathlib import Path
import re
import numpy as np
import pytest

try:
    from interfaces.api.schemas.common import (
        FEATURE_DIMENSION,
        FROZEN_THRESHOLDS,
        MAX_SEQUENCE_LENGTH,
        SequenceItem,
    )
    from interfaces.api.schemas.predict import PredictRequest, PredictResponse
    from interfaces.api.schemas.explain import ExplainRequest, ExplainResponse
    from interfaces.api.services.model_service import ModelService
    from interfaces.api.services.explain_service import ExplanationService
    from interfaces.api.app import create_app
except ModuleNotFoundError:
    from backend.interfaces.api.schemas.common import (
        FEATURE_DIMENSION,
        FROZEN_THRESHOLDS,
        MAX_SEQUENCE_LENGTH,
        SequenceItem,
    )
    from backend.interfaces.api.schemas.predict import PredictRequest, PredictResponse
    from backend.interfaces.api.schemas.explain import ExplainRequest, ExplainResponse
    from backend.interfaces.api.services.model_service import ModelService
    from backend.interfaces.api.services.explain_service import ExplanationService
    from backend.interfaces.api.app import create_app

BASE_DIR = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module", autouse=True)
def preload_models():
    ms = ModelService()
    ms.load_all_models()
    return ms


def load_demo_samples_from_ts() -> list:
    ts_file = BASE_DIR / "frontend/src/demo/demoSamples.ts"
    assert ts_file.exists(), "demoSamples.ts must exist"
    content = ts_file.read_text(encoding="utf-8")
    
    # Extract JSON array between [ ... ];
    match = re.search(r"export const DEMO_SAMPLES: DemoSample\[\] = (\[.*\]);", content, re.DOTALL)
    assert match, "Could not parse DEMO_SAMPLES array from TypeScript source"
    return json.loads(match.group(1))


# 1. Frontend Build Artifacts Test
def test_frontend_production_build_exists():
    dist_index = BASE_DIR / "frontend/dist/index.html"
    assert dist_index.exists(), "frontend/dist/index.html must exist from build"
    html = dist_index.read_text(encoding="utf-8")
    assert "FedSecureAI" in html
    assert "/assets/index-" in html


# 2. Demo Samples Schema Conformance Tests
def test_demo_samples_schema_conformance():
    samples = load_demo_samples_from_ts()
    assert len(samples) >= 4, "Must contain at least 4 demonstration samples"

    for sample in samples:
        assert sample["dataset"] in ("HI-Small", "LI-Medium")
        seq = sample["sequence"]
        features = seq["features"]
        length = seq["length"]

        assert 1 <= length <= MAX_SEQUENCE_LENGTH
        assert len(features) == MAX_SEQUENCE_LENGTH
        for t_idx, step in enumerate(features):
            assert len(step) == FEATURE_DIMENSION
            if t_idx >= length:
                assert all(v == 0.0 for v in step), f"Padding step {t_idx} must be zero-filled"

        # Pydantic validation
        validated_item = SequenceItem(features=features, length=length, sample_id=seq["sample_id"])
        assert validated_item is not None


# 3. Demo Samples Real Backend Prediction Accuracy Test
def test_demo_samples_backend_prediction():
    samples = load_demo_samples_from_ts()
    ms = ModelService()

    for sample in samples:
        dataset = sample["dataset"]
        seq_item = SequenceItem(**sample["sequence"])
        prob, thresh, decision, latency = ms.predict_sequence(dataset, seq_item)

        assert 0.0 <= prob <= 1.0
        assert thresh == FROZEN_THRESHOLDS[dataset]
        assert latency > 0

        # Validate that the sample label matches expected risk profile
        if "legit" in sample["id"]:
            assert prob < thresh, f"Sample {sample['id']} expected to clear threshold {thresh}, got {prob}"
            assert decision == "CLEARED_LEGITIMATE"
        elif "alert" in sample["id"]:
            assert prob >= thresh, f"Sample {sample['id']} expected to flag laundering above {thresh}, got {prob}"
            assert decision == "FLAGGED_LAUNDERING"


# 4. Demo Samples Real Integrated Gradients Explanation Test
def test_demo_samples_integrated_gradients_explanation():
    samples = load_demo_samples_from_ts()
    exp_service = ExplanationService()

    # Test attribution on the laundering alert sample
    alert_sample = next(s for s in samples if "alert" in s["id"])
    seq_item = SequenceItem(**alert_sample["sequence"])
    req = ExplainRequest(
        dataset=alert_sample["dataset"],
        sequence=seq_item,
        steps=25,
        top_k=5,
    )

    resp = exp_service.explain_sequence(req)

    assert resp.method == "integrated_gradients"
    assert resp.steps == 25
    assert len(resp.top_positive_drivers) > 0
    assert len(resp.top_negative_drivers) > 0
    assert resp.completeness_residual < 0.05
    assert resp.explanation_time_ms > 0

    # Ensure drivers have semantic group annotations
    for d in resp.top_positive_drivers:
        assert d.feature_group in (
            "Transaction Attributes",
            "Payment Format",
            "Currency Rails",
            "Temporal Dynamics",
            "Sender Causal History",
            "Receiver Causal History",
            "Other",
        )
        assert d.attribution > 0


# 5. Privacy & Zero-PII Audit in Demonstration Samples
def test_demo_samples_zero_pii_audit():
    samples = load_demo_samples_from_ts()
    for sample in samples:
        # Verify sample_id is synthetic/demo tagged
        assert "DEMO_" in sample["sequence"]["sample_id"]
        # Ensure no bank account number format or PII string
        desc = sample["description"]
        assert "Account" not in desc or "IBM AML" in desc
        assert not re.search(r"\b\d{10,16}\b", desc), "Description must not contain account numbers"


# 6. CORS Configuration Audit for Frontend Origins
def test_backend_cors_configuration():
    app = create_app(preload_models=False)
    # Inspect middleware for CORSMiddleware
    from starlette.middleware.cors import CORSMiddleware
    cors_mw = next((mw for mw in app.user_middleware if mw.cls is CORSMiddleware), None)
    assert cors_mw is not None, "CORSMiddleware must be configured in FastAPI app"

    allow_origins = cors_mw.kwargs.get("allow_origins", [])
    assert "http://localhost:3000" in allow_origins or "http://127.0.0.1:3000" in allow_origins


# 7. Frontend Defensive Null Safety Audit
def test_frontend_null_safety_in_aml_evaluation():
    aml_page = BASE_DIR / "frontend/src/pages/AmlEvaluationPage.tsx"
    assert aml_page.exists()
    content = aml_page.read_text(encoding="utf-8")

    # Ensure no unguarded Object.entries(inspectRes.semantic_mappings)
    assert "Object.entries(inspectRes.semantic_mappings)" not in content
    assert "Object.entries(detectedMap)" in content

    # Ensure no unguarded Object.entries(inspection.semantic_mappings)
    assert "Object.entries(inspection.semantic_mappings)" not in content
    assert "Object.entries(semanticMappingsObj)" in content

    # Ensure no unguarded Object.keys(customMappings).length
    assert "Object.keys(customMappings)." not in content


# 8. Results Analytics Page Safe Metric Formatting Audit
def test_results_analytics_page_safe_formatting_audit():
    analytics_page = BASE_DIR / "frontend/src/pages/ResultsAnalyticsPage.tsx"
    assert analytics_page.exists()
    content = analytics_page.read_text(encoding="utf-8")

    # Ensure no raw unguarded calls like metrics.brier_score.toFixed
    assert "metrics.brier_score.toFixed" not in content
    assert "metrics.alerts_per_10k.toFixed" not in content
    assert "metrics.f1_score.toFixed" not in content
    assert "metrics.accuracy * 100).toFixed" not in content

    # Ensure formatMetric, formatPercent, and formatRatioPercent are imported and used
    assert "formatMetric" in content
    assert "formatPercent" in content
    assert "formatRatioPercent" in content


# 9. Verify Backend Calculation Includes Brier Score and Alert Rate
def test_backend_metrics_calculation_includes_brier_and_alerts():
    from interfaces.api.services.csv_evaluation_service import CsvEvaluationService
    service = CsvEvaluationService()

    labels = np.array([0, 1, 0, 1, 0, 0, 0, 1])
    probs = np.array([0.1, 0.9, 0.2, 0.85, 0.05, 0.15, 0.3, 0.75])
    threshold = 0.5

    metrics = service._calculate_metrics(labels, probs, threshold)

    assert "brier_score" in metrics
    assert "alerts_per_10k" in metrics
    assert isinstance(metrics["brier_score"], float)
    assert isinstance(metrics["alerts_per_10k"], float)
    assert 0.0 <= metrics["brier_score"] <= 1.0
    assert metrics["alerts_per_10k"] > 0.0


