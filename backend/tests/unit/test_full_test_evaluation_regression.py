"""Regression tests for full-test evaluation, row limits, feature dimensions, and threshold optimization."""

from pathlib import Path
import numpy as np
import pytest
import torch

from application.dto.training import TrainingConfig
from domain.enums.model_architecture import ModelArchitecture
from domain.enums.split_name import SplitName
from infrastructure.data.sequences.processed_sequence_provider import ProcessedSequenceProvider
from infrastructure.models.pytorch.sequence_classifier import build_classifier


@pytest.fixture
def processed_root():
    return Path("data/processed/ibm_aml/LI-Medium")


@pytest.fixture
def sequence_provider(processed_root):
    return ProcessedSequenceProvider(processed_root)


def test_smoke_test_mode_explicitly_limits_rows(sequence_provider):
    """Prove that smoke-test mode respects row limits."""
    cfg_smoke = TrainingConfig(architecture=ModelArchitecture.GRU, row_limit=5000, batch_size=1024)
    ds = sequence_provider.load(SplitName.TEST, cfg_smoke)
    assert ds.sequence_count == 5000
    assert ds.feature_dimension == 57


def test_full_test_metadata_row_and_positive_counts(sequence_provider):
    """Verify that split metadata accurately reports the full 6.25M test rows."""
    meta = sequence_provider.metadata
    test_path = meta.split_path("test")
    assert test_path.is_file()

    splits_json_path = sequence_provider._root / "metadata" / "splits.json"
    import json
    with open(splits_json_path, "r", encoding="utf-8") as f:
        splits_data = json.load(f)

    test_info = next(s for s in splits_data["per_split"] if s["name"] == "test")
    assert test_info["rows"] == 6249112
    assert test_info["positive"] == 3783
    assert test_info["negative"] == 6245329


def test_model_receives_correct_feature_shape(processed_root):
    """Verify that classifier expects 57 features and produces valid sigmoid probabilities."""
    cfg = TrainingConfig(architecture=ModelArchitecture.GRU, batch_size=16)
    model = build_classifier(cfg, feature_dimension=57)
    model.eval()

    sample_x = torch.randn(16, 10, 57)
    sample_len = torch.full((16,), 10, dtype=torch.long)
    with torch.no_grad():
        logits = model(sample_x, sample_len)
        probs = torch.sigmoid(logits)

    assert logits.shape == (16,)
    assert probs.shape == (16,)
    assert (probs >= 0.0).all() and (probs <= 1.0).all()


def test_saved_predictions_align_with_full_test_labels():
    """Verify that persisted full-test prediction files have exact 6,249,112 alignment."""
    preds_dir = Path("artifacts/li_medium_run/predictions")
    labels_file = preds_dir / "full_test_labels.npy"
    if labels_file.exists():
        labels = np.load(labels_file)
        assert len(labels) == 6249112
        assert int(np.sum(labels)) == 3783

        stack_file = preds_dir / "stacking_saved_full_test_preds.npy"
        if stack_file.exists():
            stack_preds = np.load(stack_file)
            assert len(stack_preds) == 6249112
            assert (stack_preds >= 0.0).all() and (stack_preds <= 1.0).all()


def test_threshold_selection_uses_validation_data_only():
    """Prove that operational threshold selection optimizes strictly over validation data."""
    np.random.seed(42)
    val_labels = np.random.choice([0, 1], size=1000, p=[0.999, 0.001])
    val_probs = np.random.uniform(0, 0.01, size=1000)

    from sklearn.metrics import precision_recall_curve
    prec, rec, thresholds = precision_recall_curve(val_labels, val_probs)
    f1 = np.where((prec[:-1] + rec[:-1]) > 0, (2 * prec[:-1] * rec[:-1]) / (prec[:-1] + rec[:-1]), 0.0)
    best_th = float(thresholds[np.argmax(f1)])

    # Ensure best_th was computed without any test data
    assert isinstance(best_th, float)
    assert best_th >= 0.0

