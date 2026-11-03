"""Unit test verifying zero future data leakage for TemporalGraphFeatureExtractor."""

import numpy as np
import pandas as pd
import pytest

from backend.infrastructure.data.features.temporal_graph_features import TemporalGraphFeatureExtractor


def test_temporal_graph_zero_future_leakage():
    """Verify that appending or altering future transactions NEVER alters features of past transactions."""
    # Create base chronological transaction history
    base_data = {
        "timestamp": [
            "2022/09/01 08:00:00",
            "2022/09/01 08:30:00",
            "2022/09/01 09:00:00",
            "2022/09/01 10:00:00",
            "2022/09/01 11:00:00",
        ],
        "from_account": ["acc_A", "acc_B", "acc_A", "acc_C", "acc_B"],
        "to_account": ["acc_B", "acc_C", "acc_C", "acc_A", "acc_A"],
        "amount": [1000.0, 500.0, 1500.0, 200.0, 1000.0],
        "Payment Currency": ["US Dollar", "US Dollar", "Euro", "US Dollar", "US Dollar"],
        "Receiving Currency": ["US Dollar", "US Dollar", "US Dollar", "US Dollar", "US Dollar"],
    }
    df_base = pd.DataFrame(base_data)

    extractor_1 = TemporalGraphFeatureExtractor()
    feat_base = extractor_1.extract_features(df_base)

    # Now create an extended dataframe with 5 additional future transactions
    future_data = {
        "timestamp": [
            "2022/09/01 12:00:00",
            "2022/09/01 13:00:00",
            "2022/09/01 14:00:00",
            "2022/09/01 15:00:00",
            "2022/09/01 16:00:00",
        ],
        "from_account": ["acc_A", "acc_A", "acc_B", "acc_C", "acc_A"],
        "to_account": ["acc_D", "acc_E", "acc_F", "acc_B", "acc_C"],
        "amount": [99999.0, 88888.0, 77777.0, 66666.0, 55555.0],
        "Payment Currency": ["US Dollar"] * 5,
        "Receiving Currency": ["Bitcoin"] * 5,
    }
    df_future = pd.DataFrame(future_data)
    df_extended = pd.concat([df_base, df_future], ignore_index=True)

    extractor_2 = TemporalGraphFeatureExtractor()
    feat_extended = extractor_2.extract_features(df_extended)

    # Assert exact bitwise or floating equality for the first 5 rows
    for col in feat_base.columns:
        np.testing.assert_allclose(
            feat_base[col].to_numpy(),
            feat_extended[col].iloc[:5].to_numpy(),
            rtol=1e-6,
            atol=1e-6,
            err_msg=f"Future data leakage detected in feature '{col}'!",
        )


def test_two_hop_cycle_detection():
    """Verify cycle_two_hop_indicator activates only after return transaction."""
    cycle_data = {
        "timestamp": [
            "2022/09/01 08:00:00",  # A -> B
            "2022/09/01 09:00:00",  # B -> C
            "2022/09/01 10:00:00",  # B -> A (Return transfer: 2-hop loop!)
            "2022/09/01 11:00:00",  # C -> D
        ],
        "from_account": ["acc_A", "acc_B", "acc_B", "acc_C"],
        "to_account": ["acc_B", "acc_C", "acc_A", "acc_D"],
        "amount": [1000.0, 1000.0, 1000.0, 1000.0],
    }
    df = pd.DataFrame(cycle_data)
    extractor = TemporalGraphFeatureExtractor()
    feat = extractor.extract_features(df)

    # Row 0: A -> B (no previous B -> A) -> 0
    assert feat["cycle_two_hop_indicator"].iloc[0] == 0.0
    # Row 1: B -> C (no previous C -> B) -> 0
    assert feat["cycle_two_hop_indicator"].iloc[1] == 0.0
    # Row 2: B -> A (A previously sent to B at 08:00) -> 1.0!
    assert feat["cycle_two_hop_indicator"].iloc[2] == 1.0
    # Row 3: C -> D (no previous D -> C) -> 0
    assert feat["cycle_two_hop_indicator"].iloc[3] == 0.0
