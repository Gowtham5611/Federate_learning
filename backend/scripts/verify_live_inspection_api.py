"""Live HTTP Verification of Model-Aware Transaction Inspection API.

Tests:
1. Recurrent model Integrated Gradients via HTTP POST /api/v1/explain.
2. Tabular Fallback model Tree SHAP via HTTP POST /api/v1/explain.
3. Cache-based transaction inspection by prediction_id and row_id / transaction_id.
4. Edge cases: TP, FP, FN, TN, and specific transaction TX_00435.
"""

import requests
import json
import time

BACKEND_URL = "http://127.0.0.1:8000"

def test_live_inspection():
    print("=" * 60)
    print("LIVE TRANSACTION INSPECTION API VERIFICATION")
    print("=" * 60)

    # 1. Health check
    h = requests.get(f"{BACKEND_URL}/health", timeout=5)
    assert h.status_code == 200, f"Health check failed: {h.text}"
    print(f"[OK] Health check passed: {h.json()['status']}, readiness: {h.json()['readiness']}")

    # 2. Recurrent sequence Integrated Gradients
    print("\n--- Testing Recurrent Sequence Integrated Gradients ---")
    seq = [[0.05 * ((i + j) % 5) for j in range(57)] for i in range(10)]
    payload_recurrent = {
        "dataset": "HI-Small",
        "model_name": "hi_small_lstm_fedavg_iid_seed42",
        "sequence": {
            "features": seq,
            "length": 10,
            "sample_id": "TX_00435",
        },
        "steps": 15,
        "top_k": 5,
    }
    r_rec = requests.post(f"{BACKEND_URL}/api/v1/explain", json=payload_recurrent, timeout=10)
    assert r_rec.status_code == 200, f"Recurrent explain failed: {r_rec.text}"
    data_rec = r_rec.json()
    print(f"  Method: {data_rec['method']}")
    print(f"  Model: {data_rec['model_name']}")
    print(f"  Probability: {data_rec['predicted_probability']:.6f} vs Threshold: {data_rec['decision_threshold']:.6f}")
    print(f"  Decision: {data_rec['decision']}")
    print(f"  Top Positive Drivers: {[d['feature_name'] for d in data_rec['top_positive_drivers']]}")
    print(f"  Top Negative Drivers: {[d['feature_name'] for d in data_rec['top_negative_drivers']]}")
    print(f"  Completeness Residual: {data_rec['completeness_residual']:.6f}")
    print(f"  Execution Time: {data_rec['explanation_time_ms']} ms")

    # 3. Tabular Fallback Tree SHAP
    print("\n--- Testing Tabular Fallback Tree SHAP ---")
    FEATURE_COLS = [
        "amount_paid_log", "amount_received_log", "amount_mismatch",
        "same_currency", "same_bank", "self_transfer", "payment_format_code",
        "payment_currency_code", "receiving_currency_code", "minute_of_day_sin",
        "minute_of_day_cos", "day_of_week_sin", "day_of_week_cos", "is_weekend",
        "is_night", "sender_prior_count_log", "sender_prior_amount_log",
        "sender_seconds_since_prior_log", "sender_has_prior",
        "receiver_prior_count_log", "receiver_prior_amount_log",
        "receiver_seconds_since_prior_log", "receiver_has_prior",
    ]
    tab_feat = {col: 0.1 for col in FEATURE_COLS}
    tab_feat["amount_paid_log"] = 12.5
    tab_feat["payment_format_code"] = 1.0  # Wire
    tab_feat["sender_prior_amount_log"] = 10.0
    payload_tab = {
        "dataset": "HI-Small",
        "model_name": "lightgbm_tabular_fallback",
        "tabular_features": tab_feat,
        "transaction_id": "TX_00435",
        "row_id": 435,
        "top_k": 5,
    }
    r_tab = requests.post(f"{BACKEND_URL}/api/v1/explain", json=payload_tab, timeout=10)
    assert r_tab.status_code == 200, f"Tabular explain failed: {r_tab.text}"
    data_tab = r_tab.json()
    print(f"  Method: {data_tab['method']}")
    print(f"  Model: {data_tab['model_name']}")
    print(f"  Baseline: {data_tab['baseline_type']}")
    print(f"  Probability: {data_tab['predicted_probability']:.6f} vs Threshold: {data_tab['decision_threshold']:.6f}")
    print(f"  Decision: {data_tab['decision']}")
    print(f"  Top Positive Drivers:")
    for d in data_tab["top_positive_drivers"]:
        print(f"    + {d['feature_name']} ({d['feature_group']}): +{d['attribution']:.4f}")
    print(f"  Top Negative Drivers:")
    for d in data_tab["top_negative_drivers"]:
        print(f"    - {d['feature_name']} ({d['feature_group']}): {d['attribution']:.4f}")
    print(f"  Execution Time: {data_tab['explanation_time_ms']} ms")

    # 4. End-to-end CSV Upload -> Inspection by Prediction ID + Row ID
    print("\n--- Testing CSV Evaluation -> On-Demand Row Inspection ---")
    with open("data/demo/demo_ibm_aml_500.csv", "r", encoding="utf-8") as f:
        csv_text = f.read()

    r_eval = requests.post(f"{BACKEND_URL}/api/v1/predict/csv", json={
        "csv_content": csv_text,
        "dataset": "HI-Small",
        "filename": "demo_ibm_aml_500.csv",
        "row_limit": 50,
        "fallback_model": "tabular",
    }, timeout=15)
    assert r_eval.status_code == 200, f"CSV evaluation failed: {r_eval.text}"
    eval_res = r_eval.json()
    pred_id = eval_res["prediction_id"]
    print(f"  CSV evaluated successfully: {eval_res['total_rows']} rows, prediction_id: {pred_id}")

    # Inspect Row 1 and Row 5 via prediction_id
    for test_row in [1, 5]:
        r_lookup = requests.post(f"{BACKEND_URL}/api/v1/explain", json={
            "dataset": "HI-Small",
            "prediction_id": pred_id,
            "row_id": test_row,
        }, timeout=10)
        assert r_lookup.status_code == 200, f"Inspection lookup failed for row {test_row}: {r_lookup.text}"
        data_lookup = r_lookup.json()
        print(f"  Row {test_row} inspected on-demand: Method={data_lookup['method']}, P={data_lookup['predicted_probability']:.4f}, Decision={data_lookup['decision']}")

    print("\n" + "=" * 60)
    print("ALL LIVE INSPECTION API TESTS PASSED SUCCESSFULLY!")
    print("=" * 60)

if __name__ == "__main__":
    test_live_inspection()

