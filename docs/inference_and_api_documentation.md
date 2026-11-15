# FedSecureAI — REST API & Inference Engine Documentation

**Base URLs:**  
- Production / Sandboxed: `http://127.0.0.1:8000`  
- Endpoint Prefixes: Root `""`, `"/api/v1"`, and `"/api"` (all three supported identically)  
- Interactive OpenAPI Docs: `http://127.0.0.1:8000/docs`  

---

## 1. System Health & Metadata Endpoints

### `GET /health`
Verifies API availability, loaded model checkpoints, and runtime environment.

**Response (HTTP 200)**:
```json
{
  "status": "healthy",
  "app_name": "FedSecureAI Inference Service",
  "app_version": "1.0.0",
  "loaded_models": [
    "hi_small_lstm_fedavg_iid_seed42",
    "config_5_bilstm_fedavg_iid"
  ],
  "device": "cpu",
  "readiness": "ready",
  "timestamp": "2026-10-30T07:40:00Z"
}
```

---

### `GET /models` or `GET /api/v1/models`
Returns the active catalog of all 27 registered benchmark models and custom dynamic models.

**Response (HTTP 200)**:
```json
[
  {
    "model_id": "hi_small_lstm_fedavg_iid_seed42",
    "model_name": "HI-Small LSTM FedAvg (IID)",
    "architecture": "LSTM",
    "dataset": "HI-Small",
    "decision_threshold": 0.337784,
    "feature_columns": ["Amount Paid", "Amount Received", ...],
    "frozen": true,
    "is_custom": false
  }
]
```

---

## 2. Universal CSV Inspection & Compatibility

### `POST /api/v1/csv/inspect`
Inspects raw CSV content, auto-detects delimiter, encoding, semantic fields, and temporal sequence constructibility without running heavy model inference.

**Request Body**:
```json
{
  "csv_content": "Timestamp,From Bank,Account,To Bank,Account.1,Amount Received,Receiving Currency,Amount Paid,Payment Currency,Payment Format,Is Laundering\n...",
  "filename": "transactions_sample.csv"
}
```

**Response (HTTP 200)**:
```json
{
  "delimiter": ",",
  "encoding": "utf-8",
  "num_rows": 500,
  "num_cols": 11,
  "column_names": ["Timestamp", "From Bank", "Account", "To Bank", "Account.1", "Amount Received", "Receiving Currency", "Amount Paid", "Payment Currency", "Payment Format", "Is Laundering"],
  "target_column_name": "Is Laundering",
  "has_ground_truth": true,
  "can_construct_sequences": true,
  "semantic_mappings": {
    "Timestamp": "timestamp",
    "Account": "from_account",
    "Account.1": "to_account",
    "Amount Paid": "amount_paid"
  }
}
```

---

### `POST /api/v1/models/compatibility`
Verifies whether an uploaded CSV schema is directly compatible, adaptable, or incompatible with registered models.

**Request Body**:
```json
{
  "csv_content": "...",
  "model_id": "auto"
}
```

**Response (HTTP 200)**:
```json
{
  "results": [
    {
      "model_id": "HI-Small",
      "status": "COMPATIBLE",
      "missing_features": [],
      "extra_features": [],
      "compatibility_score": 1.0,
      "summary": "11/11 canonical features detected."
    }
  ]
}
```

---

## 3. High-Throughput Batch Prediction & Evaluation

### `POST /api/v1/predict/csv` or `POST /api/v1/evaluate/csv`
Executes end-to-end causal feature preprocessing, sequence generation, and neural model scoring.

**Request Body**:
```json
{
  "csv_content": "Timestamp,From Bank,Account,To Bank,Account.1,Amount Received,Receiving Currency,Amount Paid,Payment Currency,Payment Format,Is Laundering\n...",
  "model_id": "auto",
  "dataset": "HI-Small",
  "filename": "batch_october.csv",
  "row_limit": 5000,
  "enable_persistent_history": false,
  "tenant_id": "default_tenant"
}
```

**Response (HTTP 200)**:
```json
{
  "status": "success",
  "prediction_id": "pred_1730274000_a1b2c3d4",
  "filename": "batch_october.csv",
  "dataset": "HI-Small",
  "model_name": "hi_small_lstm_fedavg_iid_seed42",
  "decision_threshold": 0.337784,
  "total_rows": 500,
  "evaluated_sequences": 500,
  "distinct_accounts": 120,
  "flagged_laundering_count": 12,
  "cleared_count": 488,
  "has_labels": true,
  "evaluation_metrics": {
    "total_samples": 500,
    "positive_samples": 15,
    "negative_samples": 485,
    "accuracy": 0.984,
    "precision": 0.8333,
    "recall": 0.6667,
    "f1_score": 0.7407,
    "f2_score": 0.6944,
    "pr_auc": 0.7845,
    "roc_auc": 0.9620,
    "brier_score": 0.0124,
    "expected_calibration_error": 0.0084,
    "alerts_per_10k": 240.0,
    "confusion_matrix": {
      "tp": 10,
      "fp": 2,
      "fn": 5,
      "tn": 483
    }
  },
  "sequence_context_diagnostics": {
    "average_sequence_length": 6.8,
    "max_sequence_length": 10,
    "single_transaction_count": 35,
    "single_transaction_ratio": 0.07,
    "mean_padding_ratio": 0.32,
    "insufficient_history_warning": false,
    "operational_context_status": "HISTORICALLY_SUPPORTED"
  },
  "predictions": [
    {
      "row_id": 1,
      "from_account": "ACC_01",
      "to_account": "ACC_99",
      "amount": 1000.0,
      "predicted_probability": 0.0042,
      "decision": "CLEARED_LEGITIMATE",
      "decision_threshold": 0.337784,
      "ground_truth_label": 0,
      "sequence_length": 5
    }
  ]
}
```

---

## 4. Scientific Ensemble & Validation Tuning

### `POST /api/v1/ensemble/evaluate`
Sweeps continuous blending weight $\alpha \in [0.0, 1.0]$ evaluated strictly on centralized validation data.

**Request Body**:
```json
{
  "dataset": "HI-Small",
  "alpha_step": 0.10,
  "objective": "max_pr_auc",
  "cost_fp": 1.0,
  "cost_fn": 10.0,
  "min_recall": 0.80
}
```

**Response (HTTP 200)**:
```json
{
  "status": "success",
  "dataset": "HI-Small",
  "objective": "max_pr_auc",
  "best_alpha": 1.0,
  "frozen_threshold": 0.337784,
  "ensemble_improved": false,
  "scientific_rationale": "Scientific Finding for HI-Small: The Non-IID federated model suffered from severe client typology and volume skew... The objective optimization correctly converged to alpha*=1.00, pruning the degraded Non-IID stream.",
  "validation_metrics": {
    "alpha": 1.0,
    "threshold": 0.337784,
    "pr_auc": 0.242083,
    "f1": 0.5318
  },
  "test_metrics": {
    "pr_auc": 0.4760,
    "f1": 0.5359,
    "accuracy": 0.9988,
    "precision": 0.8183,
    "recall": 0.3984
  },
  "alpha_candidates": [
    {"alpha": 0.0, "val_pr_auc": 0.198980, "val_f1": 0.4891, "selected": false},
    {"alpha": 1.0, "val_pr_auc": 0.242083, "val_f1": 0.5318, "selected": true}
  ]
}
```

---

## 5. Mode B Dynamic Model Training

### `POST /api/v1/train/csv`
Trains a custom tabular model (`HistGradientBoosting` or `RandomForest`) on a user-uploaded labeled CSV for single-transaction scoring.

**Request Body**:
```json
{
  "csv_content": "...",
  "model_name": "Retail_Tabular_Model_v1",
  "model_type": "HistGradientBoosting"
}
```

**Response (HTTP 200)**:
```json
{
  "status": "success",
  "model_id": "dyn_retail_tabular_model_v1",
  "model_name": "Retail_Tabular_Model_v1",
  "model_type": "HistGradientBoosting",
  "frozen_threshold": 0.4500,
  "training_rows": 10000,
  "metrics": {
    "val_pr_auc": 0.6521,
    "val_f1": 0.5840
  }
}
```

---

## 6. Investigation Export & Download

### `POST /api/v1/investigation/report`
Generates a Suspicious Activity Investigation Report (SAR draft) in JSON or CSV format.

### `GET /api/v1/predictions/{prediction_id}/download`
Downloads the scored batch as a fully aligned CSV file containing original columns plus predicted probabilities and decisions.

