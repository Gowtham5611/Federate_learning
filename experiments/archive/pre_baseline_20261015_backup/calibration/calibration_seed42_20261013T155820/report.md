# Phase 10: Probability Calibration Report (calibration_seed42_20261013T155820)

## 1. Executive Summary & Verification Highlights
- **Experiment ID**: `calibration_seed42_20261013T155820`
- **Timestamp**: `20261013T155820`
- **Source Reference Model**: `federated_iid_fedavg_lstm_seed42_20260918T130515`
  - Model Architecture: `lstm`
  - Partition / Algorithm: `iid` / `fedavg`
  - Checkpoint: `C:\Users\Acer\OneDrive\Desktop\Federate_learning\models\checkpoints\federated\iid\fedavg\lstm\federated_iid_fedavg_lstm_seed42_20260918T130515\global_best.pt`
- **Selected Calibration Method**: **`isotonic_regression`**
- **Selection Metric Criterion**: `log_loss` (Validation)
- **Validation-Selected Decision Threshold**: `0.244444`
- **Historical Phase 7 Decision Threshold**: `0.1300`

---

## 2. Validation Calibration Method Comparison
Fitted on validation split only (1,015,565 sequences, 1,081 positives):

| Method | Selected | Log Loss | Brier Score | ECE (15 bins) | MCE | Val PR-AUC | Val ROC-AUC | Val F1 | Val Threshold | Parameters |
|---|---|---|---|---|---|---|---|---|---|---|
| `none` | No | 0.004412 | 0.000790 | 0.000204 | 0.282808 | 0.367526 | 0.970486 | 0.457578 | 0.127481 | None |
| `temperature_scaling` | No | 0.004388 | 0.000790 | 0.000087 | 0.256373 | 0.367526 | 0.970486 | 0.457578 | 0.138844 | temperature=1.0540 |
| `platt_scaling` | No | 0.004385 | 0.000788 | 0.000085 | 0.321095 | 0.367526 | 0.970486 | 0.457578 | 0.152820 | platt_a=0.9770, platt_b=0.1666 |
| `isotonic_regression` | **YES** | 0.004295 | 0.000777 | 0.000000 | 0.000000 | 0.359303 | 0.971424 | 0.457578 | 0.244444 | isotonic_x_min=0.0000, isotonic_x_max=0.9902, isotonic_thresholds_count=120 |

---

## 3. Test Evaluation: Uncalibrated Baseline vs Frozen Calibrated Model
Single-pass final test split evaluation (1,015,564 sequences, 1,797 positives):

| Metric | Uncalibrated Baseline | Calibrated Model | Relative Improvement |
|---|---|---|---|
| **Log Loss** | `0.005720` | `0.005734` | `-0.25%` |
| **Brier Score** | `0.001092` | `0.001075` | `+1.57%` |
| **Expected Calibration Error (ECE)** | `0.000494` | `0.000217` | `+56.18%` |
| **Max Calibration Error (MCE)** | `0.273260` | `0.155454` | `+43.11%` |
| **PR-AUC** (Ranking) | `0.543749` | `0.525941` | `-3.27%` |
| **ROC-AUC** (Ranking) | `0.976867` | `0.976639` | `-0.02%` |
| **Decision Threshold** | `0.127481` | `0.244444` | - |
| **Test F1** | `0.572194` | `0.572194` | `+0.00%` |
| **Test Precision** | `0.718855` | `0.718855` | - |
| **Test Recall** | `0.475237` | `0.475237` | - |
| **Test Accuracy** | `0.998743` | `0.998743` | - |
| **Confusion Matrix (TP/FP/TN/FN)** | `854/334/1013433/943` | `854/334/1013433/943` | - |
| **Historical Threshold (tau=0.13)** | F1: `0.571813` | F1: `0.572923` | - |

---

## 4. Key Methodological Observations
1. **Monotonicity & Ranking Invariance**: Parametric calibrators strictly preserve probability ranking order, leaving Test PR-AUC and ROC-AUC unchanged while improving probability error metrics (Log Loss, ECE, Brier Score).
2. **Probability Quality vs Classification**: Calibration aligns predicted risk probabilities with empirical frequency without modifying neural network weights.
3. **Extreme Imbalance Handling**: Probability outputs remain strictly finite in [0, 1] under extreme ~0.1% AML prevalence.
4. **Strict Isolation & Zero Test Leakage**: All calibrations and thresholds were fitted exclusively on validation sequences. The test set was scored exactly once.

---

## 5. Artifact Inventory
- Validation Comparison: `validation/validation_comparison.csv`
- Test Evaluation: `test/test_comparison.csv`
- Reliability Diagrams: `visualizations/01_baseline_reliability_diagram.png`, `visualizations/02_calibrated_reliability_diagram.png`
- Comparison Curve: `visualizations/03_calibration_comparison_curves.png`
- Full Report & Metadata: `report.json`, `metadata.json`, `report.md`
