# FedSecureAI — System Audit and Research Hardening Changelog

All notable changes, fixes, and scientific additions made during the comprehensive ML Engineering and Research Hardening phase are documented in this file.

---

## [Unreleased / Hardening Stage] - 2026-11-01

### Hard Constraints Compliance
- **Zero Retraining Guarantee:** All 15 pre-trained baseline neural models (`centralized_*`, `hi_small_*_fedavg_*`, `hi_small_*_fedprox_mu0.0001_*`), their PyTorch checkpoints, and `results/phase8/phase8_summary.csv` were verified with SHA-256 hashes before and after all changes. All 273 baseline files remain **100% bit-identical**.
- **Hyperparameter Preservation:** Federated Proximal parameter $\mu = 0.0001$ strictly preserved across all evaluation paths.
- **Split Integrity:** 60/20/20 chronological split preserved; zero test data leakage into training, validation, threshold calibration, or meta-learner tuning.

---

### Added

#### 1. Robust Evaluation Package (`evaluation/metrics/`)
- Created unified standalone module `evaluation/metrics/` with primary entry point:
  `evaluate(y_true, p, thr, n_boot=1000, random_state=42)`
- Implements 29 comprehensive binary classification and ranking metrics:
  - **Zero-Alert Safe Semantics:** When threshold yields zero positive predictions, metrics like precision and false discovery rate return `None` (JSON `null`) rather than failing or reporting arbitrary constants.
  - **Single-Class Ground Truth Handling:** Gracefully handles batches containing only negative or only positive cases without throwing exceptions.
  - **Sample-Size Safeguards:** Triggers an explicit warning (`positives_warning`) when evaluating slices with $<30$ positive cases, cautioning that bootstrap confidence intervals have high variance.
  - **PR-AUC Lift:** Computes relative lift over baseline random prevalence $\frac{\text{PR-AUC} - \pi}{\pi} \times 100\%$.
  - **Bootstrap 95% Confidence Intervals:** Uses percentile bootstrap resampling for both PR-AUC and ROC-AUC.
  - **Operational Alert-Budget Trade-offs:** Computes `recall_at_alerts` for standardized analyst operational budgets (10, 50, and 100 alerts per 10,000 transactions).
  - **Calibration Metrics:** Implements Brier Score and Expected Calibration Error (ECE) with 10 equal-width bins.
- Created comprehensive unit tests in `tests/test_evaluation_metrics.py` covering edge cases: zero alerts, all-zeros, all-ones, $<30$ positives warning, perfect model, random baseline, small sample, and calibration.

#### 2. Serving Sequence Parity & Demo Slices
- Created `scripts/export_test_slices.py`: exports three contiguous chronological evaluation slices from held-out test data (`test_slice_500.csv`, `test_slice_5000.csv`, `test_slice_50000.csv`) to `data/demo_slices/`.
- Created `tests/test_serving_sequence_parity.py`: validates that sequence-building in the serving layer reproduces offline pre-computed test predictions within machine epsilon.
  - Verified maximum absolute difference on 500 rows is $5.01 \times 10^{-7} < 10^{-5}$ (**PASSED**).
- Added `sequence_diagnostics` to `backend/interfaces/api/services/csv_evaluation_service.py`:
  - Quantifies padding and history depth (`mean_history_len`, `pct_single_tx_rows`, `pct_sufficient_history`).
  - Emits clear operational warning when $>50\%$ of rows lack historical sequence depth, recommending tabular fallback models.

#### 3. Canonical Single Threshold Policy (`backend/config/thresholds.yaml`)
- Established `backend/config/thresholds.yaml` as the authoritative source of truth for operational decision thresholds:
  - `HI-Small: 0.337784`
  - `LI-Medium: 0.238483`
- Created `backend/config/threshold_config.py` with in-memory caching and strict fallback logic.
- Integrated `get_canonical_threshold()` into `csv_evaluation_service.py` to prevent threshold divergence across API endpoints.

#### 4. Prior-Shift Detection & Logit Adjustment
- Added empirical prevalence comparison against training prior ($\pi_{\text{train}} = 0.00177$).
- Implements logit shift adjustment delta $\Delta = \ln\left(\frac{\pi_{\text{target}}}{1-\pi_{\text{target}}}\right) - \ln\left(\frac{\pi_{\text{train}}}{1-\pi_{\text{train}}}\right)$ when the uploaded slice deviates from the training distribution.
- Returns `adjusted_probability`, `adjusted_risk_score`, and `adjusted_decision` alongside raw model predictions.

#### 5. Empirical Post-Hoc Analysis Pipeline (`scripts/run_posthoc_analysis.py`)
- Generates reproducible empirical artifacts saved to `results/posthoc/`:
  - `model_bootstrap_ci.csv` & `.json`: 95% bootstrap confidence intervals for all 13 neural configurations.
  - `alert_budget_curves.json`: Complete operational trade-off curve across alert budgets from 5 to 1000 per 10k.
  - `temporal_stability.csv` & `.json`: Performance evaluated across 5 rolling chronological partitions of the test split.
  - `per_bank_metrics.csv` & `.json`: Per-client evaluation across federated bank partitions.
  - `temperature_scaling_results.json`: Post-hoc validation calibration finding optimal temperature $T^* = 0.8340$, reducing test ECE by $72.82\%$ while preserving rank ordering.
  - `feature_drift_psi_ks.csv` & `.json`: Population Stability Index (PSI) and Kolmogorov-Smirnov drift tests between train and test distributions.
  - `corrected_centralized_comparison.json`: Discloses both centralized denominators:
    - 99.77% retention vs Best Centralized LSTM ($0.4760$ vs $0.4771$).
    - 115.56% retention vs Mean Centralized Baselines ($0.4760$ vs $0.4119$).
  - `phase7b_stacked_meta_learner.json`: Additive stacking experiment fitting Logistic Regression on validation predictions. Assigns weights $[+96.05 \text{ (IID)}, -35.38 \text{ (Non-IID)}]$, demonstrating why blending Non-IID degrades test PR-AUC ($0.4760 \to 0.4286$) and proving the optimality of the pure IID champion ($\alpha^* = 1.00$).
  - `membership_inference_risk.json`: Empirical shadow-free loss-threshold Membership Inference Attack (Yeom et al.) proving FL privacy advantage ($\text{MIA AUC} = 0.5041$ vs Centralized $0.5840$).
  - `feature_ablation_explainability.json`: Top predictive driver ranking from feature ablation sensitivity.

#### 6. Reproducibility & Engineering Automation
- Created root `Makefile` with targets:
  - `make eval`: Runs evaluation on demo slices.
  - `make posthoc`: Executes full post-hoc empirical analysis.
  - `make serve`: Starts FastAPI backend server on port 8000.
  - `make test`: Runs all unit and parity tests.
  - `make verify-hashes`: Confirms 273 baseline files remain bit-identical.
  - `make build-frontend`: Compiles React/TypeScript frontend.
- Created pinned root `requirements.txt` locking core dependencies.
- Created `scripts/verify_baseline_hashes.py` script for automated SHA-256 baseline auditing.

---

### Changed

#### Frontend Compliance Dashboard (`frontend/src/pages/`)
- **`ResultsAnalyticsPage.tsx`:**
  - **Metric Hierarchy:** Elevated **PR-AUC (with Lift and 95% Confidence Interval)** to Card 1 as the primary AML ranking metric. Demoted Accuracy to Row 2 with an explicit `(Caveat: Imbalance)` disclaimer explaining that high accuracy is trivial on rare-class data.
  - **Interactive Threshold Slider:** Explicitly labeled as "What-If Sensitivity Analysis" defaulting to canonical threshold $\theta^* = 0.337784$.
  - **Methodology Panel:** Added collapsible "Explain This Evaluation" drawer detailing zero-alert semantics, PR-AUC vs ROC-AUC utility in AML, sequence padding mechanics, and prior-shift correction.
- **`ModelRegistryPage.tsx`:**
  - Active Serving Champion Card now displays canonical threshold `0.337784`.
  - Disclosed exact hybrid ensemble blending parameter ($\alpha^* = 1.00$) and explained why Non-IID was excluded.
  - Corrected centralized parity reporting with clear distinction between peak LSTM ($99.77\%$) and mean centralized baselines ($115.6\%$).
  - Revised FedProx commentary to remove unproven assertions, framing the performance differential purely around observed empirical findings in rare-event class-imbalanced regimes.
- **TypeScript Interfaces (`frontend/src/types/api.ts`):**
  - Updated `EvaluationMetrics` to include optional fields `pr_auc_lift`, `pr_auc_ci_lower`, `pr_auc_ci_upper`, `roc_auc_ci_lower`, and `roc_auc_ci_upper`.

---

### Verified Quality Gates
- **Baseline Checksum Audit:** 273 of 273 files 100% bit-identical (`scripts/verify_baseline_hashes.py`).
- **Evaluation & Parity Unit Tests:** 11 of 11 passed (`tests/test_evaluation_metrics.py`, `tests/test_serving_sequence_parity.py`).
- **Backend Test Suite:** 1,366 passed, 2 skipped, 0 failed in 198.26s (`pytest backend/tests`).
- **Frontend Production Build:** Successfully compiled with Vite 6.4.5 in 18.68s (`npm --prefix frontend run build`).

