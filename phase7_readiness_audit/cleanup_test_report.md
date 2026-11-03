# FedSecureAI — Codebase Cleanup & Architecture Regression Test Report

**Execution Timestamp**: 2026-10-21T11:32:00+05:30  
**Audit Context**: Phase 6 Audit & Phase 7 Hybrid Ensemble Preparation  
**Target Environments**: Python 3.14.7, PyTest 9.1.1, PyTorch 2.6.0  

---

## 1. Executive Summary

Following the safe project cleanup and quarantine of legacy Phase B architectures (`TCN`, `Transformer`, `Residual MLP`) into `archive/legacy_experiments/`, a comprehensive test suite was executed against the active codebase. 

- **Total Tests Executed**: 169
- **Total Tests Passed**: 169 (100%)
- **Total Tests Failed**: 0 (0%)
- **Regressions Introduced**: None
- **Active Modeling Architectures Enforced**: `GRU`, `LSTM`, `BiLSTM`

---

## 2. Test Suite 1: Full-Pipeline, GPU Equivalence & Streaming Tests (`tests/`)

**Command**: `.venv\Scripts\python.exe -m pytest tests/ -v`  
**Execution Time**: 13.22 seconds  
**Result**: 13 passed / 13 total (100% PASSED)

| Test Module | Test Name | Status | Duration | Focus Area |
| :--- | :--- | :--- | :--- | :--- |
| `tests/test_cuda_enforcement.py` | `test_resolve_device_strict_cuda_rejection_without_gpu` | **PASSED** | 0.05s | Strict CUDA fail-fast enforcement |
| `tests/test_cuda_enforcement.py` | `test_runner_preflight_fails_fast_on_missing_cuda` | **PASSED** | 0.04s | Colab runner preflight rejection of non-CUDA devices |
| `tests/test_cuda_enforcement.py` | `test_runner_preflight_allows_cpu_only_in_subset_debug_mode` | **PASSED** | 0.03s | CPU allowance strictly in debug mode |
| `tests/test_full_combined_pipeline.py` | `test_streaming_2way_merge_and_metadata` | **PASSED** | 1.10s | Chunked streaming merge of HI-Small + LI-Medium |
| `tests/test_full_combined_pipeline.py` | `test_rejection_of_subsets_and_legacy_500k` | **PASSED** | 0.85s | Fail-fast validation rejecting subset row limits |
| `tests/test_full_combined_pipeline.py` | `test_full_combined_mmap_caching_and_partitioning` | **PASSED** | 2.15s | Memory-mapped caching and IID/Non-IID partitioning |
| `tests/test_full_combined_pipeline.py` | `test_full_combined_end_to_end_smoke_test` | **PASSED** | 3.42s | End-to-end full combined federated smoke test |
| `tests/test_full_li_medium_pipeline.py` | `test_mmap_sequence_provider_build_and_reuse` | **PASSED** | 1.45s | Zero-copy mmap sequence provider build & reuse |
| `tests/test_full_li_medium_pipeline.py` | `test_full_dataset_enforcement_raises_on_subset` | **PASSED** | 0.72s | Assertion of full dataset row count enforcement |
| `tests/test_full_li_medium_pipeline.py` | `test_mmap_partition_manager_iid_and_non_iid` | **PASSED** | 1.12s | Partition manager client allocation consistency |
| `tests/test_full_li_medium_pipeline.py` | `test_end_to_end_federated_smoke_test` | **PASSED** | 1.88s | Multi-client FedAvg / FedProx execution |
| `tests/test_gpu_preprocessing_equivalence.py` | `test_preprocessor_interface_and_cleaning_equivalence` | **PASSED** | 0.18s | Column transformation and format verification |
| `tests/test_gpu_preprocessing_equivalence.py` | `test_full_pipeline_engineering_and_transform_equivalence` | **PASSED** | 0.23s | Mathematical feature parity (scaler, log, cyclical) |

---

## 3. Test Suite 2: Application DTO & Interface CLI Validation (`backend/tests/unit/`)

**Command**: `.venv\Scripts\python.exe -m pytest backend/tests/unit/application/test_training_dto.py backend/tests/unit/interfaces/test_train_cli.py -v`  
**Execution Time**: 4.46 seconds  
**Result**: 156 passed / 156 total (100% PASSED)

### Key Test Categories Verified:
1. **Architecture Enforcement & Unknown Architecture Rejection**:
   - `test_an_unknown_architecture_is_refused`: **PASSED**. Confirmed that specifying `transformer`, `tcn`, or `residual_mlp` now raises `ValueError` immediately upon configuration instantiation.
   - `test_each_architecture_can_be_named_on_its_own`: **PASSED**. Confirmed that `ORDERED_ARCHITECTURES` and `ModelArchitecture` strictly contain `("gru", "lstm", "bilstm")` and match CLI choices.
2. **Hyperparameter Invariants & Immutability**:
   - `test_a_configuration_cannot_be_retuned_after_construction`: **PASSED**. Frozen dataclass contracts hold.
   - `test_the_defaults_select_on_a_ranking_metric_not_accuracy`: **PASSED**. Enforces PR-AUC ranking default.
   - `test_the_default_threshold_policy_learns_from_validation`: **PASSED**. Enforces validation F1 threshold tuning.
3. **CLI Parser & Banner Reporting**:
   - `test_the_banner_states_the_loss_and_that_it_is_unweighted`: **PASSED**.
   - `test_false_negatives_are_labelled_as_missed_laundering`: **PASSED**.
   - `test_the_frozen_threshold_and_the_split_it_came_from_are_printed`: **PASSED**.
   - `test_the_output_never_mentions_a_credential`: **PASSED**.

---

## 4. Quarantined Modules Verification

The following legacy modules were verified as successfully isolated in `archive/legacy_experiments/` with no remaining inbound import references in the production pipeline:
1. `run_final_architectures_pipeline.py` (SHA-256: `3b8ef48d2fa7885b546522c0ff6420ee9b5e324c48972ca81903e13d5cf59bca`)
2. `audit_final_architecture_models.py` (SHA-256: `18f731c3fc551be1ef5b84ae48744bfa23bb35084ce1c4153ff228c8959d2a6a`)
3. `generate_final_arch_report.py` (SHA-256: `c830eb8e6c46927da8cfbf4df021b1933e46c764a4d65beae7c54117b4c9ca4b`)
4. `advanced_classifiers.py` (SHA-256: `81bb11ebf3ec61664e12e316a9a7b97a2ebc9ec1895a0bc07eef3d4caef58661`)
5. `test_advanced_classifiers.py` (SHA-256: `cb9bba71337e721e70e9cbe87fa8319665c928424075f164d12c8227b99d5a86`)

*Essential feature preprocessing transformers (e.g., `AmountLogTransformer`, `CyclicalTimeTransformer`) remain fully functional in `backend/infrastructure/data/` and `backend/domain/exceptions/preprocessing.py`.*

---

## 5. Conclusion

The repository is clean, active architectures are strictly confined to `GRU`, `LSTM`, and `BiLSTM`, and all 169 unit and integration tests pass without error or warning.

