# FedSecureAI Phase 14A — Comprehensive Regression Test Report

**Execution Timestamp:** October 27, 2026 06:10:44 UTC  
**Environment:** Python 3.14.7, pytest 9.1.1, pluggy 1.6.0  
**Operating System:** Windows (AMD64 / x86_64)  
**Configuration File:** `pytest.ini` (`pythonpath = backend`)  
**Total Tests Executed:** 72  
**Passing:** 72  
**Failing:** 0  
**Skipped / XFailed:** 0  
**Overall Regression Verdict:** **100% PASS (ZERO REGRESSIONS)**  

---

## 1. Exact Execution Command & Exit Code

```powershell
.venv\Scripts\pytest.exe tests/test_phase7_ensemble.py tests/test_phase12_explainability.py tests/test_phase13_schemas.py tests/test_phase13_model_service.py tests/test_phase13_api.py tests/test_phase13_security_and_contracts.py tests/test_phase14a_deployment.py -v
```

- **Exit Code:** `0`
- **Total Execution Time:** `12.61 seconds`
- **Warnings:** `0`

---

## 2. Test Suite Breakdown by Phase

### 2.1 Phase 7: Hybrid Ensemble & Alpha Calibration (`tests/test_phase7_ensemble.py`)
- `test_alpha_blending_mathematical_properties`: **PASSED**
- `test_alpha_grid_boundaries_and_steps`: **PASSED**
- `test_deterministic_tie_breaking_policy`: **PASSED**
- `test_rejection_of_shape_mismatched_predictions`: **PASSED**
- `test_rejection_of_nan_or_inf_in_predictions`: **PASSED**
- `test_frozen_configuration_serialization_and_reload`: **PASSED**
- `test_cross_dataset_blending_prevention`: **PASSED**
*Subtotal: 7 / 7 passed.*

### 2.2 Phase 12: Explainability & Integrated Gradients (`tests/test_phase12_explainability.py`)
- `test_checkpoint_and_model_contract`: **PASSED**
- `test_input_output_tensor_shapes`: **PASSED**
- `test_probability_semantics`: **PASSED**
- `test_prediction_parity_verification_artifact`: **PASSED**
- `test_deterministic_sampling_reproducibility`: **PASSED**
- `test_split_separation_integrity`: **PASSED**
- `test_attribution_output_dimensions_and_feature_mapping`: **PASSED**
- `test_numerical_finiteness_and_completeness`: **PASSED**
- `test_zero_padding_leakage_guarantee`: **PASSED**
- `test_synthetic_integrated_gradients_completeness_axiom`: **PASSED**
- `test_artifact_preservation`: **PASSED**
*Subtotal: 11 / 11 passed.*

### 2.3 Phase 13: Schema Validation (`tests/test_phase13_schemas.py`)
- `test_valid_sequence_item`: **PASSED**
- `test_sequence_item_invalid_dimension`: **PASSED**
- `test_sequence_item_too_long`: **PASSED**
- `test_sequence_item_nan_and_inf`: **PASSED**
- `test_sequence_item_non_zero_padding`: **PASSED**
- `test_predict_request_validation`: **PASSED**
- `test_batch_predict_request_bounds`: **PASSED**
- `test_explain_request_bounds`: **PASSED**
- `test_metadata_responses`: **PASSED**
*Subtotal: 9 / 9 passed.*

### 2.4 Phase 13: Model Service & Evaluation (`tests/test_phase13_model_service.py`)
- `test_model_loading_and_readiness`: **PASSED**
- `test_model_metadata`: **PASSED**
- `test_single_prediction_hi_small`: **PASSED**
- `test_single_prediction_li_medium`: **PASSED**
- `test_batch_prediction_bounded`: **PASSED**
*Subtotal: 5 / 5 passed.*

### 2.5 Phase 13: API Route Integration (`tests/test_phase13_api.py`)
- `test_health_endpoint`: **PASSED**
- `test_models_endpoint`: **PASSED**
- `test_metadata_endpoint_valid`: **PASSED**
- `test_metadata_endpoint_invalid_dataset`: **PASSED**
- `test_predict_single_hi_small`: **PASSED**
- `test_predict_single_li_medium`: **PASSED**
- `test_predict_malformed_dimensions`: **PASSED**
- `test_predict_mismatched_model_name`: **PASSED**
- `test_predict_batch_success`: **PASSED**
- `test_predict_batch_size_limit`: **PASSED**
- `test_explain_endpoint`: **PASSED**
- `test_authoritative_prediction_parity_hi_small`: **PASSED**
*Subtotal: 12 / 12 passed.*

### 2.6 Phase 13.1: Security, Contracts & Bounds (`tests/test_phase13_security_and_contracts.py`)
- `test_unsupported_dataset_rejected`: **PASSED**
- `test_malformed_request_body_missing_fields`: **PASSED**
- `test_incorrect_feature_dimension_56`: **PASSED**
- `test_incorrect_feature_dimension_58`: **PASSED**
- `test_invalid_sequence_lengths`: **PASSED**
- `test_nan_rejected`: **PASSED**
- `test_pos_inf_rejected`: **PASSED**
- `test_neg_inf_rejected`: **PASSED**
- `test_non_zero_padding_rejected`: **PASSED**
- `test_batch_size_above_64_rejected`: **PASSED**
- `test_invalid_explain_steps`: **PASSED**
- `test_invalid_explain_top_k`: **PASSED**
- `test_checkpoint_hash_mismatch_fails_closed`: **PASSED**
- `test_checkpoint_missing_fails_closed`: **PASSED**
- `test_unloaded_model_raises_503_in_route`: **PASSED**
- `test_mismatched_model_name_rejected`: **PASSED**
- `test_concurrent_prediction_requests`: **PASSED**
- `test_sensitive_data_absent_from_logs`: **PASSED**
- `test_authentication_mechanism_status`: **PASSED**
*Subtotal: 19 / 19 passed.*

### 2.7 Phase 14A: Deployment Packaging & Configuration (`tests/test_phase14a_deployment.py`)
- `test_checkpoint_path_default_resolution`: **PASSED**
- `test_checkpoint_path_explicit_env_override`: **PASSED**
- `test_checkpoint_dir_env_resolution`: **PASSED**
- `test_metadata_dir_default_resolution`: **PASSED**
- `test_metadata_dir_custom_env_resolution`: **PASSED**
- `test_inference_requirements_strictness`: **PASSED**
- `test_dockerfile_security_directives`: **PASSED**
- `test_docker_compose_security_and_network`: **PASSED**
- `test_dockerignore_rules`: **PASSED**
*Subtotal: 9 / 9 passed.*

---

## 3. Regression Verdict & Historical Consistency

The deployment packaging enhancements added in Phase 14A (dynamic checkpoint resolution, metadata path environment overrides, and packaging tests) preserved 100% of historical test behaviors. Zero regressions were observed across the entire 72-test suite.

