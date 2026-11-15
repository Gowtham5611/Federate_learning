# FedSecureAI End-to-End Live System Validation Report

**Execution Timestamp:** 2026-11-08 18:16:07 UTC  
**Frontend Host:** http://127.0.0.1:5173  
**Backend Host:** http://127.0.0.1:8000  

## E2E Target Benchmark Test Results

| File Tested | Labels | Validity Status | Alerts | F1-Score | PR-AUC | Why No Alerts? | PDF Download | CSV Download | Result |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| `demo_ibm_aml_500.csv` | Yes | `VALID EVALUATION DATASET` | 0 | `0.0000` | `0.1429` | Yes | ❌ Failed | ❌ Failed | **❌ FAIL** |
| `demo_ibm_aml_5000.csv` | Yes | `VALID EVALUATION DATASET` | 0 | `0.0000` | `0.0266` | Yes | ✅ 56,699 B | ✅ 1,197,156 B | **✅ PASS** |
| `test_slice_500.csv` | Yes | `INVALID / NON-REPRESENTATIVE BENCHMARK` | 0 | `0.0000` | `1.0000` | Yes | ✅ 51,355 B | ✅ 209,636 B | **✅ PASS** |
| `test_slice_5000.csv` | Yes | `INVALID / NON-REPRESENTATIVE BENCHMARK` | 3 | `0.4000` | `0.4337` | Yes | ✅ 59,677 B | ✅ 2,085,315 B | **✅ PASS** |
| `test_slice_50000.csv` | Yes | `INVALID / NON-REPRESENTATIVE BENCHMARK` | 23 | `0.3434` | `0.3014` | Yes | ✅ 61,352 B | ✅ 20,774,420 B | **✅ PASS** |
| `demo_no_labels_500.csv` | No | `VALID EVALUATION DATASET` | 0 | `N/A` | `N/A` | Yes | ✅ 35,952 B | ✅ 117,603 B | **✅ PASS** |
| `ibm_aml_schema_synthetic_20000.csv` | Yes | `UNKNOWN` | 67 | `0.0191` | `0.2896` | N/A | ✅ 59,932 B | ✅ 4,006,446 B | **✅ PASS** |

## Test Artifact Screenshots

- **demo_ibm_aml_500.csv**: [`demo_ibm_aml_500.png`](file:///C:/Users/Acer/OneDrive/Desktop/Federate_learning/results/e2e/demo_ibm_aml_500.png)
- **demo_ibm_aml_5000.csv**: [`demo_ibm_aml_5000.png`](file:///C:/Users/Acer/OneDrive/Desktop/Federate_learning/results/e2e/demo_ibm_aml_5000.png)
- **test_slice_500.csv**: [`test_slice_500.png`](file:///C:/Users/Acer/OneDrive/Desktop/Federate_learning/results/e2e/test_slice_500.png)
- **test_slice_5000.csv**: [`test_slice_5000.png`](file:///C:/Users/Acer/OneDrive/Desktop/Federate_learning/results/e2e/test_slice_5000.png)
- **test_slice_50000.csv**: [`test_slice_50000.png`](file:///C:/Users/Acer/OneDrive/Desktop/Federate_learning/results/e2e/test_slice_50000.png)
- **demo_no_labels_500.csv**: [`demo_no_labels_500.png`](file:///C:/Users/Acer/OneDrive/Desktop/Federate_learning/results/e2e/demo_no_labels_500.png)
- **ibm_aml_schema_synthetic_20000.csv**: [`ibm_aml_schema_synthetic_20000.png`](file:///C:/Users/Acer/OneDrive/Desktop/Federate_learning/results/e2e/ibm_aml_schema_synthetic_20000.png)

## Conclusion

**Total Files Tested:** 7  
**Passed:** 6 / 7 (85.7%)  
**Audit Status:** FAILURES DETECTED
