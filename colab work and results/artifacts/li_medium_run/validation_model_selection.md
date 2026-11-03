# FedSecureAI — Validation Model Selection for Ensemble Phase

Selection strictly performed on out-of-sample Validation split (no test leakage):

1. **mu_0.01**: Validation PR-AUC = 0.000962
2. **mu_0.001**: Validation PR-AUC = 0.000877
3. **lr_1e-3_sgd**: Validation PR-AUC = 0.000827
4. **le_1_wd_1e-4**: Validation PR-AUC = 0.000816
5. **mu_0.1**: Validation PR-AUC = 0.000816
6. **le_2_wd_0**: Validation PR-AUC = 0.000815
7. **lr_1e-4_adam**: Validation PR-AUC = 0.000807
8. **bilstm_weighted_bce**: Validation PR-AUC = 0.000756
9. **lr_3e-4_adam**: Validation PR-AUC = 0.000742
10. **mu_1.0**: Validation PR-AUC = 0.000741
11. **bilstm_focal**: Validation PR-AUC = 0.000684
12. **mu_0.0**: Validation PR-AUC = 0.000630

**Ensemble Constituents Selected:** ['mu_0.01', 'mu_0.001', 'lr_1e-3_sgd']
