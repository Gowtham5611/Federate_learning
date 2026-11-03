/**
 * Strict TypeScript types matching FedSecureAI Phase 13 & Phase 17 FastAPI schemas.
 */

export type SupportedDataset = 'HI-Small' | 'LI-Medium';

export type DecisionLabel = 'FLAGGED_LAUNDERING' | 'CLEARED_LEGITIMATE';

export type ExplanationMethod = 'integrated_gradients';

export interface HealthResponse {
  status: string;
  app_name: string;
  app_version: string;
  loaded_models: string[];
  device: string;
  readiness: 'ready' | 'initializing';
  timestamp: string;
}

export interface ModelMetadataResponse {
  dataset: SupportedDataset;
  model_name: string;
  architecture: string;
  parameter_count: number;
  sequence_length: number;
  feature_dimension: number;
  frozen_threshold: number;
  checkpoint_verified: boolean;
  supported_explanation_methods: ExplanationMethod[];
}

export interface DatasetMetadataResponse {
  dataset: SupportedDataset;
  feature_columns: string[];
  raw_feature_count: number;
  expanded_feature_count: number;
  sequence_length: number;
  frozen_threshold: number;
  active_model_name: string;
  preprocessing_version: string;
}

export interface SequenceItem {
  features: number[][]; // (L, 57)
  length?: number;
  sample_id?: string;
}

export interface PredictRequest {
  dataset: SupportedDataset;
  model_name?: string;
  sequence: SequenceItem;
}

export interface PredictResponse {
  request_id: string;
  dataset: SupportedDataset;
  model_name: string;
  predicted_probability: number;
  decision_threshold: number;
  decision: DecisionLabel;
  sample_id?: string;
  sequence_length: number;
  inference_time_ms: number;
}

export interface DriverAttribution {
  feature_name: string;
  attribution: number;
  feature_group: string;
  temporal_position?: number;
}

export interface ExplainRequest {
  dataset: SupportedDataset;
  model_name?: string;
  sequence: SequenceItem;
  steps?: number;
  top_k?: number;
}

export interface ExplainResponse {
  request_id: string;
  dataset: SupportedDataset;
  model_name: string;
  method: ExplanationMethod;
  baseline_type: string;
  steps: number;
  predicted_probability: number;
  decision_threshold: number;
  decision: DecisionLabel;
  top_positive_drivers: DriverAttribution[];
  top_negative_drivers: DriverAttribution[];
  completeness_residual: number;
  explanation_time_ms: number;
  is_history_sufficient?: boolean;
  history_warning?: string | null;
}

export interface PredictionHistoryEntry {
  id: string;
  timestamp: string;
  dataset: SupportedDataset;
  model_name?: string;
  modelName?: string;
  predicted_probability?: number;
  predictedProbability?: number;
  decision_threshold?: number;
  decisionThreshold?: number;
  decision: DecisionLabel;
  inference_time_ms?: number;
  inferenceTimeMs?: number;
  sample_id?: string;
  sampleId?: string;
  sequence_length?: number;
  explanationSnippet?: string;
}

export interface DemoSample {
  id: string;
  label: string;
  dataset: SupportedDataset;
  description: string;
  groundTruthLabel: string;
  sequence: SequenceItem;
}

// --- Phase 17 CSV Evaluation Types ---

export interface ConfusionMatrixData {
  tp: number;
  fp: number;
  tn: number;
  fn: number;
}

export interface CurveDatapoint {
  recall?: number;
  precision?: number;
  fpr?: number;
  tpr?: number;
  threshold: number;
}

export interface RiskBinData {
  bin_range: string;
  lower: number;
  upper: number;
  count: number;
  percentage: number;
}

export interface OperatingPointMetrics {
  name: string;
  threshold: number;
  alerts: number;
  alerts_per_10k: number;
  tp: number;
  fp: number;
  fn: number;
  tn: number;
  accuracy: number;
  precision: number | null;
  recall: number | null;
  specificity: number | null;
  balanced_accuracy: number | null;
  f1_score: number | null;
  f2_score?: number | null;
  mcc?: number | null;
  cohen_kappa?: number | null;
  description?: string;
  budget_per_10k?: number;
  target_alerts?: number;
  label?: string;
}

export interface OperatingPointsMap {
  frozen?: OperatingPointMetrics;
  alert_budgets?: Record<string, OperatingPointMetrics>;
  best_f1_on_file?: OperatingPointMetrics;
}

export interface EvaluationMetrics {
  total_samples: number;
  positive_samples: number;
  negative_samples: number;
  positive_prevalence?: number;
  frozen_threshold?: number;
  positive_class?: string;
  accuracy: number;
  precision: number | null;
  recall: number;
  sensitivity?: number;
  specificity?: number | null;
  balanced_accuracy?: number | null;
  f1_score: number | null;
  f2_score?: number | null;
  mcc?: number | null;
  cohen_kappa?: number | null;
  false_positive_rate?: number | null;
  false_negative_rate?: number | null;
  false_discovery_rate?: number | null;
  negative_predictive_value?: number | null;
  pr_auc: number | null;
  pr_auc_lift?: number | null;
  pr_auc_ci_lower?: number | null;
  pr_auc_ci_upper?: number | null;
  average_precision?: number | null;
  roc_auc: number | null;
  roc_auc_ci_lower?: number | null;
  roc_auc_ci_upper?: number | null;
  log_loss?: number | null;
  brier_score: number;
  expected_calibration_error?: number | null;
  maximum_calibration_error?: number | null;
  ece?: number | null;
  mce?: number | null;
  alert_count?: number;
  alert_rate?: number;
  alerts_per_10k: number;
  mean_predicted_probability?: number;
  max_predicted_probability?: number;
  min_predicted_probability?: number;
  zero_alerts_note?: string | null;
  edge_case_note?: string | null;
  pr_curve?: CurveDatapoint[];
  roc_curve?: CurveDatapoint[];
  confusion_matrix: ConfusionMatrixData;
  pr_auc_ci_reason?: string | null;
  operating_points?: OperatingPointsMap;
  ranking_vs_classification?: {
    pr_auc?: number | null;
    roc_auc?: number | null;
    interpretation?: string;
  };
  metrics_classification?: {
    threshold_dependent: string[];
    probability_dependent: string[];
  };
}

export interface TransactionPredictionItem {
  row_id: number;
  transaction_id?: string;
  timestamp: string;
  from_account: string;
  to_account: string;
  amount: number;
  predicted_probability: number;
  risk_score?: number;
  risk_category?: string;
  predicted_label?: 'SUSPICIOUS' | 'LEGITIMATE' | string;
  actual_label?: 'SUSPICIOUS' | 'LEGITIMATE' | string | null;
  status?: 'SUSPICIOUS' | 'LEGITIMATE' | string;
  decision_threshold: number;
  decision: DecisionLabel | string;
  ground_truth_label: number | null;
  sequence_length: number;
  valid_history_count?: number;
  padding_ratio?: number;
  unknown_category_ratio?: number;
  missing_feature_ratio?: number;
  is_history_sufficient?: boolean;
  usable_history_note?: string | null;
  sequence_features: number[][]; // (10, 57) for on-demand explainability
}

export interface ValidityBannerDiagnostics {
  n_rows: number;
  prevalence: number | null;
  positives_count: number | null;
  expected_positives?: number;
  recommended_min_rows?: number;
  unique_sender_ratio: number;
  pct_single_tx_rows: number;
  mean_history_len: number;
  pct_rows_with_ge_3_prior_tx: number;
  is_chronological: boolean;
  time_span: string;
}

export interface ValidityBanner {
  status: 'VALID' | 'WARNING' | 'INVALID';
  is_valid: boolean;
  flags: string[];
  reasons: string[];
  disclaimer: string;
  diagnostics: ValidityBannerDiagnostics;
}

export interface SequenceContextDiagnostics {
  average_sequence_length: number;
  max_sequence_length: number;
  single_transaction_count: number;
  single_transaction_ratio: number;
  mean_padding_ratio: number;
  insufficient_history_warning: boolean;
  operational_context_status: 'HISTORICALLY_SUPPORTED' | 'INSUFFICIENT_HISTORY_TRUNCATED';
  warning_message?: string | null;
  persistent_history_enabled?: boolean;
  tenant_id?: string | null;
  historical_accounts_tracked?: number;
  suggested_models?: Array<{ model_id: string; model_name: string; model_type?: string; reason?: string }>;
}

export interface CsvEvaluateResponse {
  status: string;
  filename: string;
  dataset: SupportedDataset;
  model_id?: string;
  model_name: string;
  decision_threshold: number;
  total_rows: number;
  evaluated_sequences: number;
  distinct_accounts: number;
  flagged_laundering_count: number;
  cleared_count: number;
  risk_distribution?: RiskBinData[];
  has_labels: boolean;
  has_ground_truth?: boolean;
  mode?: 'SUPERVISED_EVALUATION' | 'SCORE_ONLY';
  score_only_message?: string | null;
  evaluation_metrics: EvaluationMetrics | null;
  operating_points?: OperatingPointsMap;
  top_risky_transactions?: TransactionPredictionItem[];
  validity_banner?: ValidityBanner;
  sequence_context_diagnostics?: SequenceContextDiagnostics;
  suggested_models?: Array<{ model_id: string; model_name: string; model_type?: string; reason?: string }>;
  persistent_history_enabled?: boolean;
  predictions: TransactionPredictionItem[];
  execution_time_ms: number;
  // Adaptive Schema Metadata
  schema_status?: 'COMPATIBLE' | 'PARTIALLY_COMPATIBLE' | 'AMBIGUOUS' | 'UNSUPPORTED';
  schema_message?: string;
  detected_format?: string;
  field_mappings?: Record<string, string>;
  excluded_leakage_fields?: string[];
  audit_notes?: string[];
  risk_counts?: {
    low_risk: number;
    medium_risk: number;
    high_risk: number;
    total: number;
    flagged_count: number;
    cleared_count: number;
  };
  download_id?: string;
  download_url?: string;
}

// --- Universal CSV & Dynamic Model Types ---

export interface UniversalCsvInspectionResponse {
  delimiter?: string;
  encoding?: string;
  num_rows?: number;
  num_cols?: number;
  column_names: string[];
  preview_rows?: Record<string, unknown>[];
  target_column_name: string | null;
  has_ground_truth?: boolean;
  detected_schema_type?: string;
  can_construct_sequences?: boolean;
  column_types?: Record<string, string>;
  semantic_mappings?: Record<string, string | null>;
  // Legacy & Alternate field names
  detected_delimiter?: string;
  detected_encoding?: string;
  total_rows?: number;
  total_columns?: number;
  sample_rows?: Record<string, unknown>[];
  has_target_label?: boolean;
  detected_concepts?: Record<string, string | null>;
  dtypes?: Record<string, string>;
}

export interface ModelCompatibilityResult {
  model_id: string;
  status: 'COMPATIBLE' | 'ADAPTABLE' | 'INCOMPATIBLE';
  missing_features: string[];
  extra_features: string[];
  actions_needed: string[];
  compatibility_score: number;
  summary: string;
}

export interface RegisteredModelItem {
  model_id: string;
  model_name: string;
  architecture: string;
  feature_columns: string[];
  numerical_columns: string[];
  categorical_columns: string[];
  input_shape?: number[];
  version: string;
  decision_threshold: number;
  created_at: string;
  frozen: boolean;
  is_custom: boolean;
  metrics?: Record<string, unknown>;
}

export interface CustomModelTrainResponse {
  model_id: string;
  model_name: string;
  model_type: string;
  validation_metrics: Record<string, unknown>;
  message: string;
}

export type SchemaValidationStatus = 'COMPATIBLE' | 'PARTIALLY_COMPATIBLE' | 'AMBIGUOUS' | 'UNSUPPORTED';

export interface SchemaValidationResponse {
  status: SchemaValidationStatus;
  user_message: string;
  format_category: string;
  target_pipeline: string;
  decision_threshold: number;
  has_labels: boolean;
  field_mappings: Record<string, string>;
  missing_required_fields: string[];
  conflicting_fields: Record<string, string[]>;
  excluded_leakage_fields: string[];
  audit_notes: string[];
}

export interface ModelComparisonResultItem {
  model_id: string;
  model_name: string;
  model_type: string;
  dataset_name: string;
  decision_threshold: number;
  total_samples: number;
  flagged_count: number;
  cleared_count: number;
  laundering_percentage: number;
  mean_probability: number;
  max_probability: number;
  min_probability: number;
  metrics: EvaluationMetrics | null;
  pr_auc: number | null;
  roc_auc: number | null;
  precision: number | null;
  recall: number | null;
  f1_score: number | null;
  accuracy: number | null;
  specificity: number | null;
  balanced_accuracy?: number | null;
  mcc?: number | null;
  cohen_kappa?: number | null;
  brier_score: number | null;
}

export interface MultiModelEvaluateResponse {
  status: string;
  filename: string;
  total_transactions: number;
  has_labels: boolean;
  models_evaluated_count: number;
  execution_time_ms: number;
  models: ModelComparisonResultItem[];
}

// --- Research Artifacts Types ---

export interface ResearchDatasetInfo {
  name: string;
  raw_transactions: number;
  distinct_accounts: number;
  raw_columns: number;
  splits: {
    train_rows: number;
    validation_rows: number;
    test_sequences: number;
  };
  test_positive_prevalence: number;
  champion_model: string;
  champion_architecture: string;
  parameter_count: number;
  frozen_threshold: number;
  test_pr_auc: number;
  test_roc_auc: number;
  test_f1: number;
  test_precision: number;
  test_recall: number;
}

export interface ResearchOverviewResponse {
  title: string;
  description: string;
  verified_datasets: {
    'HI-Small': ResearchDatasetInfo;
    'LI-Medium': ResearchDatasetInfo;
  };
  model_architectures: string[];
  federated_algorithms: string[];
  data_distributions: string[];
  total_experiments_evaluated: number;
}

export interface BenchmarkModelRecord {
  dataset: SupportedDataset;
  config_id: string;
  model_name: string;
  architecture: string;
  distribution: string;
  algorithm: string;
  fedprox_mu: number;
  threshold: number;
  test_pr_auc: number;
  test_roc_auc: number;
  test_f1: number;
  test_precision: number;
  test_recall: number;
  alerts_per_10k: number;
  brier_score: number;
  log_loss: number;
}

export interface ConvergenceCurvePoint {
  dataset: SupportedDataset;
  config_id: string;
  architecture: string;
  distribution: string;
  algorithm: string;
  round: number;
  val_pr_auc: number;
  val_roc_auc: number;
  val_loss: number;
}

export interface ThresholdTradeoffPoint {
  dataset: SupportedDataset;
  model_name: string;
  threshold: number;
  precision: number;
  recall: number;
  f1: number;
  alerts_per_10k: number;
}

export interface AblationSummaryResponse {
  fedprox_mu_0_0001_ablation: {
    finding: string;
    hi_small_lstm_drop: string;
    hi_small_gru_drop: string;
    li_medium_bilstm_drop: string;
  };
  iid_vs_noniid_degradation: {
    finding: string;
    hi_small_lstm_drop: string;
    li_medium_bilstm_drop: string;
  };
  phase7_hybrid_ensembling: {
    finding: string;
    hi_small_alpha_star: number;
    li_medium_alpha_star: number;
  };
  phase10_probability_calibration: {
    finding: string;
    uncalibrated_ece: number;
    isotonic_ece: number;
    platt_ece: number;
    temperature_ece: number;
  };
}

export interface CommunicationEfficiencyRecord {
  architecture: string;
  parameters: number;
  size_per_transfer_kb: number;
  round_bandwidth_mb: number;
  total_5_round_bandwidth_mb: number;
  fp16_bandwidth_mb: number;
  int8_bandwidth_mb: number;
  relative_efficiency: string;
}

export interface ClientPartitionInfo {
  client_id: string;
  bank_type: string;
  training_samples: number;
  percentage: number;
  val_loss_divergence: number;
  typology_focus: string;
}

export interface ClientPartitionResponse {
  dataset: string;
  total_clients: number;
  partitioning_strategy: string;
  client_distribution: ClientPartitionInfo[];
  imbalance_ratio: number;
  aggregator_implication: string;
}

