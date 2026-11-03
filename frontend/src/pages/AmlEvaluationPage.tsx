import React, { useState, useEffect } from 'react';
import {
  CsvEvaluateResponse,
  ModelCompatibilityResult,
  MultiModelEvaluateResponse,
  RegisteredModelItem,
  UniversalCsvInspectionResponse,
} from '../types/api';
import { apiService, ApiError } from '../services/apiService';
import {
  Upload,
  FileSpreadsheet,
  AlertTriangle,
  CheckCircle2,
  ArrowRight,
  Sparkles,
  RefreshCw,
  BarChart2,
  History,
  FileText,
  ShieldAlert,
  ShieldCheck,
  Zap,
  Download,
  Settings,
  Cpu,
  Database,
  ChevronDown,
  ChevronUp,
  Layers,
} from 'lucide-react';
import { formatMetric } from '../utils/formatters';

interface AmlEvaluationPageProps {
  currentResult: CsvEvaluateResponse | null;
  multiModelResult?: MultiModelEvaluateResponse | null;
  onEvaluationComplete: (result: CsvEvaluateResponse) => void;
  onMultiModelEvaluationComplete?: (result: MultiModelEvaluateResponse) => void;
  onNavigateTab: (tab: 'evaluation' | 'analytics' | 'history' | 'models' | 'ensemble') => void;
}

// Built-in demonstration datasets
const SAMPLE_SYNTHETIC_CSV = `timestamp,sender_account,receiver_account,transaction_amount,transaction_type,currency,is_fraud,structuring_flag
2022-09-01 09:15:00,ACC_749102,ACC_839104,12450.00,TRANSFER,USD,0,0
2022-09-01 09:20:00,ACC_109284,ACC_491823,890.50,PAYMENT,USD,0,0
2022-09-01 09:45:00,ACC_749102,ACC_901842,48500.00,WIRE,EUR,1,1
2022-09-01 10:12:00,ACC_381920,ACC_839104,320.00,DEBIT,USD,0,0
2022-09-01 10:30:00,ACC_491823,ACC_109284,95000.00,CASH_OUT,USD,1,1
2022-09-01 11:05:00,ACC_610293,ACC_501928,1540.25,TRANSFER,GBP,0,0
2022-09-01 11:40:00,ACC_901842,ACC_749102,75000.00,TRANSFER,USD,1,1
2022-09-01 12:15:00,ACC_204918,ACC_381920,450.00,PAYMENT,USD,0,0
2022-09-01 13:00:00,ACC_839104,ACC_491823,120000.00,WIRE,EUR,1,1
2022-09-01 13:45:00,ACC_501928,ACC_610293,2100.00,DEBIT,USD,0,0
2022-09-01 14:10:00,ACC_109284,ACC_749102,14300.00,TRANSFER,USD,0,0
2022-09-01 14:50:00,ACC_491823,ACC_901842,62000.00,CASH_OUT,USD,1,1
2022-09-01 15:20:00,ACC_381920,ACC_204918,520.00,PAYMENT,USD,0,0
2022-09-01 15:55:00,ACC_839104,ACC_610293,88000.00,WIRE,USD,1,1
2022-09-01 16:30:00,ACC_749102,ACC_109284,1850.00,TRANSFER,USD,0,0`;

const SAMPLE_UNLABELED_CSV = `Timestamp,From Bank,Account,To Bank,Account.1,Amount Received,Receiving Currency,Amount Paid,Payment Currency,Payment Format
2022/09/01 09:00,102,ACC_91028,102,ACC_81920,5420.00,US Dollar,5420.00,US Dollar,ACH
2022/09/01 09:14,204,ACC_49182,102,ACC_10294,1280.50,US Dollar,1280.50,US Dollar,Credit Card
2022/09/01 09:30,305,ACC_77491,305,ACC_88392,67400.00,Euro,67400.00,Euro,Wire
2022/09/01 10:05,102,ACC_28190,102,ACC_49182,950.00,US Dollar,950.00,US Dollar,Cheque
2022/09/01 10:45,401,ACC_10294,204,ACC_91028,84200.00,US Dollar,84200.00,US Dollar,Wire
2022/09/01 11:20,204,ACC_81920,305,ACC_77491,15200.00,UK Pound,15200.00,UK Pound,ACH
2022/09/01 12:00,102,ACC_88392,102,ACC_28190,430.00,US Dollar,430.00,US Dollar,Credit Card
2022/09/01 12:40,305,ACC_91028,401,ACC_10294,115000.00,US Dollar,115000.00,US Dollar,Wire
2022/09/01 13:15,401,ACC_49182,204,ACC_81920,2800.00,US Dollar,2800.00,US Dollar,ACH
2022/09/01 14:00,102,ACC_77491,102,ACC_88392,92500.00,Euro,92500.00,Euro,Wire`;

export const AmlEvaluationPage: React.FC<AmlEvaluationPageProps> = ({
  currentResult,
  multiModelResult,
  onEvaluationComplete,
  onMultiModelEvaluationComplete,
  onNavigateTab,
}) => {
  // File state
  const [csvContent, setCsvContent] = useState<string>('');
  const [fileName, setFileName] = useState<string>('');
  const [fileSizeStr, setFileSizeStr] = useState<string>('');
  const [rowLimit, setRowLimit] = useState<number>(5000);
  const [customThreshold, setCustomThreshold] = useState<string>('');

  // Model Registry & Compatibility state
  const [models, setModels] = useState<RegisteredModelItem[]>([]);
  const [selectedModelId, setSelectedModelId] = useState<string>('auto');
  const [compatibilities, setCompatibilities] = useState<Record<string, ModelCompatibilityResult>>({});

  // Schema Inspection state
  const [inspection, setInspection] = useState<UniversalCsvInspectionResponse | null>(null);
  const [isInspecting, setIsInspecting] = useState<boolean>(false);

  // Column Mapping Overrides
  const [showMappingPanel, setShowMappingPanel] = useState<boolean>(false);
  const [customMappings, setCustomMappings] = useState<Record<string, string>>({});
  const [targetColumnOverride, setTargetColumnOverride] = useState<string>('');

  // Mode B: Dynamic Model Training
  const [showTrainSection, setShowTrainSection] = useState<boolean>(false);
  const [customModelName, setCustomModelName] = useState<string>('Custom_AML_Classifier');
  const [customModelType, setCustomModelType] = useState<string>('HistGradientBoosting');
  const [trainingInProgress, setTrainingInProgress] = useState<boolean>(false);
  const [trainSuccessMessage, setTrainSuccessMessage] = useState<string | null>(null);

  // Execution states
  const [evaluating, setEvaluating] = useState<boolean>(false);
  const [multiModelEvaluating, setMultiModelEvaluating] = useState<boolean>(false);
  const [apiError, setApiError] = useState<string | null>(null);

  const formatFileSize = (bytes: number): string => {
    if (bytes < 1024) return `${bytes} B`;
    if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
    return `${(bytes / (1024 * 1024)).toFixed(2)} MB`;
  };

  const defaultModels: RegisteredModelItem[] = [
    {
      model_id: 'HI-Small',
      model_name: 'HI-Small Champion (Federated LSTM)',
      architecture: 'LSTM',
      feature_columns: [],
      numerical_columns: [],
      categorical_columns: [],
      version: '1.0.0',
      decision_threshold: 0.337784,
      created_at: '',
      frozen: true,
      is_custom: false,
    },
    {
      model_id: 'LI-Medium',
      model_name: 'LI-Medium Champion (Federated BiLSTM)',
      architecture: 'BiLSTM',
      feature_columns: [],
      numerical_columns: [],
      categorical_columns: [],
      version: '1.0.0',
      decision_threshold: 0.238483,
      created_at: '',
      frozen: true,
      is_custom: false,
    },
  ];

  // Load available models on mount
  const refreshModelList = async () => {
    try {
      const modelList = await apiService.getRegisteredModels();
      if (Array.isArray(modelList) && modelList.length > 0) {
        setModels(modelList);
      } else {
        setModels(defaultModels);
      }
    } catch {
      setModels(defaultModels);
    }
  };

  useEffect(() => {
    refreshModelList();
  }, []);

  // Inspect CSV and evaluate compatibility against available models
  const analyzeUploadedCsv = async (content: string, name: string) => {
    if (!content || content.trim().length === 0) {
      setInspection(null);
      setCompatibilities({});
      return;
    }

    setIsInspecting(true);
    setApiError(null);

    try {
      // 1. Deep Schema Inspection
      const inspectRes = await apiService.inspectCsv({
        csv_content: content,
        filename: name,
      });
      setInspection(inspectRes);

      // Pre-fill target column if detected
      if (inspectRes.target_column_name) {
        setTargetColumnOverride(inspectRes.target_column_name);
      }

      // 2. Compatibility check across all models
      const compatRes = await apiService.checkModelCompatibility({
        csv_content: content,
        filename: name,
        model_id: selectedModelId || 'auto',
      });

      const compatMap: Record<string, ModelCompatibilityResult> = {};
      (compatRes?.results || []).forEach((r) => {
        compatMap[r.model_id] = r;
      });
      setCompatibilities(compatMap);

      // Initialize custom column mappings from detected semantic mappings
      const initialMappings: Record<string, string> = {};
      const detectedMap = inspectRes?.semantic_mappings || (inspectRes as any)?.detected_concepts || {};
      Object.entries(detectedMap).forEach(([canon, col]) => {
        if (col && typeof col === 'string') initialMappings[canon] = col;
      });
      setCustomMappings(initialMappings);
    } catch (err: unknown) {
      const msg = err instanceof ApiError ? err.message : err instanceof Error ? err.message : 'Inspection failed';
      setApiError(msg);
    } finally {
      setIsInspecting(false);
    }
  };

  // Handle local file selection
  const handleFileChange = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;

    if (!file.name.toLowerCase().endsWith('.csv') && !file.name.toLowerCase().endsWith('.txt')) {
      setApiError('Unsupported file format. Please upload a standard CSV or TXT file.');
      setFileName(file.name);
      return;
    }

    if (file.size > 25 * 1024 * 1024) {
      setApiError('File size limit exceeded. Uploaded file is larger than 25MB.');
      setFileName(file.name);
      return;
    }

    setFileName(file.name);
    setFileSizeStr(formatFileSize(file.size));
    setTrainSuccessMessage(null);

    try {
      const text = await file.text();
      setCsvContent(text);
      await analyzeUploadedCsv(text, file.name);
    } catch (err) {
      setApiError('Failed to read file content: ' + String(err));
    }
  };

  // Quick demonstration sample loaders
  const handleLoadSample = async (variant: 'standard' | 'synthetic' | 'unlabeled') => {
    setApiError(null);
    setEvaluating(true);
    setTrainSuccessMessage(null);

    try {
      let text = '';
      let sampleName = '';

      if (variant === 'synthetic') {
        text = SAMPLE_SYNTHETIC_CSV;
        sampleName = 'synthetic_mobile_banking.csv';
      } else if (variant === 'unlabeled') {
        text = SAMPLE_UNLABELED_CSV;
        sampleName = 'unlabeled_transactions_stream.csv';
      } else {
        const url = apiService.getSampleCsvUrl('HI-Small');
        const resp = await fetch(url);
        if (!resp.ok) {
          throw new Error(`Failed to download demo transactions (${resp.status})`);
        }
        text = await resp.text();
        sampleName = 'standard_aml_sample_25.csv';
      }

      setCsvContent(text);
      setFileName(sampleName);
      setFileSizeStr(formatFileSize(text.length));

      await analyzeUploadedCsv(text, sampleName);

      const evalResult = await apiService.predictCsv({
        csv_content: text,
        filename: sampleName,
        row_limit: rowLimit,
      });
      onEvaluationComplete(evalResult);
    } catch (err: unknown) {
      const msg = err instanceof ApiError ? err.message : err instanceof Error ? err.message : 'Demo loading failed';
      setApiError(msg);
    } finally {
      setEvaluating(false);
    }
  };

  // Mode A: Run AML Prediction
  const handleRunPrediction = async () => {
    if (!csvContent || csvContent.trim().length === 0) {
      setApiError('Please select or upload a CSV file before running prediction.');
      return;
    }

    setEvaluating(true);
    setApiError(null);

    try {
      const modelToUse = selectedModelId || 'auto';
      const thresholdVal = customThreshold !== '' ? parseFloat(customThreshold) : undefined;
      const resp = await apiService.predictCsv({
        csv_content: csvContent,
        model_id: modelToUse,
        filename: fileName || 'transactions.csv',
        column_mapping: Object.keys(customMappings || {}).length > 0 ? customMappings : undefined,
        row_limit: rowLimit,
        threshold: thresholdVal,
      });
      onEvaluationComplete(resp);
    } catch (err: unknown) {
      let msg = 'Evaluation request failed';
      if (err instanceof ApiError) {
        msg = err.message;
      } else if (err instanceof Error) {
        msg = err.message;
      }
      setApiError(msg);
    } finally {
      setEvaluating(false);
    }
  };

  // Run Multi-Model Comparative Evaluation on the exact preprocessed batch
  const handleRunMultiModelEvaluation = async () => {
    if (!csvContent || csvContent.trim().length === 0) {
      setApiError('Please select or upload a CSV file before running multi-model evaluation.');
      return;
    }

    setMultiModelEvaluating(true);
    setApiError(null);

    try {
      const thresholdVal = customThreshold !== '' ? parseFloat(customThreshold) : undefined;
      const resp = await apiService.evaluateMultiModel({
        csv_content: csvContent,
        filename: fileName || 'transactions.csv',
        column_mapping: Object.keys(customMappings || {}).length > 0 ? customMappings : undefined,
        row_limit: rowLimit,
        threshold: thresholdVal,
      });
      if (onMultiModelEvaluationComplete) {
        onMultiModelEvaluationComplete(resp);
      }
      // If we don't have currentResult yet, also fetch single prediction for 'auto' model so results tabs work
      if (!currentResult) {
        const singleResp = await apiService.predictCsv({
          csv_content: csvContent,
          model_id: 'auto',
          filename: fileName || 'transactions.csv',
          column_mapping: Object.keys(customMappings || {}).length > 0 ? customMappings : undefined,
          row_limit: rowLimit,
          threshold: thresholdVal,
        });
        onEvaluationComplete(singleResp);
      }
      onNavigateTab('analytics');
    } catch (err: unknown) {
      let msg = 'Multi-model evaluation request failed';
      if (err instanceof ApiError) {
        msg = err.message;
      } else if (err instanceof Error) {
        msg = err.message;
      }
      setApiError(msg);
    } finally {
      setMultiModelEvaluating(false);
    }
  };

  // Mode B: Train Custom Model on Labeled CSV
  const handleTrainCustomModel = async () => {
    if (!csvContent || csvContent.trim().length === 0) {
      setApiError('Please select or upload a CSV file with labeled ground truth.');
      return;
    }

    const targetCol = targetColumnOverride || inspection?.target_column_name;
    if (!targetCol) {
      setApiError('Please designate a binary target label column (e.g. 0/1 indicator).');
      return;
    }

    setTrainingInProgress(true);
    setApiError(null);
    setTrainSuccessMessage(null);

    try {
      const trainRes = await apiService.trainCustomModel({
        csv_content: csvContent,
        model_name: customModelName.trim() || 'Custom_AML_Model',
        target_column: targetCol,
        model_type: customModelType,
        filename: fileName || 'labeled_train.csv',
      });

      setTrainSuccessMessage(
        `Model "${trainRes.model_name}" trained successfully! Validation F1: ${
          trainRes.validation_metrics.validation_f1 ?? 'N/A'
        }, PR-AUC: ${trainRes.validation_metrics.validation_pr_auc ?? 'N/A'}. Model registered as ${trainRes.model_id}.`
      );

      // Refresh registered models and auto-select new model
      await refreshModelList();
      setSelectedModelId(trainRes.model_id);

      // Re-run compatibility check with newly trained model
      await analyzeUploadedCsv(csvContent, fileName);
    } catch (err: unknown) {
      const msg = err instanceof ApiError ? err.message : err instanceof Error ? err.message : 'Training failed';
      setApiError(msg);
    } finally {
      setTrainingInProgress(false);
    }
  };

  // Direct export of predictions CSV
  const handleDownloadPredictions = () => {
    if (!currentResult) return;

    if (currentResult.download_url) {
      window.open(currentResult.download_url, '_blank');
      return;
    }

    if (currentResult.download_id) {
      window.open(apiService.getPredictionsDownloadUrl(currentResult.download_id), '_blank');
      return;
    }

    // Fallback client-side generation
    const headers = [
      'Row_ID',
      'Timestamp',
      'From_Account',
      'To_Account',
      'Amount',
      'Risk_Score',
      'Risk_Level',
      'Decision_Threshold',
      'Decision',
      'Ground_Truth',
    ];

    const rows = currentResult.predictions.map((p) => [
      p.row_id,
      p.timestamp,
      p.from_account,
      p.to_account,
      p.amount,
      p.predicted_probability,
      (p as any).risk_level || (p.predicted_probability >= 0.7 ? 'High Risk' : p.predicted_probability >= 0.3 ? 'Medium Risk' : 'Low Risk'),
      p.decision_threshold,
      p.decision,
      p.ground_truth_label !== null ? p.ground_truth_label : 'UNLABELED',
    ]);

    const csvContentStr = [
      headers.join(','),
      ...rows.map((r) => r.map((c) => `"${String(c).replace(/"/g, '""')}"`).join(',')),
    ].join('\n');

    const blob = new Blob([csvContentStr], { type: 'text/csv;charset=utf-8;' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `fedsecureai_${currentResult.dataset.toLowerCase()}_predictions.csv`;
    a.click();
    URL.revokeObjectURL(url);
  };

  const selectedModelCompat =
    selectedModelId !== 'auto' && compatibilities[selectedModelId]
      ? compatibilities[selectedModelId]
      : null;

  const detectedDelimiter = inspection?.delimiter || (inspection as any)?.detected_delimiter || ',';
  const detectedEncoding = inspection?.encoding || (inspection as any)?.detected_encoding || 'utf-8';
  const totalRowsCount = inspection?.num_rows ?? (inspection as any)?.total_rows ?? 0;
  const totalColsCount = inspection?.num_cols ?? (inspection as any)?.total_columns ?? 0;
  const schemaTypeStr = inspection?.detected_schema_type || (inspection?.can_construct_sequences ? 'Transaction Sequence' : 'Tabular Dataset');
  const hasGroundTruthFlag = Boolean(inspection?.has_ground_truth ?? (inspection as any)?.has_target_label);
  const targetColName = inspection?.target_column_name ?? (inspection as any)?.target_column_name ?? null;
  const semanticMappingsObj: Record<string, string | null> = inspection?.semantic_mappings || (inspection as any)?.detected_concepts || {};

  const previewCols: string[] = inspection?.column_names || [];
  const previewData: Record<string, unknown>[] = (inspection?.preview_rows || (inspection as any)?.sample_rows || []) as Record<string, unknown>[];

  return (
    <div className="space-y-8 animate-fade-in">
      {/* Top Banner */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 border-b border-slate-800 pb-5">
        <div>
          <h1 className="text-2xl sm:text-3xl font-black text-white tracking-tight flex items-center space-x-2">
            <FileSpreadsheet className="w-7 h-7 text-indigo-400" />
            <span>Universal AML CSV Evaluation & Prediction</span>
          </h1>
          <p className="text-xs text-slate-400 mt-1">
            Universal reader for arbitrary rows, columns, delimiters, encodings, and feature schemas. Predict with frozen federated models or train custom classifiers.
          </p>
        </div>

        {currentResult && (
          <div className="flex items-center space-x-3">
            <button
              onClick={handleDownloadPredictions}
              className="flex items-center space-x-1.5 px-3.5 py-2 rounded-xl bg-emerald-600 hover:bg-emerald-500 text-white text-xs font-semibold shadow-lg shadow-emerald-600/30 transition cursor-pointer"
            >
              <Download className="w-3.5 h-3.5" />
              <span>Download CSV</span>
            </button>
            <button
              onClick={() => onNavigateTab('analytics')}
              className="flex items-center space-x-1.5 px-3.5 py-2 rounded-xl bg-indigo-600 hover:bg-indigo-500 text-white text-xs font-semibold shadow-lg shadow-indigo-600/30 transition cursor-pointer"
            >
              <BarChart2 className="w-3.5 h-3.5" />
              <span>Analytics</span>
            </button>
            <button
              onClick={() => onNavigateTab('history')}
              className="flex items-center space-x-1.5 px-3.5 py-2 rounded-xl bg-slate-800 hover:bg-slate-700 text-slate-200 text-xs font-semibold border border-slate-700 transition cursor-pointer"
            >
              <History className="w-3.5 h-3.5" />
              <span>Predictions ({currentResult.predictions?.length || 0})</span>
            </button>
          </div>
        )}
      </div>

      {/* Main Upload & Configuration Container */}
      <div className="bg-slate-900 border border-slate-800 rounded-3xl p-6 shadow-xl space-y-6">
        <div className="flex flex-col lg:flex-row lg:items-start justify-between gap-6">
          {/* Dropzone & Universal Format Banner */}
          <div className="flex-1 space-y-4">
            <label className="block text-xs font-mono uppercase text-slate-400 font-semibold">
              Select or Drop Any Transaction CSV
            </label>
            <div className="border-2 border-dashed border-slate-700 hover:border-indigo-500/80 rounded-2xl p-6 text-center transition bg-slate-950/40 relative">
              <input
                type="file"
                accept=".csv,.txt"
                onChange={handleFileChange}
                className="absolute inset-0 w-full h-full opacity-0 cursor-pointer"
              />
              <div className="space-y-2 pointer-events-none">
                <div className="w-10 h-10 rounded-xl bg-indigo-950/80 text-indigo-400 flex items-center justify-center mx-auto border border-indigo-700/50">
                  <Upload className="w-5 h-5" />
                </div>
                <div className="text-sm font-medium text-slate-200">
                  {fileName ? (
                    <span className="text-indigo-400 font-bold">
                      {fileName} {fileSizeStr ? `(${fileSizeStr})` : ''}
                    </span>
                  ) : (
                    <span>Drag and drop transaction CSV here, or click to browse</span>
                  )}
                </div>
                {isInspecting ? (
                  <p className="text-xs text-indigo-400 flex items-center justify-center space-x-1 animate-pulse">
                    <RefreshCw className="w-3.5 h-3.5 animate-spin inline" />
                    <span>Inspecting CSV schema & validating compatibility...</span>
                  </p>
                ) : (
                  <p className="text-xs text-slate-400">
                    Supports comma (,), semicolon (;), tab (\t), pipe (|) · UTF-8, UTF-8 BOM, Latin-1 · Any row or column count
                  </p>
                )}
              </div>
            </div>

            {/* Inspection & Metadata Summary Bar */}
            {inspection && (
              <div className="p-4 rounded-2xl border bg-slate-950/80 border-slate-800 space-y-3">
                <div className="flex flex-wrap items-center justify-between gap-2 text-xs font-mono">
                  <span className="text-slate-300 font-bold flex items-center space-x-1.5">
                    <Database className="w-4 h-4 text-indigo-400" />
                    <span>Schema: {schemaTypeStr}</span>
                  </span>
                  <div className="flex items-center space-x-2 text-[11px] text-slate-400">
                    <span className="px-2 py-0.5 rounded bg-slate-900 border border-slate-800">
                      Delimiter: <span className="text-indigo-300 font-bold">"{detectedDelimiter === '\t' ? '\\t' : detectedDelimiter}"</span>
                    </span>
                    <span className="px-2 py-0.5 rounded bg-slate-900 border border-slate-800">
                      Encoding: <span className="text-indigo-300 font-bold">{detectedEncoding}</span>
                    </span>
                    <span className="px-2 py-0.5 rounded bg-slate-900 border border-slate-800">
                      {totalRowsCount} rows · {totalColsCount} cols
                    </span>
                    {hasGroundTruthFlag && (
                      <span className="px-2 py-0.5 rounded bg-emerald-950/60 border border-emerald-600/50 text-emerald-300 font-bold">
                        Target: {targetColName || 'Detected'}
                      </span>
                    )}
                  </div>
                </div>

                {/* Semantic Mappings Badges */}
                <div className="flex flex-wrap gap-1.5 text-[10px]">
                  {Object.entries(semanticMappingsObj).map(([concept, col]) => (
                    <span
                      key={concept}
                      className={`px-2 py-0.5 rounded border ${
                        col
                          ? 'bg-slate-900 border-slate-700 text-slate-300'
                          : 'bg-slate-950 border-slate-800 text-slate-500 line-through'
                      }`}
                    >
                      {concept}: <span className="text-indigo-400 font-semibold">{col || 'not detected'}</span>
                    </span>
                  ))}
                </div>

                {/* Compatibility Feedback */}
                {selectedModelCompat && (
                  <div
                    className={`p-3 rounded-xl border text-xs font-mono flex items-start space-x-2 ${
                      selectedModelCompat.status === 'COMPATIBLE'
                        ? 'bg-emerald-950/40 border-emerald-500/50 text-emerald-300'
                        : selectedModelCompat.status === 'ADAPTABLE'
                        ? 'bg-amber-950/40 border-amber-500/50 text-amber-300'
                        : 'bg-rose-950/40 border-rose-500/50 text-rose-300'
                    }`}
                  >
                    {selectedModelCompat.status === 'COMPATIBLE' ? (
                      <CheckCircle2 className="w-4 h-4 text-emerald-400 shrink-0 mt-0.5" />
                    ) : (
                      <AlertTriangle className="w-4 h-4 text-amber-400 shrink-0 mt-0.5" />
                    )}
                    <div className="space-y-1 flex-1">
                      <div className="font-bold flex items-center justify-between">
                        <span>Model Compatibility: {selectedModelCompat.status} (Score: {(((selectedModelCompat.compatibility_score ?? 1)) * 100).toFixed(0)}%)</span>
                      </div>
                      <p className="text-[11px] font-sans text-slate-300">{selectedModelCompat.summary}</p>
                      {Array.isArray(selectedModelCompat.actions_needed) && selectedModelCompat.actions_needed.length > 0 && (
                        <div className="text-[10px] text-amber-300 font-mono">
                          Adaptations: {selectedModelCompat.actions_needed.join(' · ')}
                        </div>
                      )}
                    </div>
                  </div>
                )}
              </div>
            )}
          </div>

          {/* Quick-Load Controls & Model Selector */}
          <div className="w-full lg:w-88 space-y-4">
            <label className="block text-xs font-mono uppercase text-slate-400 font-semibold">
              Instant Demonstration Data
            </label>

            <div className="space-y-2">
              <button
                type="button"
                onClick={() => handleLoadSample('standard')}
                disabled={evaluating}
                className="w-full text-left p-3 rounded-xl bg-slate-950/80 hover:bg-indigo-950/40 border border-slate-800 hover:border-indigo-600/60 transition flex items-center justify-between text-xs disabled:opacity-50 cursor-pointer"
              >
                <div>
                  <div className="font-semibold text-white">Load Standard AML Batch</div>
                  <div className="text-[10px] text-slate-400 font-mono">25 multi-currency transactions (labeled)</div>
                </div>
                <ArrowRight className="w-4 h-4 text-indigo-400" />
              </button>

              <button
                type="button"
                onClick={() => handleLoadSample('synthetic')}
                disabled={evaluating}
                className="w-full text-left p-3 rounded-xl bg-slate-950/80 hover:bg-violet-950/40 border border-slate-800 hover:border-violet-600/60 transition flex items-center justify-between text-xs disabled:opacity-50 cursor-pointer"
              >
                <div>
                  <div className="font-semibold text-white">Load Synthetic Banking CSV</div>
                  <div className="text-[10px] text-slate-400 font-mono">Aliased fields + Target Leakage Protection</div>
                </div>
                <ArrowRight className="w-4 h-4 text-violet-400" />
              </button>

              <button
                type="button"
                onClick={() => handleLoadSample('unlabeled')}
                disabled={evaluating}
                className="w-full text-left p-3 rounded-xl bg-slate-950/80 hover:bg-emerald-950/40 border border-slate-800 hover:border-emerald-600/60 transition flex items-center justify-between text-xs disabled:opacity-50 cursor-pointer"
              >
                <div>
                  <div className="font-semibold text-white">Load Unlabeled Stream</div>
                  <div className="text-[10px] text-slate-400 font-mono">10 live banking stream transactions</div>
                </div>
                <ArrowRight className="w-4 h-4 text-emerald-400" />
              </button>
            </div>

            {/* Model Selection Dropdown */}
            <div className="pt-2 space-y-2">
              <label className="block text-xs font-mono uppercase text-slate-400 font-semibold flex items-center justify-between">
                <span>AML Model Selection</span>
                <span className="text-[10px] text-indigo-400 font-normal">
                  {Array.isArray(models) ? models.length : 0} registered
                </span>
              </label>
              <select
                value={selectedModelId}
                onChange={(e) => setSelectedModelId(e.target.value)}
                className="w-full bg-slate-950 border border-slate-800 text-slate-200 text-xs rounded-xl px-3 py-2 focus:border-indigo-500 focus:outline-none"
              >
                <optgroup label="Best Available & Ensemble">
                  <option value="auto">Auto-Detect Best Matching Model (Validation PR-AUC)</option>
                  <option value="ensemble_hybrid">Hybrid Ensemble (α* = 1.00 FedProx/FedAvg)</option>
                </optgroup>

                <optgroup label="Centralized Baseline Models">
                  <option value="centralized_lstm">Centralized Baseline (LSTM, hidden=64)</option>
                  <option value="centralized_gru">Centralized Baseline (GRU, hidden=64)</option>
                  <option value="centralized_bilstm">Centralized Baseline (BiLSTM, hidden=64)</option>
                </optgroup>

                <optgroup label="Federated IID (FedAvg)">
                  <option value="hi_small_lstm_fedavg_iid_seed42">FedAvg IID LSTM (HI-Small Champion, Thr=0.3378)</option>
                  <option value="hi_small_gru_fedavg_iid_seed42">FedAvg IID GRU (HI-Small, Thr=0.3601)</option>
                  <option value="hi_small_bilstm_fedavg_iid_seed42">FedAvg IID BiLSTM (HI-Small, Thr=0.3204)</option>
                </optgroup>

                <optgroup label="Federated IID (FedProx μ=0.0001)">
                  <option value="hi_small_lstm_fedprox_mu0.0001_iid_seed42">FedProx (μ=0.0001) IID LSTM (HI-Small, Thr=0.3378)</option>
                  <option value="hi_small_gru_fedprox_mu0.0001_iid_seed42">FedProx (μ=0.0001) IID GRU (HI-Small, Thr=0.3382)</option>
                  <option value="hi_small_bilstm_fedprox_mu0.0001_iid_seed42">FedProx (μ=0.0001) IID BiLSTM (HI-Small, Thr=0.3421)</option>
                </optgroup>

                <optgroup label="Federated Typology Non-IID (FedAvg)">
                  <option value="hi_small_lstm_fedavg_seed42">FedAvg Non-IID LSTM (HI-Small, Thr=0.3204)</option>
                  <option value="hi_small_gru_fedavg_seed42">FedAvg Non-IID GRU (HI-Small, Thr=0.3150)</option>
                  <option value="hi_small_bilstm_fedavg_seed42">FedAvg Non-IID BiLSTM (HI-Small, Thr=0.3120)</option>
                </optgroup>

                <optgroup label="Federated Typology Non-IID (FedProx μ=0.0001)">
                  <option value="hi_small_lstm_fedprox_mu0.0001_seed42">FedProx (μ=0.0001) Non-IID LSTM (HI-Small, Thr=0.3204)</option>
                  <option value="hi_small_gru_fedprox_mu0.0001_seed42">FedProx (μ=0.0001) Non-IID GRU (HI-Small, Thr=0.3150)</option>
                  <option value="hi_small_bilstm_fedprox_mu0.0001_seed42">FedProx (μ=0.0001) Non-IID BiLSTM (HI-Small, Thr=0.3120)</option>
                </optgroup>

                {Array.isArray(models) && models.filter(m => {
                  const id = m?.model_id || '';
                  return !id.startsWith('hi_small_') && !id.startsWith('centralized_') && id !== 'auto' && id !== 'ensemble_hybrid';
                }).length > 0 && (
                  <optgroup label="Additional / Custom Models">
                    {models.filter(m => {
                      const id = m?.model_id || '';
                      return !id.startsWith('hi_small_') && !id.startsWith('centralized_') && id !== 'auto' && id !== 'ensemble_hybrid';
                    }).map((m) => (
                      <option key={m.model_id} value={m.model_id}>
                        {m.model_name}{m.is_custom ? ' (Custom)' : ''}
                      </option>
                    ))}
                  </optgroup>
                )}
              </select>

              {/* Threshold Selection & Quick Presets */}
              <div className="space-y-1.5 pt-1">
                <div className="flex items-center justify-between text-xs text-slate-400 font-mono">
                  <span>Decision Threshold:</span>
                  <span className="text-indigo-400 font-bold">
                    {customThreshold !== '' ? customThreshold : 'Model Frozen Default'}
                  </span>
                </div>
                <div className="flex items-center space-x-1.5 flex-wrap gap-1">
                  <button
                    type="button"
                    onClick={() => setCustomThreshold('')}
                    className={`px-2 py-0.5 rounded text-[10px] font-mono transition cursor-pointer ${
                      customThreshold === ''
                        ? 'bg-indigo-600 text-white font-bold'
                        : 'bg-slate-900 text-slate-400 hover:text-slate-200 border border-slate-800'
                    }`}
                  >
                    Auto
                  </button>
                  {['0.10', '0.20', '0.30', '0.3378', '0.40', '0.50', '0.70'].map((preset) => (
                    <button
                      key={preset}
                      type="button"
                      onClick={() => setCustomThreshold(preset)}
                      className={`px-2 py-0.5 rounded text-[10px] font-mono transition cursor-pointer ${
                        customThreshold === preset
                          ? 'bg-indigo-600 text-white font-bold'
                          : 'bg-slate-900 text-slate-400 hover:text-slate-200 border border-slate-800'
                      }`}
                    >
                      {preset}
                    </button>
                  ))}
                </div>
              </div>

              {/* Row Limit Selector */}
              <div className="flex items-center justify-between text-xs text-slate-400 font-mono pt-1">
                <span>Row Limit:</span>
                <select
                  value={rowLimit}
                  onChange={(e) => setRowLimit(Number(e.target.value))}
                  className="bg-slate-950 border border-slate-800 text-slate-300 text-xs rounded-lg px-2 py-1"
                >
                  <option value={500}>500 rows</option>
                  <option value={1000}>1,000 rows</option>
                  <option value={5000}>5,000 rows (Default)</option>
                  <option value={20000}>20,000 rows</option>
                  <option value={100000}>100,000 rows</option>
                  <option value={500000}>500,000 rows</option>
                  <option value={1000000}>1,000,000 rows (Max)</option>
                </select>
              </div>
            </div>
          </div>
        </div>

        {/* First 5 Rows Table Preview */}
        {previewData.length > 0 && (
          <div className="bg-slate-950 p-4 rounded-2xl border border-slate-800 space-y-2">
            <div className="text-slate-300 font-bold text-xs flex items-center justify-between">
              <span className="flex items-center space-x-1.5">
                <FileText className="w-3.5 h-3.5 text-indigo-400" />
                <span>Uploaded CSV Preview (First {previewData.length} Rows):</span>
              </span>
              <span className="text-[11px] font-mono text-slate-500">{inspection?.num_rows} total rows</span>
            </div>
            <div className="overflow-x-auto max-h-48 rounded-xl border border-slate-800/80">
              <table className="w-full text-[11px] text-left text-slate-300 font-mono">
                <thead className="bg-slate-900/90 text-slate-400 uppercase text-[10px] sticky top-0">
                  <tr>
                    {previewCols.map((col) => (
                      <th key={col} className="px-3 py-1.5 whitespace-nowrap border-b border-slate-800">
                        {col}
                      </th>
                    ))}
                  </tr>
                </thead>
                <tbody className="divide-y divide-slate-800/60">
                  {previewData.map((row, rIdx) => (
                    <tr key={rIdx} className="hover:bg-slate-900/40">
                      {previewCols.map((col) => (
                        <td key={col} className="px-3 py-1 whitespace-nowrap truncate max-w-xs">
                          {String(row[col] ?? '')}
                        </td>
                      ))}
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        )}

        {/* Collapsible Column Mapping Override Panel */}
        {inspection && (
          <div className="border border-slate-800/80 rounded-2xl bg-slate-950/40 p-3 space-y-3">
            <button
              type="button"
              onClick={() => setShowMappingPanel(!showMappingPanel)}
              className="w-full flex items-center justify-between text-xs text-slate-300 font-mono hover:text-white cursor-pointer"
            >
              <span className="flex items-center space-x-2">
                <Settings className="w-3.5 h-3.5 text-indigo-400" />
                <span>Review / Override Column Mappings</span>
              </span>
              {showMappingPanel ? <ChevronUp className="w-4 h-4" /> : <ChevronDown className="w-4 h-4" />}
            </button>

            {showMappingPanel && (
              <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-3 pt-2 text-xs">
                {['timestamp', 'from_account', 'to_account', 'amount', 'currency', 'payment_format'].map((concept) => (
                  <div key={concept} className="space-y-1">
                    <label className="text-[11px] font-mono text-slate-400 capitalize">{concept.replace('_', ' ')}:</label>
                    <select
                      value={customMappings[concept] || ''}
                      onChange={(e) =>
                        setCustomMappings({
                          ...customMappings,
                          [concept]: e.target.value,
                        })
                      }
                      className="w-full bg-slate-900 border border-slate-800 text-slate-200 text-xs rounded-lg px-2.5 py-1.5 font-mono"
                    >
                      <option value="">(None / Optional)</option>
                      {(inspection?.column_names || []).map((col) => (
                        <option key={col} value={col}>
                          {col}
                        </option>
                      ))}
                    </select>
                  </div>
                ))}
              </div>
            )}
          </div>
        )}

        {/* Mode B: Dynamic Custom Model Training Toggle */}
        {inspection && (
          <div className="border border-slate-800/80 rounded-2xl bg-slate-950/40 p-4 space-y-3">
            <div className="flex items-center justify-between">
              <div className="flex items-center space-x-2">
                <Cpu className="w-4 h-4 text-violet-400" />
                <span className="text-xs font-bold text-white">Mode B: Train Custom AML Model on Uploaded CSV</span>
              </div>
              <button
                type="button"
                onClick={() => setShowTrainSection(!showTrainSection)}
                className="px-3 py-1 rounded-lg bg-violet-950/80 hover:bg-violet-900 text-violet-300 border border-violet-700/50 text-xs font-semibold cursor-pointer transition"
              >
                {showTrainSection ? 'Hide Trainer' : 'Configure Custom Training'}
              </button>
            </div>

            {showTrainSection && (
              <div className="p-4 rounded-xl bg-slate-900/90 border border-slate-800 space-y-4 pt-3">
                <p className="text-xs text-slate-400">
                  Train a new AML classifier on this labeled dataset. The model will be automatically evaluated, threshold-tuned, and registered without altering existing federated checkpoints.
                </p>

                <div className="grid grid-cols-1 sm:grid-cols-3 gap-3">
                  <div className="space-y-1">
                    <label className="text-[11px] font-mono text-slate-400">Model Name:</label>
                    <input
                      type="text"
                      value={customModelName}
                      onChange={(e) => setCustomModelName(e.target.value)}
                      className="w-full bg-slate-950 border border-slate-800 text-slate-200 text-xs rounded-lg px-3 py-1.5 font-mono"
                    />
                  </div>

                  <div className="space-y-1">
                    <label className="text-[11px] font-mono text-slate-400">Target Label Column:</label>
                    <select
                      value={targetColumnOverride}
                      onChange={(e) => setTargetColumnOverride(e.target.value)}
                      className="w-full bg-slate-950 border border-slate-800 text-slate-200 text-xs rounded-lg px-3 py-1.5 font-mono"
                    >
                      <option value="">Select Target Column</option>
                      {(inspection?.column_names || []).map((col) => (
                        <option key={col} value={col}>
                          {col}
                        </option>
                      ))}
                    </select>
                  </div>

                  <div className="space-y-1">
                    <label className="text-[11px] font-mono text-slate-400">Algorithm:</label>
                    <select
                      value={customModelType}
                      onChange={(e) => setCustomModelType(e.target.value)}
                      className="w-full bg-slate-950 border border-slate-800 text-slate-200 text-xs rounded-lg px-3 py-1.5 font-mono"
                    >
                      <option value="HistGradientBoosting">HistGradientBoosting (Fast & Robust)</option>
                      <option value="RandomForest">Random Forest (Ensemble)</option>
                    </select>
                  </div>
                </div>

                <div className="flex items-center justify-end pt-2">
                  <button
                    type="button"
                    onClick={handleTrainCustomModel}
                    disabled={trainingInProgress || !targetColumnOverride}
                    className="flex items-center space-x-2 px-5 py-2.5 rounded-xl bg-violet-600 hover:bg-violet-500 text-white text-xs font-semibold shadow-lg shadow-violet-600/30 disabled:opacity-50 cursor-pointer transition"
                  >
                    {trainingInProgress ? (
                      <>
                        <RefreshCw className="w-4 h-4 animate-spin text-white" />
                        <span>Training Classifier...</span>
                      </>
                    ) : (
                      <>
                        <Cpu className="w-4 h-4 text-white" />
                        <span>Train & Register Model</span>
                      </>
                    )}
                  </button>
                </div>

                {trainSuccessMessage && (
                  <div className="p-3 rounded-xl bg-emerald-950/60 border border-emerald-500/60 text-emerald-300 text-xs font-mono flex items-start space-x-2">
                    <CheckCircle2 className="w-4 h-4 text-emerald-400 shrink-0 mt-0.5" />
                    <span>{trainSuccessMessage}</span>
                  </div>
                )}
              </div>
            )}
          </div>
        )}

        {/* Action Button Bar */}
        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 pt-3 border-t border-slate-800/80">
          <div className="text-xs text-slate-400 font-mono flex items-center space-x-1.5">
            <Zap className="w-3.5 h-3.5 text-emerald-400" />
            <span>Mode A: Predict with Selected Model ({selectedModelId})</span>
          </div>

          <div className="flex items-center space-x-3">
            <button
              onClick={handleRunMultiModelEvaluation}
              disabled={multiModelEvaluating || evaluating || !csvContent}
              className="flex items-center space-x-2 px-5 py-3 rounded-xl bg-violet-900/60 hover:bg-violet-800 text-violet-200 border border-violet-700/50 font-medium text-xs sm:text-sm transition shadow-lg shadow-violet-900/20 disabled:opacity-50 cursor-pointer"
              title="Preprocess CSV once and benchmark predictions across all 16 Centralized, FedAvg, FedProx, and Ensemble models"
            >
              {multiModelEvaluating ? (
                <>
                  <RefreshCw className="w-4 h-4 animate-spin text-violet-300" />
                  <span>Evaluating All Models...</span>
                </>
              ) : (
                <>
                  <Layers className="w-4 h-4 text-violet-300" />
                  <span>Compare All Models on CSV</span>
                </>
              )}
            </button>

            <button
              onClick={handleRunPrediction}
              disabled={evaluating || multiModelEvaluating || !csvContent}
              className="flex items-center space-x-2 px-6 py-3 rounded-xl bg-indigo-600 hover:bg-indigo-500 text-white font-medium text-xs sm:text-sm transition shadow-lg shadow-indigo-600/30 disabled:opacity-50 cursor-pointer"
            >
              {evaluating ? (
                <>
                  <RefreshCw className="w-4 h-4 animate-spin text-white" />
                  <span>Evaluating Transactions...</span>
                </>
              ) : (
                <>
                  <Sparkles className="w-4 h-4 text-white" />
                  <span>Execute AML Prediction</span>
                </>
              )}
            </button>
          </div>
        </div>
      </div>

      {/* Backend API Error Banner */}
      {apiError && (
        <div className="p-4 bg-rose-950/60 border border-rose-500/50 rounded-2xl flex items-start space-x-3 text-rose-200 text-sm">
          <AlertTriangle className="w-5 h-5 text-rose-400 shrink-0 mt-0.5" />
          <div>
            <h4 className="font-semibold text-rose-300">CSV Evaluation Notice</h4>
            <p className="text-xs text-rose-200/90 mt-1 font-mono">{apiError}</p>
          </div>
        </div>
      )}

      {/* Quick Results Summary Strip (if evaluation complete) */}
      {currentResult && (
        <div className="space-y-6">
          {multiModelResult && (
            <div className="p-3 bg-violet-950/40 border border-violet-800/60 rounded-xl text-xs text-violet-200 flex items-center justify-between">
              <span className="flex items-center space-x-2">
                <Layers className="w-4 h-4 text-violet-400" />
                <span>Multi-model benchmark active ({multiModelResult.models_evaluated_count} models evaluated on this CSV).</span>
              </span>
              <button
                onClick={() => onNavigateTab('analytics')}
                className="text-violet-300 hover:text-white font-semibold underline cursor-pointer"
              >
                View Comparative Matrix →
              </button>
            </div>
          )}

          <div className="flex items-center justify-between border-b border-slate-800 pb-2">
            <h2 className="text-lg font-bold text-white flex items-center space-x-2">
              <CheckCircle2 className="w-5 h-5 text-emerald-400" />
              <span>Evaluation Summary for {currentResult.filename}</span>
            </h2>
            <div className="flex items-center space-x-2">
              <button
                onClick={handleDownloadPredictions}
                className="text-xs text-emerald-400 hover:text-emerald-300 font-semibold flex items-center space-x-1 cursor-pointer mr-3"
              >
                <Download className="w-3.5 h-3.5" />
                <span>Download Predictions CSV</span>
              </button>
              <button
                onClick={() => onNavigateTab('analytics')}
                className="text-xs text-indigo-400 hover:text-indigo-300 font-semibold flex items-center space-x-1 cursor-pointer"
              >
                <span>Full Analytics</span>
                <ArrowRight className="w-3.5 h-3.5" />
              </button>
            </div>
          </div>

          <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-6 gap-3">
            <div className="bg-slate-900 border border-slate-800 rounded-2xl p-4 shadow-md">
              <span className="text-[10px] text-slate-400 font-mono uppercase block">Total Transactions</span>
              <span className="text-2xl font-black font-mono text-white mt-1 block">
                {currentResult.total_rows || currentResult.evaluated_sequences}
              </span>
            </div>
            <div className="bg-slate-900 border border-slate-800 rounded-2xl p-4 shadow-md">
              <span className="text-[10px] text-slate-400 font-mono uppercase block flex items-center space-x-1">
                <ShieldAlert className="w-3 h-3 text-rose-400" />
                <span>Suspicious Flagged</span>
              </span>
              <span className="text-2xl font-black font-mono text-rose-400 mt-1 block">
                {currentResult.flagged_laundering_count}
              </span>
            </div>
            <div className="bg-slate-900 border border-slate-800 rounded-2xl p-4 shadow-md">
              <span className="text-[10px] text-slate-400 font-mono uppercase block flex items-center space-x-1">
                <ShieldCheck className="w-3 h-3 text-emerald-400" />
                <span>Legitimate Cleared</span>
              </span>
              <span className="text-2xl font-black font-mono text-emerald-400 mt-1 block">
                {currentResult.cleared_count}
              </span>
            </div>
            <div className="bg-slate-900 border border-slate-800 rounded-2xl p-4 shadow-md">
              <span className="text-[10px] text-slate-400 font-mono uppercase block">High-Risk Count</span>
              <span className="text-2xl font-black font-mono text-amber-400 mt-1 block">
                {currentResult.risk_counts?.high_risk ?? currentResult.flagged_laundering_count}
              </span>
            </div>
            <div className="bg-slate-900 border border-slate-800 rounded-2xl p-4 shadow-md">
              <span className="text-[10px] text-slate-400 font-mono uppercase block">Model Applied</span>
              <span className="text-sm font-bold font-mono text-indigo-300 mt-2 block truncate">
                {currentResult.model_name}
              </span>
            </div>
            <div className="bg-slate-900 border border-slate-800 rounded-2xl p-4 shadow-md">
              <span className="text-[10px] text-slate-400 font-mono uppercase block">Latency</span>
              <span className="text-2xl font-black font-mono text-slate-300 mt-1 block">
                {currentResult.execution_time_ms} ms
              </span>
            </div>
          </div>

          {/* If Ground Truth Available, show Verification Metrics Strip */}
          {currentResult.evaluation_metrics && (
            <div className="p-4 rounded-2xl bg-indigo-950/30 border border-indigo-700/40 space-y-3">
              <div className="text-xs font-mono uppercase text-indigo-300 font-bold flex items-center justify-between">
                <span>Ground Truth Verification Metrics</span>
                <span className="text-[10px] text-indigo-400 font-normal">Supervised Evaluation</span>
              </div>
              <div className="grid grid-cols-2 sm:grid-cols-5 gap-3 text-center">
                <div className="p-2.5 rounded-xl bg-slate-900 border border-slate-800">
                  <span className="text-[10px] text-slate-400 block">F1-Score</span>
                  <span className="text-xl font-bold font-mono text-white mt-0.5 block">
                    {formatMetric(currentResult.evaluation_metrics.f1_score, 4)}
                  </span>
                </div>
                <div className="p-2.5 rounded-xl bg-slate-900 border border-slate-800">
                  <span className="text-[10px] text-slate-400 block">Precision</span>
                  <span className="text-xl font-bold font-mono text-white mt-0.5 block">
                    {formatMetric(currentResult.evaluation_metrics.precision, 4)}
                  </span>
                </div>
                <div className="p-2.5 rounded-xl bg-slate-900 border border-slate-800">
                  <span className="text-[10px] text-slate-400 block">Recall</span>
                  <span className="text-xl font-bold font-mono text-white mt-0.5 block">
                    {formatMetric(currentResult.evaluation_metrics.recall, 4)}
                  </span>
                </div>
                <div className="p-2.5 rounded-xl bg-slate-900 border border-slate-800">
                  <span className="text-[10px] text-slate-400 block">ROC-AUC</span>
                  <span className="text-xl font-bold font-mono text-indigo-300 mt-0.5 block">
                    {formatMetric(currentResult.evaluation_metrics.roc_auc, 4)}
                  </span>
                </div>
                <div className="p-2.5 rounded-xl bg-slate-900 border border-slate-800">
                  <span className="text-[10px] text-slate-400 block">PR-AUC</span>
                  <span className="text-xl font-bold font-mono text-indigo-300 mt-0.5 block">
                    {formatMetric(currentResult.evaluation_metrics.pr_auc, 4)}
                  </span>
                </div>
              </div>
            </div>
          )}

          {/* Quick Action Navigation Cards */}
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <div
              onClick={() => onNavigateTab('analytics')}
              className="p-5 bg-gradient-to-br from-slate-900 to-indigo-950/40 border border-slate-800 hover:border-indigo-600/60 rounded-2xl cursor-pointer transition shadow-lg group"
            >
              <div className="flex items-center justify-between">
                <div className="flex items-center space-x-3">
                  <div className="p-2.5 rounded-xl bg-indigo-600/20 text-indigo-400 border border-indigo-500/30">
                    <BarChart2 className="w-5 h-5" />
                  </div>
                  <div>
                    <h3 className="font-bold text-white group-hover:text-indigo-300 transition">
                      Detailed Metrics & Visual Analytics
                    </h3>
                    <p className="text-xs text-slate-400 mt-0.5">
                      Inspect Confusion Matrix, ROC curve, Precision-Recall curve, and risk score distribution.
                    </p>
                  </div>
                </div>
                <ArrowRight className="w-5 h-5 text-slate-500 group-hover:text-indigo-400 transition" />
              </div>
            </div>

            <div
              onClick={() => onNavigateTab('history')}
              className="p-5 bg-gradient-to-br from-slate-900 to-violet-950/40 border border-slate-800 hover:border-violet-600/60 rounded-2xl cursor-pointer transition shadow-lg group"
            >
              <div className="flex items-center justify-between">
                <div className="flex items-center space-x-3">
                  <div className="p-2.5 rounded-xl bg-violet-600/20 text-violet-400 border border-violet-500/30">
                    <History className="w-5 h-5" />
                  </div>
                  <div>
                    <h3 className="font-bold text-white group-hover:text-violet-300 transition">
                      Transaction-Level Prediction Results
                    </h3>
                    <p className="text-xs text-slate-400 mt-0.5">
                      Search, sort, filter by risk level, explain individual transactions, and export CSV.
                    </p>
                  </div>
                </div>
                <ArrowRight className="w-5 h-5 text-slate-500 group-hover:text-violet-400 transition" />
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};
