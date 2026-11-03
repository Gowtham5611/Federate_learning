import React, { useState } from 'react';
import { formatMetric } from '../utils/formatters';
import {
  CsvEvaluateResponse,
  SupportedDataset,
  TransactionPredictionItem,
} from '../types/api';
import { apiService, ApiError } from '../services/apiService';
import {
  Upload,
  FileSpreadsheet,
  AlertTriangle,
  CheckCircle2,
  Download,
  Search,
  Eye,
  ArrowRight,
  Sparkles,
  RefreshCw,
  BarChart2,
  TrendingUp,
  Check,
  Info,
} from 'lucide-react';

interface CsvEvaluationPageProps {
  selectedDataset: SupportedDataset;
  onSelectDataset: (dataset: SupportedDataset) => void;
  onExplainTransaction: (item: TransactionPredictionItem, dataset: SupportedDataset) => void;
}

const RAW_EXPECTED_COLS = [
  'Timestamp',
  'From Bank',
  'Account',
  'To Bank',
  'Amount Received',
  'Receiving Currency',
  'Amount Paid',
  'Payment Currency',
  'Payment Format',
];

const PROCESSED_EXPECTED_COLS = [
  'timestamp',
  'from_bank',
  'from_account',
  'to_bank',
  'to_account',
  'amount_paid',
];

export const CsvEvaluationPage: React.FC<CsvEvaluationPageProps> = ({
  selectedDataset,
  onSelectDataset,
  onExplainTransaction,
}) => {
  // Input state
  const [csvContent, setCsvContent] = useState<string>('');
  const [fileName, setFileName] = useState<string>('');
  const [rowLimit, setRowLimit] = useState<number>(5000);

  // Validation states
  const [validationError, setValidationError] = useState<string | null>(null);
  const [detectedSchema, setDetectedSchema] = useState<'RAW' | 'PROCESSED' | 'UNKNOWN'>('UNKNOWN');
  const [hasDetectedLabels, setHasDetectedLabels] = useState<boolean>(false);
  const [columnCount, setColumnCount] = useState<number>(0);
  const [rowCount, setRowCount] = useState<number>(0);

  // Execution states
  const [evaluating, setEvaluating] = useState<boolean>(false);
  const [result, setResult] = useState<CsvEvaluateResponse | null>(null);
  const [apiError, setApiError] = useState<string | null>(null);

  // Filter & Search states for predictions table
  const [statusFilter, setStatusFilter] = useState<'ALL' | 'FLAGGED' | 'CLEARED' | 'NO_HISTORY'>('ALL');
  const [searchQuery, setSearchQuery] = useState<string>('');
  const [currentPage, setCurrentPage] = useState<number>(1);
  const pageSize = 20;

  // Validate CSV text locally before submitting
  const validateCsvString = (text: string): boolean => {
    setValidationError(null);
    setApiError(null);

    if (!text || text.trim().length === 0) {
      setValidationError('Uploaded CSV file is empty. Please provide valid transaction rows.');
      setDetectedSchema('UNKNOWN');
      return false;
    }

    const lines = text.trim().split(/\r?\n/).filter((l) => l.trim().length > 0);
    if (lines.length <= 1) {
      setValidationError('CSV file contains only a header line with no data rows.');
      setDetectedSchema('UNKNOWN');
      return false;
    }

    setRowCount(lines.length - 1);

    // Parse header
    const headerLine = lines[0];
    const headers = headerLine.split(',').map((h) => h.trim().replace(/^["']|["']$/g, ''));
    setColumnCount(headers.length);

    // Check Raw vs Processed schema
    const isRaw = RAW_EXPECTED_COLS.every((col) =>
      headers.some((h) => h.toLowerCase() === col.toLowerCase())
    );
    const isProcessed = PROCESSED_EXPECTED_COLS.every((col) =>
      headers.some((h) => h.toLowerCase() === col.toLowerCase())
    );

    if (isProcessed) {
      setDetectedSchema('PROCESSED');
    } else if (isRaw) {
      setDetectedSchema('RAW');
    } else {
      setDetectedSchema('UNKNOWN');
      setValidationError(
        `Schema Mismatch: Uploaded CSV columns do not match expected IBM AML schemas.\n` +
          `Expected either:\n` +
          `1) Raw IBM AML format (11 columns: Timestamp, From Bank, Account, To Bank, Account.1, Amount Received, Receiving Currency, Amount Paid, Payment Currency, Payment Format)\n` +
          `2) Processed model format (27 columns: timestamp, from_bank, from_account, to_bank, to_account, amount_paid, ...)\n` +
          `Found headers: [${headers.slice(0, 10).join(', ')}${headers.length > 10 ? ', ...' : ''}]`
      );
      return false;
    }

    // Check for ground truth label column
    const labelCol = headers.find(
      (h) =>
        h.toLowerCase() === 'is laundering' ||
        h.toLowerCase() === 'is_laundering' ||
        h.toLowerCase() === 'islaundering'
    );
    setHasDetectedLabels(Boolean(labelCol));

    // Validate a few data rows for non-empty values
    const sampleRow = lines[1].split(',');
    if (sampleRow.length !== headers.length && sampleRow.length !== headers.length + 1) {
      setValidationError(
        `Column count mismatch between header (${headers.length} columns) and first data row (${sampleRow.length} values).`
      );
      return false;
    }

    return true;
  };

  // Handle local file selection
  const handleFileChange = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;

    if (!file.name.toLowerCase().endsWith('.csv') && !file.name.toLowerCase().endsWith('.txt')) {
      setValidationError('Unsupported file type. Only standard CSV files (.csv, .txt) are accepted.');
      setFileName(file.name);
      return;
    }

    if (file.size > 10 * 1024 * 1024) {
      setValidationError('File size limit exceeded. Uploaded file is larger than 10MB.');
      setFileName(file.name);
      return;
    }

    setFileName(file.name);
    setResult(null);

    try {
      const text = await file.text();
      setCsvContent(text);
      validateCsvString(text);
    } catch (err) {
      setValidationError('Failed to read file content: ' + String(err));
    }
  };

  // One-click load built-in sample
  const handleLoadSample = async (ds: SupportedDataset) => {
    setValidationError(null);
    setApiError(null);
    setEvaluating(true);
    setResult(null);
    onSelectDataset(ds);

    try {
      const url = apiService.getSampleCsvUrl(ds);
      const resp = await fetch(url);
      if (!resp.ok) {
        throw new Error(`Failed to download sample CSV (${resp.status})`);
      }
      const text = await resp.text();
      const sampleName = ds === 'HI-Small' ? 'hi_small_sample_25.csv' : 'li_medium_sample_25.csv';
      setCsvContent(text);
      setFileName(sampleName);

      const isValid = validateCsvString(text);
      if (!isValid) return;

      // Automatically evaluate
      const evalResult = await apiService.evaluateCsv({
        csv_content: text,
        dataset: ds,
        filename: sampleName,
        row_limit: rowLimit,
      });
      setResult(evalResult);
    } catch (err: unknown) {
      const msg = err instanceof ApiError ? err.message : err instanceof Error ? err.message : 'Sample loading failed';
      setApiError(msg);
    } finally {
      setEvaluating(false);
    }
  };

  // Submit CSV evaluation
  const handleRunEvaluation = async () => {
    if (!csvContent || csvContent.trim().length === 0) {
      setValidationError('Please select or paste a CSV file before running evaluation.');
      return;
    }

    const isValid = validateCsvString(csvContent);
    if (!isValid) {
      return;
    }

    setEvaluating(true);
    setApiError(null);
    setResult(null);
    setCurrentPage(1);

    try {
      const resp = await apiService.evaluateCsv({
        csv_content: csvContent,
        dataset: selectedDataset,
        filename: fileName || 'transaction_batch.csv',
        row_limit: rowLimit,
      });
      setResult(resp);
    } catch (err: unknown) {
      let msg = 'Evaluation request failed';
      if (err instanceof ApiError) {
        if (err.status === 404) {
          msg = `Evaluation route not found on backend (${err.message}). Verify FastAPI service is running with evaluation router registered.`;
        } else if (err.status === 0) {
          msg = `Cannot reach FedSecureAI backend. Ensure the FastAPI service is running at http://127.0.0.1:8000.`;
        } else {
          msg = err.message;
        }
      } else if (err instanceof Error) {
        msg = err.message;
      }
      setApiError(msg);
    } finally {
      setEvaluating(false);
    }
  };

  // Export predictions as CSV
  const handleExportPredictions = () => {
    if (!result || !result.predictions.length) return;

    const headers = [
      'Row ID',
      'Timestamp',
      'From Account',
      'To Account',
      'Amount Paid',
      'Predicted Laundering Probability',
      'Frozen Decision Threshold',
      'Model Decision',
      'Ground Truth Label',
      'Prediction Match',
      'Sequence Length',
    ];

    const rows = result.predictions.map((p) => {
      let match = 'UNLABELED';
      if (p.ground_truth_label !== null) {
        const isFlagged = p.decision === 'FLAGGED_LAUNDERING';
        if (p.ground_truth_label === 1 && isFlagged) match = 'TRUE_POSITIVE';
        else if (p.ground_truth_label === 0 && !isFlagged) match = 'TRUE_NEGATIVE';
        else if (p.ground_truth_label === 0 && isFlagged) match = 'FALSE_POSITIVE';
        else if (p.ground_truth_label === 1 && !isFlagged) match = 'FALSE_NEGATIVE';
      }

      return [
        p.row_id,
        p.timestamp,
        p.from_account,
        p.to_account,
        p.amount,
        p.predicted_probability,
        p.decision_threshold,
        p.decision,
        p.ground_truth_label !== null ? p.ground_truth_label : 'N/A',
        match,
        p.sequence_length,
      ];
    });

    const csvStr = [
      headers.join(','),
      ...rows.map((r) => r.map((c) => `"${String(c).replace(/"/g, '""')}"`).join(',')),
    ].join('\n');

    const blob = new Blob([csvStr], { type: 'text/csv;charset=utf-8;' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `fedsecureai_${result.dataset.toLowerCase()}_predictions_${Date.now()}.csv`;
    a.click();
    URL.revokeObjectURL(url);
  };

  // Filtered prediction items
  const filteredPredictions =
    result?.predictions.filter((item) => {
      const isFlagged = item.decision === 'FLAGGED_LAUNDERING' || item.predicted_label === 'SUSPICIOUS';
      const isNoHistory = item.decision === 'LOW_CONFIDENCE_NO_HISTORY' || item.predicted_label === 'LOW_CONFIDENCE_NO_HISTORY';
      const isCleared = item.decision === 'CLEARED_LEGITIMATE' || item.predicted_label === 'LEGITIMATE';

      if (statusFilter === 'FLAGGED' && !isFlagged) return false;
      if (statusFilter === 'CLEARED' && !isCleared) return false;
      if (statusFilter === 'NO_HISTORY' && !isNoHistory) return false;

      if (searchQuery.trim()) {
        const q = searchQuery.toLowerCase();
        const match =
          item.from_account.toLowerCase().includes(q) ||
          item.to_account.toLowerCase().includes(q) ||
          item.timestamp.toLowerCase().includes(q) ||
          String(item.row_id) === q;
        if (!match) return false;
      }
      return true;
    }) || [];

  const totalPages = Math.ceil(filteredPredictions.length / pageSize) || 1;
  const paginatedPredictions = filteredPredictions.slice(
    (currentPage - 1) * pageSize,
    currentPage * pageSize
  );

  const previewLines = csvContent ? csvContent.trim().split(/\r?\n/).slice(0, 5) : [];

  return (
    <div className="space-y-8 animate-fade-in">
      {/* Page Header */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 border-b border-slate-800 pb-5">
        <div>
          <h1 className="text-2xl sm:text-3xl font-black text-white tracking-tight flex items-center space-x-2">
            <FileSpreadsheet className="w-7 h-7 text-indigo-400" />
            <span>CSV Batch Upload & AML Evaluation</span>
          </h1>
          <p className="text-xs text-slate-400 mt-1">
            End-to-end evaluation: dataset schema validation, temporal sequence assembly, champion model inference, and label-aware metrics.
          </p>
        </div>

        {/* Dataset Pipeline Switcher */}
        <div className="flex items-center space-x-2 bg-slate-900 p-1.5 rounded-2xl border border-slate-800">
          <button
            onClick={() => onSelectDataset('HI-Small')}
            className={`px-3.5 py-1.5 rounded-xl text-xs font-bold transition ${
              selectedDataset === 'HI-Small'
                ? 'bg-indigo-600 text-white shadow-lg shadow-indigo-600/30'
                : 'text-slate-400 hover:text-slate-200'
            }`}
          >
            HI-Small (LSTM • thr: 0.3378)
          </button>
          <button
            onClick={() => onSelectDataset('LI-Medium')}
            className={`px-3.5 py-1.5 rounded-xl text-xs font-bold transition ${
              selectedDataset === 'LI-Medium'
                ? 'bg-violet-600 text-white shadow-lg shadow-violet-600/30'
                : 'text-slate-400 hover:text-slate-200'
            }`}
          >
            LI-Medium (BiLSTM • thr: 0.2385)
          </button>
        </div>
      </div>

      {/* Upload Zone & Quick Sample Loaders */}
      <div className="bg-slate-900 border border-slate-800 rounded-3xl p-6 shadow-xl space-y-6">
        <div className="flex flex-col md:flex-row md:items-start justify-between gap-6">
          {/* File Upload / Drag & Drop */}
          <div className="flex-1 space-y-4">
            <label className="block text-xs font-mono uppercase text-slate-400 font-semibold">
              Select or Drop CSV File
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
                    <span className="text-indigo-400 font-bold">{fileName}</span>
                  ) : (
                    <span>Click to browse or drop CSV file here</span>
                  )}
                </div>
                <p className="text-xs text-slate-400">
                  Supports raw IBM AML format (11 cols) or processed split format (27 cols). Max 10MB.
                </p>
              </div>
            </div>

            {/* Local Pre-Validation Status Pill */}
            {detectedSchema !== 'UNKNOWN' && (
              <div className="flex flex-wrap items-center gap-2 pt-1 text-xs font-mono">
                <span className="px-2.5 py-1 rounded-lg bg-indigo-950 text-indigo-300 border border-indigo-800/80 flex items-center space-x-1">
                  <Check className="w-3.5 h-3.5 text-indigo-400" />
                  <span>Schema: {detectedSchema === 'RAW' ? 'IBM AML Raw (11-col)' : 'Processed (27-col)'}</span>
                </span>
                <span className="px-2.5 py-1 rounded-lg bg-slate-950 text-slate-300 border border-slate-800">
                  Rows: {rowCount.toLocaleString()}
                </span>
                <span className="px-2.5 py-1 rounded-lg bg-slate-950 text-slate-300 border border-slate-800">
                  Columns: {columnCount}
                </span>
                <span
                  className={`px-2.5 py-1 rounded-lg border ${
                    hasDetectedLabels
                      ? 'bg-emerald-950/80 text-emerald-300 border-emerald-800'
                      : 'bg-amber-950/80 text-amber-300 border-amber-800'
                  }`}
                >
                  {hasDetectedLabels ? '✓ Labels Detected' : 'Unlabeled (Risk Score Only)'}
                </span>
              </div>
            )}
          </div>

          {/* Quick-Load Benchmarks & Config Controls */}
          <div className="w-full md:w-80 space-y-4">
            <label className="block text-xs font-mono uppercase text-slate-400 font-semibold">
              Quick Demonstration Slices
            </label>

            <div className="space-y-2">
              <button
                type="button"
                onClick={() => handleLoadSample('HI-Small')}
                disabled={evaluating}
                className="w-full text-left p-3 rounded-xl bg-slate-950/80 hover:bg-indigo-950/40 border border-slate-800 hover:border-indigo-600/60 transition flex items-center justify-between text-xs disabled:opacity-50"
              >
                <div>
                  <div className="font-semibold text-white">Load HI-Small Test Sample</div>
                  <div className="text-[10px] text-slate-400 font-mono">25 real test rows (mixed labels)</div>
                </div>
                <ArrowRight className="w-4 h-4 text-indigo-400" />
              </button>

              <button
                type="button"
                onClick={() => handleLoadSample('LI-Medium')}
                disabled={evaluating}
                className="w-full text-left p-3 rounded-xl bg-slate-950/80 hover:bg-violet-950/40 border border-slate-800 hover:border-violet-600/60 transition flex items-center justify-between text-xs disabled:opacity-50"
              >
                <div>
                  <div className="font-semibold text-white">Load LI-Medium Test Sample</div>
                  <div className="text-[10px] text-slate-400 font-mono">25 real test rows (mixed labels)</div>
                </div>
                <ArrowRight className="w-4 h-4 text-violet-400" />
              </button>

              <div className="pt-2 flex items-center justify-between">
                <span className="text-xs text-slate-400 font-mono">Row Limit:</span>
                <select
                  value={rowLimit}
                  onChange={(e) => setRowLimit(Number(e.target.value))}
                  className="bg-slate-950 border border-slate-800 text-slate-300 text-xs rounded-xl px-2.5 py-1"
                >
                  <option value={100}>100 rows</option>
                  <option value={500}>500 rows</option>
                  <option value={1000}>1,000 rows</option>
                  <option value={5000}>5,000 rows (Default)</option>
                  <option value={10000}>10,000 rows (Max)</option>
                </select>
              </div>

              <div className="pt-1">
                <a
                  href={apiService.getSampleCsvUrl(selectedDataset)}
                  download={`${selectedDataset.toLowerCase()}_sample.csv`}
                  className="inline-flex items-center space-x-1.5 text-xs text-indigo-400 hover:text-indigo-300 font-mono"
                >
                  <Download className="w-3.5 h-3.5" />
                  <span>Download reference {selectedDataset} CSV</span>
                </a>
              </div>
            </div>
          </div>
        </div>

        {/* CSV Preview Accordion */}
        {previewLines.length > 0 && (
          <div className="bg-slate-950 p-3.5 rounded-2xl border border-slate-800 text-xs font-mono space-y-1.5">
            <div className="text-slate-400 font-bold flex items-center justify-between">
              <span>CSV Header & Data Preview (First 5 lines):</span>
              <span className="text-[10px] text-slate-500">{rowCount} total data rows</span>
            </div>
            <div className="space-y-1 overflow-x-auto text-[11px] text-slate-300">
              {previewLines.map((line, idx) => (
                <div key={idx} className={idx === 0 ? 'text-indigo-300 font-bold' : 'text-slate-400'}>
                  {line}
                </div>
              ))}
            </div>
          </div>
        )}

        {/* Action Button & Target Model Info */}
        <div className="flex items-center justify-between pt-2">
          <div className="text-xs text-slate-400 font-mono">
            Target Model:{' '}
            <strong className="text-slate-200">
              {selectedDataset === 'HI-Small'
                ? 'LSTM Champion (threshold: 0.337784)'
                : 'BiLSTM Champion (threshold: 0.238483)'}
            </strong>
          </div>
          <button
            onClick={handleRunEvaluation}
            disabled={evaluating || !csvContent || Boolean(validationError)}
            className="flex items-center space-x-2 px-6 py-3 rounded-xl bg-indigo-600 hover:bg-indigo-500 text-white font-medium text-sm transition shadow-lg shadow-indigo-600/30 disabled:opacity-50 cursor-pointer"
          >
            {evaluating ? (
              <>
                <RefreshCw className="w-4 h-4 animate-spin text-white" />
                <span>Processing & Scoring Sequences...</span>
              </>
            ) : (
              <>
                <Sparkles className="w-4 h-4 text-white" />
                <span>Execute AML Batch Evaluation</span>
              </>
            )}
          </button>
        </div>
      </div>

      {/* Validation Error Banner */}
      {validationError && (
        <div className="p-4 bg-amber-950/60 border border-amber-500/50 rounded-2xl flex items-start space-x-3 text-amber-200 text-sm">
          <AlertTriangle className="w-5 h-5 text-amber-400 shrink-0 mt-0.5" />
          <div>
            <h4 className="font-semibold text-amber-300">CSV Validation Notice</h4>
            <p className="text-xs text-amber-200/90 mt-1 whitespace-pre-line font-mono">{validationError}</p>
          </div>
        </div>
      )}

      {/* Backend API Error Banner */}
      {apiError && (
        <div className="p-4 bg-rose-950/60 border border-rose-500/50 rounded-2xl flex items-start space-x-3 text-rose-200 text-sm">
          <AlertTriangle className="w-5 h-5 text-rose-400 shrink-0 mt-0.5" />
          <div>
            <h4 className="font-semibold text-rose-300">CSV Evaluation Error</h4>
            <p className="text-xs text-rose-200/90 mt-1 font-mono">{apiError}</p>
          </div>
        </div>
      )}

      {/* Evaluation Results Section */}
      {result && (
        <div className="space-y-6">
          {/* Task 1 Upload Validity Banner */}
          {result.validity_banner && (
            <div
              className={`p-5 rounded-2xl border shadow-xl flex items-start space-x-4 ${
                result.validity_banner.status === 'VALID'
                  ? 'bg-emerald-950/40 border-emerald-500/60 text-emerald-200'
                  : result.validity_banner.status === 'INVALID'
                  ? 'bg-rose-950/40 border-rose-500/70 text-rose-200'
                  : 'bg-amber-950/40 border-amber-500/60 text-amber-200'
              }`}
            >
              {result.validity_banner.status === 'VALID' ? (
                <CheckCircle2 className="w-6 h-6 text-emerald-400 shrink-0 mt-0.5" />
              ) : (
                <AlertTriangle className="w-6 h-6 text-rose-400 shrink-0 mt-0.5" />
              )}
              <div className="space-y-2 flex-1">
                <div className="flex flex-wrap items-center justify-between gap-2 border-b border-slate-800/60 pb-2">
                  <div className="flex items-center space-x-2">
                    <span
                      className={`px-2.5 py-0.5 rounded text-xs font-mono font-bold tracking-wide uppercase ${
                        result.validity_banner.status === 'VALID'
                          ? 'bg-emerald-900/80 text-emerald-200 border border-emerald-600'
                          : result.validity_banner.status === 'INVALID'
                          ? 'bg-rose-900/80 text-rose-200 border border-rose-600'
                          : 'bg-amber-900/80 text-amber-200 border border-amber-600'
                      }`}
                    >
                      {result.validity_banner.status === 'VALID'
                        ? 'VALID EVALUATION DATASET'
                        : result.validity_banner.status === 'INVALID'
                        ? 'INVALID / NON-REPRESENTATIVE BENCHMARK'
                        : 'CAUTION: STATISTICAL SAMPLE NOTICE'}
                    </span>
                    <span className="text-xs text-slate-300 font-sans">
                      {result.validity_banner.status === 'VALID'
                        ? 'Data matches chronological sequence requirements for AML sequence models.'
                        : 'This dataset is not representative of genuine model performance.'}
                    </span>
                  </div>
                  {result.validity_banner.flags.length > 0 && (
                    <div className="flex flex-wrap gap-1 font-mono text-[10px]">
                      {result.validity_banner.flags.map((f) => (
                        <span key={f} className="px-2 py-0.5 rounded bg-slate-900/80 text-slate-300 border border-slate-700">
                          {f}
                        </span>
                      ))}
                    </div>
                  )}
                </div>

                {/* Plain-language reasons */}
                {result.validity_banner.reasons.length > 0 && (
                  <div className="space-y-1.5 pt-1">
                    {result.validity_banner.reasons.map((reason, idx) => (
                      <div key={idx} className="flex items-start space-x-2 text-xs font-sans text-slate-200">
                        <span className="text-amber-400 font-bold">•</span>
                        <span>{reason}</span>
                      </div>
                    ))}
                  </div>
                )}

                {/* Plain-language disclaimer */}
                <p className="text-xs text-slate-300 italic font-sans pt-1">
                  {result.validity_banner.disclaimer}
                </p>

                {/* Diagnostics summary bar */}
                <div className="pt-2 text-[11px] font-mono flex flex-wrap gap-x-4 gap-y-1 border-t border-slate-800/80 text-slate-400">
                  <span>Rows: <strong className="text-white">{result.validity_banner.diagnostics.n_rows}</strong></span>
                  {result.validity_banner.diagnostics.prevalence !== null && (
                    <span>Prevalence: <strong className="text-white">{(result.validity_banner.diagnostics.prevalence * 100).toFixed(2)}%</strong></span>
                  )}
                  {result.validity_banner.diagnostics.positives_count !== null && (
                    <span>Positives: <strong className="text-white">{result.validity_banner.diagnostics.positives_count}</strong></span>
                  )}
                  <span>Unique Senders: <strong className="text-white">{(result.validity_banner.diagnostics.unique_sender_ratio * 100).toFixed(1)}%</strong></span>
                  <span>Mean History: <strong className="text-white">{result.validity_banner.diagnostics.mean_history_len}/10</strong></span>
                  <span>&gt;=3 Prior Tx: <strong className="text-white">{result.validity_banner.diagnostics.pct_rows_with_ge_3_prior_tx}%</strong></span>
                  <span>Chronological: <strong className={result.validity_banner.diagnostics.is_chronological ? 'text-emerald-400' : 'text-rose-400'}>{result.validity_banner.diagnostics.is_chronological ? 'Yes' : 'No'}</strong></span>
                  <span>Time Span: <strong className="text-white">{result.validity_banner.diagnostics.time_span}</strong></span>
                </div>
              </div>
            </div>
          )}

          {/* Insufficient History / Sequence Truncation Warning */}
          {result.sequence_context_diagnostics?.insufficient_history_warning && (
            <div className="p-4 bg-amber-950/40 border border-amber-600/70 rounded-2xl text-xs text-amber-200 flex items-start space-x-3 shadow-lg">
              <AlertTriangle className="w-5 h-5 text-amber-400 shrink-0 mt-0.5" />
              <div className="space-y-1.5 flex-1">
                <div className="flex flex-wrap items-center justify-between gap-2">
                  <span className="font-bold text-amber-300">
                    INSUFFICIENT SEQUENCE CONTEXT WARNING
                  </span>
                  <span className="px-2 py-0.5 rounded text-[10px] font-mono bg-amber-900/60 text-amber-200 border border-amber-700">
                    Avg History: {result.sequence_context_diagnostics.average_sequence_length} / {result.sequence_context_diagnostics.max_sequence_length} Steps
                  </span>
                </div>
                <p className="text-slate-300 leading-relaxed font-sans">
                  {result.sequence_context_diagnostics.warning_message}
                </p>
                <div className="pt-1 text-[11px] text-slate-400 font-mono flex flex-wrap gap-4 border-t border-amber-900/40 mt-1">
                  <span>Single-Transaction Rows: {(result.sequence_context_diagnostics.single_transaction_ratio * 100).toFixed(1)}%</span>
                  <span>Mean Padding: {(result.sequence_context_diagnostics.mean_padding_ratio * 100).toFixed(1)}%</span>
                  <span>Operational Status: <strong className="text-amber-300">{result.sequence_context_diagnostics.operational_context_status}</strong></span>
                </div>
              </div>
            </div>
          )}

          {/* Summary KPI Strip */}
          <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-6 gap-3">
            <div className="bg-slate-900 border border-slate-800 rounded-2xl p-4 shadow-md">
              <span className="text-[10px] text-slate-400 font-mono uppercase block">Total Rows</span>
              <span className="text-xl font-bold font-mono text-white mt-1 block">{result.total_rows}</span>
            </div>
            <div className="bg-slate-900 border border-slate-800 rounded-2xl p-4 shadow-md">
              <span className="text-[10px] text-slate-400 font-mono uppercase block">Sequences Evaluated</span>
              <span className="text-xl font-bold font-mono text-white mt-1 block">{result.evaluated_sequences}</span>
            </div>
            <div className="bg-slate-900 border border-slate-800 rounded-2xl p-4 shadow-md">
              <span className="text-[10px] text-slate-400 font-mono uppercase block">Distinct Accounts</span>
              <span className="text-xl font-bold font-mono text-indigo-400 mt-1 block">{result.distinct_accounts}</span>
            </div>
            <div className="bg-slate-900 border border-slate-800 rounded-2xl p-4 shadow-md">
              <span className="text-[10px] text-slate-400 font-mono uppercase block">Flagged Laundering</span>
              <span className="text-xl font-bold font-mono text-rose-400 mt-1 block">{result.flagged_laundering_count}</span>
            </div>
            <div className="bg-slate-900 border border-slate-800 rounded-2xl p-4 shadow-md">
              <span className="text-[10px] text-slate-400 font-mono uppercase block">Cleared Legitimate</span>
              <span className="text-xl font-bold font-mono text-emerald-400 mt-1 block">{result.cleared_count}</span>
            </div>
            <div className="bg-slate-900 border border-slate-800 rounded-2xl p-4 shadow-md">
              <span className="text-[10px] text-slate-400 font-mono uppercase block">Inference Time</span>
              <span className="text-xl font-bold font-mono text-white mt-1 block">{result.execution_time_ms} ms</span>
            </div>
          </div>

          {/* Supervised Metrics Card (If Ground Truth Labels Present) */}
          {result.has_labels && result.evaluation_metrics ? (
            <div className="bg-slate-900 border border-slate-800 rounded-3xl p-6 shadow-xl space-y-6">
              <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 border-b border-slate-800 pb-4">
                <div>
                  <h3 className="text-lg font-bold text-white flex items-center space-x-2">
                    <CheckCircle2 className="w-5 h-5 text-emerald-400" />
                    <span>Supervised AML Evaluation Metrics (Ground-Truth Verified)</span>
                  </h3>
                  <p className="text-xs text-slate-400">
                    Evaluated using frozen threshold{' '}
                    <strong className="text-amber-400 font-mono">{result.decision_threshold.toFixed(4)}</strong> on positive class (Is Laundering = 1).
                  </p>
                </div>
                <span className="px-3 py-1 rounded-full text-xs font-mono font-bold bg-emerald-950/80 text-emerald-300 border border-emerald-800">
                  LABEL-AWARE
                </span>
              </div>

              {/* Edge Case Warning if any */}
              {result.evaluation_metrics.edge_case_note && (
                <div className="p-3 bg-amber-950/40 border border-amber-800/60 rounded-xl text-xs text-amber-200 flex items-center space-x-2">
                  <Info className="w-4 h-4 text-amber-400 shrink-0" />
                  <span>{result.evaluation_metrics.edge_case_note}</span>
                </div>
              )}

              {/* Metrics Grid */}
              <div className="grid grid-cols-2 sm:grid-cols-4 gap-4">
                <div className="bg-slate-950 p-4 rounded-xl border border-slate-800">
                  <span className="text-xs text-slate-400 block font-mono">F1-Score</span>
                  <span className="text-2xl font-bold font-mono text-white mt-1 block">
                    {formatMetric(result.evaluation_metrics.f1_score, 4, 'Undefined (0 Alerts)')}
                  </span>
                  <span className="text-[10px] text-slate-500 font-mono">Harmonic Precision & Recall</span>
                </div>
                <div className="bg-slate-950 p-4 rounded-xl border border-slate-800">
                  <span className="text-xs text-slate-400 block font-mono">Precision</span>
                  <span className="text-2xl font-bold font-mono text-white mt-1 block">
                    {formatMetric(result.evaluation_metrics.precision, 4, 'Undefined (0 Alerts)')}
                  </span>
                  <span className="text-[10px] text-slate-500 font-mono">TP / (TP + FP)</span>
                </div>
                <div className="bg-slate-950 p-4 rounded-xl border border-slate-800">
                  <span className="text-xs text-slate-400 block font-mono">Recall</span>
                  <span className="text-2xl font-bold font-mono text-white mt-1 block">
                    {formatMetric(result.evaluation_metrics.recall, 4)}
                  </span>
                  <span className="text-[10px] text-slate-500 font-mono">TP / (TP + FN)</span>
                </div>
                <div className="bg-slate-950 p-4 rounded-xl border border-slate-800">
                  <span className="text-xs text-slate-400 block font-mono">Accuracy</span>
                  <span className="text-2xl font-bold font-mono text-white mt-1 block">
                    {result.evaluation_metrics.accuracy.toFixed(4)}
                  </span>
                  <span className="text-[10px] text-slate-500 font-mono">(TP + TN) / Total</span>
                </div>
              </div>

              {/* Confusion Matrix and Probability Metrics */}
              <div className="grid grid-cols-1 md:grid-cols-2 gap-6 pt-2">
                {/* 2x2 Confusion Matrix */}
                <div className="bg-slate-950 p-4 rounded-2xl border border-slate-800 space-y-3">
                  <div className="text-xs font-mono font-bold text-slate-300 uppercase">
                    Confusion Matrix (Threshold: {result.decision_threshold.toFixed(4)})
                  </div>
                  <div className="grid grid-cols-2 gap-2 text-center text-xs font-mono">
                    <div className="bg-emerald-950/40 border border-emerald-800/60 p-3 rounded-xl">
                      <span className="text-[10px] text-emerald-400 font-bold block">TRUE POSITIVES (TP)</span>
                      <span className="text-2xl font-black text-white mt-1 block">
                        {result.evaluation_metrics.confusion_matrix.tp}
                      </span>
                      <span className="text-[10px] text-slate-400">Laundering Caught</span>
                    </div>

                    <div className="bg-amber-950/40 border border-amber-800/60 p-3 rounded-xl">
                      <span className="text-[10px] text-amber-400 font-bold block">FALSE POSITIVES (FP)</span>
                      <span className="text-2xl font-black text-white mt-1 block">
                        {result.evaluation_metrics.confusion_matrix.fp}
                      </span>
                      <span className="text-[10px] text-slate-400">False Alarms</span>
                    </div>

                    <div className="bg-rose-950/40 border border-rose-800/60 p-3 rounded-xl">
                      <span className="text-[10px] text-rose-400 font-bold block">FALSE NEGATIVES (FN)</span>
                      <span className="text-2xl font-black text-white mt-1 block">
                        {result.evaluation_metrics.confusion_matrix.fn}
                      </span>
                      <span className="text-[10px] text-slate-400">Missed Laundering</span>
                    </div>

                    <div className="bg-slate-900 border border-slate-800 p-3 rounded-xl">
                      <span className="text-[10px] text-slate-400 font-bold block">TRUE NEGATIVES (TN)</span>
                      <span className="text-2xl font-black text-white mt-1 block">
                        {result.evaluation_metrics.confusion_matrix.tn}
                      </span>
                      <span className="text-[10px] text-slate-400">Legitimate Cleared</span>
                    </div>
                  </div>
                </div>

                {/* Probability-based & Prevalence Stats */}
                <div className="bg-slate-950 p-4 rounded-2xl border border-slate-800 space-y-3">
                  <div className="text-xs font-mono font-bold text-slate-300 uppercase">
                    Probability Discrimination Metrics
                  </div>
                  <div className="space-y-2.5 text-xs font-mono">
                    <div className="flex flex-col border-b border-slate-800 pb-2 space-y-1">
                      <div className="flex items-center justify-between">
                        <span className="text-slate-400">PR-AUC (Headline Metric):</span>
                        <div className="flex items-center space-x-1.5">
                          <span className="font-bold text-emerald-400 text-sm">
                            {result.evaluation_metrics.pr_auc !== null
                              ? result.evaluation_metrics.pr_auc.toFixed(4)
                              : 'N/A (single class)'}
                          </span>
                          {result.evaluation_metrics.pr_auc !== null && result.evaluation_metrics.positive_prevalence !== undefined && (
                            <span className="text-[10px] px-1.5 py-0.5 rounded bg-slate-800 text-slate-300 font-mono">
                              Base: {(result.evaluation_metrics.positive_prevalence * 100).toFixed(2)}%
                            </span>
                          )}
                        </div>
                      </div>
                      {result.evaluation_metrics.pr_auc !== null && (
                        <div className="flex items-center justify-between text-[11px] text-slate-300">
                          <span>Lift vs Random Baseline:</span>
                          <span className={`font-bold ${
                            (result.evaluation_metrics.pr_auc_lift ?? 1.0) < 2.0 ? 'text-amber-400' : 'text-emerald-400'
                          }`}>
                            {result.evaluation_metrics.pr_auc_lift !== undefined && result.evaluation_metrics.pr_auc_lift !== null
                              ? `${result.evaluation_metrics.pr_auc_lift.toFixed(1)}x (${(result.evaluation_metrics.pr_auc_lift < 2.0 ? 'near random baseline' : 'significant lift')})`
                              : 'N/A'}
                          </span>
                        </div>
                      )}
                      {result.evaluation_metrics.pr_auc_ci_lower !== undefined && result.evaluation_metrics.pr_auc_ci_lower !== null && (
                        <div className="flex items-center justify-between text-[11px] text-slate-400">
                          <span>Bootstrap 95% CI:</span>
                          <span className="text-slate-300">[{result.evaluation_metrics.pr_auc_ci_lower.toFixed(4)}, {result.evaluation_metrics.pr_auc_ci_upper?.toFixed(4)}]</span>
                        </div>
                      )}
                    </div>
                    <div className="flex items-center justify-between border-b border-slate-800 pb-1.5">
                      <span className="text-slate-400">ROC-AUC:</span>
                      <span className="font-bold text-indigo-300">
                        {result.evaluation_metrics.roc_auc !== null
                          ? result.evaluation_metrics.roc_auc.toFixed(4)
                          : 'N/A (single class)'}
                      </span>
                    </div>
                    <div className="flex items-center justify-between border-b border-slate-800 pb-1.5">
                      <span className="text-slate-400">Brier Calibration Score:</span>
                      <span className="font-bold text-white">{result.evaluation_metrics.brier_score.toFixed(6)}</span>
                    </div>
                    <div className="flex items-center justify-between">
                      <span className="text-slate-400">Alerts per 10,000 Transactions:</span>
                      <span className="font-bold text-amber-400">{result.evaluation_metrics.alerts_per_10k.toFixed(1)}</span>
                    </div>
                  </div>
                </div>
              </div>

              {/* Visualizations: Precision-Recall Curve & ROC Curve */}
              {result.evaluation_metrics.pr_curve && result.evaluation_metrics.pr_curve.length > 0 && (
                <div className="grid grid-cols-1 md:grid-cols-2 gap-6 pt-4 border-t border-slate-800">
                  {/* PR Curve SVG */}
                  <div className="bg-slate-950 p-4 rounded-2xl border border-slate-800 space-y-2">
                    <div className="flex items-center justify-between text-xs font-mono">
                      <span className="font-bold text-slate-300 flex items-center space-x-1">
                        <TrendingUp className="w-3.5 h-3.5 text-emerald-400" />
                        <span>Empirical Precision-Recall Curve</span>
                      </span>
                      <span className="text-emerald-400 font-bold">
                        PR-AUC: {result.evaluation_metrics.pr_auc?.toFixed(4) ?? 'N/A'}
                      </span>
                    </div>

                    <div className="h-48 w-full flex items-center justify-center">
                      <svg className="w-full h-full" viewBox="0 0 360 180">
                        {/* Axes */}
                        <line x1="40" y1="20" x2="40" y2="150" stroke="#334155" strokeWidth="1" />
                        <line x1="40" y1="150" x2="340" y2="150" stroke="#334155" strokeWidth="1" />

                        {/* Grid lines */}
                        <line x1="40" y1="85" x2="340" y2="85" stroke="#1e293b" strokeDasharray="3 3" />
                        <line x1="190" y1="20" x2="190" y2="150" stroke="#1e293b" strokeDasharray="3 3" />

                        {/* Axis Labels */}
                        <text x="35" y="25" fill="#64748b" fontSize="9" textAnchor="end">1.0</text>
                        <text x="35" y="88" fill="#64748b" fontSize="9" textAnchor="end">0.5</text>
                        <text x="35" y="152" fill="#64748b" fontSize="9" textAnchor="end">0.0</text>
                        <text x="40" y="165" fill="#64748b" fontSize="9" textAnchor="middle">0.0</text>
                        <text x="190" y="165" fill="#64748b" fontSize="9" textAnchor="middle">0.5 Recall</text>
                        <text x="340" y="165" fill="#64748b" fontSize="9" textAnchor="middle">1.0</text>

                        {/* PR Path */}
                        {(() => {
                          const pts = result.evaluation_metrics.pr_curve || [];
                          if (pts.length < 2) return null;
                          const pathStr = pts
                            .map((p, idx) => {
                              const x = 40 + (p.recall ?? 0) * 300;
                              const y = 150 - (p.precision ?? 0) * 130;
                              return `${idx === 0 ? 'M' : 'L'} ${x.toFixed(1)} ${y.toFixed(1)}`;
                            })
                            .join(' ');
                          return (
                            <path
                              d={pathStr}
                              fill="none"
                              stroke="#10b981"
                              strokeWidth="2.5"
                              strokeLinecap="round"
                            />
                          );
                        })()}

                        {/* Operating Point */}
                        {(() => {
                          const prec = result.evaluation_metrics.precision ?? 0;
                          const rec = result.evaluation_metrics.recall;
                          const x = 40 + rec * 300;
                          const y = 150 - prec * 130;
                          return (
                            <g>
                              <circle cx={x} cy={y} r="5" fill="#f59e0b" />
                              <circle cx={x} cy={y} r="8" fill="none" stroke="#f59e0b" strokeWidth="1" strokeDasharray="2 2" />
                            </g>
                          );
                        })()}
                      </svg>
                    </div>
                    <div className="text-[10px] text-slate-500 font-mono text-center">
                      Amber point represents operating threshold ({result.decision_threshold.toFixed(4)})
                    </div>
                  </div>

                  {/* ROC Curve SVG */}
                  <div className="bg-slate-950 p-4 rounded-2xl border border-slate-800 space-y-2">
                    <div className="flex items-center justify-between text-xs font-mono">
                      <span className="font-bold text-slate-300 flex items-center space-x-1">
                        <TrendingUp className="w-3.5 h-3.5 text-indigo-400" />
                        <span>Empirical ROC Curve</span>
                      </span>
                      <span className="text-indigo-400 font-bold">
                        ROC-AUC: {result.evaluation_metrics.roc_auc?.toFixed(4) ?? 'N/A'}
                      </span>
                    </div>

                    <div className="h-48 w-full flex items-center justify-center">
                      <svg className="w-full h-full" viewBox="0 0 360 180">
                        {/* Axes */}
                        <line x1="40" y1="20" x2="40" y2="150" stroke="#334155" strokeWidth="1" />
                        <line x1="40" y1="150" x2="340" y2="150" stroke="#334155" strokeWidth="1" />

                        {/* Diagonal random baseline */}
                        <line x1="40" y1="150" x2="340" y2="20" stroke="#475569" strokeDasharray="4 4" strokeWidth="1" />

                        {/* Axis Labels */}
                        <text x="35" y="25" fill="#64748b" fontSize="9" textAnchor="end">1.0</text>
                        <text x="35" y="88" fill="#64748b" fontSize="9" textAnchor="end">0.5</text>
                        <text x="35" y="152" fill="#64748b" fontSize="9" textAnchor="end">0.0</text>
                        <text x="40" y="165" fill="#64748b" fontSize="9" textAnchor="middle">0.0</text>
                        <text x="190" y="165" fill="#64748b" fontSize="9" textAnchor="middle">0.5 FPR</text>
                        <text x="340" y="165" fill="#64748b" fontSize="9" textAnchor="middle">1.0</text>

                        {/* ROC Path */}
                        {(() => {
                          const pts = result.evaluation_metrics.roc_curve || [];
                          if (pts.length < 2) return null;
                          const pathStr = pts
                            .map((p, idx) => {
                              const x = 40 + (p.fpr ?? 0) * 300;
                              const y = 150 - (p.tpr ?? 0) * 130;
                              return `${idx === 0 ? 'M' : 'L'} ${x.toFixed(1)} ${y.toFixed(1)}`;
                            })
                            .join(' ');
                          return (
                            <path
                              d={pathStr}
                              fill="none"
                              stroke="#6366f1"
                              strokeWidth="2.5"
                              strokeLinecap="round"
                            />
                          );
                        })()}
                      </svg>
                    </div>
                    <div className="text-[10px] text-slate-500 font-mono text-center">
                      Dashed line: random classifier baseline (AUC = 0.50)
                    </div>
                  </div>
                </div>
              )}
            </div>
          ) : (
            <div className="p-4 bg-slate-900 border border-slate-800 rounded-2xl flex items-center space-x-3 text-xs text-slate-300">
              <AlertTriangle className="w-5 h-5 text-amber-400 shrink-0" />
              <div>
                <span className="font-bold text-white">Unlabeled Transaction Batch: </span>
                Ground-truth labels were not provided. Evaluation metrics cannot be calculated. Transaction-level probabilities and risk decisions are displayed below.
              </div>
            </div>
          )}

          {/* Risk Score Distribution Histogram & Flagged vs Cleared Count */}
          <div className="bg-slate-900 border border-slate-800 rounded-3xl p-6 shadow-xl space-y-4">
            <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 border-b border-slate-800 pb-3">
              <h3 className="text-sm font-bold text-white flex items-center space-x-2">
                <BarChart2 className="w-4 h-4 text-indigo-400" />
                <span>Predicted Laundering Probability Distribution (Deciles)</span>
              </h3>
              <div className="text-xs font-mono text-slate-400">
                Frozen Threshold:{' '}
                <strong className="text-amber-400">{result.decision_threshold.toFixed(4)}</strong>
              </div>
            </div>

            {/* Decile Bins Bar Chart */}
            {result.risk_distribution && result.risk_distribution.length > 0 && (
              <div className="space-y-2">
                <div className="grid grid-cols-10 gap-1.5 h-32 items-end pt-4 pb-1 bg-slate-950 p-3 rounded-2xl border border-slate-800">
                  {result.risk_distribution.map((bin, idx) => {
                    const isAboveThreshold = bin.upper > result.decision_threshold;
                    const maxPct = Math.max(...(result.risk_distribution?.map((b) => b.percentage) || [1]));
                    const barHeightPct = Math.max(4, (bin.percentage / (maxPct || 1)) * 100);

                    return (
                      <div key={idx} className="flex flex-col items-center h-full justify-end group relative">
                        {/* Tooltip */}
                        <div className="opacity-0 group-hover:opacity-100 transition absolute -top-8 bg-slate-800 text-white text-[10px] px-2 py-0.5 rounded shadow pointer-events-none whitespace-nowrap z-10 font-mono">
                          {bin.bin_range}: {bin.count} ({bin.percentage}%)
                        </div>

                        <div className="text-[9px] font-mono text-slate-400 mb-1">{bin.count}</div>
                        <div
                          className={`w-full rounded-t transition ${
                            isAboveThreshold
                              ? 'bg-gradient-to-t from-rose-700 to-rose-500'
                              : 'bg-gradient-to-t from-indigo-700 to-indigo-500'
                          }`}
                          style={{ height: `${barHeightPct}%` }}
                        />
                        <div className="text-[9px] font-mono text-slate-500 mt-1 truncate max-w-full">
                          {bin.lower.toFixed(1)}
                        </div>
                      </div>
                    );
                  })}
                </div>
                <div className="flex items-center justify-between text-[11px] text-slate-400 font-mono px-1">
                  <span>Probability: 0.0 (Clean)</span>
                  <span className="text-amber-400">
                    ▲ Cutoff ({result.decision_threshold.toFixed(4)})
                  </span>
                  <span>1.0 (Laundering)</span>
                </div>
              </div>
            )}

            {/* Flagged vs Cleared Ratio Bar */}
            <div className="pt-2">
              <div className="flex items-center justify-between text-xs font-mono text-slate-300 pb-1">
                <span>
                  Cleared: <strong className="text-emerald-400">{result.cleared_count}</strong> (
                  {((result.cleared_count / result.evaluated_sequences) * 100).toFixed(1)}%)
                </span>
                <span>
                  Flagged: <strong className="text-rose-400">{result.flagged_laundering_count}</strong> (
                  {((result.flagged_laundering_count / result.evaluated_sequences) * 100).toFixed(1)}%)
                </span>
              </div>
              <div className="w-full bg-slate-950 h-3 rounded-full overflow-hidden flex border border-slate-800">
                <div
                  className="bg-emerald-500 h-full"
                  style={{ width: `${(result.cleared_count / result.evaluated_sequences) * 100}%` }}
                />
                <div
                  className="bg-rose-500 h-full"
                  style={{ width: `${(result.flagged_laundering_count / result.evaluated_sequences) * 100}%` }}
                />
              </div>
            </div>
          </div>

          {/* Predictions Table Header & Actions */}
          <div className="bg-slate-900 border border-slate-800 rounded-3xl p-6 shadow-xl space-y-4">
            <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3">
              <div>
                <h3 className="text-lg font-bold text-white tracking-tight">Transaction-Level Predictions</h3>
                <p className="text-xs text-slate-400 font-mono">
                  Showing {filteredPredictions.length} of {result.evaluated_sequences} scored transactions
                </p>
              </div>

              <div className="flex items-center space-x-2">
                <button
                  onClick={handleExportPredictions}
                  className="flex items-center space-x-1.5 px-3.5 py-2 rounded-xl bg-slate-800 hover:bg-slate-700 text-slate-200 text-xs font-semibold border border-slate-700 transition cursor-pointer"
                >
                  <Download className="w-3.5 h-3.5 text-indigo-400" />
                  <span>Export Predictions (CSV)</span>
                </button>
              </div>
            </div>

            {/* Filter and Search Bar */}
            <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 pt-2">
              <div className="flex items-center space-x-2 bg-slate-950 p-1 rounded-xl border border-slate-800 text-xs">
                <button
                  onClick={() => {
                    setStatusFilter('ALL');
                    setCurrentPage(1);
                  }}
                  className={`px-3 py-1 rounded-lg transition font-medium cursor-pointer ${
                    statusFilter === 'ALL' ? 'bg-indigo-600 text-white' : 'text-slate-400'
                  }`}
                >
                  All ({result.predictions.length})
                </button>
                <button
                  onClick={() => {
                    setStatusFilter('FLAGGED');
                    setCurrentPage(1);
                  }}
                  className={`px-3 py-1 rounded-lg transition font-medium cursor-pointer ${
                    statusFilter === 'FLAGGED' ? 'bg-rose-600 text-white' : 'text-slate-400'
                  }`}
                >
                  Flagged ({result.flagged_laundering_count})
                </button>
                <button
                  onClick={() => {
                    setStatusFilter('CLEARED');
                    setCurrentPage(1);
                  }}
                  className={`px-3 py-1 rounded-lg transition font-medium cursor-pointer ${
                    statusFilter === 'CLEARED' ? 'bg-emerald-600 text-white' : 'text-slate-400'
                  }`}
                >
                  Cleared ({result.predictions.filter((p) => p.decision === 'CLEARED_LEGITIMATE' || p.predicted_label === 'LEGITIMATE').length})
                </button>
                <button
                  onClick={() => {
                    setStatusFilter('NO_HISTORY');
                    setCurrentPage(1);
                  }}
                  className={`px-3 py-1 rounded-lg transition font-medium cursor-pointer ${
                    statusFilter === 'NO_HISTORY' ? 'bg-amber-600 text-white' : 'text-slate-400'
                  }`}
                >
                  No History ({result.predictions.filter((p) => p.decision === 'LOW_CONFIDENCE_NO_HISTORY' || p.predicted_label === 'LOW_CONFIDENCE_NO_HISTORY').length})
                </button>
              </div>

              <div className="relative w-full sm:w-64">
                <Search className="w-3.5 h-3.5 text-slate-400 absolute left-3 top-2.5" />
                <input
                  type="text"
                  value={searchQuery}
                  onChange={(e) => {
                    setSearchQuery(e.target.value);
                    setCurrentPage(1);
                  }}
                  placeholder="Filter by account, row, or time..."
                  className="w-full bg-slate-950 border border-slate-800 rounded-xl pl-9 pr-3 py-1.5 text-xs text-slate-200 placeholder-slate-500 focus:outline-none"
                />
              </div>
            </div>

            {/* Data Table */}
            <div className="overflow-x-auto border border-slate-800 rounded-2xl">
              <table className="w-full text-left border-collapse text-xs">
                <thead>
                  <tr className="bg-slate-950 border-b border-slate-800 text-slate-400 font-mono uppercase text-[10px]">
                    <th className="py-2.5 px-3">Row</th>
                    <th className="py-2.5 px-3">Timestamp</th>
                    <th className="py-2.5 px-3">From Account</th>
                    <th className="py-2.5 px-3">To Account</th>
                    <th className="py-2.5 px-3 text-right">Amount</th>
                    <th className="py-2.5 px-3">Risk Probability</th>
                    <th className="py-2.5 px-3">Decision</th>
                    {result.has_labels && <th className="py-2.5 px-3 text-center">Ground Truth</th>}
                    {result.has_labels && <th className="py-2.5 px-3 text-center">Evaluation Match</th>}
                    <th className="py-2.5 px-3 text-center">Action</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-slate-800/60 font-mono">
                  {paginatedPredictions.map((item) => {
                    const isFlagged = item.decision === 'FLAGGED_LAUNDERING' || item.predicted_label === 'SUSPICIOUS';
                    const isNoHistory = item.decision === 'LOW_CONFIDENCE_NO_HISTORY' || item.predicted_label === 'LOW_CONFIDENCE_NO_HISTORY';
                    const probPct = (item.predicted_probability * 100).toFixed(2);

                    let matchBadge = null;
                    if (result.has_labels && item.ground_truth_label !== null) {
                      if (item.ground_truth_label === 1 && isFlagged) {
                        matchBadge = (
                          <span className="px-2 py-0.5 rounded text-[10px] font-bold bg-emerald-950 text-emerald-400 border border-emerald-800">
                            True Positive
                          </span>
                        );
                      } else if (item.ground_truth_label === 1 && !isFlagged) {
                        matchBadge = (
                          <span className="px-2 py-0.5 rounded text-[10px] font-bold bg-rose-950 text-rose-300 border border-rose-700 animate-pulse" title="Missed positive laundering case">
                            False Negative (Missed)
                          </span>
                        );
                      } else if (item.ground_truth_label === 0 && isFlagged) {
                        matchBadge = (
                          <span className="px-2 py-0.5 rounded text-[10px] font-bold bg-amber-950 text-amber-400 border border-amber-800">
                            False Positive
                          </span>
                        );
                      } else if (item.ground_truth_label === 0 && isNoHistory) {
                        matchBadge = (
                          <span className="px-2 py-0.5 rounded text-[10px] text-amber-300 bg-amber-950/60 border border-amber-850" title="Clean row with insufficient history depth (<3 steps)">
                            Unconfirmed (No History)
                          </span>
                        );
                      } else {
                        matchBadge = (
                          <span className="px-2 py-0.5 rounded text-[10px] text-slate-400 bg-slate-950 border border-slate-800">
                            True Negative
                          </span>
                        );
                      }
                    }

                    return (
                      <tr key={item.row_id} className="hover:bg-slate-800/40 transition">
                        <td className="py-2.5 px-3 text-slate-400">#{item.row_id}</td>
                        <td className="py-2.5 px-3 text-slate-300 font-sans">{item.timestamp}</td>
                        <td className="py-2.5 px-3 text-indigo-300 font-bold truncate max-w-[120px]" title={item.from_account}>
                          {item.from_account}
                        </td>
                        <td className="py-2.5 px-3 text-slate-400 truncate max-w-[120px]" title={item.to_account}>
                          {item.to_account}
                        </td>
                        <td className="py-2.5 px-3 text-right text-slate-200">
                          ${item.amount.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}
                        </td>
                        <td className="py-2.5 px-3">
                          <div className="flex items-center space-x-2">
                            <span className={`w-12 font-bold ${isFlagged ? 'text-rose-400' : 'text-slate-300'}`}>
                              {probPct}%
                            </span>
                            <div className="w-16 bg-slate-800 rounded-full h-1.5 overflow-hidden">
                              <div
                                className={`h-full rounded-full ${isFlagged ? 'bg-rose-500' : 'bg-indigo-500'}`}
                                style={{ width: `${Math.min(100, Math.max(2, item.predicted_probability * 100))}%` }}
                              />
                            </div>
                          </div>
                        </td>
                        <td className="py-2.5 px-3">
                          <span
                            className={`px-2 py-0.5 rounded text-[10px] font-bold ${
                              isFlagged
                                ? 'bg-rose-950/80 text-rose-300 border border-rose-800/80'
                                : isNoHistory
                                ? 'bg-amber-950/80 text-amber-300 border border-amber-800/80'
                                : 'bg-emerald-950/60 text-emerald-300 border border-emerald-800/60'
                            }`}
                            title={isNoHistory ? 'Low confidence prediction: <3 historical transactions for sender' : undefined}
                          >
                            {isFlagged ? 'FLAGGED' : isNoHistory ? 'LOW CONFIDENCE (NO HISTORY)' : 'CLEARED'}
                          </span>
                        </td>
                        {result.has_labels && (
                          <td className="py-2.5 px-3 text-center">
                            {item.ground_truth_label === 1 ? (
                              <span className="px-2 py-0.5 rounded text-[10px] font-bold bg-rose-950 text-rose-400 border border-rose-800">
                                1 (Laundering)
                              </span>
                            ) : (
                              <span className="px-2 py-0.5 rounded text-[10px] text-slate-400 bg-slate-950 border border-slate-800">
                                0 (Clean)
                              </span>
                            )}
                          </td>
                        )}
                        {result.has_labels && <td className="py-2.5 px-3 text-center">{matchBadge}</td>}
                        <td className="py-2.5 px-3 text-center">
                          <button
                            onClick={() => onExplainTransaction(item, result.dataset)}
                            className="px-2.5 py-1 rounded-lg bg-indigo-600/20 hover:bg-indigo-600/40 text-indigo-300 border border-indigo-500/40 text-[11px] font-semibold transition inline-flex items-center space-x-1 cursor-pointer"
                          >
                            <Eye className="w-3 h-3" />
                            <span>Explain</span>
                          </button>
                        </td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>

            {/* Pagination Controls */}
            {totalPages > 1 && (
              <div className="flex items-center justify-between text-xs text-slate-400 font-mono pt-2">
                <span>
                  Page {currentPage} of {totalPages}
                </span>
                <div className="flex items-center space-x-1">
                  <button
                    onClick={() => setCurrentPage((p) => Math.max(1, p - 1))}
                    disabled={currentPage === 1}
                    className="px-2.5 py-1 rounded bg-slate-800 hover:bg-slate-700 disabled:opacity-40 cursor-pointer"
                  >
                    Previous
                  </button>
                  <button
                    onClick={() => setCurrentPage((p) => Math.min(totalPages, p + 1))}
                    disabled={currentPage === totalPages}
                    className="px-2.5 py-1 rounded bg-slate-800 hover:bg-slate-700 disabled:opacity-40 cursor-pointer"
                  >
                    Next
                  </button>
                </div>
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  );
};
