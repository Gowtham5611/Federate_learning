import React, { useState, useMemo } from 'react';
import { CsvEvaluateResponse, TransactionPredictionItem } from '../types/api';
import {
  History,
  Search,
  Eye,
  Upload,
  ShieldAlert,
  ShieldCheck,
  X,
  FileText,
  Trash2,
  AlertTriangle,
  Download,
  ChevronDown,
  Info,
} from 'lucide-react';
import { apiService } from '../services/apiService';

export type PredictionFilterMode = 'ALL' | 'PRED_SUSPICIOUS' | 'PRED_LEGITIMATE' | 'NEEDS_REVIEW_FN';

interface PredictionHistoryPageProps {
  result: CsvEvaluateResponse | null;
  onNavigateToUpload: () => void;
  onSelectHistoricalRun?: (runId: string) => void;
  onClearSavedResults?: () => void;
}

export const PredictionHistoryPage: React.FC<PredictionHistoryPageProps> = ({
  result,
  onNavigateToUpload,
  onClearSavedResults,
}) => {
  const [statusFilter, setStatusFilter] = useState<PredictionFilterMode>('ALL');
  const [searchQuery, setSearchQuery] = useState<string>('');
  const [currentPage, setCurrentPage] = useState<number>(1);
  const [downloadMenuOpen, setDownloadMenuOpen] = useState<boolean>(false);
  const pageSize = 20;

  // Explainability inspection modal
  const [inspectingItem, setInspectingItem] = useState<TransactionPredictionItem | null>(null);
  const [explaining, setExplaining] = useState<boolean>(false);
  const [explanationData, setExplanationData] = useState<{
    positiveDrivers: { feature_name: string; attribution_score: number; feature_group?: string }[];
    negativeDrivers: { feature_name: string; attribution_score: number; feature_group?: string }[];
    method?: string;
    modelName?: string;
    predictedProbability?: number;
    decisionThreshold?: number;
    decision?: string;
    historyWarning?: string | null;
  } | null>(null);
  const [explainError, setExplainError] = useState<string | null>(null);

  const canonicalThreshold = result?.decision_threshold ?? 0.337784;
  const activeThreshold = canonicalThreshold;
  const hasLabels = Boolean(result?.has_labels);

  // Dynamically recomputed metrics and reconciled confusion counts from real active run data
  const {
    filteredRows,
    predSuspiciousCount,
    predLegitimateCount,
    missedFnCount,
    totalCount,
  } = useMemo(() => {
    const preds = result?.predictions || [];
    let tp = 0;
    let fp = 0;
    let fn = 0;
    let tn = 0;
    let predSuspicious = 0;
    let predLegitimate = 0;

    preds.forEach((item) => {
      const prob = Number(item.predicted_probability ?? 0);
      const isAlert = prob >= activeThreshold;
      if (isAlert) predSuspicious++;
      else predLegitimate++;

      if (hasLabels) {
        const isLaunderingLabel =
          item.ground_truth_label === 1 ||
          item.actual_label === 'SUSPICIOUS' ||
          (item as any)['Is Laundering'] === 1 ||
          (item as any)['is_laundering'] === 1;

        if (isLaunderingLabel) {
          if (isAlert) tp++;
          else fn++;
        } else {
          if (isAlert) fp++;
          else tn++;
        }
      }
    });

    const filtered = preds.filter((item) => {
      const prob = Number(item.predicted_probability ?? 0);
      const isAlert = prob >= activeThreshold;
      const isLaunderingLabel =
        item.ground_truth_label === 1 ||
        item.actual_label === 'SUSPICIOUS' ||
        (item as any)['Is Laundering'] === 1 ||
        (item as any)['is_laundering'] === 1;

      if (statusFilter === 'PRED_SUSPICIOUS' && !isAlert) return false;
      if (statusFilter === 'PRED_LEGITIMATE' && isAlert) return false;
      if (statusFilter === 'NEEDS_REVIEW_FN' && !(isLaunderingLabel && !isAlert)) return false;

      if (searchQuery.trim()) {
        const q = searchQuery.toLowerCase().trim();
        const txId = String(item.transaction_id || `TX_${String(item.row_id).padStart(5, '0')}`).toLowerCase();
        const fromAcc = String(item.from_account || '').toLowerCase();
        const toAcc = String(item.to_account || '').toLowerCase();
        const amtStr = String(item.amount || '');

        const match =
          txId.includes(q) ||
          fromAcc.includes(q) ||
          toAcc.includes(q) ||
          amtStr.includes(q) ||
          String(item.row_id) === q;
        if (!match) return false;
      }

      return true;
    });

    filtered.sort((a, b) => {
      const pa = Number(a.predicted_probability ?? 0);
      const pb = Number(b.predicted_probability ?? 0);
      const fa = pa >= activeThreshold ? 1 : 0;
      const fb = pb >= activeThreshold ? 1 : 0;
      if (fa !== fb) return fb - fa;
      return pb - pa;
    });

    return {
      filteredRows: filtered,
      predSuspiciousCount: predSuspicious,
      predLegitimateCount: predLegitimate,
      missedFnCount: fn,
      tpCount: tp,
      fpCount: fp,
      tnCount: tn,
      totalCount: preds.length,
    };
  }, [result?.predictions, activeThreshold, hasLabels, statusFilter, searchQuery]);

  const totalPages = Math.max(1, Math.ceil(filteredRows.length / pageSize));
  const paginatedRows = useMemo(() => {
    const start = (currentPage - 1) * pageSize;
    return filteredRows.slice(start, start + pageSize);
  }, [filteredRows, currentPage, pageSize]);

  // Inspect attribution
  const handleInspect = async (item: TransactionPredictionItem) => {
    if (!result) return;
    setInspectingItem(item);
    setExplanationData(null);
    setExplainError(null);
    setExplaining(true);

    try {
      const runPid = (result as any).prediction_id || result.download_id || (result as any).run_id;
      const resp = await apiService.explain({
        dataset: result.dataset || 'HI-Small',
        model_name: (result as any).model_id || result.model_name || 'HI-Small',
        prediction_id: runPid,
        row_id: item.row_id,
        transaction_id: item.transaction_id,
        sequence: item.sequence_features && item.sequence_features.length > 0 ? {
          features: item.sequence_features,
          length: item.sequence_length || item.sequence_features.length,
          sample_id: item.transaction_id || `TX_${item.row_id}`,
        } : undefined,
        tabular_features: item.tabular_features,
        steps: 25,
        threshold: activeThreshold,
      });

      setExplanationData({
        positiveDrivers: resp.top_positive_drivers.map((d) => ({
          feature_name: d.feature_name,
          attribution_score: d.attribution,
          feature_group: d.feature_group,
        })),
        negativeDrivers: resp.top_negative_drivers.map((d) => ({
          feature_name: d.feature_name,
          attribution_score: d.attribution,
          feature_group: d.feature_group,
        })),
        method: resp.method,
        modelName: resp.model_name,
        predictedProbability: resp.predicted_probability,
        decisionThreshold: resp.decision_threshold,
        decision: resp.decision,
        historyWarning: resp.history_warning,
      });
    } catch (err: unknown) {
      setExplainError(err instanceof Error ? err.message : 'Explainability generation failed');
    } finally {
      setExplaining(false);
    }
  };

  return (
    <div className="space-y-8 animate-fade-in">
      {/* Header */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 border-b border-slate-800 pb-5">
        <div>
          <h1 className="text-2xl sm:text-3xl font-black text-white tracking-tight flex items-center space-x-2">
            <History className="w-7 h-7 text-indigo-400" />
            <span>Prediction History & Downloads</span>
          </h1>
          <p className="text-xs text-slate-400 mt-1">
            {result
              ? `Uploaded CSV: ${result.filename} · Model Used: ${result.model_name || result.model_id}`
              : 'Upload a CSV file to view its summary, triage queues, and downloadable reports.'}
          </p>
        </div>

        <div className="flex items-center space-x-2">
          {onClearSavedResults && (
            <button
              type="button"
              onClick={onClearSavedResults}
              title="Clear saved results"
              aria-label="Clear saved results"
              className="p-2.5 min-h-[40px] min-w-[40px] rounded-xl bg-slate-900 hover:bg-rose-950/60 text-slate-300 hover:text-rose-300 border border-slate-800 hover:border-rose-700/50 flex items-center justify-center transition cursor-pointer"
            >
              <Trash2 className="w-4 h-4 text-rose-400" />
            </button>
          )}

          {result && (
            <>
              {/* Mobile single Download dropdown */}
              <div className="relative sm:hidden">
                <button
                  type="button"
                  onClick={() => setDownloadMenuOpen((prev) => !prev)}
                  className="flex items-center space-x-1.5 px-3.5 py-2 min-h-[40px] rounded-xl bg-indigo-600 hover:bg-indigo-500 text-white text-xs font-semibold transition cursor-pointer"
                >
                  <Download className="w-3.5 h-3.5" />
                  <span>Download</span>
                  <ChevronDown className="w-3.5 h-3.5" />
                </button>
                {downloadMenuOpen && (
                  <div className="absolute right-0 mt-2 w-52 rounded-xl bg-slate-900 border border-slate-700 shadow-xl py-1.5 z-40">
                    <button
                      type="button"
                      onClick={() => {
                        setDownloadMenuOpen(false);
                        const pid = result.prediction_id || result.download_id;
                        if (pid) window.open(apiService.getPdfReportUrl(pid), '_blank');
                      }}
                      className="w-full px-3.5 py-2.5 text-left text-xs text-indigo-300 hover:bg-slate-800 flex items-center gap-2 cursor-pointer"
                    >
                      <FileText className="w-3.5 h-3.5" />
                      <span>Report (Metrics PDF)</span>
                    </button>
                    <button
                      type="button"
                      onClick={() => {
                        setDownloadMenuOpen(false);
                        const pid = result.prediction_id || result.download_id;
                        if (pid) window.open(apiService.getFlaggedCsvDownloadUrl(pid, activeThreshold), '_blank');
                      }}
                      className="w-full px-3.5 py-2.5 text-left text-xs text-rose-300 hover:bg-slate-800 flex items-center gap-2 cursor-pointer"
                    >
                      <ShieldAlert className="w-3.5 h-3.5" />
                      <span>Flagged CSV</span>
                    </button>
                    {hasLabels && missedFnCount > 0 && (
                      <button
                        type="button"
                        onClick={() => {
                          setDownloadMenuOpen(false);
                          const pid = result.prediction_id || result.download_id;
                          if (pid) window.open(apiService.getNeedsReviewCsvDownloadUrl(pid, activeThreshold), '_blank');
                        }}
                        className="w-full px-3.5 py-2.5 text-left text-xs text-amber-300 hover:bg-slate-800 flex items-center gap-2 cursor-pointer"
                      >
                        <AlertTriangle className="w-3.5 h-3.5" />
                        <span>Under Review CSV</span>
                      </button>
                    )}
                    <button
                      type="button"
                      onClick={() => {
                        setDownloadMenuOpen(false);
                        const pid = result.prediction_id || result.download_id;
                        if (pid) window.open(apiService.getLegitimateCsvDownloadUrl(pid, activeThreshold), '_blank');
                      }}
                      className="w-full px-3.5 py-2.5 text-left text-xs text-emerald-300 hover:bg-slate-800 flex items-center gap-2 cursor-pointer"
                    >
                      <ShieldCheck className="w-3.5 h-3.5" />
                      <span>Legitimate CSV</span>
                    </button>
                  </div>
                )}
              </div>

              {/* Desktop / Tablet individual download buttons */}
              <div className="hidden sm:flex flex-wrap items-center gap-2">
                <button
                  onClick={() => {
                    const pid = result.prediction_id || result.download_id;
                    if (pid) {
                      window.open(apiService.getPdfReportUrl(pid), '_blank');
                    }
                  }}
                  className="flex items-center space-x-1.5 px-3.5 py-2 rounded-xl bg-indigo-950/80 hover:bg-indigo-900 border border-indigo-700/60 text-indigo-300 text-xs font-semibold transition cursor-pointer"
                  title="Download evaluation metrics report (PDF) for uploaded CSV"
                >
                  <FileText className="w-3.5 h-3.5" />
                  <span>Report (Metrics PDF)</span>
                </button>

                <button
                  onClick={() => {
                    const pid = result.prediction_id || result.download_id;
                    if (pid) {
                      window.open(apiService.getFlaggedCsvDownloadUrl(pid, activeThreshold), '_blank');
                    }
                  }}
                  className="flex items-center space-x-1.5 px-3.5 py-2 rounded-xl bg-rose-950/80 hover:bg-rose-900 border border-rose-700/60 text-rose-300 text-xs font-semibold transition cursor-pointer"
                  title="Download Flagged transactions CSV separately"
                >
                  <ShieldAlert className="w-3.5 h-3.5 text-rose-400" />
                  <span>Flagged CSV</span>
                </button>

                {hasLabels && missedFnCount > 0 && (
                  <button
                    onClick={() => {
                      const pid = result.prediction_id || result.download_id;
                      if (pid) {
                        window.open(apiService.getNeedsReviewCsvDownloadUrl(pid, activeThreshold), '_blank');
                      }
                    }}
                    className="flex items-center space-x-1.5 px-3.5 py-2 rounded-xl bg-amber-950/80 hover:bg-amber-900 border border-amber-700/60 text-amber-300 text-xs font-semibold transition cursor-pointer"
                    title="Download Under Review (False Negatives) transactions CSV separately"
                  >
                    <AlertTriangle className="w-3.5 h-3.5 text-amber-400" />
                    <span>Under Review CSV</span>
                  </button>
                )}

                <button
                  onClick={() => {
                    const pid = result.prediction_id || result.download_id;
                    if (pid) {
                      window.open(apiService.getLegitimateCsvDownloadUrl(pid, activeThreshold), '_blank');
                    }
                  }}
                  className="flex items-center space-x-1.5 px-3.5 py-2 rounded-xl bg-emerald-950/80 hover:bg-emerald-900 border border-emerald-700/60 text-emerald-300 text-xs font-semibold transition cursor-pointer"
                  title="Download Legitimate (Non-Flagged) transactions CSV separately"
                >
                  <ShieldCheck className="w-3.5 h-3.5 text-emerald-400" />
                  <span>Legitimate CSV</span>
                </button>
              </div>
            </>
          )}
        </div>
      </div>

      {/* Active Uploaded CSV Summary & Triage Queue */}
      {result ? (
        <div className="bg-white dark:bg-slate-900 border border-slate-200 dark:border-slate-800 rounded-3xl p-6 shadow-xl space-y-4">
          <div className="flex flex-col md:flex-row md:items-center justify-between gap-3 border-b border-slate-200 dark:border-slate-800 pb-4">
            <div>
              <div className="flex items-center space-x-2 flex-wrap gap-y-1">
                <ShieldAlert className="w-5 h-5 text-indigo-500 dark:text-indigo-400" />
                <h2 className="text-base font-bold text-slate-900 dark:text-white tracking-tight">
                  {result.filename}
                </h2>
                <span className="px-2 py-0.5 rounded-full text-[10px] font-mono bg-indigo-50 dark:bg-indigo-950/80 text-indigo-700 dark:text-indigo-300 border border-indigo-200 dark:border-indigo-800/60">
                  Model: {result.model_name || result.model_id}
                </span>
                <span className="px-2 py-0.5 rounded-full text-[10px] font-mono bg-slate-100 dark:bg-slate-800 text-slate-700 dark:text-slate-300 border border-slate-200 dark:border-slate-700">
                  {filteredRows.length.toLocaleString()} Matches
                </span>
              </div>
              <p className="text-xs text-slate-500 dark:text-slate-400 mt-1">
                CSV Summary: {totalCount.toLocaleString()} evaluated transactions · {predSuspiciousCount.toLocaleString()} Flagged · {predLegitimateCount.toLocaleString()} Non-Flagged{hasLabels ? ` · ${missedFnCount.toLocaleString()} Under Review` : ''}.
              </p>
            </div>

            {/* Status Filter Tabs (Keep ONLY Flagged, Non-Flagged, Under Review) */}
            <div className="flex flex-wrap items-center gap-1.5 bg-slate-100 dark:bg-slate-950 p-1 rounded-xl border border-slate-200 dark:border-slate-800 text-xs">
              {/* Tab 1: All */}
              <button
                onClick={() => {
                  setStatusFilter('ALL');
                  setCurrentPage(1);
                }}
                className={`px-3 py-1.5 rounded-lg transition font-medium cursor-pointer flex items-center space-x-1.5 ${
                  statusFilter === 'ALL'
                    ? 'bg-indigo-600 text-white font-bold shadow-sm'
                    : 'text-slate-600 dark:text-slate-400 hover:text-slate-900 dark:hover:text-white'
                }`}
              >
                <span>All</span>
                <span className={`px-1.5 py-0.2 rounded-full text-[10px] font-mono ${
                  statusFilter === 'ALL' ? 'bg-indigo-800 text-white' : 'bg-slate-200 dark:bg-slate-800 text-slate-700 dark:text-slate-300'
                }`}>
                  {totalCount.toLocaleString()}
                </span>
              </button>

              {/* Tab 2: Flagged */}
              <button
                onClick={() => {
                  setStatusFilter('PRED_SUSPICIOUS');
                  setCurrentPage(1);
                }}
                className={`px-3 py-1.5 rounded-lg transition font-medium cursor-pointer flex items-center space-x-1.5 ${
                  statusFilter === 'PRED_SUSPICIOUS'
                    ? 'bg-rose-600 text-white font-bold shadow-sm'
                    : 'text-slate-600 dark:text-slate-400 hover:text-slate-900 dark:hover:text-white'
                }`}
              >
                <span>Flagged</span>
                <span className={`px-1.5 py-0.2 rounded-full text-[10px] font-mono ${
                  statusFilter === 'PRED_SUSPICIOUS' ? 'bg-rose-800 text-white' : 'bg-slate-200 dark:bg-slate-800 text-slate-700 dark:text-slate-300'
                }`}>
                  {predSuspiciousCount.toLocaleString()}
                </span>
              </button>

              {/* Tab 3: Non-Flagged */}
              <button
                onClick={() => {
                  setStatusFilter('PRED_LEGITIMATE');
                  setCurrentPage(1);
                }}
                className={`px-3 py-1.5 rounded-lg transition font-medium cursor-pointer flex items-center space-x-1.5 ${
                  statusFilter === 'PRED_LEGITIMATE'
                    ? 'bg-emerald-600 text-white font-bold shadow-sm'
                    : 'text-slate-600 dark:text-slate-400 hover:text-slate-900 dark:hover:text-white'
                }`}
              >
                <span>Non-Flagged</span>
                <span className={`px-1.5 py-0.2 rounded-full text-[10px] font-mono ${
                  statusFilter === 'PRED_LEGITIMATE' ? 'bg-emerald-800 text-white' : 'bg-slate-200 dark:bg-slate-800 text-slate-700 dark:text-slate-300'
                }`}>
                  {predLegitimateCount.toLocaleString()}
                </span>
              </button>

              {/* Tab 4: Under Review */}
              {hasLabels && (
                <button
                  onClick={() => {
                    setStatusFilter('NEEDS_REVIEW_FN');
                    setCurrentPage(1);
                  }}
                  className={`px-3 py-1.5 rounded-lg transition font-medium cursor-pointer flex items-center space-x-1.5 ${
                    statusFilter === 'NEEDS_REVIEW_FN'
                      ? 'bg-amber-600 text-white font-bold shadow-sm'
                      : 'text-amber-700 dark:text-amber-400 hover:text-amber-900 dark:hover:text-amber-200'
                  }`}
                  title="Under Review: CSV laundering rows cleared by model"
                >
                  <AlertTriangle className="w-3.5 h-3.5" />
                  <span>Under Review</span>
                  <span className={`px-1.5 py-0.2 rounded-full text-[10px] font-mono ${
                    statusFilter === 'NEEDS_REVIEW_FN' ? 'bg-amber-800 text-white' : 'bg-slate-200 dark:bg-slate-800 text-slate-700 dark:text-slate-300'
                  }`}>
                    {missedFnCount.toLocaleString()}
                  </span>
                </button>
              )}
            </div>
          </div>

          {/* Search bar */}
          <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3">
            <div className="relative flex-1 max-w-md">
              <Search className="w-4 h-4 text-slate-400 absolute left-3.5 top-3" />
              <input
                type="text"
                value={searchQuery}
                onChange={(e) => {
                  setSearchQuery(e.target.value);
                  setCurrentPage(1);
                }}
                placeholder="Search by transaction ID, account, amount..."
                className="w-full pl-10 pr-4 py-2 bg-slate-50 dark:bg-slate-950 border border-slate-200 dark:border-slate-800 rounded-xl text-xs text-slate-900 dark:text-white placeholder-slate-400 dark:placeholder-slate-500 focus:outline-none focus:border-indigo-500 transition font-mono"
              />
            </div>
          </div>

          {/* Mobile Card List (< 768px) */}
          <div className="md:hidden space-y-2.5">
            {paginatedRows.length === 0 ? (
              <div className="py-8 text-center text-slate-500 text-xs rounded-2xl border border-slate-800">
                No transaction records match the current filter.
              </div>
            ) : (
              paginatedRows.map((item) => {
                const prob = Number(item.predicted_probability ?? 0);
                const isAlert = prob >= activeThreshold;
                return (
                  <div
                    key={`hist-mob-${item.row_id}`}
                    className="p-3.5 rounded-2xl border border-slate-200 dark:border-slate-800 bg-slate-50 dark:bg-slate-950/80 space-y-2"
                  >
                    <div className="flex items-center justify-between gap-2">
                      <span className="text-xs font-mono font-bold text-slate-900 dark:text-white truncate">
                        {item.transaction_id || `TX_${String(item.row_id).padStart(5, '0')}`}
                      </span>
                      <span
                        className={`px-2 py-0.5 rounded-full text-[10px] font-mono font-bold ${
                          isAlert
                            ? 'bg-rose-100 dark:bg-rose-950 text-rose-700 dark:text-rose-300 border border-rose-200 dark:border-rose-700/60'
                            : 'bg-emerald-100 dark:bg-emerald-950 text-emerald-700 dark:text-emerald-300 border border-emerald-200 dark:border-emerald-700/60'
                        }`}
                      >
                        {isAlert ? 'FLAGGED' : 'CLEARED'} · {prob.toFixed(4)}
                      </span>
                    </div>
                    <div className="flex items-center justify-between text-xs text-slate-600 dark:text-slate-400 font-mono">
                      <span className="truncate">{item.from_account} → {item.to_account}</span>
                      <span className="font-bold text-slate-900 dark:text-white shrink-0 ml-2">
                        {Number(item.amount || 0).toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}
                      </span>
                    </div>
                    <button
                      type="button"
                      onClick={() => handleInspect(item)}
                      className="w-full min-h-[40px] rounded-xl bg-slate-200/70 dark:bg-slate-900 hover:bg-indigo-100 dark:hover:bg-indigo-950 border border-slate-300 dark:border-slate-700 text-xs font-semibold text-indigo-700 dark:text-indigo-300 flex items-center justify-center gap-1.5 transition cursor-pointer"
                    >
                      <Eye className="w-3.5 h-3.5" />
                      <span>Inspect Attribution</span>
                    </button>
                  </div>
                );
              })
            )}
          </div>

          {/* Itemized Table (Horizontally Scrollable with Sticky Inspect Column on Desktop/Tablet) */}
          <div className="hidden md:block overflow-x-auto relative rounded-2xl border border-slate-200 dark:border-slate-800 shadow-sm">
            <table className="w-full text-left border-collapse text-xs font-mono">
              <thead>
                <tr className="bg-slate-100 dark:bg-slate-950 border-b border-slate-200 dark:border-slate-800 text-slate-600 dark:text-slate-400 uppercase text-[10px] sticky top-0 z-10">
                  <th className="py-3 px-3.5 min-w-[120px] whitespace-nowrap">Row Ref</th>
                  <th className="py-3 px-3 min-w-[200px] whitespace-nowrap">Routing (From → To)</th>
                  <th className="py-3 px-3 min-w-[150px] whitespace-nowrap">Timestamp</th>
                  <th className="py-3 px-3 min-w-[110px] text-right whitespace-nowrap">Amount</th>
                  <th className="py-3 px-3 min-w-[95px] text-right whitespace-nowrap">Risk Score</th>
                  <th className="py-3 px-3 min-w-[140px] text-center whitespace-nowrap">Model Decision</th>
                  <th className="py-3 px-3 min-w-[140px] text-center whitespace-nowrap">CSV Ground Truth</th>
                  <th className="py-3 px-3 min-w-[220px] whitespace-nowrap">Evaluation Outcome & Handling</th>
                  <th className="py-3 px-3 min-w-[70px] text-center whitespace-nowrap sticky right-0 bg-slate-100 dark:bg-slate-950 z-20 shadow-[-4px_0_6px_-2px_rgba(0,0,0,0.15)]">Inspect</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-200 dark:divide-slate-800/60">
                {paginatedRows.length === 0 ? (
                  <tr>
                    <td colSpan={9} className="py-8 text-center text-slate-500 text-xs">
                      No transaction records match the current filter.
                    </td>
                  </tr>
                ) : (
                  paginatedRows.map((item) => {
                    const prob = Number(item.predicted_probability ?? 0);
                    const isAlert = prob >= activeThreshold;
                    const isLaunderingLabel =
                      item.ground_truth_label === 1 ||
                      item.actual_label === 'SUSPICIOUS' ||
                      (item as any)['Is Laundering'] === 1 ||
                      (item as any)['is_laundering'] === 1;
                    const isLegitLabel =
                      item.ground_truth_label === 0 ||
                      item.actual_label === 'LEGITIMATE' ||
                      (item as any)['Is Laundering'] === 0 ||
                      (item as any)['is_laundering'] === 0;

                    let outcomeBadge = (
                      <span className="text-slate-600 dark:text-slate-400 text-[11px]">
                        Cleared Legitimate (TN)
                      </span>
                    );

                    if (isAlert) {
                      outcomeBadge = isLaunderingLabel ? (
                        <span className="text-emerald-600 dark:text-emerald-400 text-[11px] font-bold">
                          True Positive (TP) — Alert Confirmed
                        </span>
                      ) : (
                        <span className="text-rose-600 dark:text-rose-400 text-[11px]">
                          False Positive (FP) — False Alarm
                        </span>
                      );
                    } else if (hasLabels && isLaunderingLabel) {
                      outcomeBadge = (
                        <span className="text-amber-600 dark:text-amber-300 text-[11px] font-bold">
                          False Negative (FN) — Needs Review
                        </span>
                      );
                    }

                    return (
                      <tr key={item.row_id} className="hover:bg-slate-50 dark:hover:bg-slate-800/40 transition">
                        <td className="py-2.5 px-3.5 font-bold text-slate-900 dark:text-white whitespace-nowrap">
                          {item.transaction_id || `TX_${String(item.row_id).padStart(5, '0')}`}
                        </td>
                        <td className="py-2.5 px-3 text-slate-700 dark:text-slate-300">
                          <div className="truncate max-w-[190px]">
                            <span className="text-slate-500 dark:text-slate-400">{item.from_account}</span>
                            <span className="text-slate-400 dark:text-slate-500 mx-1">→</span>
                            <span className="text-slate-800 dark:text-slate-300 font-medium">{item.to_account}</span>
                          </div>
                        </td>
                        <td className="py-2.5 px-3 text-slate-500 dark:text-slate-400 whitespace-nowrap font-mono text-[11px]">
                          {item.timestamp ? String(item.timestamp).replace('T', ' ').slice(0, 19) : '—'}
                        </td>
                        <td className="py-2.5 px-3 text-right font-bold text-slate-900 dark:text-white whitespace-nowrap">
                          {Number(item.amount || 0).toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}
                        </td>
                        <td className="py-2.5 px-3 text-right whitespace-nowrap">
                          <span className={`font-bold ${isAlert ? 'text-rose-600 dark:text-rose-400' : 'text-emerald-600 dark:text-emerald-400'}`}>
                            {prob.toFixed(4)}
                          </span>
                        </td>
                        <td className="py-2.5 px-3 text-center whitespace-nowrap">
                          {isAlert ? (
                            <span className="px-2 py-0.5 rounded-full text-[10px] font-bold bg-rose-100 dark:bg-rose-950/80 text-rose-700 dark:text-rose-300 border border-rose-200 dark:border-rose-700/60 inline-flex items-center space-x-1">
                              <ShieldAlert className="w-3 h-3 text-rose-600 dark:text-rose-400" />
                              <span>FLAGGED ALERT</span>
                            </span>
                          ) : (
                            <span className="px-2 py-0.5 rounded-full text-[10px] font-bold bg-emerald-100 dark:bg-emerald-950/80 text-emerald-700 dark:text-emerald-300 border border-emerald-200 dark:border-emerald-700/60 inline-flex items-center space-x-1">
                              <ShieldCheck className="w-3 h-3 text-emerald-600 dark:text-emerald-400" />
                              <span>CLEARED</span>
                            </span>
                          )}
                        </td>
                        <td className="py-2.5 px-3 text-center whitespace-nowrap">
                          {hasLabels ? (
                            isLaunderingLabel ? (
                              <span className="px-2 py-0.5 rounded text-[10px] font-bold bg-purple-100 dark:bg-purple-950 text-purple-700 dark:text-purple-300 border border-purple-200 dark:border-purple-800">
                                Laundering (1)
                              </span>
                            ) : isLegitLabel ? (
                              <span className="px-2 py-0.5 rounded text-[10px] font-bold bg-slate-100 dark:bg-slate-800 text-slate-700 dark:text-slate-300 border border-slate-200 dark:border-slate-700">
                                Legitimate (0)
                              </span>
                            ) : (
                              <span className="text-slate-400 text-[10px]">Unknown</span>
                            )
                          ) : (
                            <span className="text-slate-400 text-[10px] italic">Unlabeled</span>
                          )}
                        </td>
                        <td className="py-2.5 px-3 whitespace-nowrap">
                          {outcomeBadge}
                        </td>
                        <td className="py-2.5 px-3 min-w-[70px] text-center whitespace-nowrap sticky right-0 bg-white/95 dark:bg-slate-900/95 backdrop-blur-sm z-10 shadow-[-4px_0_6px_-2px_rgba(0,0,0,0.15)]">
                          <button
                            onClick={() => handleInspect(item)}
                            className="p-1.5 rounded-lg bg-slate-100 dark:bg-slate-950 hover:bg-indigo-100 dark:hover:bg-indigo-950 text-slate-600 dark:text-slate-400 hover:text-indigo-600 dark:hover:text-indigo-300 border border-slate-200 dark:border-slate-800 hover:border-indigo-300 dark:hover:border-indigo-700/60 transition cursor-pointer"
                            title="Inspect Feature Attributions"
                          >
                            <Eye className="w-3.5 h-3.5" />
                          </button>
                        </td>
                      </tr>
                    );
                  })
                )}
              </tbody>
            </table>
          </div>

          {/* Pagination */}
          <div className="flex items-center justify-between pt-2 text-xs font-mono text-slate-400">
            <span>
              Page {currentPage} of {totalPages} ({filteredRows.length.toLocaleString()} transactions)
            </span>
            <div className="flex items-center space-x-2">
              <button
                disabled={currentPage <= 1}
                onClick={() => setCurrentPage((p) => p - 1)}
                className="px-3 py-1.5 rounded-lg bg-slate-950 border border-slate-800 disabled:opacity-40 hover:text-white transition cursor-pointer"
              >
                Previous
              </button>
              <button
                disabled={currentPage >= totalPages}
                onClick={() => setCurrentPage((p) => p + 1)}
                className="px-3 py-1.5 rounded-lg bg-slate-950 border border-slate-800 disabled:opacity-40 hover:text-white transition cursor-pointer"
              >
                Next
              </button>
            </div>
          </div>
        </div>
      ) : (
        <div className="bg-slate-900 border border-slate-800 rounded-3xl p-10 text-center space-y-4 max-w-xl mx-auto shadow-xl">
          <History className="w-10 h-10 text-indigo-400 mx-auto" />
          <h3 className="text-base font-bold text-white">No Evaluation Currently Active</h3>
          <p className="text-xs text-slate-400">
            Select a saved evaluation run from the table above or upload a new transaction CSV in the AML Evaluation tab.
          </p>
          <button
            onClick={onNavigateToUpload}
            className="inline-flex items-center space-x-2 px-4 py-2 rounded-xl bg-indigo-600 hover:bg-indigo-500 text-white text-xs font-semibold shadow-lg shadow-indigo-600/30 transition cursor-pointer"
          >
            <Upload className="w-4 h-4" />
            <span>Go to AML Evaluation</span>
          </button>
        </div>
      )}

      {/* Attribution Explanation Modal */}
      {inspectingItem && (() => {
        const itemProb = Number(inspectingItem.predicted_probability ?? 0);
        const itemIsAlert = itemProb >= activeThreshold;
        const diff = itemProb - activeThreshold;
        const isLaundering =
          inspectingItem.ground_truth_label === 1 ||
          inspectingItem.actual_label === 'SUSPICIOUS' ||
          (inspectingItem as any)['Is Laundering'] === 1 ||
          (inspectingItem as any)['is_laundering'] === 1;
        const isLegit =
          inspectingItem.ground_truth_label === 0 ||
          inspectingItem.actual_label === 'LEGITIMATE' ||
          (inspectingItem as any)['Is Laundering'] === 0 ||
          (inspectingItem as any)['is_laundering'] === 0;

        const isFalseNegative = hasLabels && isLaundering && !itemIsAlert;

        return (
          <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-slate-950/80 backdrop-blur-sm animate-fade-in">
            <div className="bg-slate-900 border border-slate-800 rounded-3xl p-6 shadow-2xl max-w-lg w-full space-y-4 max-h-[90vh] overflow-y-auto">
              <div className="flex items-center justify-between border-b border-slate-800 pb-3">
                <div className="flex items-center space-x-2">
                  <Eye className="w-5 h-5 text-indigo-400" />
                  <h3 className="text-sm font-bold text-white">
                    Audit Inspector: {inspectingItem.transaction_id || `TX_${String(inspectingItem.row_id).padStart(5, '0')}`}
                  </h3>
                </div>
                <button
                  onClick={() => setInspectingItem(null)}
                  className="text-slate-400 hover:text-white p-1 rounded-lg transition"
                >
                  <X className="w-4 h-4" />
                </button>
              </div>

              {/* Core Score vs. Active Threshold Breakdown */}
              <div className="grid grid-cols-2 gap-2 text-xs font-mono">
                <div className="bg-slate-950 p-2.5 rounded-xl border border-slate-800">
                  <span className="text-slate-400 block text-[10px] uppercase">Probability Score</span>
                  <span className="text-base font-bold text-white">{itemProb.toFixed(4)}</span>
                </div>
                <div className="bg-slate-950 p-2.5 rounded-xl border border-slate-800">
                  <span className="text-slate-400 block text-[10px] uppercase">Active Threshold (θ)</span>
                  <span className="text-base font-bold text-amber-400">{activeThreshold.toFixed(4)}</span>
                </div>
              </div>

              {/* Decision & Score-to-Threshold Comparison */}
              <div className={`p-3 rounded-xl border text-xs font-mono space-y-1 ${
                itemIsAlert
                  ? 'bg-rose-950/40 border-rose-700/60 text-rose-200'
                  : 'bg-emerald-950/40 border-emerald-700/60 text-emerald-200'
              }`}>
                <div className="flex items-center justify-between font-bold">
                  <span>Model Decision: {itemIsAlert ? 'FLAGGED (Alert)' : 'CLEARED (Normal)'}</span>
                  <span className="text-[10px] px-1.5 py-0.5 rounded bg-black/40">
                    {itemIsAlert ? 'Score ≥ θ' : 'Score < θ'}
                  </span>
                </div>
                <p className="text-[11px] opacity-90">
                  Score-to-threshold difference: <strong className="font-mono">{diff >= 0 ? `+${diff.toFixed(4)}` : diff.toFixed(4)}</strong> ({itemIsAlert ? 'Exceeds threshold by' : 'Below cutoff by'} {Math.abs(diff).toFixed(4)}).
                </p>
                <p className="text-[10px] opacity-75">
                  Trigger reason: {itemIsAlert ? 'Probability meets active operational cutoff.' : 'Probability remains below active operational cutoff.'}
                </p>
              </div>

              {/* Ground Truth & False Negative Explanation */}
              {hasLabels && (
                <div className={`p-3 rounded-xl border text-xs space-y-1.5 ${
                  isFalseNegative
                    ? 'bg-amber-950/50 border-amber-600/70 text-amber-200'
                    : 'bg-slate-950 border-slate-800 text-slate-300'
                }`}>
                  <div className="flex items-center justify-between font-bold">
                    <span>CSV Ground Truth: {isLaundering ? 'Laundering (1)' : isLegit ? 'Legitimate (0)' : 'Unlabeled'}</span>
                    <span className={`text-[10px] px-2 py-0.5 rounded-full font-mono font-bold ${
                      isFalseNegative
                        ? 'bg-amber-500 text-black'
                        : itemIsAlert && isLaundering
                        ? 'bg-emerald-600 text-white'
                        : 'bg-slate-800 text-slate-300'
                    }`}>
                      {isFalseNegative
                        ? 'FALSE NEGATIVE'
                        : itemIsAlert && isLaundering
                        ? 'TRUE POSITIVE'
                        : itemIsAlert && !isLaundering
                        ? 'FALSE POSITIVE'
                        : 'TRUE NEGATIVE'}
                    </span>
                  </div>

                  {isFalseNegative && (
                    <div className="space-y-1 text-[11px] leading-relaxed pt-1 border-t border-amber-700/40">
                      <div className="flex items-center space-x-1.5 font-bold text-amber-300">
                        <AlertTriangle className="w-3.5 h-3.5 shrink-0" />
                        <span>Recommended Handling: Needs Analyst Review</span>
                      </div>
                      <p>
                        This transaction is labeled <strong>Laundering</strong> in the CSV dataset, but the model assigned a risk score of <strong>{itemProb.toFixed(4)}</strong>, which is below the active decision cutoff <strong>{activeThreshold.toFixed(4)}</strong>. It is routed as a False Negative for manual investigative triage.
                      </p>
                    </div>
                  )}
                </div>
              )}

              {/* Feature Drivers & Explanation Breakdown */}
              {explaining ? (
                <div className="py-6 text-center text-xs text-slate-400 space-y-2 font-mono">
                  <div className="w-6 h-6 border-2 border-indigo-500 border-t-transparent rounded-full animate-spin mx-auto" />
                  <p>Generating feature attributions...</p>
                </div>
              ) : explanationData && (explanationData.positiveDrivers.length > 0 || explanationData.negativeDrivers.length > 0) ? (
                <div className="space-y-3 text-xs font-mono">
                  {explanationData.positiveDrivers.length > 0 && (
                    <div className="space-y-1.5">
                      <span className="text-[11px] text-emerald-400 font-bold block">Top Suspicious Drivers (+)</span>
                      {explanationData.positiveDrivers.slice(0, 5).map((d) => (
                        <div key={d.feature_name} className="flex items-center justify-between text-[11px] bg-slate-950 px-2 py-1 rounded">
                          <span className="text-slate-300 truncate max-w-[240px]">{d.feature_name}</span>
                          <span className="text-emerald-400 font-bold">+{d.attribution_score.toFixed(4)}</span>
                        </div>
                      ))}
                    </div>
                  )}

                  {explanationData.negativeDrivers.length > 0 && (
                    <div className="space-y-1.5">
                      <span className="text-[11px] text-slate-400 font-bold block">Top Mitigating Drivers (-)</span>
                      {explanationData.negativeDrivers.slice(0, 5).map((d) => (
                        <div key={d.feature_name} className="flex items-center justify-between text-[11px] bg-slate-950 px-2 py-1 rounded">
                          <span className="text-slate-400 truncate max-w-[240px]">{d.feature_name}</span>
                          <span className="text-slate-500 font-bold">{d.attribution_score.toFixed(4)}</span>
                        </div>
                      ))}
                    </div>
                  )}
                </div>
              ) : (
                <div className="p-3 rounded-xl bg-slate-950 border border-slate-800 text-[11px] text-slate-400 space-y-1 font-mono">
                  <div className="flex items-center space-x-1.5 text-slate-300 font-semibold">
                    <Info className="w-3.5 h-3.5 text-indigo-400 shrink-0" />
                    <span>Attribution Diagnostics</span>
                  </div>
                  <p>
                    {explanationData?.historyWarning || explainError || 'Feature-level attribution method is not available for this checkpoint, but the verified model probability score and active threshold comparison above remain authoritative.'}
                  </p>
                </div>
              )}

              <div className="flex justify-end pt-2">
                <button
                  onClick={() => setInspectingItem(null)}
                  className="px-4 py-1.5 rounded-xl bg-slate-800 hover:bg-slate-700 text-white font-semibold text-xs transition cursor-pointer"
                >
                  Close
                </button>
              </div>
            </div>
          </div>
        );
      })()}
    </div>
  );
};
