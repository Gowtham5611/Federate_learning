import React, { useState, useEffect, useMemo } from 'react';
import { CsvEvaluateResponse, TransactionPredictionItem } from '../types/api';
import {
  History,
  Download,
  Search,
  Eye,
  Upload,
  ShieldAlert,
  ShieldCheck,
  X,
  FileText,
  RotateCcw,
  CheckCircle2,
  Trash2,
  Sliders,
  AlertTriangle,
  Info,
} from 'lucide-react';
import { apiService } from '../services/apiService';

export type PredictionFilterMode = 'ALL' | 'PRED_SUSPICIOUS' | 'PRED_LEGITIMATE' | 'NEEDS_REVIEW_FN';

interface HistoricalRunItem {
  prediction_id: string;
  filename: string;
  model_id: string;
  model_name: string;
  decision_threshold: number;
  total_rows: number;
  flagged_count: number;
  has_labels: boolean;
  created_at: number;
  created_at_iso: string;
  eval_metrics?: any;
}

interface PredictionHistoryPageProps {
  result: CsvEvaluateResponse | null;
  onNavigateToUpload: () => void;
  onSelectHistoricalRun?: (runId: string) => void;
  onClearSavedResults?: () => void;
}

export const PredictionHistoryPage: React.FC<PredictionHistoryPageProps> = ({
  result,
  onNavigateToUpload,
  onSelectHistoricalRun,
  onClearSavedResults,
}) => {
  const [historyRuns, setHistoryRuns] = useState<HistoricalRunItem[]>([]);
  const [loadingHistory, setLoadingHistory] = useState<boolean>(false);
  const [statusFilter, setStatusFilter] = useState<PredictionFilterMode>('ALL');
  const [searchQuery, setSearchQuery] = useState<string>('');
  const [currentPage, setCurrentPage] = useState<number>(1);
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

  // Load historical runs from backend
  const fetchRuns = async () => {
    setLoadingHistory(true);
    try {
      const runs = await apiService.getPredictionHistory();
      setHistoryRuns(runs || []);
    } catch (err) {
      console.warn('Failed to fetch prediction history:', err);
    } finally {
      setLoadingHistory(false);
    }
  };

  useEffect(() => {
    fetchRuns();
  }, [result?.prediction_id]);

  const canonicalThreshold = result?.decision_threshold ?? 0.337784;
  const [activeThreshold, setActiveThreshold] = useState<number>(canonicalThreshold);
  const [sliderVal, setSliderVal] = useState<number>(canonicalThreshold);
  const hasLabels = Boolean(result?.has_labels);

  useEffect(() => {
    const t = result?.decision_threshold ?? 0.337784;
    setActiveThreshold(t);
    setSliderVal(t);
  }, [result?.decision_threshold, result?.prediction_id]);

  // Dynamically recomputed metrics and reconciled confusion counts from real active run data
  const {
    filteredRows,
    predSuspiciousCount,
    predLegitimateCount,
    missedFnCount,
    tpCount,
    fpCount,
    tnCount,
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

  // Handle Export CSV
  const handleExportCsv = () => {
    if (!result) return;
    const pid = result.prediction_id || result.download_id;
    if (pid) {
      window.open(apiService.getPredictionsDownloadUrl(pid), '_blank');
      return;
    }
  };

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
            <span>Evaluation Run History & Triage</span>
          </h1>
          <p className="text-xs text-slate-400 mt-1">
            Persisted evaluation run artifacts, compliance triage queues, and auditable transaction-level records.
          </p>
        </div>

        <div className="flex items-center space-x-2">
          {onClearSavedResults && (
            <button
              onClick={onClearSavedResults}
              className="flex items-center space-x-1.5 px-3 py-2 rounded-xl bg-slate-900 hover:bg-rose-950/60 text-slate-300 hover:text-rose-300 border border-slate-800 hover:border-rose-700/50 text-xs font-semibold transition cursor-pointer"
            >
              <Trash2 className="w-3.5 h-3.5 text-rose-400" />
              <span>Clear Saved Results</span>
            </button>
          )}

          {result && (
            <div className="flex flex-wrap items-center gap-2">
              <button
                onClick={() => {
                  const pid = result.prediction_id || result.download_id;
                  if (pid) {
                    window.open(apiService.getFlaggedCsvDownloadUrl(pid, activeThreshold), '_blank');
                  }
                }}
                className="flex items-center space-x-1.5 px-3.5 py-2 rounded-xl bg-rose-950/80 hover:bg-rose-900 border border-rose-700/60 text-rose-300 text-xs font-semibold transition cursor-pointer"
                title="Download CSV containing only transactions flagged by active model threshold"
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
                  title="Download CSV containing false negatives: CSV laundering rows cleared by model"
                >
                  <AlertTriangle className="w-3.5 h-3.5 text-amber-400" />
                  <span>Needs Review CSV</span>
                </button>
              )}

              <button
                onClick={() => {
                  const pid = result.prediction_id || result.download_id;
                  if (pid) {
                    window.open(apiService.getPdfReportUrl(pid), '_blank');
                  }
                }}
                className="flex items-center space-x-1.5 px-3.5 py-2 rounded-xl bg-indigo-950/80 hover:bg-indigo-900 border border-indigo-700/60 text-indigo-300 text-xs font-semibold transition cursor-pointer"
              >
                <FileText className="w-3.5 h-3.5" />
                <span>PDF Report</span>
              </button>

              <button
                onClick={handleExportCsv}
                className="flex items-center space-x-1.5 px-3.5 py-2 rounded-xl bg-slate-900 hover:bg-slate-800 border border-slate-700 text-slate-300 text-xs font-semibold transition cursor-pointer"
              >
                <Download className="w-3.5 h-3.5" />
                <span>Master CSV</span>
              </button>
            </div>
          )}
        </div>
      </div>

      {/* Section 1: Saved Evaluation Runs Table */}
      <div className="bg-white dark:bg-slate-900 border border-slate-200 dark:border-slate-800 rounded-3xl p-6 shadow-xl space-y-4">
        <div className="flex items-center justify-between">
          <div className="flex items-center space-x-2.5">
            <RotateCcw className="w-5 h-5 text-indigo-500 dark:text-indigo-400" />
            <div>
              <h2 className="text-base font-bold text-slate-900 dark:text-white tracking-tight">
                Persisted Evaluation Runs
              </h2>
              <p className="text-xs text-slate-500 dark:text-slate-400">
                Saved run artifacts available in backend storage. Selecting a run restores its exact evaluated state and metrics without re-computation.
              </p>
            </div>
          </div>

          <span className="text-xs font-mono text-indigo-600 dark:text-indigo-400 font-bold bg-indigo-50 dark:bg-indigo-950/80 px-2.5 py-1 rounded-xl border border-indigo-200 dark:border-indigo-800/60">
            {historyRuns.length} Runs Available
          </span>
        </div>

        <div className="overflow-x-auto rounded-2xl border border-slate-200 dark:border-slate-800">
          <table className="w-full text-left border-collapse text-xs font-mono">
            <thead>
              <tr className="bg-slate-100 dark:bg-slate-950 border-b border-slate-200 dark:border-slate-800 text-slate-600 dark:text-slate-400 uppercase text-[10px]">
                <th className="py-3 px-3">Run ID</th>
                <th className="py-3 px-3">Source CSV File</th>
                <th className="py-3 px-3">Model Architecture</th>
                <th className="py-3 px-3 text-center">Threshold</th>
                <th className="py-3 px-3 text-right">Rows</th>
                <th className="py-3 px-3 text-right">Model Alerts (TP + FP)</th>
                <th className="py-3 px-3 text-center">Outcomes (TP / FP / FN)</th>
                <th className="py-3 px-3 text-right">Actions</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-200 dark:divide-slate-800/60">
              {historyRuns.length === 0 ? (
                <tr>
                  <td colSpan={8} className="py-6 text-center text-slate-500 text-xs">
                    {loadingHistory ? 'Loading saved evaluation runs...' : 'No historical evaluation runs found in storage.'}
                  </td>
                </tr>
              ) : (
                historyRuns.slice(0, 8).map((run) => {
                  const isActive = result?.prediction_id === run.prediction_id;
                  const metrics = run.eval_metrics || {};
                  const modelAlerts = run.flagged_count > 0
                    ? run.flagged_count
                    : (metrics.tp !== undefined && metrics.fp !== undefined ? (Number(metrics.tp) + Number(metrics.fp)) : 0);

                  return (
                    <tr
                      key={run.prediction_id}
                      className={`hover:bg-slate-50 dark:hover:bg-slate-800/40 transition ${isActive ? 'bg-indigo-50/50 dark:bg-indigo-950/20' : ''}`}
                    >
                      <td className="py-2.5 px-3 font-bold text-slate-900 dark:text-white whitespace-nowrap">
                        <div className="flex items-center space-x-1.5">
                          {isActive && <CheckCircle2 className="w-3.5 h-3.5 text-emerald-500 dark:text-emerald-400 shrink-0" />}
                          <span className={isActive ? 'text-indigo-600 dark:text-indigo-300 font-bold' : 'text-slate-700 dark:text-slate-300'}>
                            {run.prediction_id}
                          </span>
                        </div>
                      </td>
                      <td className="py-2.5 px-3 text-slate-700 dark:text-slate-200 font-medium truncate max-w-[180px]">
                        {run.filename}
                      </td>
                      <td className="py-2.5 px-3 text-slate-500 dark:text-slate-400 text-[11px] truncate max-w-[180px]">
                        {run.model_name}
                      </td>
                      <td className="py-2.5 px-3 text-center font-bold text-amber-600 dark:text-amber-300 whitespace-nowrap">
                        θ = {run.decision_threshold.toFixed(4)}
                      </td>
                      <td className="py-2.5 px-3 text-right font-bold text-slate-900 dark:text-white whitespace-nowrap">
                        {run.total_rows.toLocaleString()}
                      </td>
                      <td className="py-2.5 px-3 text-right font-bold whitespace-nowrap">
                        <span className={modelAlerts > 0 ? 'text-rose-600 dark:text-rose-400 font-bold' : 'text-slate-400'}>
                          {modelAlerts.toLocaleString()}
                        </span>
                      </td>
                      <td className="py-2.5 px-3 text-center text-[10px] whitespace-nowrap">
                        {run.has_labels && metrics.tp !== undefined ? (
                          <span className="text-slate-600 dark:text-slate-300">
                            TP: <strong className="text-emerald-600 dark:text-emerald-400">{metrics.tp}</strong> · FP: <strong className="text-rose-600 dark:text-rose-400">{metrics.fp}</strong> · FN: <strong className="text-amber-600 dark:text-amber-400">{metrics.fn}</strong>
                          </span>
                        ) : (
                          <span className="text-slate-400 italic">Score-Only</span>
                        )}
                      </td>
                      <td className="py-2.5 px-3 text-right whitespace-nowrap">
                        <div className="flex items-center justify-end space-x-1.5">
                          {onSelectHistoricalRun && (
                            <button
                              onClick={() => onSelectHistoricalRun(run.prediction_id)}
                              className={`px-2.5 py-1 rounded-lg text-[11px] font-semibold transition cursor-pointer ${
                                isActive
                                  ? 'bg-emerald-100 dark:bg-emerald-950 text-emerald-800 dark:text-emerald-300 border border-emerald-300 dark:border-emerald-700/60'
                                  : 'bg-indigo-600 hover:bg-indigo-500 text-white'
                              }`}
                            >
                              {isActive ? 'Active Run' : 'Load Run'}
                            </button>
                          )}
                          <button
                            onClick={() => window.open(apiService.getPdfReportUrl(run.prediction_id), '_blank')}
                            title="Download PDF Report"
                            className="p-1 rounded-lg bg-slate-100 dark:bg-slate-950 hover:bg-slate-200 dark:hover:bg-slate-800 text-slate-600 dark:text-slate-400 hover:text-slate-900 dark:hover:text-white border border-slate-200 dark:border-slate-800 transition"
                          >
                            <FileText className="w-3.5 h-3.5 text-indigo-500 dark:text-indigo-400" />
                          </button>
                        </div>
                      </td>
                    </tr>
                  );
                })
              )}
            </tbody>
          </table>
        </div>
      </div>

      {/* Section 2: Active Run Itemized Transaction Triage Table */}
      {result ? (
        <div className="bg-white dark:bg-slate-900 border border-slate-200 dark:border-slate-800 rounded-3xl p-6 shadow-xl space-y-4">
          <div className="flex flex-col md:flex-row md:items-center justify-between gap-3 border-b border-slate-200 dark:border-slate-800 pb-4">
            <div>
              <div className="flex items-center space-x-2">
                <ShieldAlert className="w-5 h-5 text-indigo-500 dark:text-indigo-400" />
                <h2 className="text-base font-bold text-slate-900 dark:text-white tracking-tight">
                  Itemized Transactions for Run <span className="font-mono text-indigo-600 dark:text-indigo-300">{result.prediction_id}</span>
                </h2>
                <span className="px-2 py-0.5 rounded-full text-[10px] font-mono bg-slate-100 dark:bg-slate-800 text-slate-700 dark:text-slate-300 border border-slate-200 dark:border-slate-700">
                  {filteredRows.length.toLocaleString()} Matches
                </span>
                <span className="px-2 py-0.5 rounded-full text-[10px] font-mono font-bold bg-amber-100 dark:bg-amber-950/80 text-amber-800 dark:text-amber-300 border border-amber-300 dark:border-amber-700/60">
                  θ = {activeThreshold.toFixed(4)}
                </span>
              </div>
              <p className="text-xs text-slate-500 dark:text-slate-400 mt-1">
                Decisions evaluated dynamically: <code className="text-amber-600 dark:text-amber-300 font-bold">flagged = score ≥ {activeThreshold.toFixed(4)}</code>. Transactions below cutoff remain model-cleared regardless of CSV label.
              </p>
            </div>

            {/* Threshold Interactive Presets & Slider */}
            <div className="flex flex-col sm:flex-row sm:items-center gap-2.5 bg-slate-50 dark:bg-slate-950 p-2.5 rounded-2xl border border-slate-200 dark:border-slate-800">
              <div className="flex items-center space-x-1.5">
                <Sliders className="w-4 h-4 text-indigo-500 dark:text-indigo-400" />
                <span className="text-[11px] font-bold text-slate-700 dark:text-slate-300 font-mono">Cutoff θ:</span>
              </div>
              <div className="flex items-center gap-1.5 flex-wrap">
                <button
                  onClick={() => {
                    setActiveThreshold(canonicalThreshold);
                    setSliderVal(canonicalThreshold);
                  }}
                  className={`px-2 py-1 rounded-lg text-[10px] font-mono font-semibold transition cursor-pointer border ${
                    Math.abs(activeThreshold - canonicalThreshold) < 1e-4
                      ? 'bg-indigo-600 text-white border-indigo-500 shadow-sm'
                      : 'bg-white dark:bg-slate-900 text-slate-600 dark:text-slate-400 border-slate-200 dark:border-slate-800 hover:text-white'
                  }`}
                >
                  Canonical ({canonicalThreshold.toFixed(4)})
                </button>
                <button
                  onClick={() => {
                    setActiveThreshold(0.20);
                    setSliderVal(0.20);
                  }}
                  className={`px-2 py-1 rounded-lg text-[10px] font-mono font-semibold transition cursor-pointer border ${
                    Math.abs(activeThreshold - 0.20) < 1e-4
                      ? 'bg-amber-600 text-white border-amber-500 shadow-sm'
                      : 'bg-white dark:bg-slate-900 text-slate-600 dark:text-slate-400 border-slate-200 dark:border-slate-800 hover:text-white'
                  }`}
                >
                  High Recall (0.20)
                </button>
                <button
                  onClick={() => {
                    setActiveThreshold(0.50);
                    setSliderVal(0.50);
                  }}
                  className={`px-2 py-1 rounded-lg text-[10px] font-mono font-semibold transition cursor-pointer border ${
                    Math.abs(activeThreshold - 0.50) < 1e-4
                      ? 'bg-rose-600 text-white border-rose-500 shadow-sm'
                      : 'bg-white dark:bg-slate-900 text-slate-600 dark:text-slate-400 border-slate-200 dark:border-slate-800 hover:text-white'
                  }`}
                >
                  Strict (0.50)
                </button>
              </div>
              <div className="flex items-center space-x-2 flex-1 min-w-[120px]">
                <input
                  type="range"
                  min="0.00"
                  max="1.00"
                  step="0.005"
                  value={sliderVal}
                  onChange={(e) => {
                    const v = parseFloat(e.target.value);
                    setSliderVal(v);
                    setActiveThreshold(v);
                  }}
                  className="w-full h-1.5 bg-slate-200 dark:bg-slate-800 rounded-lg appearance-none cursor-pointer accent-indigo-500"
                />
                <span className="text-[11px] font-mono font-bold text-amber-600 dark:text-amber-400 min-w-[42px] text-right">
                  {activeThreshold.toFixed(3)}
                </span>
              </div>
            </div>

            {/* Reconciled Confusion & Operating Counts */}
            {hasLabels && (
              <div className="flex flex-wrap items-center gap-2.5 text-[11px] font-mono px-3.5 py-2 rounded-xl bg-slate-50 dark:bg-slate-950 border border-slate-200 dark:border-slate-800 text-slate-600 dark:text-slate-400">
                <span className="font-bold text-slate-800 dark:text-slate-200">Reconciled Counts:</span>
                <span className="text-emerald-600 dark:text-emerald-400 font-bold" title="True Positives: CSV Laundering and Score >= θ">
                  TP: {tpCount.toLocaleString()}
                </span>
                <span className="text-rose-600 dark:text-rose-400 font-bold" title="False Positives: CSV Legitimate and Score >= θ">
                  FP: {fpCount.toLocaleString()}
                </span>
                <span className="text-amber-600 dark:text-amber-400 font-bold" title="False Negatives (Needs Review): CSV Laundering and Score < θ">
                  FN: {missedFnCount.toLocaleString()}
                </span>
                <span className="text-indigo-600 dark:text-indigo-400 font-bold" title="True Negatives: CSV Legitimate and Score < θ">
                  TN: {tnCount.toLocaleString()}
                </span>
                <span className="text-slate-400 dark:text-slate-600">|</span>
                <span>Flagged (TP+FP): <strong className="text-rose-600 dark:text-rose-400">{(tpCount + fpCount).toLocaleString()}</strong></span>
                <span>Cleared (TN+FN): <strong className="text-emerald-600 dark:text-emerald-400">{(tnCount + missedFnCount).toLocaleString()}</strong></span>
                <span>Total: <strong className="text-slate-800 dark:text-slate-200">{totalCount.toLocaleString()}</strong></span>
              </div>
            )}

            {/* Status Filter Tabs (Dynamic Counts) */}
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

              {/* Tab 2: Predicted Suspicious (Alerts) */}
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
                <span>Model Flagged</span>
                <span className={`px-1.5 py-0.2 rounded-full text-[10px] font-mono ${
                  statusFilter === 'PRED_SUSPICIOUS' ? 'bg-rose-800 text-white' : 'bg-slate-200 dark:bg-slate-800 text-slate-700 dark:text-slate-300'
                }`}>
                  {predSuspiciousCount.toLocaleString()}
                </span>
              </button>

              {/* Tab 3: Predicted Legitimate (Cleared) */}
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
                <span>Model Cleared</span>
                <span className={`px-1.5 py-0.2 rounded-full text-[10px] font-mono ${
                  statusFilter === 'PRED_LEGITIMATE' ? 'bg-emerald-800 text-white' : 'bg-slate-200 dark:bg-slate-800 text-slate-700 dark:text-slate-300'
                }`}>
                  {predLegitimateCount.toLocaleString()}
                </span>
              </button>

              {/* Tab 4: Needs Review (False Negatives) */}
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
                  title="False Negatives: CSV laundering rows where model score is below active cutoff"
                >
                  <AlertTriangle className="w-3.5 h-3.5" />
                  <span>Needs Review (False Negatives)</span>
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

          {/* Itemized Table (Horizontally Scrollable with Sticky Inspect Column) */}
          <div className="overflow-x-auto relative rounded-2xl border border-slate-200 dark:border-slate-800 shadow-sm">
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
                        <td className="py-2.5 px-3">
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
