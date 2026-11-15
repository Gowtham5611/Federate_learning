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
} from 'lucide-react';
import { apiService } from '../services/apiService';

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
  const [statusFilter, setStatusFilter] = useState<'ALL' | 'FLAGGED' | 'NEEDS_REVIEW' | 'CLEARED'>('FLAGGED');
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

  const activeThreshold = result?.decision_threshold ?? 0.337784;
  const hasLabels = Boolean(result?.has_labels);

  // Filtered transactions for active run
  const { filteredRows, flaggedCount, clearedCount, fnCount } = useMemo(() => {
    const preds = result?.predictions || [];
    let flagged = 0;
    let cleared = 0;
    let fn = 0;

    preds.forEach((item) => {
      const prob = Number(item.predicted_probability ?? 0);
      const isAlert = prob >= activeThreshold;
      if (isAlert) flagged++;
      else cleared++;

      if (hasLabels) {
        const isLaunderingLabel =
          item.ground_truth_label === 1 ||
          item.actual_label === 'SUSPICIOUS' ||
          (item as any)['Is Laundering'] === 1 ||
          (item as any)['is_laundering'] === 1;
        if (isLaunderingLabel && !isAlert) fn++;
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

      if (statusFilter === 'FLAGGED' && !isAlert) return false;
      if (statusFilter === 'CLEARED' && isAlert) return false;
      if (statusFilter === 'NEEDS_REVIEW' && !(isLaunderingLabel && !isAlert)) return false;

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

    return { filteredRows: filtered, flaggedCount: flagged, clearedCount: cleared, fnCount: fn };
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
            <button
              onClick={handleExportCsv}
              className="flex items-center space-x-1.5 px-4 py-2 rounded-xl bg-indigo-600 hover:bg-indigo-500 text-white text-xs font-semibold shadow-lg shadow-indigo-600/30 transition cursor-pointer"
            >
              <Download className="w-3.5 h-3.5" />
              <span>Download Active CSV</span>
            </button>
          )}
        </div>
      </div>

      {/* Section 1: Saved Evaluation Runs Table */}
      <div className="bg-slate-900 border border-slate-800 rounded-3xl p-6 shadow-xl space-y-4">
        <div className="flex items-center justify-between">
          <div className="flex items-center space-x-2.5">
            <RotateCcw className="w-5 h-5 text-indigo-400" />
            <div>
              <h2 className="text-base font-bold text-white tracking-tight">
                Persisted Evaluation Runs
              </h2>
              <p className="text-xs text-slate-400">
                Saved run artifacts available in backend storage. Selecting a run restores its exact evaluated state and metrics without re-computation.
              </p>
            </div>
          </div>

          <span className="text-xs font-mono text-indigo-400 font-bold bg-indigo-950/80 px-2.5 py-1 rounded-xl border border-indigo-800/60">
            {historyRuns.length} Runs Available
          </span>
        </div>

        <div className="overflow-x-auto rounded-2xl border border-slate-800">
          <table className="w-full text-left border-collapse text-xs font-mono">
            <thead>
              <tr className="bg-slate-950 border-b border-slate-800 text-slate-400 uppercase text-[10px]">
                <th className="py-3 px-3">Run ID</th>
                <th className="py-3 px-3">Source CSV File</th>
                <th className="py-3 px-3">Model Architecture</th>
                <th className="py-3 px-3 text-center">Threshold</th>
                <th className="py-3 px-3 text-right">Rows</th>
                <th className="py-3 px-3 text-right">Model Alerts</th>
                <th className="py-3 px-3 text-center">Outcomes (TP / FP / FN)</th>
                <th className="py-3 px-3 text-right">Actions</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-800/60">
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
                  return (
                    <tr
                      key={run.prediction_id}
                      className={`hover:bg-slate-800/40 transition ${isActive ? 'bg-indigo-950/20' : ''}`}
                    >
                      <td className="py-2.5 px-3 font-bold text-white whitespace-nowrap">
                        <div className="flex items-center space-x-1.5">
                          {isActive && <CheckCircle2 className="w-3.5 h-3.5 text-emerald-400 shrink-0" />}
                          <span className={isActive ? 'text-indigo-300 font-bold' : 'text-slate-300'}>
                            {run.prediction_id}
                          </span>
                        </div>
                      </td>
                      <td className="py-2.5 px-3 text-slate-200 font-medium truncate max-w-[180px]">
                        {run.filename}
                      </td>
                      <td className="py-2.5 px-3 text-slate-400 text-[11px] truncate max-w-[180px]">
                        {run.model_name}
                      </td>
                      <td className="py-2.5 px-3 text-center font-bold text-amber-300 whitespace-nowrap">
                        θ = {run.decision_threshold.toFixed(4)}
                      </td>
                      <td className="py-2.5 px-3 text-right font-bold text-white whitespace-nowrap">
                        {run.total_rows.toLocaleString()}
                      </td>
                      <td className="py-2.5 px-3 text-right font-bold whitespace-nowrap">
                        <span className={run.flagged_count > 0 ? 'text-rose-400' : 'text-slate-400'}>
                          {run.flagged_count.toLocaleString()}
                        </span>
                      </td>
                      <td className="py-2.5 px-3 text-center text-[10px] whitespace-nowrap">
                        {run.has_labels && metrics.tp !== undefined ? (
                          <span className="text-slate-300">
                            TP: <strong className="text-emerald-400">{metrics.tp}</strong> · FP: <strong className="text-rose-400">{metrics.fp}</strong> · FN: <strong className="text-amber-400">{metrics.fn}</strong>
                          </span>
                        ) : (
                          <span className="text-slate-500 italic">Score-Only</span>
                        )}
                      </td>
                      <td className="py-2.5 px-3 text-right whitespace-nowrap">
                        <div className="flex items-center justify-end space-x-1.5">
                          {onSelectHistoricalRun && (
                            <button
                              onClick={() => onSelectHistoricalRun(run.prediction_id)}
                              className={`px-2.5 py-1 rounded-lg text-[11px] font-semibold transition cursor-pointer ${
                                isActive
                                  ? 'bg-emerald-950 text-emerald-300 border border-emerald-700/60'
                                  : 'bg-indigo-600 hover:bg-indigo-500 text-white'
                              }`}
                            >
                              {isActive ? 'Active Run' : 'Load Run'}
                            </button>
                          )}
                          <button
                            onClick={() => window.open(apiService.getPdfReportUrl(run.prediction_id), '_blank')}
                            title="Download PDF Report"
                            className="p-1 rounded-lg bg-slate-950 hover:bg-slate-800 text-slate-400 hover:text-white border border-slate-800 transition"
                          >
                            <FileText className="w-3.5 h-3.5 text-indigo-400" />
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
        <div className="bg-slate-900 border border-slate-800 rounded-3xl p-6 shadow-xl space-y-4">
          <div className="flex flex-col md:flex-row md:items-center justify-between gap-3 border-b border-slate-800 pb-4">
            <div>
              <div className="flex items-center space-x-2">
                <ShieldAlert className="w-5 h-5 text-indigo-400" />
                <h2 className="text-base font-bold text-white tracking-tight">
                  Itemized Transactions for Run <span className="font-mono text-indigo-300">{result.prediction_id}</span>
                </h2>
                <span className="px-2 py-0.5 rounded-full text-[10px] font-mono bg-slate-800 text-slate-300 border border-slate-700">
                  {filteredRows.length.toLocaleString()} Matches
                </span>
              </div>
              <p className="text-xs text-slate-400 mt-1">
                Distinguish model alerts from under-review false negatives and cleared normal records under active cutoff <code className="text-amber-300">θ = {activeThreshold.toFixed(4)}</code>.
              </p>
            </div>

            {/* Status Filter Tabs */}
            <div className="flex flex-wrap items-center gap-1.5 bg-slate-950 p-1 rounded-xl border border-slate-800 text-xs">
              <button
                onClick={() => {
                  setStatusFilter('FLAGGED');
                  setCurrentPage(1);
                }}
                className={`px-3 py-1.5 rounded-lg transition font-medium cursor-pointer flex items-center space-x-1.5 ${
                  statusFilter === 'FLAGGED' ? 'bg-rose-600 text-white font-bold' : 'text-slate-400 hover:text-white'
                }`}
              >
                <span>Flagged (Model Alerts)</span>
                <span className={`px-1.5 py-0.2 rounded-full text-[10px] font-mono ${
                  statusFilter === 'FLAGGED' ? 'bg-rose-900 text-white' : 'bg-slate-800 text-slate-300'
                }`}>
                  {flaggedCount.toLocaleString()}
                </span>
              </button>

              {hasLabels && (
                <button
                  onClick={() => {
                    setStatusFilter('NEEDS_REVIEW');
                    setCurrentPage(1);
                  }}
                  className={`px-3 py-1.5 rounded-lg transition font-medium cursor-pointer flex items-center space-x-1.5 ${
                    statusFilter === 'NEEDS_REVIEW' ? 'bg-amber-600 text-white font-bold' : 'text-amber-400 hover:text-amber-200'
                  }`}
                >
                  <span>Needs Review (False Negatives)</span>
                  <span className={`px-1.5 py-0.2 rounded-full text-[10px] font-mono ${
                    statusFilter === 'NEEDS_REVIEW' ? 'bg-amber-950 text-amber-200 border border-amber-700' : 'bg-slate-800 text-amber-300'
                  }`}>
                    {fnCount.toLocaleString()}
                  </span>
                </button>
              )}

              <button
                onClick={() => {
                  setStatusFilter('CLEARED');
                  setCurrentPage(1);
                }}
                className={`px-3 py-1.5 rounded-lg transition font-medium cursor-pointer flex items-center space-x-1.5 ${
                  statusFilter === 'CLEARED' ? 'bg-emerald-600 text-white font-bold' : 'text-slate-400 hover:text-white'
                }`}
              >
                <span>Legitimate Cleared</span>
                <span className={`px-1.5 py-0.2 rounded-full text-[10px] font-mono ${
                  statusFilter === 'CLEARED' ? 'bg-emerald-900 text-white' : 'bg-slate-800 text-slate-300'
                }`}>
                  {clearedCount.toLocaleString()}
                </span>
              </button>

              <button
                onClick={() => {
                  setStatusFilter('ALL');
                  setCurrentPage(1);
                }}
                className={`px-3 py-1.5 rounded-lg transition font-medium cursor-pointer flex items-center space-x-1.5 ${
                  statusFilter === 'ALL' ? 'bg-indigo-600 text-white font-bold' : 'text-slate-400 hover:text-white'
                }`}
              >
                <span>All Transactions</span>
                <span className={`px-1.5 py-0.2 rounded-full text-[10px] font-mono ${
                  statusFilter === 'ALL' ? 'bg-indigo-900 text-white' : 'bg-slate-800 text-slate-300'
                }`}>
                  {(result.predictions || []).length.toLocaleString()}
                </span>
              </button>
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
                className="w-full pl-10 pr-4 py-2 bg-slate-950 border border-slate-800 rounded-xl text-xs text-white placeholder-slate-500 focus:outline-none focus:border-indigo-500 transition font-mono"
              />
            </div>
          </div>

          {/* Itemized Table */}
          <div className="overflow-x-auto rounded-2xl border border-slate-800">
            <table className="w-full text-left border-collapse text-xs font-mono">
              <thead>
                <tr className="bg-slate-950 border-b border-slate-800 text-slate-400 uppercase text-[10px]">
                  <th className="py-3 px-3">Row Ref</th>
                  <th className="py-3 px-3">Routing (From → To)</th>
                  <th className="py-3 px-3">Timestamp</th>
                  <th className="py-3 px-3 text-right">Amount</th>
                  <th className="py-3 px-3 text-right">Risk Score</th>
                  <th className="py-3 px-3 text-center">Model Decision</th>
                  <th className="py-3 px-3 text-center">CSV Ground Truth</th>
                  <th className="py-3 px-3">Evaluation Outcome & Handling</th>
                  <th className="py-3 px-2 text-center">Inspect</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-800/60">
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
                      <span className="text-slate-400 text-[11px]">
                        Cleared Legitimate (TN)
                      </span>
                    );

                    if (isAlert) {
                      outcomeBadge = isLaunderingLabel ? (
                        <span className="text-emerald-400 text-[11px] font-bold">
                          True Positive (TP) — Alert Confirmed
                        </span>
                      ) : (
                        <span className="text-rose-400 text-[11px]">
                          False Positive (FP) — False Alarm
                        </span>
                      );
                    } else if (hasLabels && isLaunderingLabel) {
                      outcomeBadge = (
                        <span className="text-amber-300 text-[11px] font-bold">
                          False Negative (FN) — Needs Review
                        </span>
                      );
                    }

                    return (
                      <tr key={item.row_id} className="hover:bg-slate-800/40 transition">
                        <td className="py-2.5 px-3 font-bold text-white whitespace-nowrap">
                          {item.transaction_id || `TX_${String(item.row_id).padStart(5, '0')}`}
                        </td>
                        <td className="py-2.5 px-3 text-slate-300">
                          <div className="truncate max-w-[150px]">
                            <span className="text-slate-400">{item.from_account}</span>
                            <span className="text-slate-500 mx-1">→</span>
                            <span className="text-slate-300">{item.to_account}</span>
                          </div>
                        </td>
                        <td className="py-2.5 px-3 text-slate-400 whitespace-nowrap">
                          {String(item.timestamp || '').slice(0, 16)}
                        </td>
                        <td className="py-2.5 px-3 text-right font-bold text-white whitespace-nowrap">
                          {Number(item.amount || 0).toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}
                        </td>
                        <td className="py-2.5 px-3 text-right whitespace-nowrap">
                          <span className={`font-bold ${isAlert ? 'text-rose-400' : 'text-emerald-400'}`}>
                            {prob.toFixed(4)}
                          </span>
                        </td>
                        <td className="py-2.5 px-3 text-center whitespace-nowrap">
                          {isAlert ? (
                            <span className="px-2 py-0.5 rounded-full text-[10px] font-bold bg-rose-950/80 text-rose-300 border border-rose-700/60 inline-flex items-center space-x-1">
                              <ShieldAlert className="w-3 h-3 text-rose-400" />
                              <span>FLAGGED ALERT</span>
                            </span>
                          ) : (
                            <span className="px-2 py-0.5 rounded-full text-[10px] font-bold bg-emerald-950/80 text-emerald-300 border border-emerald-700/60 inline-flex items-center space-x-1">
                              <ShieldCheck className="w-3 h-3 text-emerald-400" />
                              <span>CLEARED</span>
                            </span>
                          )}
                        </td>
                        <td className="py-2.5 px-3 text-center whitespace-nowrap">
                          {hasLabels ? (
                            isLaunderingLabel ? (
                              <span className="px-2 py-0.5 rounded text-[10px] font-bold bg-purple-950 text-purple-300 border border-purple-800">
                                Laundering (1)
                              </span>
                            ) : isLegitLabel ? (
                              <span className="px-2 py-0.5 rounded text-[10px] font-bold bg-slate-800 text-slate-300 border border-slate-700">
                                Legitimate (0)
                              </span>
                            ) : (
                              <span className="text-slate-500 text-[10px]">Unknown</span>
                            )
                          ) : (
                            <span className="text-slate-500 text-[10px] italic">Unlabeled</span>
                          )}
                        </td>
                        <td className="py-2.5 px-3">
                          {outcomeBadge}
                        </td>
                        <td className="py-2.5 px-2 text-center whitespace-nowrap">
                          <button
                            onClick={() => handleInspect(item)}
                            className="p-1.5 rounded-lg bg-slate-950 hover:bg-indigo-950/80 text-slate-400 hover:text-indigo-300 border border-slate-800 hover:border-indigo-700/60 transition cursor-pointer"
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
      {inspectingItem && (
        <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-slate-950/80 backdrop-blur-sm animate-fade-in">
          <div className="bg-slate-900 border border-slate-800 rounded-3xl p-6 shadow-2xl max-w-lg w-full space-y-4">
            <div className="flex items-center justify-between border-b border-slate-800 pb-3">
              <div className="flex items-center space-x-2">
                <Eye className="w-5 h-5 text-indigo-400" />
                <h3 className="text-sm font-bold text-white">
                  Transaction Feature Attribution: {inspectingItem.transaction_id || `TX_${inspectingItem.row_id}`}
                </h3>
              </div>
              <button
                onClick={() => setInspectingItem(null)}
                className="text-slate-400 hover:text-white p-1 rounded-lg transition"
              >
                <X className="w-4 h-4" />
              </button>
            </div>

            {explaining ? (
              <div className="py-8 text-center text-xs text-slate-400 space-y-2">
                <div className="w-6 h-6 border-2 border-indigo-500 border-t-transparent rounded-full animate-spin mx-auto" />
                <p>Generating Integrated Gradients attributions...</p>
              </div>
            ) : explainError ? (
              <div className="p-3 rounded-xl bg-rose-950/50 border border-rose-800/50 text-xs text-rose-300">
                {explainError}
              </div>
            ) : explanationData ? (
              <div className="space-y-4 text-xs font-mono">
                <div className="flex items-center justify-between bg-slate-950 p-2.5 rounded-xl border border-slate-800">
                  <span className="text-slate-400">Predicted Probability:</span>
                  <span className="font-bold text-rose-400">
                    {(explanationData.predictedProbability ?? inspectingItem.predicted_probability).toFixed(4)}
                  </span>
                </div>

                <div className="space-y-2">
                  <span className="text-[11px] text-emerald-400 font-bold block">Top Suspicious Drivers (+)</span>
                  {explanationData.positiveDrivers.slice(0, 5).map((d) => (
                    <div key={d.feature_name} className="flex items-center justify-between text-[11px]">
                      <span className="text-slate-300 truncate max-w-[200px]">{d.feature_name}</span>
                      <span className="text-emerald-400 font-bold">+{d.attribution_score.toFixed(4)}</span>
                    </div>
                  ))}
                </div>

                <div className="space-y-2">
                  <span className="text-[11px] text-slate-400 font-bold block">Top Mitigating Drivers (-)</span>
                  {explanationData.negativeDrivers.slice(0, 5).map((d) => (
                    <div key={d.feature_name} className="flex items-center justify-between text-[11px]">
                      <span className="text-slate-400 truncate max-w-[200px]">{d.feature_name}</span>
                      <span className="text-slate-500 font-bold">{d.attribution_score.toFixed(4)}</span>
                    </div>
                  ))}
                </div>
              </div>
            ) : null}
          </div>
        </div>
      )}
    </div>
  );
};
