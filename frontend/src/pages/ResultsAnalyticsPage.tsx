import React, { useState, useMemo, useEffect, useRef } from 'react';
import { apiService } from '../services/apiService';
import { CsvEvaluateResponse, MultiModelEvaluateResponse, BenchmarkModelRecord } from '../types/api';
import { ValidityBanner } from '../components/ValidityBanner';
import { WhyNoAlertsCard } from '../components/WhyNoAlertsCard';
import { type OperatingPointKey } from '../components/OperatingPointSelector';
import {
  BarChart2,
  TrendingUp,
  AlertTriangle,
  Upload,
  Info,
  ShieldAlert,
  ShieldCheck,
  Target,
  Sliders,
  RefreshCw,
  Download,
  HelpCircle,
  FileText,
} from 'lucide-react';
import { formatMetric, formatPercent, formatRatioPercent } from '../utils/formatters';
import { FlaggedTransactionsTable, TableFilterMode } from '../components/FlaggedTransactionsTable';
import { PerformanceHeatmap } from '../components/PerformanceHeatmap';
import { EvaluationPieChart } from '../components/EvaluationPieChart';
import { TransactionRiskHeatmap } from '../components/TransactionRiskHeatmap';

interface ResultsAnalyticsPageProps {
  result: CsvEvaluateResponse | null;
  multiModelResult?: MultiModelEvaluateResponse | null;
  onNavigateToUpload: () => void;
}

export const ResultsAnalyticsPage: React.FC<ResultsAnalyticsPageProps> = ({
  result,
  multiModelResult,
  onNavigateToUpload,
}) => {
  if (!result) {
    return (
      <div className="bg-slate-900 border border-slate-800 rounded-3xl p-12 text-center space-y-4 max-w-2xl mx-auto my-12 shadow-xl animate-fade-in">
        <div className="w-16 h-16 rounded-2xl bg-indigo-950/80 text-indigo-400 flex items-center justify-center mx-auto border border-indigo-700/50">
          <BarChart2 className="w-8 h-8" />
        </div>
        <h2 className="text-xl font-bold text-white">No Evaluation Results Available</h2>
        <p className="text-xs text-slate-400 max-w-md mx-auto">
          Please upload and evaluate a banking transaction CSV in the AML Evaluation tab to inspect performance metrics, confusion matrices, and empirical ROC/PR curves.
        </p>
        <button
          onClick={onNavigateToUpload}
          className="inline-flex items-center space-x-2 px-5 py-2.5 rounded-xl bg-indigo-600 hover:bg-indigo-500 text-white text-xs font-semibold shadow-lg shadow-indigo-600/30 transition cursor-pointer"
        >
          <Upload className="w-4 h-4" />
          <span>Go to AML Evaluation</span>
        </button>
      </div>
    );
  }

  const defaultThreshold = result.decision_threshold ?? 0.337784;
  const [threshold, setThreshold] = useState<number>(defaultThreshold);
  const [sliderDisplayVal, setSliderDisplayVal] = useState<number>(defaultThreshold);
  const sliderDebounceRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const [selectedOperatingPoint, setSelectedOperatingPoint] = useState<OperatingPointKey>('frozen');
  const [isManualSliderActive, setIsManualSliderActive] = useState<boolean>(false);
  const [showExplainPanel, setShowExplainPanel] = useState<boolean>(false);
  const [activeTableFilter, setActiveTableFilter] = useState<TableFilterMode>('flagged');
  const [heatmapCellFilter, setHeatmapCellFilter] = useState<{
    timeBucket: string;
    riskBand: string;
    minScore: number;
    maxScore: number;
  } | null>(null);

  // Active operating point helper
  const activeOp = useMemo(() => {
    const ops = result.operating_points || result.evaluation_metrics?.operating_points;
    if (!ops) return null;
    if (selectedOperatingPoint === 'zero_diagnostic') return ops.zero_diagnostic || null;
    if (selectedOperatingPoint === 'budget_500') return ops.alert_budgets?.['500'] || null;
    if (selectedOperatingPoint === 'budget_10') return ops.alert_budgets?.['10'] || null;
    if (selectedOperatingPoint === 'budget_50') return ops.alert_budgets?.['50'] || null;
    if (selectedOperatingPoint === 'budget_100') return ops.alert_budgets?.['100'] || null;
    if (selectedOperatingPoint === 'best_f1') return ops.best_f1_on_file || null;
    return ops.frozen || null;
  }, [result, selectedOperatingPoint]);

  // Sync threshold with new result
  useEffect(() => {
    setThreshold(result.decision_threshold ?? 0.337784);
    setSelectedOperatingPoint('frozen');
    setIsManualSliderActive(false);
  }, [result.decision_threshold, result.filename]);

  const handleSelectOperatingPoint = (point: OperatingPointKey) => {
    setSelectedOperatingPoint(point);
    setIsManualSliderActive(false);
    const ops = result.operating_points || result.evaluation_metrics?.operating_points;
    if (!ops) return;

    if (point === 'zero_diagnostic') {
      if (ops.zero_diagnostic) {
        setThreshold(ops.zero_diagnostic.threshold);
      } else {
        setThreshold(0.0);
      }
    } else if (point === 'budget_500' && ops.alert_budgets?.['500']) {
      setThreshold(ops.alert_budgets['500'].threshold);
    } else if (point === 'budget_10' && ops.alert_budgets?.['10']) {
      setThreshold(ops.alert_budgets['10'].threshold);
    } else if (point === 'budget_50' && ops.alert_budgets?.['50']) {
      setThreshold(ops.alert_budgets['50'].threshold);
    } else if (point === 'budget_100' && ops.alert_budgets?.['100']) {
      setThreshold(ops.alert_budgets['100'].threshold);
    } else if (point === 'best_f1' && ops.best_f1_on_file) {
      setThreshold(ops.best_f1_on_file.threshold);
    } else if (ops.frozen) {
      setThreshold(ops.frozen.threshold);
    } else {
      setThreshold(defaultThreshold);
    }
  };

  const hasMetrics = Boolean(result.has_labels && result.evaluation_metrics);
  const staticMetrics = result.evaluation_metrics;

  // Real-time client-side recomputation of all threshold-dependent metrics
  const computed = useMemo(() => {
    const preds = result.predictions || [];
    const total = preds.length > 0 ? preds.length : (result.total_rows || result.evaluated_sequences || 0);

    // If an operating point is selected and the user has not manually adjusted the continuous slider,
    // use the exact authoritative operating point metrics from server-side evaluation
    if (!isManualSliderActive && activeOp) {
      const tp = activeOp.tp ?? 0;
      const fp = activeOp.fp ?? 0;
      const fn = activeOp.fn ?? 0;
      const tn = activeOp.tn ?? 0;
      const flagged = activeOp.alerts ?? (tp + fp);
      const cleared = total - flagged;
      const precision = activeOp.precision ?? (flagged > 0 ? tp / flagged : null);
      const recall = activeOp.recall ?? ((tp + fn) > 0 ? tp / (tp + fn) : 0);
      const specificity = activeOp.specificity ?? ((tn + fp) > 0 ? tn / (tn + fp) : 0);
      const f1_score = activeOp.f1_score ?? (precision && recall ? (2 * precision * recall) / (precision + recall) : 0);
      const accuracy = activeOp.accuracy ?? ((tp + tn) / total);
      const balanced_accuracy = activeOp.balanced_accuracy ?? ((recall + specificity) / 2);
      const mcc = activeOp.mcc ?? 0;
      const cohen_kappa = activeOp.cohen_kappa ?? 0;
      const fpr = fp + tn > 0 ? fp / (fp + tn) : 0;
      const fnr = tp + fn > 0 ? fn / (tp + fn) : 0;
      const fdr = tp + fp > 0 ? fp / (tp + fp) : 0;
      const npv = tn + fn > 0 ? tn / (tn + fn) : 0;

      return {
        total,
        flagged,
        cleared,
        tp,
        fp,
        fn,
        tn,
        accuracy,
        precision,
        recall,
        sensitivity: recall,
        specificity,
        f1_score,
        f2_score: activeOp.f2_score ?? staticMetrics?.f2_score ?? null,
        balanced_accuracy,
        mcc,
        cohen_kappa,
        fpr,
        fnr,
        fdr,
        npv,
        alertCount: flagged,
      };
    }

    let tp = 0;
    let fp = 0;
    let fn = 0;
    let tn = 0;
    let flagged = 0;

    preds.forEach((p) => {
      const prob = p.predicted_probability ?? 0;
      const isFlagged = prob >= threshold;
      if (isFlagged) flagged++;

      if (hasMetrics && p.ground_truth_label !== null && p.ground_truth_label !== undefined) {
        const isActualLaundering = p.ground_truth_label === 1;
        if (isActualLaundering && isFlagged) tp++;
        else if (!isActualLaundering && isFlagged) fp++;
        else if (isActualLaundering && !isFlagged) fn++;
        else tn++;
      }
    });

    const cleared = total - flagged;
    const alertCount = flagged;

    // Fallback if predictions array is empty but we have server-evaluated counts
    if (preds.length === 0 && Math.abs(threshold - defaultThreshold) < 1e-4) {
      return {
        total,
        flagged: result.flagged_laundering_count,
        cleared: result.cleared_count,
        tp: staticMetrics?.confusion_matrix?.tp ?? 0,
        fp: staticMetrics?.confusion_matrix?.fp ?? 0,
        fn: staticMetrics?.confusion_matrix?.fn ?? 0,
        tn: staticMetrics?.confusion_matrix?.tn ?? 0,
        accuracy: staticMetrics?.accuracy ?? 0,
        precision: staticMetrics?.precision ?? null,
        recall: staticMetrics?.recall ?? 0,
        sensitivity: staticMetrics?.sensitivity ?? staticMetrics?.recall ?? 0,
        specificity: staticMetrics?.specificity ?? 0,
        f1_score: staticMetrics?.f1_score ?? null,
        f2_score: staticMetrics?.f2_score ?? null,
        balanced_accuracy: staticMetrics?.balanced_accuracy ?? 0,
        mcc: staticMetrics?.mcc ?? 0,
        cohen_kappa: staticMetrics?.cohen_kappa ?? 0,
        fpr: staticMetrics?.false_positive_rate ?? 0,
        fnr: staticMetrics?.false_negative_rate ?? 0,
        fdr: staticMetrics?.false_discovery_rate ?? 0,
        npv: staticMetrics?.negative_predictive_value ?? 0,
        alertCount: (staticMetrics?.confusion_matrix?.tp ?? 0) + (staticMetrics?.confusion_matrix?.fp ?? 0),
      };
    }

    if (!hasMetrics) {
      return {
        total,
        flagged,
        cleared,
        tp: 0,
        fp: 0,
        fn: 0,
        tn: 0,
        accuracy: 0,
        precision: null,
        recall: 0,
        sensitivity: 0,
        specificity: 0,
        f1_score: null,
        f2_score: null,
        balanced_accuracy: 0,
        mcc: 0,
        cohen_kappa: 0,
        fpr: 0,
        fnr: 0,
        fdr: 0,
        npv: 0,
        alertCount,
      };
    }

    const accuracy = total > 0 ? (tp + tn) / total : 0;
    const precision = (tp + fp) > 0 ? tp / (tp + fp) : null;
    const recall = (tp + fn) > 0 ? tp / (tp + fn) : 0;
    const specificity = (tn + fp) > 0 ? tn / (tn + fp) : 0;
    const sensitivity = recall;

    const f1_score =
      precision !== null && recall > 0 && precision + recall > 0
        ? (2 * precision * recall) / (precision + recall)
        : (tp === 0 && (tp + fn) > 0)
        ? 0.0
        : null;

    const f2_score =
      precision !== null && recall > 0 && 4 * precision + recall > 0
        ? (5 * precision * recall) / (4 * precision + recall)
        : (tp === 0 && (tp + fn) > 0)
        ? 0.0
        : null;

    const balanced_accuracy = (recall + specificity) / 2;

    const mccDenom = Math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn));
    const mcc = mccDenom > 0 ? (tp * tn - fp * fn) / mccDenom : 0;

    const po = total > 0 ? (tp + tn) / total : 0;
    const pe =
      total > 0
        ? ((tp + fp) * (tp + fn) + (fn + tn) * (fp + tn)) / (total * total)
        : 0;
    const cohen_kappa = 1 - pe > 0 ? (po - pe) / (1 - pe) : 0;

    const fpr = fp + tn > 0 ? fp / (fp + tn) : 0;
    const fnr = tp + fn > 0 ? fn / (tp + fn) : 0;
    const fdr = tp + fp > 0 ? fp / (tp + fp) : 0;
    const npv = tn + fn > 0 ? tn / (tn + fn) : 0;

    return {
      total,
      flagged,
      cleared,
      tp,
      fp,
      fn,
      tn,
      accuracy,
      precision,
      recall,
      sensitivity,
      specificity,
      f1_score,
      f2_score,
      balanced_accuracy,
      mcc,
      cohen_kappa,
      fpr,
      fnr,
      fdr,
      npv,
      alertCount,
    };
  }, [result, threshold, hasMetrics, defaultThreshold, staticMetrics]);

  // Download predictions CSV with active threshold applied
  const handleDownloadPredictions = () => {
    const headers = [
      'Row_ID',
      'Timestamp',
      'From_Account',
      'To_Account',
      'Amount',
      'predicted_probability',
      'predicted_label',
      'prediction_status',
      'model_name',
      'classification_threshold',
      'is_laundering',
    ];

    const preds = result.predictions || [];
    const rows = preds.map((p) => {
      const prob = p.predicted_probability ?? 0;
      const isAlert = prob >= threshold;
      return [
        p.row_id,
        p.timestamp || '',
        p.from_account || '',
        p.to_account || '',
        p.amount ?? 0,
        prob.toFixed(6),
        isAlert ? 'SUSPICIOUS' : 'LEGITIMATE',
        isAlert ? 'FLAGGED_LAUNDERING' : 'CLEARED_LEGITIMATE',
        result.model_name || 'FedSecureAI',
        threshold.toFixed(6),
        p.ground_truth_label !== null && p.ground_truth_label !== undefined
          ? p.ground_truth_label
          : 'UNLABELED',
      ];
    });

    const csvContent = [
      headers.join(','),
      ...rows.map((r) => r.map((c) => `"${String(c).replace(/"/g, '""')}"`).join(',')),
    ].join('\n');

    const blob = new Blob([csvContent], { type: 'text/csv;charset=utf-8;' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `fedsecureai_${(result.dataset || 'dataset').toLowerCase()}_predictions_thr${threshold.toFixed(2)}.csv`;
    a.click();
    URL.revokeObjectURL(url);
  };

  return (
    <div className="space-y-8 animate-fade-in">
      {/* Header */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 border-b border-slate-800 pb-5">
        <div>
          <h1 className="text-2xl sm:text-3xl font-black text-white tracking-tight flex items-center space-x-2">
            <BarChart2 className="w-7 h-7 text-indigo-400" />
            <span>Evaluation Results & Visual Analytics</span>
          </h1>
          <p className="text-xs text-slate-400 mt-1">
            Empirical validation metrics computed from {result.evaluated_sequences} transactions ({result.filename}) using{' '}
            <span className="text-indigo-300 font-semibold">{result.model_name}</span>.
          </p>
        </div>

        <div className="flex flex-wrap items-center gap-2">
          <button
            onClick={() => {
              const pid = result.prediction_id || result.download_id;
              if (pid) {
                window.open(apiService.getFlaggedCsvDownloadUrl(pid, threshold), '_blank');
              }
            }}
            className="flex items-center space-x-1.5 px-3.5 py-1.5 rounded-xl bg-rose-950/80 hover:bg-rose-900 border border-rose-700/60 text-rose-300 text-xs font-semibold transition cursor-pointer"
            title="Download CSV containing all transactions flagged by active model threshold or documented rule"
          >
            <ShieldAlert className="w-3.5 h-3.5 text-rose-400" />
            <span>Download Flagged CSV</span>
          </button>

          {hasMetrics && computed.fn > 0 && (
            <button
              onClick={() => {
                const pid = result.prediction_id || result.download_id;
                if (pid) {
                  window.open(apiService.getNeedsReviewCsvDownloadUrl(pid, threshold), '_blank');
                }
              }}
              className="flex items-center space-x-1.5 px-3.5 py-1.5 rounded-xl bg-amber-950/80 hover:bg-amber-900 border border-amber-700/60 text-amber-300 text-xs font-semibold transition cursor-pointer"
              title="Download CSV containing ground-truth positive laundering cases cleared by model"
            >
              <AlertTriangle className="w-3.5 h-3.5 text-amber-400" />
              <span>Download Needs Review CSV</span>
            </button>
          )}

          <button
            onClick={() => {
              const pid = result.prediction_id || result.download_id;
              if (pid) {
                window.open(apiService.getPdfReportUrl(pid, selectedOperatingPoint, activeTableFilter), '_blank');
              }
            }}
            className="flex items-center space-x-1.5 px-3.5 py-1.5 rounded-xl bg-indigo-950/80 hover:bg-indigo-900 border border-indigo-700/60 text-indigo-300 text-xs font-semibold transition cursor-pointer"
          >
            <FileText className="w-3.5 h-3.5" />
            <span>Download Report (PDF)</span>
          </button>

          <button
            onClick={handleDownloadPredictions}
            className="flex items-center space-x-1.5 px-3.5 py-1.5 rounded-xl bg-slate-900 hover:bg-slate-800 border border-slate-700 text-slate-300 text-xs font-semibold transition cursor-pointer"
          >
            <Download className="w-3.5 h-3.5" />
            <span>Master CSV</span>
          </button>

          <button
            onClick={onNavigateToUpload}
            className="flex items-center space-x-1.5 px-3.5 py-1.5 rounded-xl bg-slate-900 hover:bg-slate-800 border border-slate-800 text-slate-300 text-xs font-semibold transition cursor-pointer"
          >
            <Upload className="w-3.5 h-3.5 text-indigo-400" />
            <span>Upload Another CSV</span>
          </button>
        </div>
      </div>

      {/* Upload Validity Banner */}
      <ValidityBanner banner={result.validity_banner} />

      {/* Diagnostic "Why no alerts?" card when alerts <= 5 or payload present */}
      <WhyNoAlertsCard
        whyNoAlerts={result.why_no_alerts}
        alertCount={computed.alertCount}
        threshold={threshold}
        onSelectOperatingPoint={handleSelectOperatingPoint}
      />

      {/* Active Decision Threshold Control (Single Unified Setting) */}
      <div className="bg-slate-900/90 border border-indigo-500/30 rounded-2xl p-5 shadow-xl space-y-4">
        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3">
          <div className="flex items-center space-x-2.5">
            <Sliders className="w-5 h-5 text-indigo-400" />
            <div>
              <h3 className="text-sm font-bold text-white flex items-center space-x-2 flex-wrap gap-1">
                <span>Active Decision Threshold:</span>
                <span className={`px-2.5 py-0.5 rounded text-xs font-mono font-bold border ${
                  Math.abs(threshold - 0.0) < 1e-4
                    ? 'bg-amber-950 text-amber-300 border-amber-600/60'
                    : 'bg-indigo-950 text-indigo-300 border-indigo-700/60'
                }`}>
                  θ = {threshold.toFixed(4)}
                  {Math.abs(threshold - 0.0) < 1e-4
                    ? ' (Diagnostic Baseline)'
                    : Math.abs(threshold - defaultThreshold) < 1e-4
                    ? ' (Operational Serving)'
                    : ' (Custom Cutoff)'}
                </span>
              </h3>
              <p className="text-[11px] text-slate-400 mt-1">
                {Math.abs(threshold - 0.0) < 1e-4 ? (
                  <span className="text-amber-300">
                    Diagnostic setting active: decision rule <code className="font-mono font-bold">score ≥ 0.0</code> flags every valid non-negative score ({computed.flagged.toLocaleString()} flagged). This is a diagnostic baseline and not an operational serving setting.
                  </span>
                ) : (
                  <span>
                    Frozen operational serving threshold: <code className="text-indigo-300 font-mono font-bold">θ* = {defaultThreshold.toFixed(4)}</code> (validation-optimal F1). All displayed metrics, confusion counts, pie charts, heatmap, and transaction rows reflect this active threshold only.
                  </span>
                )}
              </p>
            </div>
          </div>

          <div className="flex items-center space-x-2 shrink-0">
            {Math.abs(threshold - 0.0) < 1e-4 ? (
              <button
                type="button"
                onClick={() => {
                  if (sliderDebounceRef.current) clearTimeout(sliderDebounceRef.current);
                  setThreshold(defaultThreshold);
                  setSliderDisplayVal(defaultThreshold);
                  setSelectedOperatingPoint('frozen');
                  setIsManualSliderActive(false);
                }}
                className="px-3 py-1.5 rounded-xl bg-indigo-600 hover:bg-indigo-500 text-white text-xs font-semibold shadow-md shadow-indigo-600/30 transition flex items-center space-x-1.5 cursor-pointer"
              >
                <RefreshCw className="w-3.5 h-3.5" />
                <span>Return to Serving Threshold ({defaultThreshold.toFixed(4)})</span>
              </button>
            ) : (
              <button
                type="button"
                onClick={() => {
                  if (sliderDebounceRef.current) clearTimeout(sliderDebounceRef.current);
                  setThreshold(0.0);
                  setSliderDisplayVal(0.0);
                  setSelectedOperatingPoint('zero_diagnostic');
                  setIsManualSliderActive(true);
                }}
                className="px-3 py-1.5 rounded-xl bg-slate-950 hover:bg-amber-950/60 text-amber-300 border border-amber-700/50 text-xs font-semibold transition flex items-center space-x-1.5 cursor-pointer"
              >
                <AlertTriangle className="w-3.5 h-3.5 text-amber-400" />
                <span>Threshold 0 Diagnostic</span>
              </button>
            )}

            {Math.abs(threshold - defaultThreshold) > 1e-4 && Math.abs(threshold - 0.0) >= 1e-4 && (
              <button
                type="button"
                onClick={() => {
                  if (sliderDebounceRef.current) clearTimeout(sliderDebounceRef.current);
                  setThreshold(defaultThreshold);
                  setSliderDisplayVal(defaultThreshold);
                  setSelectedOperatingPoint('frozen');
                  setIsManualSliderActive(false);
                }}
                className="px-2.5 py-1.5 rounded-xl text-xs text-indigo-400 hover:text-indigo-300 font-mono transition border border-indigo-800/50 bg-indigo-950/40 cursor-pointer"
              >
                Reset ({defaultThreshold.toFixed(4)})
              </button>
            )}
          </div>
        </div>

        {/* Continuous Active Slider */}
        <div className="space-y-1.5">
          <input
            type="range"
            min="0.00"
            max="1.00"
            step="0.005"
            value={sliderDisplayVal}
            onChange={(e) => {
              const val = parseFloat(e.target.value);
              setSliderDisplayVal(val);
              setIsManualSliderActive(true);
              if (sliderDebounceRef.current) clearTimeout(sliderDebounceRef.current);
              sliderDebounceRef.current = setTimeout(() => {
                setThreshold(val);
                if (Math.abs(val - 0.0) < 1e-4) {
                  setSelectedOperatingPoint('zero_diagnostic');
                } else if (Math.abs(val - defaultThreshold) < 1e-4) {
                  setSelectedOperatingPoint('frozen');
                }
              }, 60);
            }}
            className="w-full h-2 bg-slate-950 rounded-lg appearance-none cursor-pointer accent-indigo-500 border border-slate-800"
          />
          <div className="flex items-center justify-between text-[10px] text-slate-500 font-mono">
            <span>0.00 (Diagnostic: Flags All Valid Scores)</span>
            <span>θ* = {defaultThreshold.toFixed(4)} (Operational Serving Threshold)</span>
            <span>1.00 (Zero Tolerance / 0 Alerts)</span>
          </div>
        </div>
      </div>

      {/* Scenario A: Supervised Metrics Section (If ground truth labels available) */}
      {hasMetrics && staticMetrics ? (
        <div className="space-y-6">
          {/* Explain this evaluation toggle button */}
          <div className="flex justify-end">
            <button
              onClick={() => setShowExplainPanel(!showExplainPanel)}
              className="inline-flex items-center space-x-1.5 text-xs text-indigo-400 hover:text-indigo-300 font-mono bg-indigo-950/40 border border-indigo-800/40 px-3 py-1.5 rounded-xl transition cursor-pointer"
            >
              <HelpCircle className="w-3.5 h-3.5" />
              <span>{showExplainPanel ? 'Hide Evaluation Methodology' : 'Explain This Evaluation'}</span>
            </button>
          </div>

          {/* Explain this evaluation collapsible banner */}
          {showExplainPanel && (
            <div className="bg-slate-900/90 border border-indigo-500/30 rounded-2xl p-5 space-y-3 text-xs text-slate-300 animate-fade-in shadow-xl">
              <h4 className="font-bold text-white flex items-center space-x-2">
                <Info className="w-4 h-4 text-indigo-400" />
                <span>AML Evaluation Methodology & Interpretation Guide</span>
              </h4>
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4 pt-1">
                <div className="bg-slate-950 p-3 rounded-xl border border-slate-800 space-y-1">
                  <strong className="text-emerald-400 block font-mono">1. Headline PR-AUC (Not Raw Accuracy)</strong>
                  <p className="text-slate-400 leading-relaxed text-[11px]">
                    At typical banking AML prevalence (0.18%), an uninformative model that classifies every transaction as legitimate achieves 99.82% Accuracy. PR-AUC measures true positive ranking power invariant to majority-class size, providing the only scientifically valid benchmark.
                  </p>
                </div>
                <div className="bg-slate-950 p-3 rounded-xl border border-slate-800 space-y-1">
                  <strong className="text-amber-400 block font-mono">2. Decision Threshold (What-If Budgeting)</strong>
                  <p className="text-slate-400 leading-relaxed text-[11px]">
                    The canonical serving threshold θ* = 0.337784 was frozen exclusively by maximizing validation F1. The slider enables what-if sensitivity analysis: lower thresholds increase recall (catching more laundering), while higher thresholds minimize analyst fatigue (false alarms).
                  </p>
                </div>
                <div className="bg-slate-950 p-3 rounded-xl border border-slate-800 space-y-1">
                  <strong className="text-indigo-400 block font-mono">3. Sequence Context & History Truncation</strong>
                  <p className="text-slate-400 leading-relaxed text-[11px]">
                    FedSecureAI recurrent sequence models process up to 10 historical transactions per sending account. Single-transaction rows with 90% padding output conservative prior probabilities (~0.001 - 0.015). Multi-step history is required for full model activation.
                  </p>
                </div>
                <div className="bg-slate-950 p-3 rounded-xl border border-slate-800 space-y-1">
                  <strong className="text-cyan-400 block font-mono">4. Prior-Shift Adjustment</strong>
                  <p className="text-slate-400 leading-relaxed text-[11px]">
                    If an uploaded testing slice has elevated positive prevalence (e.g. synthetic test with 30% positives), log-odds adjustment is available to rescale raw model outputs to match the sample prior.
                  </p>
                </div>
              </div>
            </div>
          )}

          {/* Row 1: Primary AML Metrics (PR-AUC, Precision, Recall, Specificity, F1, F2, Balanced Acc) */}
          <div className="grid grid-cols-2 sm:grid-cols-4 lg:grid-cols-7 gap-3">
            {/* 1. Headline PR-AUC */}
            <div className="bg-slate-900 border border-emerald-500/30 rounded-2xl p-4 shadow-md bg-gradient-to-br from-slate-900 via-slate-900 to-emerald-950/20">
              <span className="text-[10px] text-emerald-400 font-mono uppercase font-bold block">★ Headline PR-AUC (Threshold-Invariant)</span>
              <span className="text-2xl font-black font-mono text-emerald-400 mt-1 block">
                {formatMetric(staticMetrics.pr_auc, 4)}
              </span>
              <span className="text-[10px] text-slate-400 font-mono block truncate">
                {staticMetrics.pr_auc_lift ? `${staticMetrics.pr_auc_lift.toFixed(1)}x Lift` : 'Ranking Invariant'}
                {staticMetrics.pr_auc_ci_lower ? ` [${staticMetrics.pr_auc_ci_lower.toFixed(2)}, ${staticMetrics.pr_auc_ci_upper?.toFixed(2)}]` : ''}
              </span>
            </div>

            {/* 2. Precision */}
            <div className="bg-slate-900 border border-slate-800 rounded-2xl p-4 shadow-md">
              <span className="text-[10px] text-slate-400 font-mono uppercase block">Precision</span>
              <span className="text-2xl font-black font-mono text-white mt-1 block">
                {formatPercent(computed.precision, 1, '0 Alerts')}
              </span>
              <span className="text-[10px] text-slate-500 font-mono">TP / (TP + FP)</span>
            </div>

            {/* 3. Recall */}
            <div className="bg-slate-900 border border-slate-800 rounded-2xl p-4 shadow-md">
              <span className="text-[10px] text-slate-400 font-mono uppercase block">Recall (Sens.)</span>
              <span className="text-2xl font-black font-mono text-indigo-300 mt-1 block">
                {formatPercent(computed.recall, 1)}
              </span>
              <span className="text-[10px] text-slate-500 font-mono">Laundering Caught</span>
            </div>

            {/* 4. Specificity */}
            <div className="bg-slate-900 border border-slate-800 rounded-2xl p-4 shadow-md">
              <span className="text-[10px] text-slate-400 font-mono uppercase block">Specificity</span>
              <span className="text-2xl font-black font-mono text-cyan-300 mt-1 block">
                {formatPercent(computed.specificity, 1)}
              </span>
              <span className="text-[10px] text-slate-500 font-mono">True Negative Rate</span>
            </div>

            {/* 5. F1-Score */}
            <div className="bg-slate-900 border border-slate-800 rounded-2xl p-4 shadow-md">
              <span className="text-[10px] text-slate-400 font-mono uppercase block">F1-Score</span>
              <span className="text-2xl font-black font-mono text-amber-400 mt-1 block">
                {formatMetric(computed.f1_score, 4, '0 Alerts')}
              </span>
              <span className="text-[10px] text-slate-500 font-mono">Harmonic Mean</span>
            </div>

            {/* 6. F2-Score */}
            <div className="bg-slate-900 border border-slate-800 rounded-2xl p-4 shadow-md">
              <span className="text-[10px] text-slate-400 font-mono uppercase block">F2-Score</span>
              <span className="text-2xl font-black font-mono text-amber-300 mt-1 block">
                {formatMetric(computed.f2_score, 4, '0 Alerts')}
              </span>
              <span className="text-[10px] text-slate-500 font-mono">Recall-Weighted</span>
            </div>

            {/* 7. Balanced Accuracy */}
            <div className="bg-slate-900 border border-slate-800 rounded-2xl p-4 shadow-md">
              <span className="text-[10px] text-slate-400 font-mono uppercase block">Balanced Acc</span>
              <span className="text-2xl font-black font-mono text-teal-400 mt-1 block">
                {formatPercent(computed.balanced_accuracy, 1)}
              </span>
              <span className="text-[10px] text-slate-500 font-mono">(Sens + Spec) / 2</span>
            </div>
          </div>

          {/* Row 2: Advanced Metrics (MCC, Cohen's Kappa, PR-AUC, ROC-AUC, Log Loss, Brier, ECE) */}
          <div className="grid grid-cols-2 sm:grid-cols-4 lg:grid-cols-7 gap-3">
            <div className="bg-slate-900 border border-slate-800 rounded-2xl p-4 shadow-md">
              <span className="text-[10px] text-slate-400 font-mono uppercase block">Accuracy</span>
              <span className="text-xl font-black font-mono text-slate-200 mt-1 block">
                {formatPercent(computed.accuracy, 1)}
              </span>
              <span className="text-[9px] text-slate-500 font-mono block">Caveat: Imbalance</span>
            </div>

            <div className="bg-slate-900 border border-slate-800 rounded-2xl p-4 shadow-md">
              <span className="text-[10px] text-slate-400 font-mono uppercase block">MCC</span>
              <span className="text-xl font-black font-mono text-violet-300 mt-1 block">
                {formatMetric(computed.mcc, 4)}
              </span>
              <span className="text-[10px] text-slate-500 font-mono">Matthews Corr.</span>
            </div>

            <div className="bg-slate-900 border border-slate-800 rounded-2xl p-4 shadow-md">
              <span className="text-[10px] text-slate-400 font-mono uppercase block">Cohen's Kappa</span>
              <span className="text-xl font-black font-mono text-violet-400 mt-1 block">
                {formatMetric(computed.cohen_kappa, 4)}
              </span>
              <span className="text-[10px] text-slate-500 font-mono">Inter-Rater Agr.</span>
            </div>

            <div className="bg-slate-900 border border-slate-800 rounded-2xl p-4 shadow-md">
              <span className="text-[10px] text-slate-400 font-mono uppercase block">ROC-AUC (Threshold-Invariant)</span>
              <span className="text-xl font-black font-mono text-indigo-400 mt-1 block">
                {formatMetric(staticMetrics.roc_auc, 4)}
              </span>
              <span className="text-[10px] text-slate-500 font-mono">Ranking Invariant</span>
            </div>

            <div className="bg-slate-900 border border-slate-800 rounded-2xl p-4 shadow-md">
              <span className="text-[10px] text-slate-400 font-mono uppercase block">Log Loss</span>
              <span className="text-xl font-black font-mono text-rose-300 mt-1 block">
                {formatMetric(staticMetrics.log_loss ?? null, 4, 'N/A')}
              </span>
              <span className="text-[10px] text-slate-500 font-mono">Cross-Entropy</span>
            </div>

            <div className="bg-slate-900 border border-slate-800 rounded-2xl p-4 shadow-md">
              <span className="text-[10px] text-slate-400 font-mono uppercase block">Brier Score</span>
              <span className="text-xl font-black font-mono text-cyan-400 mt-1 block">
                {formatMetric(staticMetrics.brier_score, 6)}
              </span>
              <span className="text-[10px] text-slate-500 font-mono">Prob. Mean Sq.</span>
            </div>

            <div className="bg-slate-900 border border-slate-800 rounded-2xl p-4 shadow-md">
              <span className="text-[10px] text-slate-400 font-mono uppercase block">ECE</span>
              <span className="text-xl font-black font-mono text-cyan-300 mt-1 block">
                {formatMetric(staticMetrics.expected_calibration_error, 5, 'N/A')}
              </span>
              <span className="text-[10px] text-slate-500 font-mono">Expected Calib.</span>
            </div>
          </div>

          {/* Zero Alerts Notice */}
          {computed.alertCount === 0 && (
            <div className="p-3.5 bg-amber-950/40 border border-amber-800/60 rounded-xl text-xs text-amber-200 flex items-start space-x-2.5">
              <Info className="w-4 h-4 text-amber-400 shrink-0 mt-0.5" />
              <div className="space-y-0.5">
                <span className="font-bold text-amber-300 font-mono uppercase text-[11px] block">
                  Zero Laundering Alerts at θ = {threshold.toFixed(4)}
                </span>
                <span className="text-slate-300 leading-normal">
                  All transactions scored below the current decision threshold. Lower the threshold using the slider above to increase sensitivity and flag lower-confidence anomalies.
                </span>
              </div>
            </div>
          )}

          {/* Insufficient History / Truncated Sequence Warning */}
          {result.sequence_context_diagnostics?.insufficient_history_warning && (
            <div className="p-4 bg-amber-950/30 border border-amber-600/60 rounded-2xl text-xs text-amber-200 flex items-start space-x-3 shadow-lg">
              <AlertTriangle className="w-5 h-5 text-amber-400 shrink-0 mt-0.5" />
              <div className="space-y-1.5 flex-1">
                <div className="flex flex-wrap items-center justify-between gap-2">
                  <span className="font-bold text-amber-300">
                    INSUFFICIENT SEQUENCE CONTEXT WARNING
                  </span>
                  <span className="px-2 py-0.5 rounded text-[10px] font-mono bg-amber-900/60 text-amber-200 border border-amber-700">
                    Avg History: {result.sequence_context_diagnostics.average_sequence_length} /{' '}
                    {result.sequence_context_diagnostics.max_sequence_length} Steps
                  </span>
                </div>
                <p className="text-slate-300 leading-relaxed">
                  {result.sequence_context_diagnostics.warning_message}
                </p>
                <div className="pt-1 text-[11px] text-slate-400 font-mono flex flex-wrap gap-4 border-t border-amber-900/40 mt-1">
                  <span>Single-Tx Rows: {(result.sequence_context_diagnostics.single_transaction_ratio * 100).toFixed(1)}%</span>
                  <span>Mean Padding: {(result.sequence_context_diagnostics.mean_padding_ratio * 100).toFixed(1)}%</span>
                  <span>Operational Status: <strong className="text-amber-300">{result.sequence_context_diagnostics.operational_context_status}</strong></span>
                </div>
              </div>
            </div>
          )}

          {/* Dynamic Confusion Matrix and Probability Discrimination */}
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            {/* Confusion Matrix */}
            <div className="bg-slate-900 border border-slate-800 rounded-3xl p-5 sm:p-6 shadow-xl space-y-4">
              <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2 border-b border-slate-800 pb-3">
                <div className="flex items-center space-x-2">
                  <Target className="w-4 h-4 text-emerald-400 shrink-0" />
                  <h3 className="text-sm font-bold text-white tracking-tight">
                    Confusion Matrix
                  </h3>
                  <span
                    className={`px-2 py-0.5 rounded-full text-[10px] font-mono font-bold border ${
                      !isManualSliderActive && selectedOperatingPoint === 'frozen'
                        ? 'bg-indigo-950 text-indigo-300 border-indigo-700/60'
                        : !isManualSliderActive && selectedOperatingPoint === 'budget_500'
                        ? 'bg-emerald-950 text-emerald-300 border-emerald-700/60'
                        : 'bg-amber-950 text-amber-300 border-amber-700/60'
                    }`}
                  >
                    {!isManualSliderActive && selectedOperatingPoint === 'frozen'
                      ? 'FIXED THRESHOLD (θ* = 0.3378)'
                      : !isManualSliderActive && selectedOperatingPoint === 'budget_500'
                      ? 'FIXED CAPACITY: 50 / 1k (TOP-1,000)'
                      : `CUSTOM CUTOFF (θ = ${threshold.toFixed(4)})`}
                  </span>
                </div>
                <span className="text-xs font-mono text-slate-400">
                  Total: <strong className="text-white">{computed.total.toLocaleString()}</strong> rows
                </span>
              </div>

              {/* 2x2 Grid with high contrast & explicit subtitles */}
              <div className="grid grid-cols-2 gap-3 text-center font-mono">
                <div className="bg-emerald-950/40 border border-emerald-800/60 p-3.5 sm:p-4 rounded-2xl">
                  <span className="text-[10px] text-emerald-400 font-bold block uppercase tracking-wider">
                    TRUE POSITIVES (TP)
                  </span>
                  <span className="text-2xl sm:text-3xl font-black text-white mt-1 block">
                    {computed.tp.toLocaleString()}
                  </span>
                  <span className="text-[10px] text-slate-400 mt-1 block">
                    Laundering Correctly Flagged
                  </span>
                </div>

                <div className="bg-amber-950/40 border border-amber-800/60 p-3.5 sm:p-4 rounded-2xl">
                  <span className="text-[10px] text-amber-400 font-bold block uppercase tracking-wider">
                    FALSE POSITIVES (FP)
                  </span>
                  <span className="text-2xl sm:text-3xl font-black text-white mt-1 block">
                    {computed.fp.toLocaleString()}
                  </span>
                  <span className="text-[10px] text-slate-400 mt-1 block">
                    False Alarms (Legitimate Flagged)
                  </span>
                </div>

                <div className="bg-rose-950/40 border border-rose-800/60 p-3.5 sm:p-4 rounded-2xl">
                  <span className="text-[10px] text-rose-400 font-bold block uppercase tracking-wider">
                    FALSE NEGATIVES (FN)
                  </span>
                  <span className="text-2xl sm:text-3xl font-black text-white mt-1 block">
                    {computed.fn.toLocaleString()}
                  </span>
                  <span className="text-[10px] text-slate-400 mt-1 block">
                    Missed Laundering Transactions
                  </span>
                </div>

                <div className="bg-slate-950 border border-slate-800 p-3.5 sm:p-4 rounded-2xl">
                  <span className="text-[10px] text-slate-400 font-bold block uppercase tracking-wider">
                    TRUE NEGATIVES (TN)
                  </span>
                  <span className="text-2xl sm:text-3xl font-black text-white mt-1 block">
                    {computed.tn.toLocaleString()}
                  </span>
                  <span className="text-[10px] text-slate-400 mt-1 block">
                    Legitimate Correctly Cleared
                  </span>
                </div>
              </div>

              {/* Mathematical Arithmetic Reconciliation Bar */}
              <div className="p-2.5 rounded-xl bg-slate-950 border border-slate-800/80 text-[10px] sm:text-[11px] font-mono text-slate-300 space-y-1">
                <div className="flex flex-wrap items-center justify-between gap-2 border-b border-slate-900 pb-1">
                  <span>Actual Positives (TP+FN): <strong className="text-white">{(computed.tp + computed.fn).toLocaleString()}</strong></span>
                  <span>Actual Negatives (FP+TN): <strong className="text-white">{(computed.fp + computed.tn).toLocaleString()}</strong></span>
                  <span>Alerts (TP+FP): <strong className="text-amber-400">{computed.alertCount.toLocaleString()}</strong></span>
                  <span>Sum: <strong className="text-emerald-400">{(computed.tp + computed.fp + computed.fn + computed.tn).toLocaleString()}</strong></span>
                </div>
                <p className="text-[10px] text-slate-400 leading-normal pt-0.5">
                  {Math.abs(threshold - 0.0) < 1e-4
                    ? 'Diagnostic threshold 0.0000: flags 100% of non-negative transactions. Workload diagnostic only; not an operational recommendation.'
                    : `Active decision policy: cutoff threshold θ* = ${threshold.toFixed(4)}. Alerts: ${computed.alertCount.toLocaleString()} (${((computed.alertCount / Math.max(1, computed.total)) * 100).toFixed(2)}% alert rate). Cleared: ${(computed.tn + computed.fn).toLocaleString()}.`}
                </p>
              </div>
            </div>

            {/* Probability Discrimination Breakdown */}
            <div className="bg-slate-900 border border-slate-800 rounded-3xl p-6 shadow-xl space-y-4">
              <div className="flex items-center justify-between border-b border-slate-800 pb-3">
                <h3 className="text-sm font-bold text-white flex items-center space-x-2">
                  <TrendingUp className="w-4 h-4 text-indigo-400" />
                  <span>Detection Discrimination Breakdown</span>
                </h3>
              </div>

              <div className="space-y-3 text-xs font-mono">
                <div className="flex items-center justify-between p-2.5 rounded-xl bg-slate-950 border border-slate-800">
                  <span className="text-slate-400">Positive Class (Actual Laundering):</span>
                  <span className="font-bold text-white">
                    {staticMetrics.positive_samples ?? computed.tp + computed.fn} / {computed.total} ({formatRatioPercent(staticMetrics.positive_samples ?? computed.tp + computed.fn, computed.total, 1)})
                  </span>
                </div>

                <div className="flex items-center justify-between p-2.5 rounded-xl bg-slate-950 border border-slate-800">
                  <span className="text-slate-400">Negative Class (Actual Legitimate):</span>
                  <span className="font-bold text-white">
                    {staticMetrics.negative_samples ?? computed.tn + computed.fp} / {computed.total} ({formatRatioPercent(staticMetrics.negative_samples ?? computed.tn + computed.fp, computed.total, 1)})
                  </span>
                </div>

                <div className="flex items-center justify-between p-2.5 rounded-xl bg-slate-950 border border-slate-800">
                  <span className="text-slate-400">Error Rates (FPR / FNR):</span>
                  <span className="font-bold text-slate-300">
                    FPR: <strong className="text-amber-400">{formatPercent(computed.fpr, 2)}</strong> • FNR:{' '}
                    <strong className="text-rose-400">{formatPercent(computed.fnr, 2)}</strong>
                  </span>
                </div>

                <div className="flex items-center justify-between p-2.5 rounded-xl bg-slate-950 border border-slate-800">
                  <span className="text-slate-400">Discovery Rates (FDR / NPV):</span>
                  <span className="font-bold text-slate-300">
                    FDR: <strong className="text-amber-300">{formatPercent(computed.fdr, 2)}</strong> • NPV:{' '}
                    <strong className="text-emerald-300">{formatPercent(computed.npv, 2)}</strong>
                  </span>
                </div>

                <div className="flex items-center justify-between p-2.5 rounded-xl bg-slate-950 border border-slate-800">
                  <span className="text-slate-400">Probability Range (Min / Mean / Max):</span>
                  <span className="font-bold text-teal-300">
                    {formatMetric(staticMetrics.min_predicted_probability, 4)} /{' '}
                    {formatMetric(staticMetrics.mean_predicted_probability, 4)} /{' '}
                    {formatMetric(staticMetrics.max_predicted_probability, 4)}
                  </span>
                </div>

                <div className="flex items-center justify-between p-2.5 rounded-xl bg-slate-950 border border-slate-800">
                  <span className="text-slate-400">Alert Generation Rate:</span>
                  <span className="font-bold text-amber-400">
                    {computed.alertCount} alerts flagged (
                    {computed.total > 0
                      ? ((computed.alertCount / computed.total) * 10000).toFixed(1)
                      : '0.0'}{' '}
                    per 10k transactions)
                  </span>
                </div>
              </div>
            </div>
          </div>

          {/* Visual Curves: Precision-Recall & ROC Curves */}
          {staticMetrics.pr_curve && staticMetrics.pr_curve.length > 0 && (
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              {/* Precision-Recall Curve */}
              <div className="bg-slate-900 border border-slate-800 rounded-3xl p-6 shadow-xl space-y-3">
                <div className="flex items-center justify-between border-b border-slate-800 pb-2 text-xs font-mono">
                  <span className="font-bold text-white flex items-center space-x-1.5">
                    <TrendingUp className="w-4 h-4 text-emerald-400" />
                    <span>Precision-Recall Curve</span>
                  </span>
                  <span className="text-emerald-400 font-bold">
                    PR-AUC: {formatMetric(staticMetrics.pr_auc, 4)}
                  </span>
                </div>

                <div className="h-56 w-full flex items-center justify-center pt-2">
                  <svg className="w-full h-full" viewBox="0 0 360 200">
                    <line x1="45" y1="20" x2="45" y2="165" stroke="#334155" strokeWidth="1" />
                    <line x1="45" y1="165" x2="340" y2="165" stroke="#334155" strokeWidth="1" />

                    <line x1="45" y1="92" x2="340" y2="92" stroke="#1e293b" strokeDasharray="3 3" />
                    <line x1="192" y1="20" x2="192" y2="165" stroke="#1e293b" strokeDasharray="3 3" />

                    <text x="38" y="24" fill="#64748b" fontSize="9" textAnchor="end">1.0</text>
                    <text x="38" y="95" fill="#64748b" fontSize="9" textAnchor="end">0.5</text>
                    <text x="38" y="168" fill="#64748b" fontSize="9" textAnchor="end">0.0</text>
                    <text x="45" y="182" fill="#64748b" fontSize="9" textAnchor="middle">0.0</text>
                    <text x="192" y="182" fill="#64748b" fontSize="9" textAnchor="middle">0.5 Recall</text>
                    <text x="340" y="182" fill="#64748b" fontSize="9" textAnchor="middle">1.0</text>

                    {(() => {
                      const pts = staticMetrics.pr_curve || [];
                      if (pts.length < 2) return null;
                      const pathStr = pts
                        .map((p, idx) => {
                          const rec = Number.isFinite(p.recall) ? (p.recall as number) : 0;
                          const prec = Number.isFinite(p.precision) ? (p.precision as number) : 0;
                          const x = 45 + rec * 295;
                          const y = 165 - prec * 145;
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

                    {/* Dynamic Operating Point */}
                    {(() => {
                      const recVal = Number.isFinite(computed.recall) ? computed.recall : 0;
                      const precVal = computed.precision !== null && Number.isFinite(computed.precision) ? computed.precision : 0;
                      const x = 45 + recVal * 295;
                      const y = 165 - precVal * 145;
                      return (
                        <g>
                          <circle cx={x} cy={y} r="5" fill="#f59e0b" />
                          <circle cx={x} cy={y} r="8" fill="none" stroke="#f59e0b" strokeWidth="1" strokeDasharray="2 2" />
                        </g>
                      );
                    })()}
                  </svg>
                </div>
                <p className="text-[11px] text-slate-500 font-mono text-center">
                  Amber marker denotes operating trade-off at θ = {threshold.toFixed(4)}.
                </p>
              </div>

              {/* ROC Curve */}
              <div className="bg-slate-900 border border-slate-800 rounded-3xl p-6 shadow-xl space-y-3">
                <div className="flex items-center justify-between border-b border-slate-800 pb-2 text-xs font-mono">
                  <span className="font-bold text-white flex items-center space-x-1.5">
                    <TrendingUp className="w-4 h-4 text-indigo-400" />
                    <span>ROC Curve</span>
                  </span>
                  <span className="text-indigo-400 font-bold">
                    ROC-AUC: {formatMetric(staticMetrics.roc_auc, 4)}
                  </span>
                </div>

                <div className="h-56 w-full flex items-center justify-center pt-2">
                  <svg className="w-full h-full" viewBox="0 0 360 200">
                    <line x1="45" y1="20" x2="45" y2="165" stroke="#334155" strokeWidth="1" />
                    <line x1="45" y1="165" x2="340" y2="165" stroke="#334155" strokeWidth="1" />

                    <line x1="45" y1="165" x2="340" y2="20" stroke="#475569" strokeDasharray="4 4" strokeWidth="1" />

                    <text x="38" y="24" fill="#64748b" fontSize="9" textAnchor="end">1.0</text>
                    <text x="38" y="95" fill="#64748b" fontSize="9" textAnchor="end">0.5</text>
                    <text x="38" y="168" fill="#64748b" fontSize="9" textAnchor="end">0.0</text>
                    <text x="45" y="182" fill="#64748b" fontSize="9" textAnchor="middle">0.0</text>
                    <text x="192" y="182" fill="#64748b" fontSize="9" textAnchor="middle">0.5 FPR</text>
                    <text x="340" y="182" fill="#64748b" fontSize="9" textAnchor="middle">1.0</text>

                    {(() => {
                      const pts = staticMetrics.roc_curve || [];
                      if (pts.length < 2) return null;
                      const pathStr = pts
                        .map((p, idx) => {
                          const fpr = Number.isFinite(p.fpr) ? (p.fpr as number) : 0;
                          const tpr = Number.isFinite(p.tpr) ? (p.tpr as number) : 0;
                          const x = 45 + fpr * 295;
                          const y = 165 - tpr * 145;
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

                    {/* Dynamic Operating Point */}
                    {(() => {
                      const fprVal = Number.isFinite(computed.fpr) ? computed.fpr : 0;
                      const tprVal = Number.isFinite(computed.recall) ? computed.recall : 0;
                      const x = 45 + fprVal * 295;
                      const y = 165 - tprVal * 145;
                      return (
                        <g>
                          <circle cx={x} cy={y} r="5" fill="#f59e0b" />
                          <circle cx={x} cy={y} r="8" fill="none" stroke="#f59e0b" strokeWidth="1" strokeDasharray="2 2" />
                        </g>
                      );
                    })()}
                  </svg>
                </div>
                <p className="text-[11px] text-slate-500 font-mono text-center">
                  Dashed diagonal represents random baseline (AUC = 0.50).
                </p>
              </div>
            </div>
          )}

          {/* Empirical Distribution Pie Charts (Confusion, Actual, Predicted) */}
          <EvaluationPieChart
            tp={computed.tp}
            fp={computed.fp}
            fn={computed.fn}
            tn={computed.tn}
            totalRows={computed.total}
            hasLabels={hasMetrics}
            threshold={threshold}
            activeModeName={
              selectedOperatingPoint === 'budget_500'
                ? 'Fixed Capacity: 50 alerts / 1,000 tx (1,000 Alerts Capped)'
                : selectedOperatingPoint === 'budget_10'
                ? 'Fixed Capacity: 10 alerts / 10k tx'
                : selectedOperatingPoint === 'budget_50'
                ? 'Fixed Capacity: 50 alerts / 10k tx'
                : selectedOperatingPoint === 'budget_100'
                ? 'Fixed Capacity: 100 alerts / 10k tx'
                : selectedOperatingPoint === 'best_f1'
                ? 'Best F1 On Batch (What-If)'
                : `Fixed Canonical Threshold (θ = ${threshold.toFixed(4)})`
            }
          />
        </div>
      ) : (
        /* Scenario B: Unlabeled CSV Notice */
        <div className="p-6 bg-slate-900 border border-amber-600/60 rounded-3xl space-y-4 shadow-xl">
          <div className="flex items-start space-x-3 text-amber-200">
            <AlertTriangle className="w-6 h-6 text-amber-400 shrink-0 mt-0.5" />
            <div className="space-y-1">
              <h4 className="text-base font-bold text-amber-300">
                Ground-truth labels are unavailable. Evaluation metrics cannot be calculated for this uploaded dataset.
              </h4>
              <p className="text-xs text-slate-300 leading-relaxed">
                Supervised classification metrics (Accuracy, Precision, Recall, F1, PR-AUC, ROC-AUC, and Confusion Matrix) require verified ground-truth labels (<code className="text-amber-300 font-mono">is_laundering</code>).
                All transactional laundering probabilities, decile distributions, and alert flags have been computed dynamically using the selected threshold (<code className="text-amber-300 font-mono">θ = {threshold.toFixed(4)}</code>).
              </p>
            </div>
          </div>

          <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 pt-2">
            <div className="bg-slate-950 p-4 rounded-2xl border border-slate-800">
              <span className="text-[10px] text-slate-400 font-mono uppercase block">Total Transactions</span>
              <span className="text-2xl font-black font-mono text-white mt-1 block">{computed.total}</span>
            </div>
            <div className="bg-slate-950 p-4 rounded-2xl border border-slate-800">
              <span className="text-[10px] text-slate-400 font-mono uppercase block">Predicted Suspicious</span>
              <span className="text-2xl font-black font-mono text-rose-400 mt-1 block">
                {computed.flagged} ({formatRatioPercent(computed.flagged, computed.total, 1)})
              </span>
            </div>
            <div className="bg-slate-950 p-4 rounded-2xl border border-slate-800">
              <span className="text-[10px] text-slate-400 font-mono uppercase block">Legitimate Cleared</span>
              <span className="text-2xl font-black font-mono text-emerald-400 mt-1 block">
                {computed.cleared} ({formatRatioPercent(computed.cleared, computed.total, 1)})
              </span>
            </div>
            <div className="bg-slate-950 p-4 rounded-2xl border border-slate-800">
              <span className="text-[10px] text-slate-400 font-mono uppercase block">Operating Threshold</span>
              <span className="text-2xl font-black font-mono text-indigo-300 mt-1 block">
                θ = {threshold.toFixed(4)}
              </span>
            </div>
          </div>
        </div>
      )}

      {/* Multi-Model Comparison (When Multi-Model Evaluation is run) */}
      {multiModelResult && multiModelResult.models && multiModelResult.models.length > 0 && (
        <PerformanceHeatmap
          models={multiModelResult.models.map((m) => ({
            ...m,
            config_id: m.model_name,
            dataset: 'HI-Small' as const,
            architecture: (m.model_name.includes('bilstm') ? 'BiLSTM' : m.model_name.includes('lstm') ? 'LSTM' : 'GRU') as any,
            distribution: (m.model_name.includes('non_iid') ? 'Non-IID' : m.model_name.includes('iid') ? 'IID' : 'Centralized') as any,
            algorithm: (m.model_name.includes('fedprox') ? 'FedProx' : m.model_name.includes('fedavg') ? 'FedAvg' : 'Centralized') as any,
            test_pr_auc: m.pr_auc,
            test_roc_auc: m.roc_auc,
            test_f1: m.f1_score,
            test_precision: m.precision,
            test_recall: m.recall,
            test_specificity: m.specificity,
            test_accuracy: m.accuracy,
            test_balanced_accuracy: m.balanced_accuracy,
            test_mcc: m.mcc,
            test_cohen_kappa: m.cohen_kappa,
            brier_score: m.brier_score,
            alerts_per_10k: m.laundering_percentage != null ? m.laundering_percentage * 100 : undefined,
          })) as unknown as BenchmarkModelRecord[]}
          title={`Multi-Model Evaluation Matrix (${multiModelResult.models_evaluated_count} Models)`}
          subtitle={`Evaluated on ${multiModelResult.total_transactions.toLocaleString()} preprocessed sequences across centralized and federated architectures.`}
        />
      )}

      {/* Transaction Risk Score Distribution (Always available) */}
      <div className="bg-slate-900 border border-slate-800 rounded-3xl p-6 shadow-xl space-y-4">
        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 border-b border-slate-800 pb-3">
          <h3 className="text-sm font-bold text-white flex items-center space-x-2">
            <BarChart2 className="w-4 h-4 text-indigo-400" />
            <span>Transaction Risk Score Distribution (Probability Deciles)</span>
          </h3>
          <span className="text-xs font-mono text-slate-400">
            {computed.total} transactions analyzed
          </span>
        </div>

        {result.risk_distribution && result.risk_distribution.length > 0 && (
          <div className="space-y-3">
            <div className="grid grid-cols-10 gap-2 h-36 items-end pt-4 pb-1 bg-slate-950 p-4 rounded-2xl border border-slate-800">
              {result.risk_distribution.map((bin, idx) => {
                const maxPct = Math.max(...(result.risk_distribution?.map((b) => b.percentage || 0) || [1]));
                const pct = Number.isFinite(bin.percentage) ? bin.percentage : 0;
                const barHeightPct = Math.max(4, (pct / (maxPct || 1)) * 100);
                const isHighRisk = (bin.lower ?? 0) >= threshold;

                return (
                  <div key={idx} className="flex flex-col items-center h-full justify-end group relative">
                    <div className="opacity-0 group-hover:opacity-100 transition absolute -top-8 bg-slate-800 text-white text-[10px] px-2 py-0.5 rounded shadow pointer-events-none whitespace-nowrap z-10 font-mono">
                      {bin.bin_range}: {bin.count ?? 0} ({formatMetric(bin.percentage, 1, '0.0')}%)
                    </div>

                    <div className="text-[10px] font-mono text-slate-400 mb-1">{bin.count ?? 0}</div>
                    <div
                      className={`w-full rounded-t transition ${
                        isHighRisk
                          ? 'bg-gradient-to-t from-rose-700 to-rose-500'
                          : 'bg-gradient-to-t from-indigo-700 to-indigo-500'
                      }`}
                      style={{ height: `${barHeightPct}%` }}
                    />
                    <div className="text-[9px] font-mono text-slate-500 mt-1 truncate max-w-full">
                      {formatMetric(bin.lower, 1, '0.0')}
                    </div>
                  </div>
                );
              })}
            </div>
            <div className="flex items-center justify-between text-[11px] text-slate-400 font-mono px-1">
              <span>0.0 (Legitimate)</span>
              <span>Laundering Probability Deciles (Threshold θ = {threshold.toFixed(2)})</span>
              <span>1.0 (Suspicious)</span>
            </div>
          </div>
        )}

        {/* Flagged vs Cleared Ratio */}
        <div className="pt-2">
          <div className="flex items-center justify-between text-xs font-mono text-slate-300 pb-1.5">
            <span className="flex items-center space-x-1.5">
              <ShieldCheck className="w-3.5 h-3.5 text-emerald-400" />
              <span>
                Legitimate Cleared: <strong className="text-emerald-400">{computed.cleared}</strong> (
                {formatRatioPercent(computed.cleared, computed.total, 1)})
              </span>
            </span>
            <span className="flex items-center space-x-1.5">
              <ShieldAlert className="w-3.5 h-3.5 text-rose-400" />
              <span>
                Suspicious Flagged: <strong className="text-rose-400">{computed.flagged}</strong> (
                {formatRatioPercent(computed.flagged, computed.total, 1)})
              </span>
            </span>
          </div>
          <div className="w-full bg-slate-950 h-3 rounded-full overflow-hidden flex border border-slate-800">
            <div
              className="bg-emerald-500 h-full transition-all duration-200"
              style={{
                width: `${Math.min(
                  100,
                  Math.max(0, computed.total > 0 ? (computed.cleared / computed.total) * 100 : 0)
                )}%`,
              }}
            />
            <div
              className="bg-rose-500 h-full transition-all duration-200"
              style={{
                width: `${Math.min(
                  100,
                  Math.max(0, computed.total > 0 ? (computed.flagged / computed.total) * 100 : 0)
                )}%`,
              }}
            />
          </div>
        </div>
      </div>

      {/* Real Uploaded-CSV Temporal Risk Heatmap */}
      <TransactionRiskHeatmap
        predictions={result.predictions || []}
        activeThreshold={threshold}
        hasLabels={hasMetrics}
        selectedCell={
          heatmapCellFilter
            ? { timeBucket: heatmapCellFilter.timeBucket, riskBand: heatmapCellFilter.riskBand }
            : null
        }
        onSelectCell={setHeatmapCellFilter}
      />

      {/* Itemized Transaction Evaluation Records & Triage Table */}
      <FlaggedTransactionsTable
        predictions={result.predictions || []}
        activeThreshold={threshold}
        activeModeName={
          selectedOperatingPoint === 'zero_diagnostic'
            ? 'Diagnostic Only — Not Serving Threshold (θ = 0.0000)'
            : selectedOperatingPoint === 'budget_500'
            ? 'Fixed Capacity: 50 alerts / 1,000 tx (1,000 Alerts Capped)'
            : selectedOperatingPoint === 'budget_10'
            ? 'Fixed Capacity: 10 alerts / 10k tx'
            : selectedOperatingPoint === 'budget_50'
            ? 'Fixed Capacity: 50 alerts / 10k tx'
            : selectedOperatingPoint === 'budget_100'
            ? 'Fixed Capacity: 100 alerts / 10k tx'
            : selectedOperatingPoint === 'best_f1'
            ? 'Best F1 On Batch (What-If)'
            : `Fixed Canonical Threshold (θ = ${threshold.toFixed(4)})`
        }
        hasLabels={hasMetrics}
        dataset={result.dataset}
        modelName={result.model_name || result.model_id || 'Federated Neural Classifier'}
        heatmapFilter={heatmapCellFilter}
        onClearHeatmapFilter={() => setHeatmapCellFilter(null)}
        onFilterChange={setActiveTableFilter}
      />
    </div>
  );
};
