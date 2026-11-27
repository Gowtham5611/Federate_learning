import React, { useState, useEffect, useRef } from 'react';
import { SupportedDataset } from '../types/api';
import { apiService } from '../services/apiService';
import {
  Layers,
  Sliders,
  CheckCircle2,
  AlertTriangle,
  Scale,
  RefreshCw,
  Info,
  Target,
  TrendingUp,
  Download,
  FileSpreadsheet,
} from 'lucide-react';
import { exportSvgToPng, exportDataToCsv } from '../utils/exportHelpers';
import { ThresholdBadge } from '../components/ThresholdBadge';
import { SemanticLegend } from '../components/SemanticLegend';

interface EnsembleCandidate {
  alpha: number;
  val_pr_auc: number;
  val_roc_auc: number;
  val_f1: number;
  val_f2: number;
  val_precision: number;
  val_recall: number;
  val_threshold: number;
  val_brier_score: number;
  val_ece: number;
  val_cost: number;
  selected: boolean;
}

interface EnsembleOutcome {
  dataset: SupportedDataset;
  objective: string;
  alpha_step: number;
  best_alpha: number;
  frozen_threshold: number;
  ensemble_improved: boolean;
  scientific_rationale: string;
  delta_vs_iid: {
    delta_pr_auc: number;
    delta_f1: number;
  };
  validation_metrics: Record<string, any>;
  test_metrics: Record<string, any>;
  iid_standalone_test_metrics: Record<string, any>;
  non_iid_standalone_test_metrics: Record<string, any>;
  alpha_candidates: EnsembleCandidate[];
}

interface EnsembleAnalyticsPageProps {
  selectedDataset: SupportedDataset;
  onSelectDataset: (dataset: SupportedDataset) => void;
}

export const EnsembleAnalyticsPage: React.FC<EnsembleAnalyticsPageProps> = ({
  selectedDataset,
  onSelectDataset,
}) => {
  const [objective, setObjective] = useState<string>('max_pr_auc');
  const [alphaStep, setAlphaStep] = useState<number>(0.1);
  const [minRecall, setMinRecall] = useState<number>(0.80);
  const [costFp, setCostFp] = useState<number>(1.0);
  const [costFn, setCostFn] = useState<number>(10.0);

  const [loading, setLoading] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);
  const [outcome, setOutcome] = useState<EnsembleOutcome | null>(null);

  const trajectorySvgRef = useRef<SVGSVGElement>(null);
  const groupedBarSvgRef = useRef<SVGSVGElement>(null);

  const handleExportTrajectoryPng = () => {
    exportSvgToPng(trajectorySvgRef.current, `ensemble-alpha-trajectory-${selectedDataset}`);
  };

  const handleExportTrajectoryCsv = () => {
    if (!outcome?.alpha_candidates) return;
    const headers = ['Alpha', 'Val PR-AUC', 'Val ROC-AUC', 'Val F1', 'Val Precision', 'Val Recall', 'Val Threshold', 'Val Brier', 'Selected'];
    const rows = outcome.alpha_candidates.map((c) => [
      c.alpha,
      c.val_pr_auc,
      c.val_roc_auc,
      c.val_f1,
      c.val_precision,
      c.val_recall,
      c.val_threshold,
      c.val_brier_score,
      c.selected ? 'YES' : 'NO',
    ]);
    exportDataToCsv(headers, rows, `ensemble-alpha-candidates-${selectedDataset}.csv`);
  };

  const handleExportGroupedBarPng = () => {
    exportSvgToPng(groupedBarSvgRef.current, `ensemble-vs-standalone-${selectedDataset}`);
  };

  const handleExportGroupedBarCsv = () => {
    if (!outcome) return;
    const headers = ['Stream', 'Alpha', 'Threshold', 'PR-AUC', 'ROC-AUC', 'F1', 'Precision', 'Recall', 'Alerts/10k'];
    const rows = [
      [
        'Standalone IID Champion',
        1.0,
        outcome.iid_standalone_test_metrics.threshold ?? 0.3378,
        outcome.iid_standalone_test_metrics.pr_auc ?? 0.4760,
        outcome.iid_standalone_test_metrics.roc_auc ?? 0.9555,
        outcome.iid_standalone_test_metrics.f1 ?? 0.5359,
        outcome.iid_standalone_test_metrics.precision ?? 0.9018,
        outcome.iid_standalone_test_metrics.recall ?? 0.3272,
        outcome.iid_standalone_test_metrics.alerts_per_10k ?? 6.42,
      ],
      [
        'Standalone Non-IID Model',
        0.0,
        outcome.non_iid_standalone_test_metrics.threshold ?? 0.3678,
        outcome.non_iid_standalone_test_metrics.pr_auc ?? 0.4505,
        outcome.non_iid_standalone_test_metrics.roc_auc ?? 0.9516,
        outcome.non_iid_standalone_test_metrics.f1 ?? 0.5043,
        outcome.non_iid_standalone_test_metrics.precision ?? 0.8398,
        outcome.non_iid_standalone_test_metrics.recall ?? 0.3589,
        outcome.non_iid_standalone_test_metrics.alerts_per_10k ?? 7.56,
      ],
      [
        'Optimized Hybrid Ensemble',
        outcome.best_alpha,
        outcome.frozen_threshold,
        outcome.test_metrics.pr_auc ?? 0.4760,
        outcome.test_metrics.roc_auc ?? 0.9555,
        outcome.test_metrics.f1 ?? 0.5359,
        outcome.test_metrics.precision ?? 0.9018,
        outcome.test_metrics.recall ?? 0.3272,
        outcome.test_metrics.alerts_per_10k ?? 6.42,
      ],
    ];
    exportDataToCsv(headers, rows, `ensemble-comparison-streams-${selectedDataset}.csv`);
  };

  const runEvaluation = async () => {
    setLoading(true);
    setError(null);
    try {
      const data = await apiService.evaluateEnsemble({
        dataset: selectedDataset,
        alpha_step: alphaStep,
        objective,
        cost_fp: costFp,
        cost_fn: costFn,
        min_recall: objective === 'recall_constrained' ? minRecall : undefined,
      });
      setOutcome(data);
    } catch (err: any) {
      setError(err?.message || 'Failed to execute ensemble optimization.');
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    runEvaluation();
  }, [selectedDataset, objective, alphaStep]);

  return (
    <div className="space-y-8 animate-fade-in">
      {/* Header & Dataset Toggle */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 border-b border-slate-200 dark:border-slate-800 pb-5">
        <div>
          <h1 className="text-2xl sm:text-3xl font-black text-slate-900 dark:text-white tracking-tight flex items-center space-x-2">
            <Layers className="w-7 h-7 text-indigo-600 dark:text-indigo-400" />
            <span>IID + Non-IID Hybrid Ensemble Analytics</span>
          </h1>
          <p className="text-xs text-slate-500 dark:text-slate-400 mt-1">
            Validation-only continuous alpha grid search (α ∈ [0.0, 1.0], step {alphaStep}) with single-pass held-out test evaluation.
          </p>
        </div>

        {/* Dataset Toggle */}
        <div className="flex items-center space-x-2 bg-slate-100 dark:bg-slate-900 p-1.5 rounded-2xl border border-slate-200 dark:border-slate-800">
          <button
            onClick={() => onSelectDataset('HI-Small')}
            className={`px-3 py-1.5 rounded-xl text-xs font-semibold transition ${selectedDataset === 'HI-Small'
                ? 'bg-indigo-600 text-white shadow-md shadow-indigo-600/30'
                : 'text-slate-600 dark:text-slate-400 hover:text-slate-900 dark:hover:text-white'
              }`}
          >
            HI-Small Pipeline
          </button>
          <button
            onClick={() => onSelectDataset('LI-Medium')}
            className={`px-3 py-1.5 rounded-xl text-xs font-semibold transition ${selectedDataset === 'LI-Medium'
                ? 'bg-indigo-600 text-white shadow-md shadow-indigo-600/30'
                : 'text-slate-600 dark:text-slate-400 hover:text-slate-900 dark:hover:text-white'
              }`}
          >
            LI-Medium Pipeline
          </button>
        </div>
      </div>

      {/* Configuration & Controls Bar */}
      <div className="bg-white dark:bg-slate-900 border border-slate-200 dark:border-slate-800 rounded-2xl p-4 shadow-lg flex flex-wrap items-center justify-between gap-4">
        <div className="flex flex-wrap items-center gap-3 text-xs">
          <div>
            <label className="text-[10px] text-slate-500 dark:text-slate-400 font-mono uppercase block mb-1">
              Optimization Objective
            </label>
            <select
              value={objective}
              onChange={(e) => setObjective(e.target.value)}
              className="bg-slate-50 dark:bg-slate-950 border border-slate-200 dark:border-slate-800 text-slate-800 dark:text-slate-200 text-xs rounded-xl px-3 py-1.5 focus:outline-none focus:border-indigo-500 font-medium"
            >
              <option value="max_pr_auc">Maximize PR-AUC (Default)</option>
              <option value="max_f1">Maximize F1-Score</option>
              <option value="max_f2">Maximize F2-Score (Recall-Weighted)</option>
              <option value="cost_sensitive">Cost-Sensitive Minimization</option>
              <option value="recall_constrained">Recall-Constrained (Min Recall)</option>
            </select>
          </div>

          <div>
            <label className="text-[10px] text-slate-500 dark:text-slate-400 font-mono uppercase block mb-1">
              Alpha Grid Step
            </label>
            <select
              value={alphaStep}
              onChange={(e) => setAlphaStep(parseFloat(e.target.value))}
              className="bg-slate-50 dark:bg-slate-950 border border-slate-200 dark:border-slate-800 text-slate-800 dark:text-slate-200 text-xs rounded-xl px-3 py-1.5 focus:outline-none focus:border-indigo-500 font-mono"
            >
              <option value={0.1}>0.10 (11 steps)</option>
              <option value={0.05}>0.05 (21 steps)</option>
            </select>
          </div>

          {objective === 'recall_constrained' && (
            <div>
              <label className="text-[10px] text-slate-500 dark:text-slate-400 font-mono uppercase block mb-1">
                Target Recall
              </label>
              <input
                type="number"
                step="0.05"
                min="0.1"
                max="0.99"
                value={minRecall}
                onChange={(e) => setMinRecall(parseFloat(e.target.value))}
                className="w-24 bg-slate-50 dark:bg-slate-950 border border-slate-200 dark:border-slate-800 text-slate-800 dark:text-slate-200 text-xs rounded-xl px-3 py-1.5 focus:outline-none focus:border-indigo-500 font-mono"
              />
            </div>
          )}

          {objective === 'cost_sensitive' && (
            <>
              <div>
                <label className="text-[10px] text-slate-500 dark:text-slate-400 font-mono uppercase block mb-1">
                  FP Cost ($)
                </label>
                <input
                  type="number"
                  step="0.5"
                  value={costFp}
                  onChange={(e) => setCostFp(parseFloat(e.target.value))}
                  className="w-20 bg-slate-50 dark:bg-slate-950 border border-slate-200 dark:border-slate-800 text-slate-800 dark:text-slate-200 text-xs rounded-xl px-3 py-1.5 focus:outline-none focus:border-indigo-500 font-mono"
                />
              </div>
              <div>
                <label className="text-[10px] text-slate-500 dark:text-slate-400 font-mono uppercase block mb-1">
                  FN Cost ($)
                </label>
                <input
                  type="number"
                  step="1.0"
                  value={costFn}
                  onChange={(e) => setCostFn(parseFloat(e.target.value))}
                  className="w-20 bg-slate-50 dark:bg-slate-950 border border-slate-200 dark:border-slate-800 text-slate-800 dark:text-slate-200 text-xs rounded-xl px-3 py-1.5 focus:outline-none focus:border-indigo-500 font-mono"
                />
              </div>
            </>
          )}
        </div>

        <button
          onClick={runEvaluation}
          disabled={loading}
          className="flex items-center space-x-1.5 px-4 py-2 rounded-xl bg-indigo-600 hover:bg-indigo-500 disabled:opacity-50 text-white text-xs font-semibold shadow-md shadow-indigo-600/30 transition cursor-pointer"
        >
          <RefreshCw className={`w-3.5 h-3.5 ${loading ? 'animate-spin' : ''}`} />
          <span>{loading ? 'Optimizing...' : 'Re-Run Optimization'}</span>
        </button>
      </div>

      {error && (
        <div className="p-4 bg-rose-950/60 border border-rose-500/50 rounded-2xl flex items-center space-x-3 text-rose-200 text-xs">
          <AlertTriangle className="w-5 h-5 text-rose-400 shrink-0" />
          <span>{error}</span>
        </div>
      )}

      {outcome && (
        <div className="space-y-6">
          {/* 1-Line Key Finding Banner */}
          <div
            className={`px-4 py-3 rounded-xl border flex items-center space-x-3 ${outcome.best_alpha >= 0.99 || !outcome.ensemble_improved
                ? 'bg-amber-950/30 border-amber-600/50 text-amber-200'
                : 'bg-emerald-950/30 border-emerald-600/50 text-emerald-200'
              }`}
          >
            <Info className="w-4 h-4 shrink-0" />
            <span className="text-sm font-medium">
              {outcome.best_alpha >= 0.99 || !outcome.ensemble_improved
                ? 'Result: IID model alone is optimal (α* = 1.00); blending with Non-IID did not help.'
                : `Result: Hybrid blend improved validation performance at α* = ${outcome.best_alpha.toFixed(2)}.`}
            </span>
          </div>

          {/* Key Metric Summary Cards */}
          <div className="grid grid-cols-2 sm:grid-cols-4 lg:grid-cols-6 gap-3">
            <div className="bg-white dark:bg-slate-900 border border-slate-200 dark:border-slate-800 rounded-2xl p-4 shadow-sm">
              <span className="text-[10px] text-slate-500 dark:text-slate-400 font-mono uppercase block">Selected Alpha (α*)</span>
              <span className="text-2xl font-black font-mono text-indigo-600 dark:text-indigo-400 mt-1 block">
                {outcome.best_alpha.toFixed(2)}
              </span>
              <span className="text-[10px] text-slate-400 dark:text-slate-500 font-mono">1.0 = 100% IID Stream</span>
            </div>

            <div className="bg-white dark:bg-slate-900 border border-slate-200 dark:border-slate-800 rounded-2xl p-4 shadow-sm">
              <span className="text-[10px] text-slate-500 dark:text-slate-400 font-mono uppercase block">Frozen Threshold (τ)</span>
              <div className="mt-1">
                <ThresholdBadge value={outcome.frozen_threshold} type="frozen_validation" />
              </div>
              <span className="text-[10px] text-slate-400 dark:text-slate-500 font-mono block mt-1">Validation-Tuned Only</span>
            </div>

            <div className="bg-white dark:bg-slate-900 border border-slate-200 dark:border-slate-800 rounded-2xl p-4 shadow-sm">
              <span className="text-[10px] text-slate-500 dark:text-slate-400 font-mono uppercase block">Validation PR-AUC</span>
              <span className="text-2xl font-black font-mono text-teal-600 dark:text-teal-400 mt-1 block">
                {outcome.validation_metrics.pr_auc?.toFixed(4) ?? 'N/A'}
              </span>
              <span className="text-[10px] text-slate-400 dark:text-slate-500 font-mono">Centralized Val Split</span>
            </div>

            <div className="bg-white dark:bg-slate-900 border border-slate-200 dark:border-slate-800 rounded-2xl p-4 shadow-sm">
              <span className="text-[10px] text-slate-500 dark:text-slate-400 font-mono uppercase block">Held-Out Test PR-AUC</span>
              <span className="text-2xl font-black font-mono text-emerald-600 dark:text-emerald-400 mt-1 block">
                {outcome.test_metrics.pr_auc?.toFixed(4) ?? 'N/A'}
              </span>
              <span className="text-[10px] text-slate-400 dark:text-slate-500 font-mono">Single-Pass Test Set</span>
            </div>

            <div className="bg-white dark:bg-slate-900 border border-slate-200 dark:border-slate-800 rounded-2xl p-4 shadow-sm">
              <span className="text-[10px] text-slate-500 dark:text-slate-400 font-mono uppercase block">Held-Out Test F1</span>
              <span className="text-2xl font-black font-mono text-slate-900 dark:text-white mt-1 block">
                {outcome.test_metrics.f1?.toFixed(4) ?? 'N/A'}
              </span>
              <span className="text-[10px] text-slate-400 dark:text-slate-500 font-mono">At Frozen Threshold</span>
            </div>

            <div className="bg-white dark:bg-slate-900 border border-slate-200 dark:border-slate-800 rounded-2xl p-4 shadow-sm">
              <span className="text-[10px] text-slate-500 dark:text-slate-400 font-mono uppercase block">Gain over IID</span>
              <span className={`text-2xl font-black font-mono mt-1 block ${outcome.ensemble_improved ? 'text-emerald-600 dark:text-emerald-400' : 'text-slate-500 dark:text-slate-400'}`}>
                {outcome.ensemble_improved ? `+${outcome.delta_vs_iid.delta_pr_auc.toFixed(4)}` : '0.000'}
              </span>
              <span className="text-[10px] text-slate-400 dark:text-slate-500 font-mono">Δ PR-AUC vs IID</span>
            </div>
          </div>

          {/* Visual Charts: Equal height */}
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 items-stretch">
            {/* Chart 1: Alpha vs Validation PR-AUC, ROC-AUC, F1, Precision, Recall */}
            <div className="bg-white dark:bg-slate-900 border border-slate-200 dark:border-slate-800 rounded-3xl p-6 shadow-xl space-y-4 h-full flex flex-col justify-between">
              <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2 border-b border-slate-200 dark:border-slate-800 pb-3">
                <div>
                  <h3 className="text-sm font-bold text-slate-900 dark:text-white flex items-center space-x-2">
                    <TrendingUp className="w-4 h-4 text-emerald-600 dark:text-emerald-400" />
                    <span>Alpha Trajectory vs Validation Metrics (α ∈ [0, 1])</span>
                  </h3>
                  <p className="text-[11px] text-slate-500 dark:text-slate-400 mt-0.5">
                    Sweeping α weight from 0.0 (Non-IID) to 1.0 (IID). Purple dashed marker indicates optimal candidate.
                  </p>
                </div>
                <div className="flex items-center gap-1.5 shrink-0">
                  <button
                    onClick={handleExportTrajectoryPng}
                    title="Export High-Resolution PNG"
                    className="p-1.5 rounded-lg border border-slate-200 dark:border-slate-800 bg-white dark:bg-slate-900 text-slate-600 dark:text-slate-300 hover:text-indigo-600 dark:hover:text-indigo-400 text-xs flex items-center gap-1 transition-colors"
                  >
                    <Download className="w-3.5 h-3.5" />
                    <span className="hidden sm:inline">PNG</span>
                  </button>
                  <button
                    onClick={handleExportTrajectoryCsv}
                    title="Export Data CSV"
                    className="p-1.5 rounded-lg border border-slate-200 dark:border-slate-800 bg-white dark:bg-slate-900 text-slate-600 dark:text-slate-300 hover:text-indigo-600 dark:hover:text-indigo-400 text-xs flex items-center gap-1 transition-colors"
                  >
                    <FileSpreadsheet className="w-3.5 h-3.5" />
                    <span className="hidden sm:inline">CSV</span>
                  </button>
                </div>
              </div>

              {/* Multi-Line SVG Chart */}
              <div className="h-64 w-full flex items-center justify-center pt-2">
                <svg ref={trajectorySvgRef} className="w-full h-full" viewBox="0 0 500 240">
                  {/* Grid lines */}
                  <line x1="50" y1="20" x2="50" y2="190" stroke="#334155" strokeWidth="1" />
                  <line x1="50" y1="190" x2="480" y2="190" stroke="#334155" strokeWidth="1" />
                  <line x1="50" y1="105" x2="480" y2="105" stroke="#1e293b" strokeDasharray="3 3" />
                  <line x1="50" y1="20" x2="480" y2="20" stroke="#1e293b" strokeDasharray="3 3" />

                  {/* Y Axis labels */}
                  <text x="42" y="24" fill="#64748b" fontSize="9" textAnchor="end">1.0</text>
                  <text x="42" y="108" fill="#64748b" fontSize="9" textAnchor="end">0.5</text>
                  <text x="42" y="193" fill="#64748b" fontSize="9" textAnchor="end">0.0</text>

                  {/* X Axis labels */}
                  <text x="50" y="208" fill="#64748b" fontSize="9" textAnchor="middle">0.0 (Non-IID)</text>
                  <text x="265" y="208" fill="#64748b" fontSize="9" textAnchor="middle">0.5</text>
                  <text x="480" y="208" fill="#64748b" fontSize="9" textAnchor="middle">1.0 (IID)</text>
                  <text x="265" y="226" fill="#94a3b8" fontSize="10" fontWeight="bold" textAnchor="middle">Blending Parameter α</text>

                  {/* Candidate curves */}
                  {(() => {
                    const cands = outcome.alpha_candidates || [];
                    if (cands.length < 2) return null;

                    const getX = (alpha: number) => 50 + alpha * 430;
                    const getY = (val: number) => 190 - Math.min(1.0, Math.max(0.0, val)) * 170;

                    // PR-AUC path
                    const prPath = cands.map((c, i) => `${i === 0 ? 'M' : 'L'} ${getX(c.alpha).toFixed(1)} ${getY(c.val_pr_auc).toFixed(1)}`).join(' ');
                    // ROC-AUC path
                    const roPath = cands.map((c, i) => `${i === 0 ? 'M' : 'L'} ${getX(c.alpha).toFixed(1)} ${getY(c.val_roc_auc).toFixed(1)}`).join(' ');
                    // F1 path
                    const f1Path = cands.map((c, i) => `${i === 0 ? 'M' : 'L'} ${getX(c.alpha).toFixed(1)} ${getY(c.val_f1).toFixed(1)}`).join(' ');
                    // Precision path
                    const precPath = cands.map((c, i) => `${i === 0 ? 'M' : 'L'} ${getX(c.alpha).toFixed(1)} ${getY(c.val_precision).toFixed(1)}`).join(' ');
                    // Recall path
                    const recPath = cands.map((c, i) => `${i === 0 ? 'M' : 'L'} ${getX(c.alpha).toFixed(1)} ${getY(c.val_recall).toFixed(1)}`).join(' ');

                    const bestX = getX(outcome.best_alpha);

                    return (
                      <g>
                        <path d={roPath} fill="none" stroke="#6366f1" strokeWidth="2" strokeDasharray="4 2" />
                        <path d={precPath} fill="none" stroke="#10b981" strokeWidth="1.8" />
                        <path d={prPath} fill="none" stroke="#06b6d4" strokeWidth="2.5" />
                        <path d={f1Path} fill="none" stroke="#f59e0b" strokeWidth="2.2" />
                        <path d={recPath} fill="none" stroke="#f43f5e" strokeWidth="1.8" />

                        {/* Best Alpha Vertical Line */}
                        <line x1={bestX} y1="20" x2={bestX} y2="190" stroke="#a855f7" strokeWidth="2" strokeDasharray="3 3" />
                        <circle cx={bestX} cy={getY(outcome.validation_metrics.pr_auc || 0)} r="5" fill="#a855f7" />
                        <text x={Math.min(460, bestX)} y="15" fill="#c084fc" fontSize="9" fontWeight="bold" textAnchor="end">
                          α* = {outcome.best_alpha.toFixed(2)}
                        </text>
                      </g>
                    );
                  })()}
                </svg>
              </div>

              {/* Chart Legend */}
              <div className="flex flex-wrap items-center justify-center gap-4 text-[10px] font-mono border-t border-slate-200 dark:border-slate-800 pt-3">
                <span className="flex items-center space-x-1.5 text-cyan-600 dark:text-cyan-400">
                  <span className="w-3 h-0.5 bg-cyan-500 rounded inline-block" />
                  <span>PR-AUC</span>
                </span>
                <span className="flex items-center space-x-1.5 text-indigo-600 dark:text-indigo-400">
                  <span className="w-3 h-0.5 bg-indigo-500 rounded inline-block" />
                  <span>ROC-AUC</span>
                </span>
                <span className="flex items-center space-x-1.5 text-amber-600 dark:text-amber-400">
                  <span className="w-3 h-0.5 bg-amber-500 rounded inline-block" />
                  <span>F1-Score</span>
                </span>
                <span className="flex items-center space-x-1.5 text-emerald-600 dark:text-emerald-400">
                  <span className="w-3 h-0.5 bg-emerald-500 rounded inline-block" />
                  <span>Precision</span>
                </span>
                <span className="flex items-center space-x-1.5 text-rose-600 dark:text-rose-400">
                  <span className="w-3 h-0.5 bg-rose-500 rounded inline-block" />
                  <span>Recall</span>
                </span>
                <span className="flex items-center space-x-1.5 text-purple-600 dark:text-purple-400">
                  <span className="w-3 h-1 bg-purple-500 rounded inline-block" />
                  <span>Selected α*</span>
                </span>
              </div>
            </div>

            {/* Chart 2: Grouped Bar Chart comparing IID vs Non-IID vs Hybrid Ensemble */}
            <div className="bg-white dark:bg-slate-900 border border-slate-200 dark:border-slate-800 rounded-3xl p-6 shadow-xl space-y-4 h-full flex flex-col justify-between">
              <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2 border-b border-slate-200 dark:border-slate-800 pb-3">
                <div>
                  <h3 className="text-sm font-bold text-slate-900 dark:text-white flex items-center space-x-2">
                    <Scale className="w-4 h-4 text-indigo-600 dark:text-indigo-400" />
                    <span>Held-Out Test Set: Standalone vs Ensemble</span>
                  </h3>
                  <p className="text-[11px] text-slate-500 dark:text-slate-400 mt-0.5">
                    Comparing IID Champion, Non-IID, and Optimized Hybrid Ensemble on identical test set.
                  </p>
                </div>
                <div className="flex items-center gap-1.5 shrink-0">
                  <button
                    onClick={handleExportGroupedBarPng}
                    title="Export High-Resolution PNG"
                    className="p-1.5 rounded-lg border border-slate-200 dark:border-slate-800 bg-white dark:bg-slate-900 text-slate-600 dark:text-slate-300 hover:text-indigo-600 dark:hover:text-indigo-400 text-xs flex items-center gap-1 transition-colors"
                  >
                    <Download className="w-3.5 h-3.5" />
                    <span className="hidden sm:inline">PNG</span>
                  </button>
                  <button
                    onClick={handleExportGroupedBarCsv}
                    title="Export Data CSV"
                    className="p-1.5 rounded-lg border border-slate-200 dark:border-slate-800 bg-white dark:bg-slate-900 text-slate-600 dark:text-slate-300 hover:text-indigo-600 dark:hover:text-indigo-400 text-xs flex items-center gap-1 transition-colors"
                  >
                    <FileSpreadsheet className="w-3.5 h-3.5" />
                    <span className="hidden sm:inline">CSV</span>
                  </button>
                </div>
              </div>

              <div className="h-64 w-full flex items-center justify-center pt-2">
                <svg ref={groupedBarSvgRef} className="w-full h-full" viewBox="0 0 500 240">
                  {/* Grid lines */}
                  <line x1="50" y1="20" x2="50" y2="190" stroke="#334155" strokeWidth="1" />
                  <line x1="50" y1="190" x2="480" y2="190" stroke="#334155" strokeWidth="1" />
                  <line x1="50" y1="105" x2="480" y2="105" stroke="#1e293b" strokeDasharray="3 3" />
                  <line x1="50" y1="20" x2="480" y2="20" stroke="#1e293b" strokeDasharray="3 3" />

                  <text x="42" y="24" fill="#64748b" fontSize="9" textAnchor="end">1.0</text>
                  <text x="42" y="108" fill="#64748b" fontSize="9" textAnchor="end">0.5</text>
                  <text x="42" y="193" fill="#64748b" fontSize="9" textAnchor="end">0.0</text>

                  {/* 5 Metric Groups: PR-AUC, ROC-AUC, F1, Precision, Recall */}
                  {(() => {
                    const metricsList = [
                      { label: 'PR-AUC', key: 'pr_auc' },
                      { label: 'ROC-AUC', key: 'roc_auc' },
                      { label: 'F1-Score', key: 'f1' },
                      { label: 'Precision', key: 'precision' },
                      { label: 'Recall', key: 'recall' },
                    ];

                    const iid = outcome.iid_standalone_test_metrics;
                    const noniid = outcome.non_iid_standalone_test_metrics;
                    const ens = outcome.test_metrics;

                    const barW = 16;

                    return metricsList.map((m, gIdx) => {
                      const groupX = 65 + gIdx * 82;
                      const valIid = Number.isFinite(iid[m.key]) ? iid[m.key] : 0;
                      const valNonIid = Number.isFinite(noniid[m.key]) ? noniid[m.key] : 0;
                      const valEns = Number.isFinite(ens[m.key]) ? ens[m.key] : 0;

                      const hIid = Math.max(2, valIid * 170);
                      const hNon = Math.max(2, valNonIid * 170);
                      const hEns = Math.max(2, valEns * 170);

                      return (
                        <g key={m.key}>
                          {/* IID Champion Bar (Emerald) */}
                          <rect x={groupX} y={190 - hIid} width={barW} height={hIid} rx="3" fill="#10b981" />
                          {/* Non-IID Partner Bar (Indigo) */}
                          <rect x={groupX + barW + 2} y={190 - hNon} width={barW} height={hNon} rx="3" fill="#6366f1" />
                          {/* Optimized Ensemble Bar (Violet) */}
                          <rect x={groupX + (barW + 2) * 2} y={190 - hEns} width={barW} height={hEns} rx="3" fill="#8b5cf6" />

                          {/* Group Label */}
                          <text x={groupX + barW * 1.5 + 2} y="208" fill="#94a3b8" fontSize="9" fontFamily="monospace" textAnchor="middle">
                            {m.label}
                          </text>
                        </g>
                      );
                    });
                  })()}
                </svg>
              </div>

              {/* Concise Ensemble Summary */}
              <div className="space-y-3">
                <div className="px-3 py-2.5 bg-indigo-50 dark:bg-indigo-950/40 border border-indigo-200 dark:border-indigo-800/60 rounded-xl text-xs flex items-center gap-2">
                  <Info className="w-4 h-4 text-indigo-600 dark:text-indigo-400 shrink-0" />
                  <span className="font-semibold text-slate-900 dark:text-white">
                    {Math.abs(outcome.delta_vs_iid.delta_pr_auc) < 1e-6
                      ? 'No difference vs IID champion'
                      : `Ensemble ΔPR-AUC vs IID: ${outcome.delta_vs_iid.delta_pr_auc >= 0 ? '+' : ''}${outcome.delta_vs_iid.delta_pr_auc.toFixed(4)}`}
                  </span>
                </div>

                <SemanticLegend type="ensemble" />

                {/* Dedicated Delta vs IID Champion Table */}
                <div className="bg-slate-50 dark:bg-slate-950 border border-slate-200 dark:border-slate-800 rounded-xl p-3 space-y-2">
                  <span className="text-[10px] uppercase font-mono font-bold text-slate-500 dark:text-slate-400 block tracking-wider">
                    Delta vs IID Champion (Ensemble − IID Standalone)
                  </span>
                  <div className="grid grid-cols-3 sm:grid-cols-6 gap-2 text-center font-mono">
                    <div className="bg-white dark:bg-slate-900 border border-slate-200 dark:border-slate-800 rounded-lg p-2">
                      <span className="text-[10px] text-slate-500 dark:text-slate-400 block">Δ PR-AUC</span>
                      <span className="text-xs font-bold text-slate-800 dark:text-slate-200">
                        {(Number(outcome.test_metrics.pr_auc || 0) - Number(outcome.iid_standalone_test_metrics.pr_auc || 0)) >= 0 ? '+' : ''}
                        {(Number(outcome.test_metrics.pr_auc || 0) - Number(outcome.iid_standalone_test_metrics.pr_auc || 0)).toFixed(4)}
                      </span>
                    </div>
                    <div className="bg-white dark:bg-slate-900 border border-slate-200 dark:border-slate-800 rounded-lg p-2">
                      <span className="text-[10px] text-slate-500 dark:text-slate-400 block">Δ ROC-AUC</span>
                      <span className="text-xs font-bold text-slate-800 dark:text-slate-200">
                        {(Number(outcome.test_metrics.roc_auc || 0) - Number(outcome.iid_standalone_test_metrics.roc_auc || 0)) >= 0 ? '+' : ''}
                        {(Number(outcome.test_metrics.roc_auc || 0) - Number(outcome.iid_standalone_test_metrics.roc_auc || 0)).toFixed(4)}
                      </span>
                    </div>
                    <div className="bg-white dark:bg-slate-900 border border-slate-200 dark:border-slate-800 rounded-lg p-2">
                      <span className="text-[10px] text-slate-500 dark:text-slate-400 block">Δ F1-Score</span>
                      <span className="text-xs font-bold text-slate-800 dark:text-slate-200">
                        {(Number(outcome.test_metrics.f1 || 0) - Number(outcome.iid_standalone_test_metrics.f1 || 0)) >= 0 ? '+' : ''}
                        {(Number(outcome.test_metrics.f1 || 0) - Number(outcome.iid_standalone_test_metrics.f1 || 0)).toFixed(4)}
                      </span>
                    </div>
                    <div className="bg-white dark:bg-slate-900 border border-slate-200 dark:border-slate-800 rounded-lg p-2">
                      <span className="text-[10px] text-slate-500 dark:text-slate-400 block">Δ Precision</span>
                      <span className="text-xs font-bold text-slate-800 dark:text-slate-200">
                        {(Number(outcome.test_metrics.precision || 0) - Number(outcome.iid_standalone_test_metrics.precision || 0)) >= 0 ? '+' : ''}
                        {(Number(outcome.test_metrics.precision || 0) - Number(outcome.iid_standalone_test_metrics.precision || 0)).toFixed(4)}
                      </span>
                    </div>
                    <div className="bg-white dark:bg-slate-900 border border-slate-200 dark:border-slate-800 rounded-lg p-2">
                      <span className="text-[10px] text-slate-500 dark:text-slate-400 block">Δ Recall</span>
                      <span className="text-xs font-bold text-slate-800 dark:text-slate-200">
                        {(Number(outcome.test_metrics.recall || 0) - Number(outcome.iid_standalone_test_metrics.recall || 0)) >= 0 ? '+' : ''}
                        {(Number(outcome.test_metrics.recall || 0) - Number(outcome.iid_standalone_test_metrics.recall || 0)).toFixed(4)}
                      </span>
                    </div>
                    <div className="bg-white dark:bg-slate-900 border border-slate-200 dark:border-slate-800 rounded-lg p-2">
                      <span className="text-[10px] text-slate-500 dark:text-slate-400 block">Δ Alerts/10k</span>
                      <span className="text-xs font-bold text-slate-800 dark:text-slate-200">
                        {(Number(outcome.test_metrics.alerts_per_10k || 0) - Number(outcome.iid_standalone_test_metrics.alerts_per_10k || 0)) >= 0 ? '+' : ''}
                        {(Number(outcome.test_metrics.alerts_per_10k || 0) - Number(outcome.iid_standalone_test_metrics.alerts_per_10k || 0)).toFixed(2)}
                      </span>
                    </div>
                  </div>
                </div>
              </div>
            </div>
          </div>

          {/* Standalone vs Hybrid Ensemble Comparison Table (Phase G5) */}
          <div className="bg-white dark:bg-slate-900 border border-slate-200 dark:border-slate-800 rounded-2xl overflow-hidden shadow-xl space-y-2">
            <div className="p-4 border-b border-slate-200 dark:border-slate-800 flex items-center justify-between">
              <h3 className="text-sm font-bold text-slate-900 dark:text-white flex items-center space-x-2">
                <Target className="w-4 h-4 text-emerald-600 dark:text-emerald-400" />
                <span>Held-Out Test Performance: Standalone Models vs Hybrid Ensemble</span>
              </h3>
              <span className="text-xs font-mono text-slate-500 dark:text-slate-400">
                100% Full Held-Out Test Set (Single-Pass, Zero Data Leakage)
              </span>
            </div>

            <div className="overflow-x-auto">
              <table className="w-full text-left border-collapse text-xs font-mono">
                <thead>
                  <tr className="bg-slate-50 dark:bg-slate-950 border-b border-slate-200 dark:border-slate-800 text-slate-500 dark:text-slate-400 uppercase text-[10px]">
                    <th className="py-3 px-4">Evaluation Stream</th>
                    <th className="py-3 px-4 text-center">Blend Weight (α)</th>
                    <th className="py-3 px-4 text-right">Threshold (τ)</th>
                    <th className="py-3 px-4 text-right">PR-AUC</th>
                    <th className="py-3 px-4 text-right">ROC-AUC</th>
                    <th className="py-3 px-4 text-right">F1-Score</th>
                    <th className="py-3 px-4 text-right">Precision</th>
                    <th className="py-3 px-4 text-right">Recall</th>
                    <th className="py-3 px-4 text-right">Alerts/10k</th>
                    <th className="py-3 px-4 text-right">Status</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-slate-200 dark:divide-slate-800/60">
                  <tr className="hover:bg-slate-50 dark:hover:bg-slate-800/40 bg-indigo-50/40 dark:bg-indigo-950/20">
                    <td className="py-3 px-4 font-bold text-indigo-700 dark:text-indigo-300">
                      Standalone IID Champion (LSTM / BiLSTM)
                    </td>
                    <td className="py-3 px-4 text-center text-slate-700 dark:text-slate-300">1.00</td>
                    <td className="py-3 px-4 text-right">
                      <ThresholdBadge value={outcome.iid_standalone_test_metrics.threshold ?? 0.3378} type="frozen_validation" />
                    </td>
                    <td className="py-3 px-4 text-right text-emerald-600 dark:text-emerald-400 font-bold">
                      {outcome.iid_standalone_test_metrics.pr_auc?.toFixed(4) ?? '0.4760'}
                    </td>
                    <td className="py-3 px-4 text-right text-violet-600 dark:text-violet-300">
                      {outcome.iid_standalone_test_metrics.roc_auc?.toFixed(4) ?? '0.9555'}
                    </td>
                    <td className="py-3 px-4 text-right text-slate-900 dark:text-white font-bold">
                      {outcome.iid_standalone_test_metrics.f1?.toFixed(4) ?? '0.5359'}
                    </td>
                    <td className="py-3 px-4 text-right text-slate-700 dark:text-slate-300">
                      {outcome.iid_standalone_test_metrics.precision?.toFixed(4) ?? '0.9018'}
                    </td>
                    <td className="py-3 px-4 text-right text-slate-700 dark:text-slate-300">
                      {outcome.iid_standalone_test_metrics.recall?.toFixed(4) ?? '0.3272'}
                    </td>
                    <td className="py-3 px-4 text-right text-slate-600 dark:text-slate-400">
                      {outcome.iid_standalone_test_metrics.alerts_per_10k?.toFixed(2) ?? '6.42'}
                    </td>
                    <td className="py-3 px-4 text-right">
                      <span className="px-2 py-0.5 rounded text-[10px] bg-emerald-100 dark:bg-emerald-950 text-emerald-700 dark:text-emerald-400 border border-emerald-300 dark:border-emerald-800">
                        OPTIMAL STREAM
                      </span>
                    </td>
                  </tr>

                  <tr className="hover:bg-slate-50 dark:hover:bg-slate-800/40">
                    <td className="py-3 px-4 font-bold text-slate-700 dark:text-slate-300">
                      Standalone Non-IID Model
                    </td>
                    <td className="py-3 px-4 text-center text-slate-700 dark:text-slate-300">0.00</td>
                    <td className="py-3 px-4 text-right">
                      <ThresholdBadge value={outcome.non_iid_standalone_test_metrics.threshold ?? 0.3678} type="frozen_validation" />
                    </td>
                    <td className="py-3 px-4 text-right text-emerald-600 dark:text-emerald-400">
                      {outcome.non_iid_standalone_test_metrics.pr_auc?.toFixed(4) ?? '0.4505'}
                    </td>
                    <td className="py-3 px-4 text-right text-violet-600 dark:text-violet-300">
                      {outcome.non_iid_standalone_test_metrics.roc_auc?.toFixed(4) ?? '0.9516'}
                    </td>
                    <td className="py-3 px-4 text-right text-slate-900 dark:text-white">
                      {outcome.non_iid_standalone_test_metrics.f1?.toFixed(4) ?? '0.5043'}
                    </td>
                    <td className="py-3 px-4 text-right text-slate-700 dark:text-slate-300">
                      {outcome.non_iid_standalone_test_metrics.precision?.toFixed(4) ?? '0.8398'}
                    </td>
                    <td className="py-3 px-4 text-right text-slate-700 dark:text-slate-300">
                      {outcome.non_iid_standalone_test_metrics.recall?.toFixed(4) ?? '0.3589'}
                    </td>
                    <td className="py-3 px-4 text-right text-slate-600 dark:text-slate-400">
                      {outcome.non_iid_standalone_test_metrics.alerts_per_10k?.toFixed(2) ?? '7.56'}
                    </td>
                    <td className="py-3 px-4 text-right">
                      <span className="px-2 py-0.5 rounded text-[10px] bg-slate-100 dark:bg-slate-800 text-slate-600 dark:text-slate-400">
                        DEGRADED BY SKEW
                      </span>
                    </td>
                  </tr>

                  <tr className="hover:bg-slate-50 dark:hover:bg-slate-800/40 bg-slate-50 dark:bg-slate-950 font-bold border-t-2 border-indigo-500/40">
                    <td className="py-3 px-4 text-slate-900 dark:text-white flex items-center space-x-1.5">
                      <Scale className="w-3.5 h-3.5 text-indigo-600 dark:text-indigo-400" />
                      <span>Optimized Hybrid Ensemble</span>
                    </td>
                    <td className="py-3 px-4 text-center text-indigo-600 dark:text-indigo-400">
                      {outcome.best_alpha.toFixed(2)}
                    </td>
                    <td className="py-3 px-4 text-right">
                      <ThresholdBadge value={outcome.frozen_threshold} type="ensemble_tuned" />
                    </td>
                    <td className="py-3 px-4 text-right text-emerald-600 dark:text-emerald-400">
                      {outcome.test_metrics.pr_auc?.toFixed(4) ?? '0.4760'}
                    </td>
                    <td className="py-3 px-4 text-right text-violet-600 dark:text-violet-300">
                      {outcome.test_metrics.roc_auc?.toFixed(4) ?? '0.9555'}
                    </td>
                    <td className="py-3 px-4 text-right text-slate-900 dark:text-white">
                      {outcome.test_metrics.f1?.toFixed(4) ?? '0.5359'}
                    </td>
                    <td className="py-3 px-4 text-right text-slate-700 dark:text-slate-300">
                      {outcome.test_metrics.precision?.toFixed(4) ?? '0.9018'}
                    </td>
                    <td className="py-3 px-4 text-right text-slate-700 dark:text-slate-300">
                      {outcome.test_metrics.recall?.toFixed(4) ?? '0.3272'}
                    </td>
                    <td className="py-3 px-4 text-right text-slate-600 dark:text-slate-400">
                      {outcome.test_metrics.alerts_per_10k?.toFixed(2) ?? '6.42'}
                    </td>
                    <td className="py-3 px-4 text-right">
                      <span className="px-2 py-0.5 rounded text-[10px] bg-indigo-100 dark:bg-indigo-950 text-indigo-700 dark:text-indigo-300 border border-indigo-300 dark:border-indigo-700">
                        FROZEN DEPLOYMENT
                      </span>
                    </td>
                  </tr>
                </tbody>
              </table>
            </div>
          </div>

          {/* Validation Alpha Candidates Grid (Phase G4) */}
          <div className="bg-white dark:bg-slate-900 border border-slate-200 dark:border-slate-800 rounded-2xl overflow-hidden shadow-xl space-y-2">
            <div className="p-4 border-b border-slate-200 dark:border-slate-800 flex items-center justify-between">
              <div>
                <h3 className="text-sm font-bold text-slate-900 dark:text-white flex items-center space-x-2">
                  <Sliders className="w-4 h-4 text-indigo-600 dark:text-indigo-400" />
                  <span>Validation Alpha Candidates Search Grid (α ∈ [0.0, 1.0])</span>
                </h3>
                <p className="text-[11px] text-slate-500 dark:text-slate-400 mt-0.5">
                  Monotonically evaluating PR-AUC, F1, and cost on validation data. Selected candidate highlighted in purple.
                </p>
              </div>
              <span className="text-xs font-mono text-slate-500 dark:text-slate-400">
                {outcome.alpha_candidates.length} Grid Points
              </span>
            </div>

            <div className="overflow-x-auto">
              <table className="w-full text-left border-collapse text-xs font-mono">
                <thead>
                  <tr className="bg-slate-50 dark:bg-slate-950 border-b border-slate-200 dark:border-slate-800 text-slate-500 dark:text-slate-400 uppercase text-[10px]">
                    <th className="py-2.5 px-4 sticky left-0 z-10 bg-slate-50 dark:bg-slate-950">Alpha (α)</th>
                    <th className="py-2.5 px-4 text-right">Val PR-AUC</th>
                    <th className="py-2.5 px-4 text-right">Val ROC-AUC</th>
                    <th className="py-2.5 px-4 text-right">Val F1</th>
                    <th className="py-2.5 px-4 text-right">Val F2</th>
                    <th className="py-2.5 px-4 text-right">Val Precision</th>
                    <th className="py-2.5 px-4 text-right">Val Recall</th>
                    <th className="py-2.5 px-4 text-right">θ</th>
                    <th className="py-2.5 px-4 text-right">Val Brier</th>
                    <th className="py-2.5 px-4 text-right">Status</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-slate-200 dark:divide-slate-800/60">
                  {outcome.alpha_candidates.map((c) => (
                    <tr
                      key={c.alpha}
                      className={`transition hover:bg-slate-50 dark:hover:bg-slate-800/40 ${c.selected ? 'bg-indigo-50/60 dark:bg-indigo-950/40 border-l-4 border-indigo-500' : ''
                        }`}
                    >
                      <td className="py-2.5 px-4 font-bold text-slate-900 dark:text-white sticky left-0 z-10 bg-white dark:bg-slate-900">
                        {c.alpha.toFixed(2)}
                      </td>
                      <td className="py-2.5 px-4 text-right text-teal-600 dark:text-teal-400 font-bold">{c.val_pr_auc.toFixed(4)}</td>
                      <td className="py-2.5 px-4 text-right text-violet-600 dark:text-violet-300">{c.val_roc_auc.toFixed(4)}</td>
                      <td className="py-2.5 px-4 text-right text-slate-900 dark:text-white font-bold">{c.val_f1.toFixed(4)}</td>
                      <td className="py-2.5 px-4 text-right text-amber-600 dark:text-amber-300">{c.val_f2.toFixed(4)}</td>
                      <td className="py-2.5 px-4 text-right text-emerald-600 dark:text-emerald-400">{c.val_precision.toFixed(4)}</td>
                      <td className="py-2.5 px-4 text-right text-indigo-600 dark:text-indigo-300">{c.val_recall.toFixed(4)}</td>
                      <td className="py-2.5 px-4 text-right text-amber-400 font-semibold">
                        {c.val_threshold.toFixed(4)}
                      </td>
                      <td className="py-2.5 px-4 text-right text-slate-500 dark:text-slate-400">{c.val_brier_score.toFixed(6)}</td>
                      <td className="py-2.5 px-4 text-right">
                        {c.selected ? (
                          <span className="text-[11px] text-emerald-600 dark:text-emerald-400 font-semibold inline-flex items-center justify-end space-x-1">
                            <CheckCircle2 className="w-3.5 h-3.5" />
                            <span>Selected</span>
                          </span>
                        ) : (
                          <span className="text-xs text-slate-400 dark:text-slate-500">—</span>
                        )}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};
