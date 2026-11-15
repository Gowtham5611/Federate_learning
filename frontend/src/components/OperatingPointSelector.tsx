import React, { useState } from 'react';
import { OperatingPointsMap, OperatingPointMetrics } from '../types/api';
import { formatMetric, formatPercent } from '../utils/formatters';
import { Sliders, ShieldCheck, Target, ChevronDown, ChevronUp, AlertCircle, HelpCircle } from 'lucide-react';

export type OperatingPointKey = 'frozen' | 'zero_diagnostic' | 'budget_10' | 'budget_50' | 'budget_100' | 'budget_500' | 'best_f1';

interface OperatingPointSelectorProps {
  operatingPoints?: OperatingPointsMap | null;
  selectedPoint: OperatingPointKey;
  onSelectPoint: (point: OperatingPointKey) => void;
  defaultThreshold: number;
}

export const OperatingPointSelector: React.FC<OperatingPointSelectorProps> = ({
  operatingPoints,
  selectedPoint,
  onSelectPoint,
  defaultThreshold,
}) => {
  const [showAdvanced, setShowAdvanced] = useState<boolean>(false);

  if (!operatingPoints) return null;

  const getPointByKey = (key: OperatingPointKey): OperatingPointMetrics | null => {
    if (key === 'zero_diagnostic') return operatingPoints.zero_diagnostic || null;
    if (key === 'budget_500') return operatingPoints.alert_budgets?.['500'] || null;
    if (key === 'budget_10') return operatingPoints.alert_budgets?.['10'] || null;
    if (key === 'budget_50') return operatingPoints.alert_budgets?.['50'] || null;
    if (key === 'budget_100') return operatingPoints.alert_budgets?.['100'] || null;
    if (key === 'best_f1') return operatingPoints.best_f1_on_file || null;
    return operatingPoints.frozen || null;
  };

  const frozenOp = getPointByKey('frozen');
  const budget500Op = getPointByKey('budget_500');
  const zeroOp = getPointByKey('zero_diagnostic');
  const activeOp = getPointByKey(selectedPoint);

  const activeCutoff = activeOp?.threshold ?? (selectedPoint === 'zero_diagnostic' ? 0.0 : defaultThreshold);
  const isFixedThreshold = selectedPoint === 'frozen';
  const isFixedCapacity50 = selectedPoint === 'budget_500';
  const isZeroDiagnostic = selectedPoint === 'zero_diagnostic';

  return (
    <div className="bg-slate-950 p-4 sm:p-5 rounded-3xl border border-slate-800 space-y-4 shadow-xl">
      {/* Title & Active Policy Summary Bar */}
      <div className="flex flex-col md:flex-row md:items-center justify-between gap-3 border-b border-slate-900 pb-3">
        <div className="flex items-center space-x-2.5">
          <div className="p-2 rounded-xl bg-indigo-500/20 text-indigo-300 border border-indigo-500/30">
            <Sliders className="w-4 h-4 text-indigo-400" />
          </div>
          <div>
            <div className="flex flex-wrap items-center gap-2">
              <h3 className="text-sm font-bold text-white tracking-tight">
                Operating Mode & Decision Policy
              </h3>
              <span
                className={`px-2 py-0.5 rounded-full text-[10px] font-mono font-bold border ${
                  isFixedThreshold
                    ? 'bg-indigo-950 text-indigo-300 border-indigo-700/60'
                    : isFixedCapacity50
                    ? 'bg-emerald-950 text-emerald-300 border-emerald-700/60'
                    : isZeroDiagnostic
                    ? 'bg-amber-950 text-amber-300 border-amber-700/60'
                    : 'bg-slate-900 text-slate-300 border-slate-700'
                }`}
              >
                ACTIVE:{' '}
                {isFixedThreshold
                  ? 'FIXED SERVING THRESHOLD'
                  : isFixedCapacity50
                  ? 'FIXED CAPACITY (50/1k)'
                  : isZeroDiagnostic
                  ? 'THRESHOLD 0 DIAGNOSTIC'
                  : activeOp?.name || selectedPoint}
              </span>
            </div>
            <p className="text-[11px] text-slate-400">
              Select between the operational serving threshold ($\theta^* = {(frozenOp?.threshold ?? defaultThreshold).toFixed(4)}$), investigative capacity, or the zero-threshold diagnostic.
            </p>
          </div>
        </div>

        {/* Live Summary Chips */}
        <div className="flex flex-wrap items-center gap-2 font-mono text-xs">
          <div className="px-3 py-1.5 rounded-xl bg-slate-900 border border-slate-800 flex items-center space-x-1.5">
            <span className="text-slate-400 text-[10px] uppercase">Effective Cutoff:</span>
            <strong className="text-amber-400 font-bold">{activeCutoff.toFixed(4)}</strong>
          </div>
          <div className="px-3 py-1.5 rounded-xl bg-slate-900 border border-slate-800 flex items-center space-x-1.5">
            <span className="text-slate-400 text-[10px] uppercase">Alerts Flagged:</span>
            <strong className="text-white font-bold">{activeOp?.alerts?.toLocaleString() ?? 0}</strong>
          </div>
          <div className="px-3 py-1.5 rounded-xl bg-slate-900 border border-slate-800 flex items-center space-x-1.5">
            <span className="text-slate-400 text-[10px] uppercase">Rate:</span>
            <strong className="text-emerald-400 font-bold">
              {((activeOp?.alerts_per_10k ?? 0) / 10).toFixed(1)} / 1k
            </strong>
          </div>
        </div>
      </div>

      {/* Primary Operating Modes: Three High-Contrast Cards */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-3.5">
        {/* Mode 1: Fixed Canonical Threshold */}
        <button
          type="button"
          onClick={() => onSelectPoint('frozen')}
          className={`relative text-left p-4 rounded-2xl transition-all border cursor-pointer group focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-indigo-500 ${
            isFixedThreshold
              ? 'bg-gradient-to-br from-indigo-950/70 to-slate-900 border-indigo-500 shadow-lg shadow-indigo-600/20 ring-1 ring-indigo-500/50'
              : 'bg-slate-900/60 hover:bg-slate-900 border-slate-800 hover:border-slate-700'
          }`}
        >
          <div className="flex items-start justify-between gap-2">
            <div className="space-y-1">
              <div className="flex items-center space-x-2">
                <span
                  className={`w-3.5 h-3.5 rounded-full border-2 flex items-center justify-center shrink-0 ${
                    isFixedThreshold ? 'border-indigo-400 bg-indigo-500' : 'border-slate-600 bg-slate-950'
                  }`}
                >
                  {isFixedThreshold && <span className="w-1.5 h-1.5 rounded-full bg-white" />}
                </span>
                <span className="text-sm font-bold text-white group-hover:text-indigo-200 transition">
                  Fixed Serving Threshold
                </span>
              </div>
              <p className="text-[11px] text-slate-400 font-sans pl-5.5">
                Calibrated serving cutoff ($\theta^* = {(frozenOp?.threshold ?? defaultThreshold).toFixed(4)}$). Flags all scores exceeding threshold.
              </p>
            </div>
            <span className="px-2 py-0.5 rounded text-[10px] font-mono font-semibold bg-indigo-950/80 text-indigo-300 border border-indigo-800 shrink-0">
              Operational
            </span>
          </div>

          {/* Key Metrics Preview Strip */}
          <div className="mt-3.5 pt-3 border-t border-slate-800/80 grid grid-cols-2 sm:grid-cols-4 gap-1.5 text-center font-mono">
            <div className="p-1.5 rounded-xl bg-slate-950/80 border border-slate-800/60">
              <span className="text-[9px] text-slate-400 uppercase block">Threshold</span>
              <span className="text-xs font-black text-amber-300 block">
                {(frozenOp?.threshold ?? defaultThreshold).toFixed(4)}
              </span>
            </div>
            <div className="p-1.5 rounded-xl bg-slate-950/80 border border-slate-800/60">
              <span className="text-[9px] text-slate-400 uppercase block">Alerts</span>
              <span className="text-xs font-black text-white block">
                {frozenOp?.alerts !== undefined ? frozenOp.alerts.toLocaleString() : '--'}
              </span>
            </div>
            <div className="p-1.5 rounded-xl bg-slate-950/80 border border-slate-800/60">
              <span className="text-[9px] text-slate-400 uppercase block">Precision</span>
              <span className="text-xs font-black text-emerald-400 block">
                {frozenOp?.precision !== undefined && frozenOp?.precision !== null ? formatPercent(frozenOp.precision, 2) : '--'}
              </span>
            </div>
            <div className="p-1.5 rounded-xl bg-slate-950/80 border border-slate-800/60">
              <span className="text-[9px] text-slate-400 uppercase block">Recall</span>
              <span className="text-xs font-black text-indigo-300 block">
                {frozenOp?.recall !== undefined && frozenOp?.recall !== null ? formatPercent(frozenOp.recall, 2) : '--'}
              </span>
            </div>
          </div>

          {/* Confusion Counts Footer */}
          <div className="mt-2 text-[10px] font-mono text-slate-400 flex flex-wrap items-center justify-between gap-1 px-1">
            <span>TP: <strong className="text-emerald-300">{frozenOp?.tp?.toLocaleString() ?? '--'}</strong></span>
            <span>FP: <strong className="text-amber-300">{frozenOp?.fp?.toLocaleString() ?? '--'}</strong></span>
            <span>FN: <strong className="text-rose-300">{frozenOp?.fn?.toLocaleString() ?? '--'}</strong></span>
            <span>TN: <strong className="text-slate-300">{frozenOp?.tn?.toLocaleString() ?? '--'}</strong></span>
          </div>
        </button>

        {/* Mode 2: Fixed Alert Capacity (50 alerts / 1,000 transactions) */}
        <button
          type="button"
          onClick={() => onSelectPoint('budget_500')}
          className={`relative text-left p-4 rounded-2xl transition-all border cursor-pointer group focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-emerald-500 ${
            isFixedCapacity50
              ? 'bg-gradient-to-br from-emerald-950/70 to-slate-900 border-emerald-500 shadow-lg shadow-emerald-600/20 ring-1 ring-emerald-500/50'
              : 'bg-slate-900/60 hover:bg-slate-900 border-slate-800 hover:border-slate-700'
          }`}
        >
          <div className="flex items-start justify-between gap-2">
            <div className="space-y-1">
              <div className="flex items-center space-x-2">
                <span
                  className={`w-3.5 h-3.5 rounded-full border-2 flex items-center justify-center shrink-0 ${
                    isFixedCapacity50 ? 'border-emerald-400 bg-emerald-500' : 'border-slate-600 bg-slate-950'
                  }`}
                >
                  {isFixedCapacity50 && <span className="w-1.5 h-1.5 rounded-full bg-white" />}
                </span>
                <span className="text-sm font-bold text-white group-hover:text-emerald-200 transition">
                  Fixed Capacity (50/1k)
                </span>
              </div>
              <p className="text-[11px] text-slate-400 font-sans pl-5.5">
                Strict top-K workload ceiling (50 alerts / 1,000 tx). Effective cutoff adjusts dynamically.
              </p>
            </div>
            <span className="px-2 py-0.5 rounded text-[10px] font-mono font-semibold bg-emerald-950/80 text-emerald-300 border border-emerald-800 shrink-0">
              Capacity Cap
            </span>
          </div>

          {/* Key Metrics Preview Strip */}
          <div className="mt-3.5 pt-3 border-t border-slate-800/80 grid grid-cols-2 sm:grid-cols-4 gap-1.5 text-center font-mono">
            <div className="p-1.5 rounded-xl bg-slate-950/80 border border-slate-800/60">
              <span className="text-[9px] text-slate-400 uppercase block">Effective Cutoff</span>
              <span className="text-xs font-black text-amber-300 block">
                {budget500Op?.threshold !== undefined ? budget500Op.threshold.toFixed(4) : '--'}
              </span>
            </div>
            <div className="p-1.5 rounded-xl bg-slate-950/80 border border-slate-800/60">
              <span className="text-[9px] text-slate-400 uppercase block">Alerts</span>
              <span className="text-xs font-black text-white block">
                {budget500Op?.alerts !== undefined ? budget500Op.alerts.toLocaleString() : '--'}
              </span>
            </div>
            <div className="p-1.5 rounded-xl bg-slate-950/80 border border-slate-800/60">
              <span className="text-[9px] text-slate-400 uppercase block">Precision</span>
              <span className="text-xs font-black text-emerald-400 block">
                {budget500Op?.precision !== undefined && budget500Op?.precision !== null ? formatPercent(budget500Op.precision, 2) : '--'}
              </span>
            </div>
            <div className="p-1.5 rounded-xl bg-slate-950/80 border border-slate-800/60">
              <span className="text-[9px] text-slate-400 uppercase block">Recall</span>
              <span className="text-xs font-black text-indigo-300 block">
                {budget500Op?.recall !== undefined && budget500Op?.recall !== null ? formatPercent(budget500Op.recall, 2) : '--'}
              </span>
            </div>
          </div>

          {/* Confusion Counts Footer */}
          <div className="mt-2 text-[10px] font-mono text-slate-400 flex flex-wrap items-center justify-between gap-1 px-1">
            <span>TP: <strong className="text-emerald-300">{budget500Op?.tp?.toLocaleString() ?? '--'}</strong></span>
            <span>FP: <strong className="text-amber-300">{budget500Op?.fp?.toLocaleString() ?? '--'}</strong></span>
            <span>FN: <strong className="text-rose-300">{budget500Op?.fn?.toLocaleString() ?? '--'}</strong></span>
            <span>TN: <strong className="text-slate-300">{budget500Op?.tn?.toLocaleString() ?? '--'}</strong></span>
          </div>
        </button>

        {/* Mode 3: Threshold 0 Diagnostic (Diagnostic only - not serving threshold) */}
        <button
          type="button"
          onClick={() => onSelectPoint('zero_diagnostic')}
          className={`relative text-left p-4 rounded-2xl transition-all border cursor-pointer group focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-amber-500 ${
            isZeroDiagnostic
              ? 'bg-gradient-to-br from-amber-950/70 to-slate-900 border-amber-500 shadow-lg shadow-amber-600/20 ring-1 ring-amber-500/50'
              : 'bg-slate-900/60 hover:bg-slate-900 border-slate-800 hover:border-slate-700'
          }`}
        >
          <div className="flex items-start justify-between gap-2">
            <div className="space-y-1">
              <div className="flex items-center space-x-2">
                <span
                  className={`w-3.5 h-3.5 rounded-full border-2 flex items-center justify-center shrink-0 ${
                    isZeroDiagnostic ? 'border-amber-400 bg-amber-500' : 'border-slate-600 bg-slate-950'
                  }`}
                >
                  {isZeroDiagnostic && <span className="w-1.5 h-1.5 rounded-full bg-white" />}
                </span>
                <span className="text-sm font-bold text-white group-hover:text-amber-200 transition">
                  Threshold 0 Diagnostic
                </span>
              </div>
              <p className="text-[11px] text-slate-400 font-sans pl-5.5">
                Diagnostic baseline ($\theta = 0.0000$). Flags all valid non-negative scores; does not change serving threshold.
              </p>
            </div>
            <span className="px-2 py-0.5 rounded text-[10px] font-mono font-semibold bg-amber-950/80 text-amber-300 border border-amber-800 shrink-0">
              Diagnostic Only
            </span>
          </div>

          {/* Key Metrics Preview Strip */}
          <div className="mt-3.5 pt-3 border-t border-slate-800/80 grid grid-cols-2 sm:grid-cols-4 gap-1.5 text-center font-mono">
            <div className="p-1.5 rounded-xl bg-slate-950/80 border border-slate-800/60">
              <span className="text-[9px] text-slate-400 uppercase block">Threshold</span>
              <span className="text-xs font-black text-amber-300 block">
                0.0000
              </span>
            </div>
            <div className="p-1.5 rounded-xl bg-slate-950/80 border border-slate-800/60">
              <span className="text-[9px] text-slate-400 uppercase block">Alerts</span>
              <span className="text-xs font-black text-white block">
                {zeroOp?.alerts !== undefined ? zeroOp.alerts.toLocaleString() : '--'}
              </span>
            </div>
            <div className="p-1.5 rounded-xl bg-slate-950/80 border border-slate-800/60">
              <span className="text-[9px] text-slate-400 uppercase block">Precision</span>
              <span className="text-xs font-black text-emerald-400 block">
                {zeroOp?.precision !== undefined && zeroOp?.precision !== null ? formatPercent(zeroOp.precision, 2) : '--'}
              </span>
            </div>
            <div className="p-1.5 rounded-xl bg-slate-950/80 border border-slate-800/60">
              <span className="text-[9px] text-slate-400 uppercase block">Recall</span>
              <span className="text-xs font-black text-indigo-300 block">
                {zeroOp?.recall !== undefined && zeroOp?.recall !== null ? formatPercent(zeroOp.recall, 2) : '100.0%'}
              </span>
            </div>
          </div>

          {/* Confusion Counts Footer */}
          <div className="mt-2 text-[10px] font-mono text-slate-400 flex flex-wrap items-center justify-between gap-1 px-1">
            <span>TP: <strong className="text-emerald-300">{zeroOp?.tp?.toLocaleString() ?? '--'}</strong></span>
            <span>FP: <strong className="text-amber-300">{zeroOp?.fp?.toLocaleString() ?? '--'}</strong></span>
            <span>FN: <strong className="text-rose-300">{zeroOp?.fn?.toLocaleString() ?? '0'}</strong></span>
            <span>TN: <strong className="text-slate-300">{zeroOp?.tn?.toLocaleString() ?? '0'}</strong></span>
          </div>
        </button>
      </div>

      {/* Explanatory Policy Callout */}
      <div
        className={`p-3.5 rounded-2xl border text-xs leading-relaxed transition-all ${
          isFixedThreshold
            ? 'bg-indigo-950/30 border-indigo-800/50 text-indigo-200'
            : isFixedCapacity50
            ? 'bg-emerald-950/30 border-emerald-800/50 text-emerald-200'
            : isZeroDiagnostic
            ? 'bg-amber-950/30 border-amber-800/50 text-amber-200'
            : 'bg-slate-900/60 border-slate-800 text-slate-300'
        }`}
      >
        <div className="flex items-start space-x-2">
          {isFixedThreshold ? (
            <Target className="w-4 h-4 text-indigo-400 shrink-0 mt-0.5" />
          ) : isFixedCapacity50 ? (
            <ShieldCheck className="w-4 h-4 text-emerald-400 shrink-0 mt-0.5" />
          ) : isZeroDiagnostic ? (
            <AlertCircle className="w-4 h-4 text-amber-400 shrink-0 mt-0.5" />
          ) : (
            <HelpCircle className="w-4 h-4 text-slate-400 shrink-0 mt-0.5" />
          )}
          <div>
            <strong className="text-white block font-mono text-[11px] uppercase">
              {isFixedThreshold
                ? 'Operational Serving Threshold Active'
                : isFixedCapacity50
                ? 'Fixed Capacity Policy Active (50 alerts / 1,000 tx)'
                : isZeroDiagnostic
                ? 'Diagnostic Only — Not the Serving Threshold'
                : `Custom Operating Point Active (${selectedPoint})`}
            </strong>
            <p className="mt-0.5 text-[11px] text-slate-300">
              {isFixedThreshold
                ? `Fixed threshold evaluates probability >= ${(frozenOp?.threshold ?? defaultThreshold).toFixed(4)} without alert caps. Alert volume floats dynamically with batch risk (${frozenOp?.alerts?.toLocaleString() ?? 0} alerts flagged on this upload).`
                : isFixedCapacity50
                ? `Fixed capacity enforces a strict top-K investigative capacity ceiling (50 alerts per 1,000 transactions, capping at ${budget500Op?.alerts?.toLocaleString() ?? 0} alerts). Effective cutoff adjusts dynamically to ${budget500Op?.threshold !== undefined ? budget500Op.threshold.toFixed(4) : '--'}.`
                : isZeroDiagnostic
                ? `Threshold 0 diagnostic evaluates decision rule probability >= 0.0 across all valid non-negative scores (${zeroOp?.alerts?.toLocaleString() ?? 0} alerts flagged). It captures 100% of labeled laundering cases (TP = ${zeroOp?.tp?.toLocaleString() ?? 0}, FN = 0) at the expense of zero specificity (FP = ${zeroOp?.fp?.toLocaleString() ?? 0}, TN = 0). This view is diagnostic only and does not change or overwrite the operational serving threshold.`
                : `Operating point '${selectedPoint}' is active with effective cutoff ${activeCutoff.toFixed(4)}.`}
            </p>
          </div>
        </div>
      </div>

      {/* Collapsible Additional Research Capacities */}
      <div className="pt-1">
        <button
          type="button"
          onClick={() => setShowAdvanced(!showAdvanced)}
          className="flex items-center space-x-2 text-xs font-mono text-slate-400 hover:text-slate-200 cursor-pointer transition focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-indigo-500 rounded-lg px-2 py-1"
        >
          <span>{showAdvanced ? 'Hide Additional Operating Points' : 'View Other Alert Capacities (10/10k, 50/10k, 100/10k, Best F1)'}</span>
          {showAdvanced ? <ChevronUp className="w-3.5 h-3.5" /> : <ChevronDown className="w-3.5 h-3.5" />}
        </button>

        {showAdvanced && (
          <div className="mt-2.5 p-3 rounded-2xl bg-slate-900/60 border border-slate-800/80 space-y-2 text-xs">
            <span className="text-[10px] text-slate-500 font-mono uppercase block">Alternative Capacity Settings:</span>
            <div className="flex flex-wrap gap-2">
              <button
                type="button"
                onClick={() => onSelectPoint('budget_10')}
                className={`px-3 py-1.5 rounded-xl text-xs font-mono font-semibold transition border cursor-pointer ${
                  selectedPoint === 'budget_10'
                    ? 'bg-indigo-600 text-white border-indigo-500 shadow-md shadow-indigo-600/30'
                    : 'bg-slate-950 text-slate-400 border-slate-800 hover:text-slate-200'
                }`}
              >
                10 / 10k (1 / 1k)
              </button>

              <button
                type="button"
                onClick={() => onSelectPoint('budget_50')}
                className={`px-3 py-1.5 rounded-xl text-xs font-mono font-semibold transition border cursor-pointer ${
                  selectedPoint === 'budget_50'
                    ? 'bg-indigo-600 text-white border-indigo-500 shadow-md shadow-indigo-600/30'
                    : 'bg-slate-950 text-slate-400 border-slate-800 hover:text-slate-200'
                }`}
              >
                50 / 10k (5 / 1k)
              </button>

              <button
                type="button"
                onClick={() => onSelectPoint('budget_100')}
                className={`px-3 py-1.5 rounded-xl text-xs font-mono font-semibold transition border cursor-pointer ${
                  selectedPoint === 'budget_100'
                    ? 'bg-indigo-600 text-white border-indigo-500 shadow-md shadow-indigo-600/30'
                    : 'bg-slate-950 text-slate-400 border-slate-800 hover:text-slate-200'
                }`}
              >
                100 / 10k (10 / 1k)
              </button>

              <button
                type="button"
                onClick={() => onSelectPoint('best_f1')}
                className={`px-3 py-1.5 rounded-xl text-xs font-mono font-semibold transition border cursor-pointer ${
                  selectedPoint === 'best_f1'
                    ? 'bg-amber-600 text-white border-amber-500 shadow-md shadow-amber-600/30'
                    : 'bg-slate-950 text-amber-400/80 border-slate-800 hover:text-amber-300'
                }`}
              >
                Best F1 on File (Diagnostic What-If)
              </button>
            </div>

            {selectedPoint === 'best_f1' && (
              <div className="text-[11px] font-sans text-amber-300 bg-amber-950/40 p-2.5 rounded-xl border border-amber-800/60 flex items-center space-x-2 mt-2">
                <AlertCircle className="w-4 h-4 shrink-0 text-amber-400" />
                <span>
                  <strong>In-Sample Diagnostic What-If:</strong> Cutoff {activeCutoff.toFixed(4)} represents an in-sample upper bound on this uploaded batch's ground-truth labels ({formatMetric(activeOp?.f1_score, 4)} F1). Not an out-of-sample or deployable claim.
                </span>
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
};
