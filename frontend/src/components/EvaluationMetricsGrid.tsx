import React from 'react';
import { EvaluationMetrics, OperatingPoint } from '../types/api';
import { formatMetric, formatPercent } from '../utils/formatters';

interface EvaluationMetricsGridProps {
  metrics: EvaluationMetrics;
  activeOp?: OperatingPoint | null;
}

export const EvaluationMetricsGrid: React.FC<EvaluationMetricsGridProps> = ({
  metrics,
  activeOp,
}) => {
  const f1 = activeOp?.f1_score ?? metrics.f1_score;
  const precision = activeOp?.precision ?? metrics.precision;
  const recall = activeOp?.recall ?? metrics.recall;
  const specificity = activeOp?.specificity ?? metrics.specificity;
  const balancedAcc = activeOp?.balanced_accuracy ?? metrics.balanced_accuracy;
  const accuracy = activeOp?.accuracy ?? metrics.accuracy;
  const mcc = activeOp?.mcc ?? metrics.mcc;
  const kappa = activeOp?.cohen_kappa ?? metrics.cohen_kappa;

  return (
    <div className="space-y-4">
      {/* 8 Primary Supervised Classification Metrics */}
      <div className="grid grid-cols-2 sm:grid-cols-4 lg:grid-cols-4 xl:grid-cols-8 gap-2.5 sm:gap-3">
        {/* 1. Precision */}
        <div className="bg-slate-950 p-3.5 sm:p-4 rounded-2xl border border-slate-800 shadow-sm flex flex-col justify-between">
          <div>
            <span className="text-[11px] text-slate-400 block font-mono font-medium">Precision</span>
            <span className="text-xl sm:text-2xl font-black font-mono text-white mt-1 block tracking-tight">
              {precision !== null && precision !== undefined ? formatPercent(precision, 2) : '0 Alerts'}
            </span>
          </div>
          <span className="text-[10px] text-slate-500 font-mono mt-2 block pt-1 border-t border-slate-900">
            TP / (TP + FP)
          </span>
        </div>

        {/* 2. Recall / Sensitivity */}
        <div className="bg-slate-950 p-3.5 sm:p-4 rounded-2xl border border-slate-800 shadow-sm flex flex-col justify-between">
          <div>
            <span className="text-[11px] text-slate-400 block font-mono font-medium">Recall / Sens.</span>
            <span className="text-xl sm:text-2xl font-black font-mono text-indigo-300 mt-1 block tracking-tight">
              {formatPercent(recall, 2)}
            </span>
          </div>
          <span className="text-[10px] text-slate-500 font-mono mt-2 block pt-1 border-t border-slate-900">
            TP / (TP + FN)
          </span>
        </div>

        {/* 3. F1-Score */}
        <div className="bg-slate-950 p-3.5 sm:p-4 rounded-2xl border border-slate-800 shadow-sm flex flex-col justify-between">
          <div>
            <span className="text-[11px] text-slate-400 block font-mono font-medium">F1-Score</span>
            <span className="text-xl sm:text-2xl font-black font-mono text-amber-400 mt-1 block tracking-tight">
              {formatMetric(f1, 4, '0.0000')}
            </span>
          </div>
          <span className="text-[10px] text-slate-500 font-mono mt-2 block pt-1 border-t border-slate-900">
            Harmonic Mean
          </span>
        </div>

        {/* 4. Specificity */}
        <div className="bg-slate-950 p-3.5 sm:p-4 rounded-2xl border border-slate-800 shadow-sm flex flex-col justify-between">
          <div>
            <span className="text-[11px] text-slate-400 block font-mono font-medium">Specificity</span>
            <span className="text-xl sm:text-2xl font-black font-mono text-cyan-300 mt-1 block tracking-tight">
              {formatPercent(specificity, 2)}
            </span>
          </div>
          <span className="text-[10px] text-slate-500 font-mono mt-2 block pt-1 border-t border-slate-900">
            TN / (TN + FP)
          </span>
        </div>

        {/* 5. Balanced Accuracy */}
        <div className="bg-slate-950 p-3.5 sm:p-4 rounded-2xl border border-slate-800 shadow-sm flex flex-col justify-between">
          <div>
            <span className="text-[11px] text-slate-400 block font-mono font-medium">Balanced Acc</span>
            <span className="text-xl sm:text-2xl font-black font-mono text-teal-400 mt-1 block tracking-tight">
              {formatPercent(balancedAcc, 2)}
            </span>
          </div>
          <span className="text-[10px] text-slate-500 font-mono mt-2 block pt-1 border-t border-slate-900">
            (Sens + Spec) / 2
          </span>
        </div>

        {/* 6. Overall Accuracy */}
        <div className="bg-slate-950 p-3.5 sm:p-4 rounded-2xl border border-slate-800 shadow-sm flex flex-col justify-between">
          <div>
            <span className="text-[11px] text-slate-400 block font-mono font-medium">Accuracy</span>
            <span className="text-xl sm:text-2xl font-black font-mono text-slate-200 mt-1 block tracking-tight">
              {formatPercent(accuracy, 2)}
            </span>
          </div>
          <span className="text-[10px] text-slate-500 font-mono mt-2 block pt-1 border-t border-slate-900">
            (TP + TN) / Total
          </span>
        </div>

        {/* 7. Matthews Corr (MCC) */}
        <div className="bg-slate-950 p-3.5 sm:p-4 rounded-2xl border border-slate-800 shadow-sm flex flex-col justify-between">
          <div>
            <span className="text-[11px] text-slate-400 block font-mono font-medium">Matthews Corr</span>
            <span className="text-xl sm:text-2xl font-black font-mono text-violet-300 mt-1 block tracking-tight">
              {formatMetric(mcc, 4, '0.0000')}
            </span>
          </div>
          <span className="text-[10px] text-slate-500 font-mono mt-2 block pt-1 border-t border-slate-900">
            Correlation [-1, +1]
          </span>
        </div>

        {/* 8. Cohen's Kappa */}
        <div className="bg-slate-950 p-3.5 sm:p-4 rounded-2xl border border-slate-800 shadow-sm flex flex-col justify-between">
          <div>
            <span className="text-[11px] text-slate-400 block font-mono font-medium">Cohen's Kappa</span>
            <span className="text-xl sm:text-2xl font-black font-mono text-violet-400 mt-1 block tracking-tight">
              {formatMetric(kappa, 4, '0.0000')}
            </span>
          </div>
          <span className="text-[10px] text-slate-500 font-mono mt-2 block pt-1 border-t border-slate-900">
            Chance-Corrected
          </span>
        </div>
      </div>

      {/* Headline Ranking & Calibration Invariants */}
      <div className="grid grid-cols-2 sm:grid-cols-4 gap-2.5 sm:gap-3 pt-1">
        <div className="bg-gradient-to-br from-slate-950 to-emerald-950/30 p-3.5 sm:p-4 rounded-2xl border border-emerald-500/40">
          <div className="flex items-center justify-between">
            <span className="text-[10px] text-emerald-400 block font-mono font-bold uppercase">★ Headline PR-AUC</span>
            <span className="px-1.5 py-0.2 rounded text-[9px] font-mono bg-emerald-950 text-emerald-300 border border-emerald-800">
              Invariant
            </span>
          </div>
          <span className="text-xl sm:text-2xl font-black font-mono text-emerald-400 mt-1 block">
            {formatMetric(metrics.pr_auc, 4, 'N/A')}
          </span>
          <span className="text-[10px] text-slate-400 font-mono mt-1 block">
            {metrics.pr_auc_lift ? `${metrics.pr_auc_lift.toFixed(1)}x Lift vs Prevalence` : 'Ranking Invariant'}
          </span>
        </div>

        <div className="bg-slate-950 p-3.5 sm:p-4 rounded-2xl border border-slate-800">
          <div className="flex items-center justify-between">
            <span className="text-[10px] text-slate-400 block font-mono uppercase">ROC-AUC</span>
            <span className="px-1.5 py-0.2 rounded text-[9px] font-mono bg-slate-900 text-slate-400 border border-slate-800">
              Invariant
            </span>
          </div>
          <span className="text-xl sm:text-2xl font-bold font-mono text-indigo-400 mt-1 block">
            {formatMetric(metrics.roc_auc, 4, 'N/A')}
          </span>
          <span className="text-[10px] text-slate-500 font-mono mt-1 block">Discrimination Power</span>
        </div>

        <div className="bg-slate-950 p-3.5 sm:p-4 rounded-2xl border border-slate-800">
          <span className="text-[10px] text-slate-400 block font-mono uppercase">Brier Calibration Score</span>
          <span className="text-xl sm:text-2xl font-bold font-mono text-cyan-400 mt-1 block">
            {formatMetric(metrics.brier_score, 6, '0.000000')}
          </span>
          <span className="text-[10px] text-slate-500 font-mono mt-1 block">Prob. Mean Squared Error</span>
        </div>

        <div className="bg-slate-950 p-3.5 sm:p-4 rounded-2xl border border-slate-800">
          <span className="text-[10px] text-slate-400 block font-mono uppercase">Alert Volume Rate</span>
          <span className="text-xl sm:text-2xl font-bold font-mono text-amber-400 mt-1 block">
            {activeOp ? ((activeOp.alerts_per_10k) / 10).toFixed(1) : ((metrics.alerts_per_10k) / 10).toFixed(1)} / 1k tx
          </span>
          <span className="text-[10px] text-slate-500 font-mono mt-1 block">
            {activeOp ? `${activeOp.alerts?.toLocaleString()} alerts enforced` : 'Investigation Volume'}
          </span>
        </div>
      </div>
    </div>
  );
};
