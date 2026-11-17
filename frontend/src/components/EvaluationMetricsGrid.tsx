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
        <div
          title="Precision: Proportion of generated alerts that are confirmed money laundering cases (TP / (TP + FP))."
          className="bg-white dark:bg-slate-950 p-3.5 sm:p-4 rounded-2xl border border-slate-200 dark:border-slate-800 shadow-sm flex flex-col justify-between cursor-help transition-colors"
        >
          <div>
            <span className="text-[11px] text-slate-500 dark:text-slate-400 block font-mono font-medium">Precision</span>
            <span className="text-xl sm:text-2xl font-black font-mono text-slate-900 dark:text-white mt-1 block tracking-tight">
              {precision !== null && precision !== undefined ? formatPercent(precision, 2) : '0 Alerts'}
            </span>
          </div>
          <span className="text-[10px] text-slate-400 dark:text-slate-500 font-mono mt-2 block pt-1 border-t border-slate-100 dark:border-slate-900">
            TP / (TP + FP)
          </span>
        </div>

        {/* 2. Recall / Sensitivity */}
        <div
          title="Recall / Sensitivity: Proportion of all true laundering cases captured by the model at this decision threshold (TP / (TP + FN))."
          className="bg-white dark:bg-slate-950 p-3.5 sm:p-4 rounded-2xl border border-slate-200 dark:border-slate-800 shadow-sm flex flex-col justify-between cursor-help transition-colors"
        >
          <div>
            <span className="text-[11px] text-slate-500 dark:text-slate-400 block font-mono font-medium">Recall / Sens.</span>
            <span className="text-xl sm:text-2xl font-black font-mono text-indigo-600 dark:text-indigo-300 mt-1 block tracking-tight">
              {formatPercent(recall, 2)}
            </span>
          </div>
          <span className="text-[10px] text-slate-400 dark:text-slate-500 font-mono mt-2 block pt-1 border-t border-slate-100 dark:border-slate-900">
            TP / (TP + FN)
          </span>
        </div>

        {/* 3. F1-Score */}
        <div
          title="F1-Score: Harmonic mean of precision and recall at serving threshold. Balances missed illicit activity against false alarms."
          className="bg-white dark:bg-slate-950 p-3.5 sm:p-4 rounded-2xl border border-slate-200 dark:border-slate-800 shadow-sm flex flex-col justify-between cursor-help transition-colors"
        >
          <div>
            <span className="text-[11px] text-slate-500 dark:text-slate-400 block font-mono font-medium">F1-Score</span>
            <span className="text-xl sm:text-2xl font-black font-mono text-amber-600 dark:text-amber-400 mt-1 block tracking-tight">
              {formatMetric(f1, 4, '0.0000')}
            </span>
          </div>
          <span className="text-[10px] text-slate-400 dark:text-slate-500 font-mono mt-2 block pt-1 border-t border-slate-100 dark:border-slate-900">
            Harmonic Mean
          </span>
        </div>

        {/* 4. Specificity */}
        <div
          title="Specificity: True Negative Rate; proportion of legitimate transactions correctly cleared (TN / (TN + FP))."
          className="bg-white dark:bg-slate-950 p-3.5 sm:p-4 rounded-2xl border border-slate-200 dark:border-slate-800 shadow-sm flex flex-col justify-between cursor-help transition-colors"
        >
          <div>
            <span className="text-[11px] text-slate-500 dark:text-slate-400 block font-mono font-medium">Specificity</span>
            <span className="text-xl sm:text-2xl font-black font-mono text-cyan-600 dark:text-cyan-300 mt-1 block tracking-tight">
              {formatPercent(specificity, 2)}
            </span>
          </div>
          <span className="text-[10px] text-slate-400 dark:text-slate-500 font-mono mt-2 block pt-1 border-t border-slate-100 dark:border-slate-900">
            TN / (TN + FP)
          </span>
        </div>

        {/* 5. Balanced Accuracy */}
        <div
          title="Balanced Accuracy: Arithmetic mean of sensitivity and specificity ((Sens + Spec) / 2), unskewed by extreme class imbalance."
          className="bg-white dark:bg-slate-950 p-3.5 sm:p-4 rounded-2xl border border-slate-200 dark:border-slate-800 shadow-sm flex flex-col justify-between cursor-help transition-colors"
        >
          <div>
            <span className="text-[11px] text-slate-500 dark:text-slate-400 block font-mono font-medium">Balanced Acc</span>
            <span className="text-xl sm:text-2xl font-black font-mono text-teal-600 dark:text-teal-400 mt-1 block tracking-tight">
              {formatPercent(balancedAcc, 2)}
            </span>
          </div>
          <span className="text-[10px] text-slate-400 dark:text-slate-500 font-mono mt-2 block pt-1 border-t border-slate-100 dark:border-slate-900">
            (Sens + Spec) / 2
          </span>
        </div>

        {/* 6. Overall Accuracy */}
        <div
          title="Accuracy: Crude fraction of all transactions correctly classified ((TP + TN) / Total)."
          className="bg-white dark:bg-slate-950 p-3.5 sm:p-4 rounded-2xl border border-slate-200 dark:border-slate-800 shadow-sm flex flex-col justify-between cursor-help transition-colors"
        >
          <div>
            <span className="text-[11px] text-slate-500 dark:text-slate-400 block font-mono font-medium">Accuracy</span>
            <span className="text-xl sm:text-2xl font-black font-mono text-slate-800 dark:text-slate-200 mt-1 block tracking-tight">
              {formatPercent(accuracy, 2)}
            </span>
          </div>
          <span className="text-[10px] text-slate-400 dark:text-slate-500 font-mono mt-2 block pt-1 border-t border-slate-100 dark:border-slate-900">
            (TP + TN) / Total
          </span>
        </div>

        {/* 7. Matthews Corr (MCC) */}
        <div
          title="Matthews Correlation Coefficient (MCC): Quality metric for binary classifications on highly imbalanced data. Ranges [-1, +1]."
          className="bg-white dark:bg-slate-950 p-3.5 sm:p-4 rounded-2xl border border-slate-200 dark:border-slate-800 shadow-sm flex flex-col justify-between cursor-help transition-colors"
        >
          <div>
            <span className="text-[11px] text-slate-500 dark:text-slate-400 block font-mono font-medium">Matthews Corr</span>
            <span className="text-xl sm:text-2xl font-black font-mono text-violet-600 dark:text-violet-300 mt-1 block tracking-tight">
              {formatMetric(mcc, 4, '0.0000')}
            </span>
          </div>
          <span className="text-[10px] text-slate-400 dark:text-slate-500 font-mono mt-2 block pt-1 border-t border-slate-100 dark:border-slate-900">
            Correlation [-1, +1]
          </span>
        </div>

        {/* 8. Cohen's Kappa */}
        <div
          title="Cohen's Kappa: Chance-corrected measure of agreement between predictions and ground-truth labels."
          className="bg-white dark:bg-slate-950 p-3.5 sm:p-4 rounded-2xl border border-slate-200 dark:border-slate-800 shadow-sm flex flex-col justify-between cursor-help transition-colors"
        >
          <div>
            <span className="text-[11px] text-slate-500 dark:text-slate-400 block font-mono font-medium">Cohen's Kappa</span>
            <span className="text-xl sm:text-2xl font-black font-mono text-violet-700 dark:text-violet-400 mt-1 block tracking-tight">
              {formatMetric(kappa, 4, '0.0000')}
            </span>
          </div>
          <span className="text-[10px] text-slate-400 dark:text-slate-500 font-mono mt-2 block pt-1 border-t border-slate-100 dark:border-slate-900">
            Chance-Corrected
          </span>
        </div>
      </div>

      {/* Headline Ranking & Calibration Invariants */}
      <div className="grid grid-cols-2 sm:grid-cols-4 gap-2.5 sm:gap-3 pt-1">
        <div
          title="PR-AUC: Precision-Recall Area Under Curve. Primary ranking power invariant to majority-class size; highest scientific validity in imbalanced AML."
          className="bg-gradient-to-br from-white to-emerald-50/50 dark:from-slate-950 dark:to-emerald-950/30 p-3.5 sm:p-4 rounded-2xl border border-emerald-300 dark:border-emerald-500/40 shadow-sm cursor-help transition-colors"
        >
          <div className="flex items-center justify-between">
            <span className="text-[10px] text-emerald-700 dark:text-emerald-400 block font-mono font-bold uppercase">★ Headline PR-AUC</span>
            <span className="px-1.5 py-0.2 rounded text-[9px] font-mono bg-emerald-100 dark:bg-emerald-950 text-emerald-800 dark:text-emerald-300 border border-emerald-300 dark:border-emerald-800">
              Ranking Invariant
            </span>
          </div>
          <span className="text-xl sm:text-2xl font-black font-mono text-emerald-600 dark:text-emerald-400 mt-1 block">
            {formatMetric(metrics.pr_auc, 4, 'N/A')}
          </span>
          <span className="text-[10px] text-slate-500 dark:text-slate-400 font-mono mt-1 block">
            {metrics.pr_auc_lift ? `${metrics.pr_auc_lift.toFixed(1)}x Lift vs Prevalence` : 'Ranking Invariant'}
          </span>
        </div>

        <div
          title="ROC-AUC: Receiver Operating Characteristic AUC. Probability that a randomly chosen laundering case is ranked above a legitimate transaction."
          className="bg-white dark:bg-slate-950 p-3.5 sm:p-4 rounded-2xl border border-slate-200 dark:border-slate-800 shadow-sm cursor-help transition-colors"
        >
          <div className="flex items-center justify-between">
            <span className="text-[10px] text-slate-500 dark:text-slate-400 block font-mono uppercase">ROC-AUC</span>
            <span className="px-1.5 py-0.2 rounded text-[9px] font-mono bg-slate-100 dark:bg-slate-900 text-slate-600 dark:text-slate-400 border border-slate-200 dark:border-slate-800">
              Ranking Invariant
            </span>
          </div>
          <span className="text-xl sm:text-2xl font-bold font-mono text-indigo-600 dark:text-indigo-400 mt-1 block">
            {formatMetric(metrics.roc_auc, 4, 'N/A')}
          </span>
          <span className="text-[10px] text-slate-400 dark:text-slate-500 font-mono mt-1 block">Discrimination Power</span>
        </div>

        <div
          title="Brier Calibration Score: Mean squared difference between predicted risk probabilities and actual binary outcomes; measures probability calibration."
          className="bg-white dark:bg-slate-950 p-3.5 sm:p-4 rounded-2xl border border-slate-200 dark:border-slate-800 shadow-sm cursor-help transition-colors"
        >
          <span className="text-[10px] text-slate-500 dark:text-slate-400 block font-mono uppercase">Brier Calibration Score</span>
          <span className="text-xl sm:text-2xl font-bold font-mono text-cyan-600 dark:text-cyan-400 mt-1 block">
            {formatMetric(metrics.brier_score, 6, '0.000000')}
          </span>
          <span className="text-[10px] text-slate-400 dark:text-slate-500 font-mono mt-1 block">Prob. Mean Squared Error</span>
        </div>

        <div
          title="Alert Volume Density: Investigator workload volume density evaluated against capacity policies; reflects operational volume without intrinsic detection judgment."
          className="bg-white dark:bg-slate-950 p-3.5 sm:p-4 rounded-2xl border border-slate-200 dark:border-slate-800 shadow-sm cursor-help transition-colors"
        >
          <span className="text-[10px] text-slate-500 dark:text-slate-400 block font-mono uppercase">Alert Volume Rate</span>
          <span className="text-xl sm:text-2xl font-bold font-mono text-amber-600 dark:text-amber-400 mt-1 block">
            {activeOp ? ((activeOp.alerts_per_10k) / 10).toFixed(1) : ((metrics.alerts_per_10k) / 10).toFixed(1)} / 1k tx
          </span>
          <span className="text-[10px] text-slate-400 dark:text-slate-500 font-mono mt-1 block">
            {activeOp ? `${activeOp.alerts?.toLocaleString()} alerts enforced` : 'Investigation Volume'}
          </span>
        </div>
      </div>
    </div>
  );
};
