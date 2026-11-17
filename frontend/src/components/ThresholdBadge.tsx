import React from 'react';
import { formatThreshold } from '../utils/formatters';
import { Info } from 'lucide-react';

export type ThresholdType = 'frozen_validation' | 'ensemble_tuned' | 'custom_operational' | 'capacity_constrained';

interface ThresholdBadgeProps {
  value: number;
  type?: ThresholdType;
  label?: string;
  className?: string;
}

export const ThresholdBadge: React.FC<ThresholdBadgeProps> = ({
  value,
  type = 'frozen_validation',
  label,
  className = '',
}) => {
  const displayLabel = label || (
    type === 'frozen_validation'
      ? 'Frozen validation threshold'
      : type === 'ensemble_tuned'
      ? 'Ensemble-tuned threshold'
      : type === 'capacity_constrained'
      ? 'Capacity policy threshold'
      : 'Operational threshold'
  );

  const tooltipText = type === 'frozen_validation'
    ? 'Frozen validation threshold: Selected on validation split during model training and frozen permanently to protect against test set data leakage.'
    : type === 'ensemble_tuned'
    ? 'Ensemble-tuned threshold: Selected via validation PR-AUC during Phase 7 hybrid grid search. Frozen prior to test evaluation.'
    : 'Active decision threshold: Probability cutoff applied to model risk scores (p ≥ θ flags transaction).';

  return (
    <span
      className={`inline-flex items-center space-x-1.5 px-2 py-0.5 rounded-lg text-xs font-mono font-bold bg-amber-50 dark:bg-amber-950/80 text-amber-800 dark:text-amber-300 border border-amber-200 dark:border-amber-800/60 shadow-sm cursor-help ${className}`}
      title={`${displayLabel}: θ = ${formatThreshold(value)}. ${tooltipText}`}
    >
      <span>θ = {formatThreshold(value)}</span>
      <span className="text-[10px] font-sans font-medium text-amber-700 dark:text-amber-400 opacity-90 hidden sm:inline">
        ({displayLabel})
      </span>
      <Info className="w-3 h-3 text-amber-600 dark:text-amber-400 opacity-70 shrink-0" />
    </span>
  );
};
