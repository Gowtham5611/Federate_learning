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
      ? 'Validation threshold'
      : type === 'ensemble_tuned'
      ? 'Ensemble threshold'
      : type === 'capacity_constrained'
      ? 'Capacity threshold'
      : 'Operational threshold'
  );

  const tooltipText = type === 'frozen_validation'
    ? 'Selected on the validation split and held fixed during evaluation.'
    : type === 'ensemble_tuned'
    ? 'Selected via validation PR-AUC during grid search.'
    : 'Probability cutoff applied to model risk scores (p ≥ θ flags transaction).';

  return (
    <span
      className={`inline-flex items-center space-x-1 px-2 py-0.5 rounded-lg text-xs font-mono font-bold bg-amber-50 dark:bg-amber-950/80 text-amber-800 dark:text-amber-300 border border-amber-200 dark:border-amber-800/60 shadow-sm cursor-help ${className}`}
      title={`${displayLabel}: θ = ${formatThreshold(value)}. ${tooltipText}`}
    >
      <span>θ = {formatThreshold(value)}</span>
      <Info className="w-3 h-3 text-amber-600 dark:text-amber-400 opacity-75 shrink-0" />
    </span>
  );
};
