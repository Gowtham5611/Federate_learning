import React from 'react';

export type SemanticLegendType = 'models' | 'triage' | 'ensemble';

interface SemanticLegendProps {
  type?: SemanticLegendType;
  className?: string;
}

export const SemanticLegend: React.FC<SemanticLegendProps> = ({
  type = 'models',
  className = '',
}) => {
  if (type === 'triage') {
    return (
      <div className={`flex flex-wrap items-center justify-center gap-5 text-[11px] font-mono border-t border-slate-200 dark:border-slate-800 pt-3 ${className}`}>
        <span className="flex items-center space-x-1.5 text-rose-700 dark:text-rose-400 font-bold">
          <span className="w-2.5 h-2.5 bg-rose-500 rounded-full inline-block" />
          <span>Flagged Alert (p ≥ θ)</span>
        </span>
        <span className="flex items-center space-x-1.5 text-amber-700 dark:text-amber-400 font-bold">
          <span className="w-2.5 h-2.5 bg-amber-500 rounded-full inline-block" />
          <span>Needs Review / FN</span>
        </span>
        <span className="flex items-center space-x-1.5 text-emerald-700 dark:text-emerald-400 font-medium">
          <span className="w-2.5 h-2.5 bg-emerald-500 rounded-full inline-block" />
          <span>Cleared Normal (p &lt; θ)</span>
        </span>
        <span className="flex items-center space-x-1.5 text-purple-700 dark:text-purple-400 font-medium">
          <span className="w-2.5 h-2.5 bg-purple-500 rounded-full inline-block" />
          <span>CSV Ground Truth (1)</span>
        </span>
      </div>
    );
  }

  if (type === 'ensemble') {
    return (
      <div className={`flex flex-wrap items-center justify-center gap-5 text-[11px] font-mono border-t border-slate-200 dark:border-slate-800 pt-3 ${className}`}>
        <span className="flex items-center space-x-1.5 text-emerald-700 dark:text-emerald-400 font-bold">
          <span className="w-2.5 h-2.5 bg-emerald-500 rounded-full inline-block" />
          <span>Serving Champion (IID)</span>
        </span>
        <span className="flex items-center space-x-1.5 text-violet-700 dark:text-violet-400 font-bold">
          <span className="w-2.5 h-2.5 bg-violet-600 rounded-full inline-block" />
          <span>Optimized Ensemble (α*)</span>
        </span>
        <span className="flex items-center space-x-1.5 text-indigo-700 dark:text-indigo-400 font-medium">
          <span className="w-2.5 h-2.5 bg-indigo-500 rounded-full inline-block" />
          <span>Non-IID Partner</span>
        </span>
      </div>
    );
  }

  return (
    <div className={`flex flex-wrap items-center justify-center gap-5 text-[11px] font-mono border-t border-slate-200 dark:border-slate-800 pt-3 ${className}`}>
      <span className="flex items-center space-x-1.5 text-emerald-700 dark:text-emerald-400 font-bold">
        <span className="w-2.5 h-2.5 bg-emerald-500 rounded-full inline-block" />
        <span>Serving Champion</span>
      </span>
      <span className="flex items-center space-x-1.5 text-purple-700 dark:text-purple-400 font-medium">
        <span className="w-2.5 h-2.5 bg-purple-500 rounded-full inline-block" />
        <span>FedAvg (IID)</span>
      </span>
      <span className="flex items-center space-x-1.5 text-indigo-700 dark:text-indigo-400 font-medium">
        <span className="w-2.5 h-2.5 bg-indigo-500 rounded-full inline-block" />
        <span>FedAvg (Non-IID)</span>
      </span>
      <span className="flex items-center space-x-1.5 text-amber-700 dark:text-amber-400 font-medium">
        <span className="w-2.5 h-2.5 bg-amber-500 rounded-full inline-block" />
        <span>FedProx (μ=0.0001)</span>
      </span>
      <span className="flex items-center space-x-1.5 text-sky-700 dark:text-sky-400 font-medium">
        <span className="w-2.5 h-2.5 bg-sky-500 rounded-full inline-block" />
        <span>Centralized Baseline</span>
      </span>
    </div>
  );
};
