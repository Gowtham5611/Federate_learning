import React from 'react';
import { WhyNoAlerts } from '../types/api';
import { Info, Sliders, TrendingUp, Cpu } from 'lucide-react';

interface WhyNoAlertsCardProps {
  whyNoAlerts?: WhyNoAlerts | null;
  alertCount: number;
  threshold: number;
  onSelectOperatingPoint?: (op: 'frozen' | 'budget_10' | 'budget_50' | 'budget_100' | 'best_f1') => void;
  onApplyLogitShift?: (delta: number) => void;
  onRunFallbackModel?: () => void;
}

export const WhyNoAlertsCard: React.FC<WhyNoAlertsCardProps> = ({
  whyNoAlerts,
  alertCount,
  threshold,
  onSelectOperatingPoint,
  onApplyLogitShift,
  onRunFallbackModel,
}) => {
  if (alertCount > 5 && !whyNoAlerts) return null;

  const maxScore = whyNoAlerts?.max_score ?? 0;
  const pctGe3 = whyNoAlerts?.pct_rows_ge_3_prior ?? 0;
  const shiftDelta = whyNoAlerts?.logit_shift_delta ?? (whyNoAlerts as any)?.prevalence_shift?.delta_logit ?? null;

  return (
    <div className="bg-slate-900/90 border border-amber-500/40 rounded-2xl p-5 shadow-lg space-y-4">
      <div className="flex items-start gap-2.5">
        <Info className="w-5 h-5 text-amber-400 shrink-0 mt-0.5" />
        <div className="space-y-1">
          <h3 className="text-base font-semibold text-white">Model Behaviour on This Dataset</h3>
          <p className="text-sm text-slate-300">
            {alertCount === 0
              ? `No transactions exceeded θ = ${threshold.toFixed(4)} (max score ${maxScore.toFixed(4)}).`
              : `${alertCount} transactions exceeded θ = ${threshold.toFixed(4)} (max score ${maxScore.toFixed(4)}).`}
          </p>
          <p className="text-xs text-slate-400">
            Sequence depth: {pctGe3.toFixed(1)}% of rows have &ge;3 prior transactions; try an alternative view below if needed.
          </p>
        </div>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-3 gap-3">
        <button
          type="button"
          onClick={() => onSelectOperatingPoint && onSelectOperatingPoint('budget_10')}
          className="p-3.5 min-h-[44px] rounded-xl bg-slate-950 hover:bg-indigo-950/40 border border-slate-800 hover:border-indigo-600/60 transition text-left cursor-pointer group"
        >
          <div className="flex items-center space-x-2 text-indigo-400 group-hover:text-indigo-300 font-semibold text-xs">
            <Sliders className="w-4 h-4 shrink-0" />
            <span>Fixed Alert-Budget View</span>
          </div>
          <p className="text-xs text-slate-400 mt-1">Top 10 alerts per 10k transactions.</p>
        </button>

        <button
          type="button"
          onClick={() => {
            if (shiftDelta && onApplyLogitShift) {
              onApplyLogitShift(shiftDelta);
            } else if (onSelectOperatingPoint) {
              onSelectOperatingPoint('best_f1');
            }
          }}
          className="p-3.5 min-h-[44px] rounded-xl bg-slate-950 hover:bg-cyan-950/40 border border-slate-800 hover:border-cyan-600/60 transition text-left cursor-pointer group"
        >
          <div className="flex items-center space-x-2 text-cyan-400 group-hover:text-cyan-300 font-semibold text-xs">
            <TrendingUp className="w-4 h-4 shrink-0" />
            <span>Prevalence-Adjusted View</span>
          </div>
          <p className="text-xs text-slate-400 mt-1">
            Calibrate scores{shiftDelta ? ` (Δ = ${shiftDelta.toFixed(2)})` : ' for file prevalence'}.
          </p>
        </button>

        <button
          type="button"
          onClick={() => onRunFallbackModel && onRunFallbackModel()}
          className="p-3.5 min-h-[44px] rounded-xl bg-slate-950 hover:bg-violet-950/40 border border-slate-800 hover:border-violet-600/60 transition text-left cursor-pointer group"
        >
          <div className="flex items-center space-x-2 text-violet-400 group-hover:text-violet-300 font-semibold text-xs">
            <Cpu className="w-4 h-4 shrink-0" />
            <span>Tabular Fallback Model</span>
          </div>
          <p className="text-xs text-slate-400 mt-1">Non-recurrent classifier for single-row accounts.</p>
        </button>
      </div>
    </div>
  );
};
