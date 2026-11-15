import React from 'react';
import { WhyNoAlerts } from '../types/api';
import { HelpCircle, Sliders, TrendingUp, Cpu } from 'lucide-react';

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
  const meanScore = whyNoAlerts?.mean_score ?? 0;
  const pctGe3 = whyNoAlerts?.pct_rows_ge_3_prior ?? 0;
  const uniqueSenderRatio = whyNoAlerts?.unique_sender_ratio ?? 1.0;
  const uploadedPrev = whyNoAlerts?.uploaded_positive_prevalence ?? (whyNoAlerts as any)?.prevalence;
  const trainingPrior = whyNoAlerts?.training_prior ?? 0.0018;
  const shiftDelta = whyNoAlerts?.logit_shift_delta ?? (whyNoAlerts as any)?.prevalence_shift?.delta_logit ?? null;

  return (
    <div className="bg-slate-900/90 border border-amber-500/40 rounded-3xl p-6 shadow-xl space-y-4">
      <div className="flex items-center space-x-2.5 border-b border-slate-800 pb-3">
        <HelpCircle className="w-5 h-5 text-amber-400 shrink-0" />
        <div>
          <h3 className="text-base font-bold text-white">
            {alertCount === 0 ? 'Why did the model produce zero alerts?' : `Why did the model produce only ${alertCount} alerts?`}
          </h3>
          <p className="text-xs text-slate-400">
            Real diagnosis of model scoring dynamics, account history depth, and distribution shifts.
          </p>
        </div>
      </div>

      {/* Real Diagnostic Numbers Grid */}
      <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-6 gap-3 font-mono text-xs">
        <div className="bg-slate-950 p-3 rounded-xl border border-slate-800">
          <span className="text-[10px] text-slate-400 uppercase block">Serving Threshold</span>
          <span className="text-base font-bold text-amber-400 mt-1 block">θ = {threshold.toFixed(4)}</span>
        </div>
        <div className="bg-slate-950 p-3 rounded-xl border border-slate-800">
          <span className="text-[10px] text-slate-400 uppercase block">Max Output Score</span>
          <span className="text-base font-bold text-rose-400 mt-1 block">{maxScore.toFixed(4)}</span>
        </div>
        <div className="bg-slate-950 p-3 rounded-xl border border-slate-800">
          <span className="text-[10px] text-slate-400 uppercase block">Mean Output Score</span>
          <span className="text-base font-bold text-slate-300 mt-1 block">{meanScore.toFixed(4)}</span>
        </div>
        <div className="bg-slate-950 p-3 rounded-xl border border-slate-800">
          <span className="text-[10px] text-slate-400 uppercase block">≥3 Prior Tx Rows</span>
          <span className="text-base font-bold text-cyan-400 mt-1 block">{pctGe3.toFixed(1)}%</span>
        </div>
        <div className="bg-slate-950 p-3 rounded-xl border border-slate-800">
          <span className="text-[10px] text-slate-400 uppercase block">Unique Senders</span>
          <span className="text-base font-bold text-indigo-300 mt-1 block">{(uniqueSenderRatio * 100).toFixed(1)}%</span>
        </div>
        <div className="bg-slate-950 p-3 rounded-xl border border-slate-800">
          <span className="text-[10px] text-slate-400 uppercase block">Prevalence Ratio</span>
          <span className="text-base font-bold text-emerald-400 mt-1 block">
            {uploadedPrev !== null && uploadedPrev !== undefined
              ? `${(uploadedPrev / trainingPrior).toFixed(0)}x Prior`
              : 'Unlabeled'}
          </span>
        </div>
      </div>

      {/* Root Cause Explanation */}
      <div className="bg-slate-950 p-4 rounded-xl border border-slate-800 text-xs text-slate-300 leading-relaxed space-y-1">
        <p className="font-semibold text-amber-300">Root Cause Analysis:</p>
        <p>
          {whyNoAlerts?.explanation ||
            `All transactions in this file scored below the canonical threshold θ = ${threshold.toFixed(4)} (highest score was ${maxScore.toFixed(4)}). FedSecureAI recurrent sequence models require multi-step transaction history per sending account to recognize complex laundering patterns. In single-transaction files (or unique sender streams), models default to the baseline prior probability (~0.001 - 0.015).`}
        </p>
      </div>

      {/* 3 One-Click Working Alternative Actions */}
      <div className="space-y-2 pt-1">
        <span className="text-xs font-mono uppercase text-slate-400 font-bold block">
          One-Click Working Alternatives:
        </span>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-3">
          {/* Action 1: Alert-Budget View */}
          <button
            type="button"
            onClick={() => onSelectOperatingPoint && onSelectOperatingPoint('budget_10')}
            className="p-3.5 rounded-xl bg-slate-950 hover:bg-indigo-950/40 border border-slate-800 hover:border-indigo-600/60 transition text-left cursor-pointer group"
          >
            <div className="flex items-center space-x-2 text-indigo-400 group-hover:text-indigo-300 font-bold text-xs">
              <Sliders className="w-4 h-4 shrink-0" />
              <span>1. Fixed Alert-Budget View</span>
            </div>
            <p className="text-[11px] text-slate-400 mt-1 leading-snug">
              Evaluate top-ranked transactions calibrated to investigate 10 alerts per 10k transactions.
            </p>
          </button>

          {/* Action 2: Prevalence-Adjusted Shift */}
          <button
            type="button"
            onClick={() => {
              if (shiftDelta && onApplyLogitShift) {
                onApplyLogitShift(shiftDelta);
              } else if (onSelectOperatingPoint) {
                onSelectOperatingPoint('best_f1');
              }
            }}
            className="p-3.5 rounded-xl bg-slate-950 hover:bg-cyan-950/40 border border-slate-800 hover:border-cyan-600/60 transition text-left cursor-pointer group"
          >
            <div className="flex items-center space-x-2 text-cyan-400 group-hover:text-cyan-300 font-bold text-xs">
              <TrendingUp className="w-4 h-4 shrink-0" />
              <span>2. Prevalence-Adjusted View</span>
            </div>
            <p className="text-[11px] text-slate-400 mt-1 leading-snug">
              Adjust logit scores by Δ = {shiftDelta ? shiftDelta.toFixed(2) : 'logit shift'} to match uploaded positive prevalence.
            </p>
          </button>

          {/* Action 3: Tabular Fallback Model */}
          <button
            type="button"
            onClick={() => onRunFallbackModel && onRunFallbackModel()}
            className="p-3.5 rounded-xl bg-slate-950 hover:bg-violet-950/40 border border-slate-800 hover:border-violet-600/60 transition text-left cursor-pointer group"
          >
            <div className="flex items-center space-x-2 text-violet-400 group-hover:text-violet-300 font-bold text-xs">
              <Cpu className="w-4 h-4 shrink-0" />
              <span>3. Tabular Fallback Model</span>
            </div>
            <p className="text-[11px] text-slate-400 mt-1 leading-snug">
              Evaluate with non-recurrent gradient boosted classifier designed for zero-history rows.
            </p>
          </button>
        </div>
      </div>
    </div>
  );
};
