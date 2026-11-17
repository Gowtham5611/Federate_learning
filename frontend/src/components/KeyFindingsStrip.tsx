import React, { useState } from 'react';
import { ShieldCheck, Scale, Users, ChevronDown, ChevronUp, Sparkles } from 'lucide-react';

interface KeyFindingsStripProps {
  className?: string;
}

export const KeyFindingsStrip: React.FC<KeyFindingsStripProps> = ({
  className = '',
}) => {
  const [collapsed, setCollapsed] = useState<boolean>(false);

  return (
    <section
      aria-label="Executive Key Findings"
      className={`rounded-2xl border border-indigo-200 dark:border-indigo-900/60 bg-gradient-to-r from-indigo-50/90 via-slate-50 to-purple-50/80 dark:from-slate-900 dark:via-indigo-950/30 dark:to-slate-900 shadow-md p-4 space-y-3 transition-colors ${className}`}
    >
      <div className="flex items-center justify-between gap-3">
        <div className="flex items-center space-x-2">
          <div className="w-7 h-7 rounded-lg bg-indigo-600 text-white flex items-center justify-center shadow-sm">
            <Sparkles className="w-4 h-4" />
          </div>
          <div>
            <h2 className="text-xs font-bold uppercase tracking-wider text-indigo-900 dark:text-indigo-300 font-mono flex items-center gap-2">
              <span>Executive Summary: Core AML Research Findings</span>
              <span className="text-[10px] px-2 py-0.5 rounded-full bg-indigo-100 dark:bg-indigo-900/80 text-indigo-700 dark:text-indigo-300 font-sans font-semibold">
                Peer-Review Audited
              </span>
            </h2>
            <p className="text-[11px] text-slate-600 dark:text-slate-400">
              Key empirical results from IBM AML benchmark evaluation across 27 federated & centralized configurations.
            </p>
          </div>
        </div>

        <div className="flex items-center space-x-2 shrink-0">
          <button
            type="button"
            onClick={() => setCollapsed((prev) => !prev)}
            className="p-1.5 rounded-lg border border-slate-200 dark:border-slate-800 text-slate-500 hover:text-slate-900 dark:hover:text-white transition cursor-pointer focus-visible:ring-2 focus-visible:ring-indigo-500 outline-none"
            title={collapsed ? 'Expand findings strip' : 'Collapse findings strip'}
            aria-expanded={!collapsed}
          >
            {collapsed ? <ChevronDown className="w-4 h-4" /> : <ChevronUp className="w-4 h-4" />}
          </button>
        </div>
      </div>

      {!collapsed && (
        <div className="grid grid-cols-1 md:grid-cols-3 gap-3 pt-1">
          {/* Finding 1: Champion Retention */}
          <article className="bg-white/80 dark:bg-slate-900/80 border border-slate-200 dark:border-slate-800/80 rounded-xl p-3 shadow-sm space-y-1">
            <div className="flex items-center justify-between text-xs">
              <span className="font-semibold text-slate-700 dark:text-slate-300 flex items-center space-x-1.5">
                <ShieldCheck className="w-3.5 h-3.5 text-emerald-600 dark:text-emerald-400 shrink-0" />
                <span>Champion Retention</span>
              </span>
              <span className="font-mono font-bold text-emerald-600 dark:text-emerald-400">99.77%</span>
            </div>
            <p className="text-[11px] text-slate-600 dark:text-slate-400 leading-relaxed">
              Federated IID LSTM retains 99.77% of centralized baseline PR-AUC (0.4760 vs 0.4771) with zero raw transaction data movement across client partitions.
            </p>
          </article>

          {/* Finding 2: Hybrid Ensemble Finding */}
          <article className="bg-white/80 dark:bg-slate-900/80 border border-slate-200 dark:border-slate-800/80 rounded-xl p-3 shadow-sm space-y-1">
            <div className="flex items-center justify-between text-xs">
              <span className="font-semibold text-slate-700 dark:text-slate-300 flex items-center space-x-1.5">
                <Scale className="w-3.5 h-3.5 text-indigo-600 dark:text-indigo-400 shrink-0" />
                <span>Ensemble Validation</span>
              </span>
              <span className="font-mono font-bold text-indigo-600 dark:text-indigo-400">α* = 1.00 (No Gain)</span>
            </div>
            <p className="text-[11px] text-slate-600 dark:text-slate-400 leading-relaxed">
              Rigorous validation sweep determined that blending with non-IID weights causes negative transfer. The optimizer froze α* = 1.00, preserving standalone champion efficacy.
            </p>
          </article>

          {/* Finding 3: Alert Volume Density vs Capacity */}
          <article className="bg-white/80 dark:bg-slate-900/80 border border-slate-200 dark:border-slate-800/80 rounded-xl p-3 shadow-sm space-y-1">
            <div className="flex items-center justify-between text-xs">
              <span className="font-semibold text-slate-700 dark:text-slate-300 flex items-center space-x-1.5">
                <Users className="w-3.5 h-3.5 text-amber-600 dark:text-amber-400 shrink-0" />
                <span>Workload Density</span>
              </span>
              <span className="font-mono font-bold text-amber-600 dark:text-amber-400">6.42 / 10k tx</span>
            </div>
            <p className="text-[11px] text-slate-600 dark:text-slate-400 leading-relaxed">
              Frozen operating threshold (θ = 0.3378) controls alert volume to 6.42 per 10k transactions, aligning with FIU analyst queue capacity while preventing alert fatigue.
            </p>
          </article>
        </div>
      )}
    </section>
  );
};
