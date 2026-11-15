import React from 'react';
import { ShieldAlert, Database, Info, Layers } from 'lucide-react';

interface DemoDisclaimerBannerProps {
  filename?: string;
  totalRows?: number;
  positiveCount?: number;
  negativeCount?: number;
}

export const DemoDisclaimerBanner: React.FC<DemoDisclaimerBannerProps> = ({
  filename = '',
  totalRows = 20000,
  positiveCount,
  negativeCount,
}) => {
  const isTriggerDemo =
    filename.toLowerCase().includes('trigger_demo') ||
    filename.toLowerCase().includes('demo_only') ||
    filename.toLowerCase().includes('model_trigger') ||
    (totalRows === 20000 && positiveCount === 6000);

  if (!isTriggerDemo) {
    return null;
  }

  const pos = positiveCount ?? 6000;
  const neg = negativeCount ?? 14000;
  const total = totalRows || pos + neg;
  const posPct = ((pos / total) * 100).toFixed(1);
  const negPct = ((neg / total) * 100).toFixed(1);

  return (
    <section
      aria-label="Synthetic Demonstration Scope Notice"
      className="bg-gradient-to-br from-amber-950/40 via-slate-900 to-indigo-950/30 border border-amber-500/40 rounded-3xl p-5 sm:p-6 shadow-xl space-y-4"
    >
      {/* Header Strip */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 border-b border-amber-800/40 pb-3.5">
        <div className="flex items-center space-x-2.5">
          <div className="p-2 rounded-xl bg-amber-500/20 text-amber-300 border border-amber-500/30 shrink-0">
            <ShieldAlert className="w-5 h-5 text-amber-400" />
          </div>
          <div>
            <div className="flex flex-wrap items-center gap-2">
              <h2 className="text-sm sm:text-base font-black text-white tracking-tight">
                Purpose-Built Synthetic Threshold-Trigger Demonstration
              </h2>
              <span className="px-2.5 py-0.5 rounded-full text-[10px] font-mono font-bold bg-amber-500/20 text-amber-300 border border-amber-500/40">
                DEMO ONLY • NOT A BENCHMARK
              </span>
            </div>
            <p className="text-xs text-amber-200/80 mt-0.5 font-medium">
              Sensitivity test for frozen sequence LSTM (<code>hi_small_lstm_fedavg_iid_seed42</code>) at canonical threshold $\tau^* = 0.3378$.
            </p>
          </div>
        </div>

        <div className="flex items-center space-x-2 shrink-0">
          <span className="px-3 py-1 rounded-xl bg-slate-950/80 border border-slate-800 text-[11px] font-mono text-slate-300">
            Evaluated File: <strong className="text-amber-300">{filename || 'threshold_trigger_demo_only.csv'}</strong>
          </span>
        </div>
      </div>

      {/* Composition & Distinction Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4 text-xs">
        {/* Left: File Composition */}
        <div className="bg-slate-950/70 p-4 rounded-2xl border border-slate-800/80 space-y-2.5">
          <div className="flex items-center justify-between font-mono">
            <span className="font-bold text-slate-200 flex items-center space-x-1.5">
              <Layers className="w-3.5 h-3.5 text-indigo-400" />
              <span>Exact Scenario Label Composition</span>
            </span>
            <span className="text-indigo-400 font-semibold">{total.toLocaleString()} total rows</span>
          </div>

          <div className="grid grid-cols-2 gap-2 text-[11px] font-mono">
            <div className="p-2.5 rounded-xl bg-slate-900 border border-slate-800">
              <span className="text-slate-400 block text-[10px] uppercase">Legitimate (Is Laundering = 0)</span>
              <span className="text-base font-bold text-emerald-400 mt-0.5 block">
                {neg.toLocaleString()} <span className="text-[10px] text-slate-400 font-normal">({negPct}%)</span>
              </span>
              <span className="text-[9px] text-slate-500 block mt-0.5">12,420 Background + 1,580 Warmup</span>
            </div>

            <div className="p-2.5 rounded-xl bg-slate-900 border border-slate-800">
              <span className="text-slate-400 block text-[10px] uppercase">Laundering (Is Laundering = 1)</span>
              <span className="text-base font-bold text-rose-400 mt-0.5 block">
                {pos.toLocaleString()} <span className="text-[10px] text-slate-400 font-normal">({posPct}%)</span>
              </span>
              <span className="text-[9px] text-slate-500 block mt-0.5">Layering, Cycles, Fan-Out, Fan-In</span>
            </div>
          </div>

          <p className="text-[11px] text-slate-400 leading-relaxed">
            Ground truth was determined purely from pattern generator parameters (layering hops, cycle loops, fan-in smurfs) before inference. No labels were derived from model prediction scores.
          </p>
        </div>

        {/* Right: Methodological Safeguards & Benchmark Boundary */}
        <div className="bg-slate-950/70 p-4 rounded-2xl border border-slate-800/80 space-y-2.5">
          <div className="flex items-center space-x-1.5 font-mono text-amber-300 font-bold">
            <Info className="w-3.5 h-3.5 shrink-0 text-amber-400" />
            <span>Methodological Boundary & Benchmark Reference</span>
          </div>

          <div className="space-y-2 text-[11px] text-slate-300 leading-relaxed">
            <p>
              <strong className="text-white">Sensitivity Verification:</strong> This demo verifies whether recurring chronological sequences activate the sequence model above 0.3378 without changing model weights or thresholds. High precision (99.21%) reflects scenario alignment and must <strong className="text-amber-300">not</strong> be cited as general real-world performance.
            </p>
            <div className="p-2 rounded-xl bg-indigo-950/50 border border-indigo-800/50 text-indigo-200">
              <div className="flex items-center justify-between text-[11px] font-mono">
                <span className="font-bold text-white flex items-center space-x-1">
                  <Database className="w-3 h-3 text-indigo-400" />
                  <span>Official Out-of-Sample Benchmark:</span>
                </span>
                <span className="text-indigo-300 font-semibold">IBM AML HI-Small</span>
              </div>
              <p className="text-[10px] text-slate-400 mt-1 font-mono">
                1,015,564 rows • 698 TP, 413 FP • 62.83% Prec, 38.84% Rec • PR-AUC 0.4760, ROC-AUC 0.9555
              </p>
            </div>
          </div>
        </div>
      </div>
    </section>
  );
};
