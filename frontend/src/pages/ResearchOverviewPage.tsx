import React from 'react';
import { HealthResponse, ResearchOverviewResponse, SupportedDataset } from '../types/api';
import {
  ShieldCheck,
  Database,
  Lock,
  ArrowRight,
  BarChart2,
  FileSpreadsheet,
  TrendingDown,
} from 'lucide-react';

interface ResearchOverviewPageProps {
  overview: ResearchOverviewResponse | null;
  health?: HealthResponse | null;
  loading?: boolean;
  onSelectDataset: (dataset: SupportedDataset) => void;
  onNavigateTab: (tab: any) => void;
}

export const ResearchOverviewPage: React.FC<ResearchOverviewPageProps> = ({
  overview,
  onSelectDataset,
  onNavigateTab,
}) => {
  const hiStats = overview?.verified_datasets['HI-Small'];
  const liStats = overview?.verified_datasets['LI-Medium'];

  return (
    <div className="space-y-8 animate-fade-in">
      {/* Hero Banner */}
      <div className="relative overflow-hidden rounded-3xl bg-gradient-to-br from-slate-900 via-indigo-950/40 to-slate-900 border border-slate-800 p-8 shadow-2xl">
        <div className="relative z-10 max-w-4xl space-y-4">
          <div className="inline-flex items-center space-x-2 px-3 py-1 rounded-full bg-indigo-950/80 border border-indigo-500/40 text-indigo-300 text-xs font-mono font-medium">
            <Lock className="w-3.5 h-3.5 text-indigo-400" />
            <span>Cross-Silo Privacy-Preserving Banking Network • 27 Verified Benchmarks</span>
          </div>

          <h1 className="text-3xl sm:text-4xl lg:text-5xl font-black text-white tracking-tight leading-tight">
            FedSecureAI <span className="text-transparent bg-clip-text bg-gradient-to-r from-indigo-400 to-violet-400">Research Platform</span>
          </h1>

          <p className="text-slate-300 text-sm sm:text-base leading-relaxed max-w-3xl">
            A privacy-preserving, explainable, and drift-aware federated anti-money laundering detection system.
            Evaluated on IBM AML benchmarks across 36+ million real transactions. Addresses extreme class imbalance,
            non-IID bank typologies, and complex temporal money dispersion graph topologies without raw data centralization.
          </p>

          <div className="pt-2 flex flex-wrap items-center gap-3">
            <button
              onClick={() => onNavigateTab('csv-eval')}
              className="flex items-center space-x-2 px-5 py-2.5 rounded-xl bg-indigo-600 hover:bg-indigo-500 text-white font-medium text-sm transition shadow-lg shadow-indigo-600/30"
            >
              <FileSpreadsheet className="w-4 h-4" />
              <span>Launch CSV Batch Evaluator</span>
              <ArrowRight className="w-4 h-4" />
            </button>
            <button
              onClick={() => onNavigateTab('registry')}
              className="flex items-center space-x-2 px-5 py-2.5 rounded-xl bg-slate-800 hover:bg-slate-700 text-slate-200 font-medium text-sm border border-slate-700 transition"
            >
              <Database className="w-4 h-4 text-indigo-400" />
              <span>Explore 27 Verified Models</span>
            </button>
            <button
              onClick={() => onNavigateTab('experiments')}
              className="flex items-center space-x-2 px-5 py-2.5 rounded-xl bg-slate-800 hover:bg-slate-700 text-slate-200 font-medium text-sm border border-slate-700 transition"
            >
              <BarChart2 className="w-4 h-4 text-violet-400" />
              <span>Training Convergence Curves</span>
            </button>
          </div>
        </div>
      </div>

      {/* Dataset Pipelines Separation Matrix */}
      <div className="space-y-4">
        <div className="flex items-center justify-between">
          <div>
            <h2 className="text-xl font-bold text-white tracking-tight">Dataset Pipeline Architecture</h2>
            <p className="text-xs text-slate-400 mt-0.5">
              Strict model and data separation: HI-Small and LI-Medium operate as independent verification pipelines.
            </p>
          </div>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          {/* HI-Small Pipeline Card */}
          <div className="bg-slate-900/90 border border-slate-800 rounded-2xl p-6 shadow-xl relative overflow-hidden flex flex-col justify-between">
            <div className="space-y-4">
              <div className="flex items-center justify-between border-b border-slate-800/80 pb-3">
                <div className="flex items-center space-x-3">
                  <div className="w-10 h-10 rounded-xl bg-indigo-950/80 border border-indigo-700/50 flex items-center justify-center text-indigo-400">
                    <Database className="w-5 h-5" />
                  </div>
                  <div>
                    <h3 className="font-bold text-lg text-white">IBM AML HI-Small Pipeline</h3>
                    <span className="text-[11px] font-mono text-indigo-400">High-Imbalance Small Dataset</span>
                  </div>
                </div>
                <span className="px-2.5 py-1 rounded-full text-xs font-mono font-bold bg-indigo-950/80 text-indigo-300 border border-indigo-800">
                  CHAMPION: LSTM
                </span>
              </div>

              <p className="text-xs text-slate-300 leading-relaxed">
                Benchmark for high-imbalance temporal money laundering with 5.08M transactions.
                Temporal windows link up to 10 historical sender transactions into 57-dimensional feature vectors.
              </p>

              {/* Verified Metrics Grid */}
              <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 pt-1">
                <div className="bg-slate-950/60 p-3 rounded-xl border border-slate-800/80">
                  <span className="text-[10px] text-slate-400 uppercase font-mono block">Raw Rows</span>
                  <span className="text-base font-bold font-mono text-white">
                    {hiStats?.raw_transactions.toLocaleString() ?? '5,078,345'}
                  </span>
                </div>
                <div className="bg-slate-950/60 p-3 rounded-xl border border-slate-800/80">
                  <span className="text-[10px] text-slate-400 uppercase font-mono block">Distinct Accounts</span>
                  <span className="text-base font-bold font-mono text-white">
                    {hiStats?.distinct_accounts.toLocaleString() ?? '515,080'}
                  </span>
                </div>
                <div className="bg-slate-950/60 p-3 rounded-xl border border-slate-800/80">
                  <span className="text-[10px] text-slate-400 uppercase font-mono block">Positive Rate</span>
                  <span className="text-base font-bold font-mono text-rose-400">
                    {hiStats ? `${(hiStats.test_positive_prevalence * 100).toFixed(3)}%` : '0.177%'}
                  </span>
                </div>
                <div className="bg-slate-950/60 p-3 rounded-xl border border-slate-800/80">
                  <span className="text-[10px] text-slate-400 uppercase font-mono block">Frozen Threshold</span>
                  <span className="text-base font-bold font-mono text-amber-400">
                    {hiStats?.frozen_threshold.toFixed(4) ?? '0.3378'}
                  </span>
                </div>
              </div>

              {/* Champion Performance Card */}
              <div className="bg-gradient-to-r from-indigo-950/30 to-slate-950/50 p-4 rounded-xl border border-indigo-900/40">
                <div className="flex items-center justify-between text-xs font-mono text-indigo-300 mb-2">
                  <span>Authoritative Test Split Performance (1,015,564 sequences)</span>
                  <ShieldCheck className="w-4 h-4 text-emerald-400" />
                </div>
                <div className="grid grid-cols-4 gap-2 text-center">
                  <div>
                    <span className="text-[10px] text-slate-400 block">PR-AUC</span>
                    <span className="text-base font-bold font-mono text-emerald-400">0.4760</span>
                  </div>
                  <div>
                    <span className="text-[10px] text-slate-400 block">ROC-AUC</span>
                    <span className="text-base font-bold font-mono text-indigo-300">0.9555</span>
                  </div>
                  <div>
                    <span className="text-[10px] text-slate-400 block">F1-Score</span>
                    <span className="text-base font-bold font-mono text-white">0.5359</span>
                  </div>
                  <div>
                    <span className="text-[10px] text-slate-400 block">Precision</span>
                    <span className="text-base font-bold font-mono text-white">0.8183</span>
                  </div>
                </div>
              </div>
            </div>

            <div className="mt-5 pt-4 border-t border-slate-800 flex items-center justify-between">
              <span className="text-[11px] text-slate-400 font-mono">Model: hi_small_lstm_fedavg_iid_seed42</span>
              <button
                onClick={() => {
                  onSelectDataset('HI-Small');
                  onNavigateTab('csv-eval');
                }}
                className="text-xs text-indigo-400 hover:text-indigo-300 font-semibold flex items-center space-x-1"
              >
                <span>Evaluate HI-Small</span>
                <ArrowRight className="w-3.5 h-3.5" />
              </button>
            </div>
          </div>

          {/* LI-Medium Pipeline Card */}
          <div className="bg-slate-900/90 border border-slate-800 rounded-2xl p-6 shadow-xl relative overflow-hidden flex flex-col justify-between">
            <div className="space-y-4">
              <div className="flex items-center justify-between border-b border-slate-800/80 pb-3">
                <div className="flex items-center space-x-3">
                  <div className="w-10 h-10 rounded-xl bg-violet-950/80 border border-violet-700/50 flex items-center justify-center text-violet-400">
                    <Database className="w-5 h-5" />
                  </div>
                  <div>
                    <h3 className="font-bold text-lg text-white">IBM AML LI-Medium Pipeline</h3>
                    <span className="text-[11px] font-mono text-violet-400">Low-Imbalance Medium Scale</span>
                  </div>
                </div>
                <span className="px-2.5 py-1 rounded-full text-xs font-mono font-bold bg-violet-950/80 text-violet-300 border border-violet-800">
                  CHAMPION: BiLSTM
                </span>
              </div>

              <p className="text-xs text-slate-300 leading-relaxed">
                Industrial-scale benchmark spanning 31.25M transactions and 3.1M accounts.
                Captures bidirectional temporal structuring patterns with extreme rarity (only 6 out of 10,000 positive).
              </p>

              {/* Verified Metrics Grid */}
              <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 pt-1">
                <div className="bg-slate-950/60 p-3 rounded-xl border border-slate-800/80">
                  <span className="text-[10px] text-slate-400 uppercase font-mono block">Raw Rows</span>
                  <span className="text-base font-bold font-mono text-white">
                    {liStats?.raw_transactions.toLocaleString() ?? '31,247,404'}
                  </span>
                </div>
                <div className="bg-slate-950/60 p-3 rounded-xl border border-slate-800/80">
                  <span className="text-[10px] text-slate-400 uppercase font-mono block">Distinct Accounts</span>
                  <span className="text-base font-bold font-mono text-white">
                    {liStats?.distinct_accounts.toLocaleString() ?? '~3,100,000'}
                  </span>
                </div>
                <div className="bg-slate-950/60 p-3 rounded-xl border border-slate-800/80">
                  <span className="text-[10px] text-slate-400 uppercase font-mono block">Positive Rate</span>
                  <span className="text-base font-bold font-mono text-rose-400">
                    {liStats ? `${(liStats.test_positive_prevalence * 100).toFixed(3)}%` : '0.061%'}
                  </span>
                </div>
                <div className="bg-slate-950/60 p-3 rounded-xl border border-slate-800/80">
                  <span className="text-[10px] text-slate-400 uppercase font-mono block">Frozen Threshold</span>
                  <span className="text-base font-bold font-mono text-amber-400">
                    {liStats?.frozen_threshold.toFixed(4) ?? '0.2385'}
                  </span>
                </div>
              </div>

              {/* Champion Performance Card */}
              <div className="bg-gradient-to-r from-violet-950/30 to-slate-950/50 p-4 rounded-xl border border-violet-900/40">
                <div className="flex items-center justify-between text-xs font-mono text-violet-300 mb-2">
                  <span>Authoritative Test Split Performance (6,249,112 sequences)</span>
                  <ShieldCheck className="w-4 h-4 text-emerald-400" />
                </div>
                <div className="grid grid-cols-4 gap-2 text-center">
                  <div>
                    <span className="text-[10px] text-slate-400 block">PR-AUC</span>
                    <span className="text-base font-bold font-mono text-emerald-400">0.2534</span>
                  </div>
                  <div>
                    <span className="text-[10px] text-slate-400 block">ROC-AUC</span>
                    <span className="text-base font-bold font-mono text-violet-300">0.9695</span>
                  </div>
                  <div>
                    <span className="text-[10px] text-slate-400 block">F1-Score</span>
                    <span className="text-base font-bold font-mono text-white">0.3168</span>
                  </div>
                  <div>
                    <span className="text-[10px] text-slate-400 block">Precision</span>
                    <span className="text-base font-bold font-mono text-white">0.5942</span>
                  </div>
                </div>
              </div>
            </div>

            <div className="mt-5 pt-4 border-t border-slate-800 flex items-center justify-between">
              <span className="text-[11px] text-slate-400 font-mono">Model: config_5_bilstm_fedavg_iid</span>
              <button
                onClick={() => {
                  onSelectDataset('LI-Medium');
                  onNavigateTab('csv-eval');
                }}
                className="text-xs text-violet-400 hover:text-violet-300 font-semibold flex items-center space-x-1"
              >
                <span>Evaluate LI-Medium</span>
                <ArrowRight className="w-3.5 h-3.5" />
              </button>
            </div>
          </div>
        </div>
      </div>

      {/* Core Architectural Pillars */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-5">
        <div className="bg-slate-900 border border-slate-800 rounded-2xl p-5 shadow-lg">
          <div className="flex items-center space-x-3 text-indigo-400 mb-3">
            <Lock className="w-5 h-5" />
            <h4 className="font-bold text-white text-sm">Privacy-Preserving Cross-Silo FL</h4>
          </div>
          <p className="text-xs text-slate-300 leading-relaxed">
            Local bank partitions retain all account details and raw transactions. Only model parameter updates
            (weights & gradients) are transmitted during aggregation, satisfying strict AML secrecy laws.
          </p>
        </div>

        <div className="bg-slate-900 border border-slate-800 rounded-2xl p-5 shadow-lg">
          <div className="flex items-center space-x-3 text-violet-400 mb-3">
            <TrendingDown className="w-5 h-5" />
            <h4 className="font-bold text-white text-sm">Drift & Heterogeneity Analysis</h4>
          </div>
          <p className="text-xs text-slate-300 leading-relaxed">
            Evaluated across standard IID distributions and non-IID bank typologies (retail vs commercial vs crypto).
            Quantifies the empirical collapse induced by FedProx proximal terms in extreme class imbalance.
          </p>
        </div>

        <div className="bg-slate-900 border border-slate-800 rounded-2xl p-5 shadow-lg">
          <div className="flex items-center space-x-3 text-emerald-400 mb-3">
            <ShieldCheck className="w-5 h-5" />
            <h4 className="font-bold text-white text-sm">Axiomatic Integrated Gradients</h4>
          </div>
          <p className="text-xs text-slate-300 leading-relaxed">
            Integrated Gradients attribution (50 Riemann integration steps) provides exact feature rankings for regulatory
            compliance and suspicious activity reports (SAR), eliminating black-box opacity.
          </p>
        </div>
      </div>
    </div>
  );
};
