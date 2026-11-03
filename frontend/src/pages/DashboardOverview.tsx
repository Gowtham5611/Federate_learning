import React from 'react';
import { HealthResponse, ModelMetadataResponse, PredictionHistoryEntry } from '../types/api';
import { ModelCard } from '../components/ModelCard';
import { Shield, Activity, Database, Cpu, Lock, ArrowRight, AlertTriangle } from 'lucide-react';

interface DashboardOverviewProps {
  health: HealthResponse | null;
  models: ModelMetadataResponse[];
  history: PredictionHistoryEntry[];
  loading: boolean;
  error: string | null;
  onSelectDataset: (dataset: 'HI-Small' | 'LI-Medium') => void;
  onNavigateToPredict: () => void;
}

export const DashboardOverview: React.FC<DashboardOverviewProps> = ({
  health,
  models,
  history,
  loading,
  error,
  onSelectDataset,
  onNavigateToPredict,
}) => {
  const isReady = health?.readiness === 'ready';

  return (
    <div className="space-y-8">
      {/* Top Hero Banner */}
      <div className="bg-gradient-to-r from-slate-900 via-indigo-950/40 to-slate-900 border border-slate-800 rounded-3xl p-8 relative overflow-hidden shadow-2xl">
        <div className="relative z-10 max-w-3xl">
          <div className="inline-flex items-center space-x-2 px-3 py-1 rounded-full bg-indigo-950/80 border border-indigo-500/40 text-indigo-300 text-xs font-mono mb-4">
            <Lock className="w-3.5 h-3.5" />
            <span>Restricted Network / Authenticated Ingress Mode</span>
          </div>

          <h1 className="text-3xl sm:text-4xl font-extrabold text-white tracking-tight">
            FedSecureAI AML Analyst Workspace
          </h1>
          <p className="mt-3 text-slate-300 text-sm sm:text-base leading-relaxed">
            Real-time inference and axiomatic explainability engine for federated anti-money laundering detection.
            Powered by authoritative federated champion models trained on IBM AML HI-Small and LI-Medium datasets.
          </p>

          <div className="mt-6 flex flex-wrap items-center gap-4">
            <button
              onClick={onNavigateToPredict}
              className="flex items-center space-x-2 px-5 py-2.5 rounded-xl bg-indigo-600 hover:bg-indigo-500 text-white font-medium text-sm transition shadow-lg shadow-indigo-600/30"
            >
              <span>Launch Transaction Scoring</span>
              <ArrowRight className="w-4 h-4" />
            </button>
            <div className="text-xs text-slate-400 font-mono">
              Thresholds locked in Phase 7 • Explanations via Integrated Gradients
            </div>
          </div>
        </div>
      </div>

      {/* Error State if API Disconnected */}
      {error && (
        <div className="p-4 bg-rose-950/60 border border-rose-500/50 rounded-2xl flex items-start space-x-3 text-rose-200 text-sm">
          <AlertTriangle className="w-5 h-5 text-rose-400 shrink-0 mt-0.5" />
          <div>
            <h4 className="font-semibold text-rose-300">Backend Communication Issue</h4>
            <p className="text-xs text-rose-200/80 mt-1">{error}</p>
          </div>
        </div>
      )}

      {/* Overview Stat Cards */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
        {/* Service Liveness */}
        <div className="bg-slate-900 border border-slate-800 rounded-2xl p-5 shadow-lg">
          <div className="flex items-center justify-between text-xs text-slate-400">
            <span>Inference Engine Status</span>
            <Activity className="w-4 h-4 text-indigo-400" />
          </div>
          <div className="mt-3 flex items-baseline space-x-2">
            <span
              className={`text-xl font-bold font-mono ${
                isReady ? 'text-emerald-400' : health ? 'text-amber-400' : 'text-rose-400'
              }`}
            >
              {isReady ? 'READY (200 OK)' : health ? 'INITIALIZING' : 'OFFLINE'}
            </span>
          </div>
          <div className="mt-2 text-[11px] text-slate-400 truncate">
            {health ? `${health.app_name} (${health.device})` : 'Backend connection unavailable'}
          </div>
        </div>

        {/* Loaded Champions */}
        <div className="bg-slate-900 border border-slate-800 rounded-2xl p-5 shadow-lg">
          <div className="flex items-center justify-between text-xs text-slate-400">
            <span>Active Model Champions</span>
            <Database className="w-4 h-4 text-indigo-400" />
          </div>
          <div className="mt-3 flex items-baseline space-x-2">
            <span className="text-xl font-bold font-mono text-white">
              {models.length} / 2
            </span>
            <span className="text-xs text-emerald-400 font-mono">Authenticated</span>
          </div>
          <div className="mt-2 text-[11px] text-slate-400 truncate">
            HI-Small LSTM & LI-Medium BiLSTM
          </div>
        </div>

        {/* Causal Sequence Matrix */}
        <div className="bg-slate-900 border border-slate-800 rounded-2xl p-5 shadow-lg">
          <div className="flex items-center justify-between text-xs text-slate-400">
            <span>Input Representation</span>
            <Cpu className="w-4 h-4 text-indigo-400" />
          </div>
          <div className="mt-3 flex items-baseline space-x-2">
            <span className="text-xl font-bold font-mono text-white">57 Features</span>
          </div>
          <div className="mt-2 text-[11px] text-slate-400 truncate">
            Expanded canonical one-hot encoding
          </div>
        </div>

        {/* Session Inferences */}
        <div className="bg-slate-900 border border-slate-800 rounded-2xl p-5 shadow-lg">
          <div className="flex items-center justify-between text-xs text-slate-400">
            <span>Session Inferences</span>
            <Shield className="w-4 h-4 text-indigo-400" />
          </div>
          <div className="mt-3 flex items-baseline space-x-2">
            <span className="text-xl font-bold font-mono text-indigo-300">
              {history.length}
            </span>
            <span className="text-xs text-slate-400">evaluations</span>
          </div>
          <div className="mt-2 text-[11px] text-slate-400 truncate">
            {history.filter((h) => h.decision === 'FLAGGED_LAUNDERING').length} flagged laundering
          </div>
        </div>
      </div>

      {/* Model Catalog Section */}
      <div>
        <div className="flex items-center justify-between mb-4">
          <div>
            <h2 className="text-lg font-bold text-white">Federated Champion Models</h2>
            <p className="text-xs text-slate-400">
              Authoritative model architectures, frozen operational thresholds, and parameter topologies.
            </p>
          </div>
          <span className="text-xs font-mono text-slate-400">
            Source: Phase 7 Authoritative Lock
          </span>
        </div>

        {loading ? (
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            <div className="h-64 bg-slate-900/40 rounded-2xl animate-pulse border border-slate-800" />
            <div className="h-64 bg-slate-900/40 rounded-2xl animate-pulse border border-slate-800" />
          </div>
        ) : models.length === 0 ? (
          <div className="p-8 text-center bg-slate-900 border border-slate-800 rounded-2xl text-slate-400 text-sm">
            No models reported by backend `/models` endpoint.
          </div>
        ) : (
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            {models.map((model) => (
              <ModelCard
                key={model.dataset}
                model={model}
                onSelectDataset={onSelectDataset}
              />
            ))}
          </div>
        )}
      </div>
    </div>
  );
};

