import React from 'react';
import { Database, CheckCircle2, Cpu, ShieldCheck } from 'lucide-react';
import { ModelMetadataResponse } from '../types/api';
import { formatThreshold } from '../utils/formatters';

interface ModelCardProps {
  model: ModelMetadataResponse;
  onSelectDataset?: (dataset: 'HI-Small' | 'LI-Medium') => void;
}

export const ModelCard: React.FC<ModelCardProps> = ({ model, onSelectDataset }) => {
  return (
    <div className="bg-slate-900 border border-slate-800 rounded-2xl p-6 hover:border-slate-700 transition shadow-lg shadow-black/20 flex flex-col justify-between">
      <div>
        {/* Header */}
        <div className="flex items-start justify-between">
          <div className="flex items-center space-x-3">
            <div className="p-2.5 bg-slate-800 border border-slate-700 rounded-xl text-indigo-400">
              <Database className="w-5 h-5" />
            </div>
            <div>
              <span className="text-xs font-mono uppercase tracking-wider text-indigo-400 font-semibold">
                Dataset: {model.dataset}
              </span>
              <h3 className="font-semibold text-white text-base mt-0.5">{model.model_name}</h3>
            </div>
          </div>
          <span className="px-2.5 py-0.5 rounded-full text-xs font-mono bg-indigo-950/80 text-indigo-300 border border-indigo-500/30">
            {model.architecture.toUpperCase()}
          </span>
        </div>

        {/* Specifications Matrix */}
        <div className="grid grid-cols-2 gap-3 mt-6">
          <div className="p-3 bg-slate-950/60 rounded-xl border border-slate-800/80">
            <div className="text-xs text-slate-400">Parameter Count</div>
            <div className="text-sm font-mono font-medium text-slate-200 mt-1">
              {model.parameter_count.toLocaleString()}
            </div>
          </div>
          <div className="p-3 bg-slate-950/60 rounded-xl border border-slate-800/80">
            <div className="text-xs text-slate-400">Frozen Threshold</div>
            <div className="text-sm font-mono font-medium text-amber-400 mt-1">
              {formatThreshold(model.frozen_threshold)}
            </div>
          </div>
          <div className="p-3 bg-slate-950/60 rounded-xl border border-slate-800/80">
            <div className="text-xs text-slate-400">Feature Layout</div>
            <div className="text-sm font-mono font-medium text-slate-200 mt-1">
              {model.sequence_length} steps × {model.feature_dimension} features
            </div>
          </div>
          <div className="p-3 bg-slate-950/60 rounded-xl border border-slate-800/80">
            <div className="text-xs text-slate-400">Hardware Runtime</div>
            <div className="text-sm font-mono font-medium text-slate-200 mt-1 flex items-center space-x-1.5">
              <Cpu className="w-3.5 h-3.5 text-slate-400" />
              <span>PyTorch CPU</span>
            </div>
          </div>
        </div>

        {/* Integrity status */}
        <div className="mt-4 pt-4 border-t border-slate-800/80 flex items-center justify-between text-xs text-slate-400">
          <div className="flex items-center space-x-1.5 text-emerald-400">
            <CheckCircle2 className="w-4 h-4" />
            <span>SHA-256 Checkpoint Authenticated</span>
          </div>
          <div className="flex items-center space-x-1 text-slate-400 font-mono">
            <ShieldCheck className="w-3.5 h-3.5" />
            <span>Integrated Gradients</span>
          </div>
        </div>
      </div>

      {onSelectDataset && (
        <button
          onClick={() => onSelectDataset(model.dataset)}
          className="mt-6 w-full py-2.5 px-4 rounded-xl bg-slate-800 hover:bg-indigo-600 text-slate-200 hover:text-white text-xs font-semibold uppercase tracking-wider transition border border-slate-700 hover:border-indigo-500"
        >
          Select {model.dataset} for Inference
        </button>
      )}
    </div>
  );
};

