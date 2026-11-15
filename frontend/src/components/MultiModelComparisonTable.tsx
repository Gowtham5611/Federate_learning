import React, { useState, useMemo } from 'react';
import {
  Layers,
  Award,
  TrendingUp,
  CheckCircle2,
  Zap,
  Filter,
  BarChart2,
  Cpu,
} from 'lucide-react';
import { MultiModelEvaluateResponse } from '../types/api';

interface MultiModelComparisonTableProps {
  multiModelResult: MultiModelEvaluateResponse;
  selectedModelId: string;
  onSelectAndPredictModel: (modelId: string) => void;
  onNavigateToAnalytics?: () => void;
}

export const MultiModelComparisonTable: React.FC<MultiModelComparisonTableProps> = ({
  multiModelResult,
  selectedModelId,
  onSelectAndPredictModel,
  onNavigateToAnalytics,
}) => {
  const [archFilter, setArchFilter] = useState<string>('all');
  const [searchQuery, setSearchQuery] = useState<string>('');

  const models = multiModelResult?.models || [];
  const hasLabels = Boolean(multiModelResult?.has_labels);

  // Leaderboard statistics
  const bestPrAucModel = useMemo(() => {
    if (!hasLabels) return null;
    const sorted = [...models].filter((m) => m.pr_auc != null).sort((a, b) => (b.pr_auc || 0) - (a.pr_auc || 0));
    return sorted[0] || null;
  }, [models, hasLabels]);

  const bestF1Model = useMemo(() => {
    if (!hasLabels) return null;
    const sorted = [...models].filter((m) => m.f1_score != null).sort((a, b) => (b.f1_score || 0) - (a.f1_score || 0));
    return sorted[0] || null;
  }, [models, hasLabels]);

  // Filtered models
  const filteredModels = useMemo(() => {
    return models.filter((m) => {
      const arch = (m.model_type || '').toLowerCase();
      const name = (m.model_name || '').toLowerCase();
      const mid = (m.model_id || '').toLowerCase();

      if (archFilter !== 'all') {
        if (archFilter === 'lstm' && !arch.includes('lstm')) return false;
        if (archFilter === 'gru' && !arch.includes('gru')) return false;
        if (archFilter === 'bilstm' && !arch.includes('bilstm')) return false;
        if (archFilter === 'ensemble' && !arch.includes('ensemble')) return false;
        if (archFilter === 'tabular' && !arch.includes('tabular') && !arch.includes('lightgbm')) return false;
      }

      if (searchQuery.trim() !== '') {
        const q = searchQuery.toLowerCase();
        if (!name.includes(q) && !mid.includes(q) && !arch.includes(q)) return false;
      }

      return true;
    });
  }, [models, archFilter, searchQuery]);

  const getArchBadge = (modelType: string) => {
    const t = (modelType || '').toUpperCase();
    if (t.includes('BILSTM')) {
      return <span className="px-2 py-0.5 rounded text-[10px] font-mono font-bold bg-emerald-950/80 text-emerald-300 border border-emerald-700/60">BiLSTM</span>;
    }
    if (t.includes('LSTM')) {
      return <span className="px-2 py-0.5 rounded text-[10px] font-mono font-bold bg-blue-950/80 text-blue-300 border border-blue-700/60">LSTM</span>;
    }
    if (t.includes('GRU')) {
      return <span className="px-2 py-0.5 rounded text-[10px] font-mono font-bold bg-purple-950/80 text-purple-300 border border-purple-700/60">GRU</span>;
    }
    if (t.includes('ENSEMBLE')) {
      return <span className="px-2 py-0.5 rounded text-[10px] font-mono font-bold bg-violet-950/80 text-violet-300 border border-violet-700/60">Ensemble</span>;
    }
    return <span className="px-2 py-0.5 rounded text-[10px] font-mono font-bold bg-amber-950/80 text-amber-300 border border-amber-700/60">LightGBM</span>;
  };

  const getParadigmBadge = (name: string, id: string) => {
    const str = `${name} ${id}`.toLowerCase();
    if (str.includes('centralized')) {
      return <span className="text-[10px] font-mono text-slate-400 bg-slate-900 border border-slate-800 px-1.5 py-0.5 rounded">Centralized</span>;
    }
    if (str.includes('fedprox') && str.includes('non_iid')) {
      return <span className="text-[10px] font-mono text-rose-300 bg-rose-950/60 border border-rose-800/60 px-1.5 py-0.5 rounded">FedProx Non-IID</span>;
    }
    if (str.includes('fedprox')) {
      return <span className="text-[10px] font-mono text-amber-300 bg-amber-950/60 border border-amber-800/60 px-1.5 py-0.5 rounded">FedProx IID</span>;
    }
    if (str.includes('non_iid') || str.includes('non-iid')) {
      return <span className="text-[10px] font-mono text-orange-300 bg-orange-950/60 border border-orange-800/60 px-1.5 py-0.5 rounded">FedAvg Non-IID</span>;
    }
    if (str.includes('fedavg') || str.includes('federated') || str.includes('champion')) {
      return <span className="text-[10px] font-mono text-indigo-300 bg-indigo-950/60 border border-indigo-800/60 px-1.5 py-0.5 rounded">FedAvg IID</span>;
    }
    return <span className="text-[10px] font-mono text-slate-400 bg-slate-900 border border-slate-800 px-1.5 py-0.5 rounded">Standard</span>;
  };

  return (
    <div className="bg-slate-900 border border-violet-800/60 rounded-3xl p-6 shadow-2xl space-y-6">
      {/* Header Banner */}
      <div className="flex flex-col md:flex-row md:items-center justify-between gap-4 border-b border-slate-800 pb-4">
        <div>
          <div className="flex items-center space-x-2">
            <Layers className="w-5 h-5 text-violet-400" />
            <h3 className="text-lg font-bold text-white tracking-tight">
              Multi-Model Performance Matrix & Comparison
            </h3>
            <span className="px-2.5 py-0.5 rounded-full text-[11px] font-mono font-bold bg-violet-950 border border-violet-700/60 text-violet-300">
              {models.length} Models Benchmarked
            </span>
          </div>
          <p className="text-xs text-slate-400 mt-1">
            Evaluated on exactly identical preprocessed sequence batches ({multiModelResult.total_transactions.toLocaleString()} transactions). Select any model to inspect full individual predictions.
          </p>
        </div>

        {onNavigateToAnalytics && (
          <button
            onClick={onNavigateToAnalytics}
            className="flex items-center space-x-1.5 px-3.5 py-1.5 rounded-xl bg-violet-950/80 hover:bg-violet-900 border border-violet-700/60 text-violet-200 text-xs font-semibold cursor-pointer transition"
          >
            <BarChart2 className="w-3.5 h-3.5 text-violet-400" />
            <span>Open Performance Heatmap →</span>
          </button>
        )}
      </div>

      {/* Leaderboard Cards */}
      <div className="grid grid-cols-1 sm:grid-cols-3 gap-3">
        <div className="bg-slate-950 border border-slate-800 rounded-2xl p-4 flex items-center space-x-3">
          <div className="w-10 h-10 rounded-xl bg-indigo-950/80 text-indigo-400 flex items-center justify-center border border-indigo-700/50 shrink-0">
            <Cpu className="w-5 h-5" />
          </div>
          <div className="overflow-hidden">
            <span className="text-[10px] text-slate-400 font-mono uppercase block">Active Selected Model</span>
            <span className="text-sm font-bold text-white font-mono truncate block">
              {selectedModelId}
            </span>
          </div>
        </div>

        <div className="bg-slate-950 border border-slate-800 rounded-2xl p-4 flex items-center space-x-3">
          <div className="w-10 h-10 rounded-xl bg-emerald-950/80 text-emerald-400 flex items-center justify-center border border-emerald-700/50 shrink-0">
            <Award className="w-5 h-5" />
          </div>
          <div className="overflow-hidden">
            <span className="text-[10px] text-slate-400 font-mono uppercase block">Champion (PR-AUC)</span>
            <span className="text-sm font-bold text-emerald-400 font-mono truncate block">
              {bestPrAucModel ? `${bestPrAucModel.model_name} (${(bestPrAucModel.pr_auc || 0).toFixed(4)})` : 'N/A (Score Mode)'}
            </span>
          </div>
        </div>

        <div className="bg-slate-950 border border-slate-800 rounded-2xl p-4 flex items-center space-x-3">
          <div className="w-10 h-10 rounded-xl bg-amber-950/80 text-amber-400 flex items-center justify-center border border-amber-700/50 shrink-0">
            <TrendingUp className="w-5 h-5" />
          </div>
          <div className="overflow-hidden">
            <span className="text-[10px] text-slate-400 font-mono uppercase block">Top F1-Score Model</span>
            <span className="text-sm font-bold text-amber-400 font-mono truncate block">
              {bestF1Model ? `${bestF1Model.model_name} (${(bestF1Model.f1_score || 0).toFixed(4)})` : 'N/A (Score Mode)'}
            </span>
          </div>
        </div>
      </div>

      {/* Filter and Search Bar */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 bg-slate-950 p-3 rounded-2xl border border-slate-800">
        <div className="flex items-center space-x-1.5 flex-wrap gap-1">
          <span className="text-xs text-slate-400 font-mono flex items-center space-x-1 mr-1">
            <Filter className="w-3.5 h-3.5" />
            <span>Architecture:</span>
          </span>
          {[
            { id: 'all', label: 'All Architectures' },
            { id: 'lstm', label: 'LSTM' },
            { id: 'gru', label: 'GRU' },
            { id: 'bilstm', label: 'BiLSTM' },
            { id: 'ensemble', label: 'Ensemble' },
            { id: 'tabular', label: 'LightGBM' },
          ].map((tab) => (
            <button
              key={tab.id}
              onClick={() => setArchFilter(tab.id)}
              className={`px-2.5 py-1 rounded-lg text-xs font-mono transition cursor-pointer ${
                archFilter === tab.id
                  ? 'bg-violet-600 text-white font-bold'
                  : 'bg-slate-900 text-slate-400 hover:text-slate-200 border border-slate-800'
              }`}
            >
              {tab.label}
            </button>
          ))}
        </div>

        <div className="w-full sm:w-64">
          <input
            type="text"
            placeholder="Search model name..."
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            className="w-full bg-slate-900 border border-slate-800 text-slate-200 text-xs rounded-xl px-3 py-1.5 font-mono focus:border-violet-500 focus:outline-none"
          />
        </div>
      </div>

      {/* Comparison Data Table */}
      <div className="overflow-x-auto rounded-2xl border border-slate-800">
        <table className="w-full text-xs text-left text-slate-300 font-mono">
          <thead className="bg-slate-950 text-slate-400 uppercase text-[10px] border-b border-slate-800 sticky top-0">
            <tr>
              <th className="px-3 py-2.5 whitespace-nowrap">Model Name & Type</th>
              <th className="px-3 py-2.5 whitespace-nowrap text-center">Architecture</th>
              <th className="px-3 py-2.5 whitespace-nowrap text-center">Threshold (τ)</th>
              <th className="px-3 py-2.5 whitespace-nowrap text-right">Suspicious Flagged</th>
              <th className="px-3 py-2.5 whitespace-nowrap text-right">Alert Rate</th>
              <th className="px-3 py-2.5 whitespace-nowrap text-right">Mean Prob</th>
              {hasLabels && (
                <>
                  <th className="px-3 py-2.5 whitespace-nowrap text-right text-emerald-400">PR-AUC</th>
                  <th className="px-3 py-2.5 whitespace-nowrap text-right text-indigo-400">ROC-AUC</th>
                  <th className="px-3 py-2.5 whitespace-nowrap text-right text-amber-400">F1-Score</th>
                  <th className="px-3 py-2.5 whitespace-nowrap text-right">Recall</th>
                  <th className="px-3 py-2.5 whitespace-nowrap text-right">Precision</th>
                </>
              )}
              <th className="px-3 py-2.5 whitespace-nowrap text-center">Action</th>
            </tr>
          </thead>
          <tbody className="divide-y divide-slate-800/80 bg-slate-900/40">
            {filteredModels.map((m) => {
              const isSelected = selectedModelId === m.model_id;
              return (
                <tr
                  key={m.model_id}
                  className={`hover:bg-slate-800/50 transition ${
                    isSelected ? 'bg-violet-950/30 border-l-4 border-l-violet-500' : ''
                  }`}
                >
                  <td className="px-3 py-2.5">
                    <div className="space-y-0.5">
                      <div className="font-bold text-white flex items-center space-x-1.5">
                        <span>{m.model_name}</span>
                        {isSelected && (
                          <span className="px-1.5 py-0.2 rounded text-[9px] font-bold bg-violet-600 text-white">
                            ACTIVE
                          </span>
                        )}
                      </div>
                      <div className="flex items-center space-x-2">
                        {getParadigmBadge(m.model_name, m.model_id)}
                        <span className="text-[10px] text-slate-500 font-mono">{m.model_id}</span>
                      </div>
                    </div>
                  </td>
                  <td className="px-3 py-2.5 text-center">{getArchBadge(m.model_type)}</td>
                  <td className="px-3 py-2.5 text-center font-bold text-amber-400">
                    {typeof m.decision_threshold === 'number' ? m.decision_threshold.toFixed(4) : '0.5000'}
                  </td>
                  <td className="px-3 py-2.5 text-right font-bold text-rose-400">
                    {m.flagged_count?.toLocaleString()}
                  </td>
                  <td className="px-3 py-2.5 text-right font-semibold text-slate-300">
                    {m.laundering_percentage != null ? `${m.laundering_percentage.toFixed(2)}%` : 'N/A'}
                  </td>
                  <td className="px-3 py-2.5 text-right text-slate-400">
                    {m.mean_probability != null ? m.mean_probability.toFixed(4) : 'N/A'}
                  </td>
                  {hasLabels && (
                    <>
                      <td className="px-3 py-2.5 text-right font-black text-emerald-400">
                        {m.pr_auc != null ? m.pr_auc.toFixed(4) : '—'}
                      </td>
                      <td className="px-3 py-2.5 text-right font-semibold text-indigo-300">
                        {m.roc_auc != null ? m.roc_auc.toFixed(4) : '—'}
                      </td>
                      <td className="px-3 py-2.5 text-right font-bold text-amber-300">
                        {m.f1_score != null ? m.f1_score.toFixed(4) : '—'}
                      </td>
                      <td className="px-3 py-2.5 text-right text-slate-300">
                        {m.recall != null ? m.recall.toFixed(4) : '—'}
                      </td>
                      <td className="px-3 py-2.5 text-right text-slate-300">
                        {m.precision != null ? m.precision.toFixed(4) : '—'}
                      </td>
                    </>
                  )}
                  <td className="px-3 py-2.5 text-center">
                    {isSelected ? (
                      <span className="inline-flex items-center space-x-1 px-3 py-1 rounded-xl bg-violet-600/30 text-violet-300 border border-violet-500/50 text-[11px] font-bold">
                        <CheckCircle2 className="w-3.5 h-3.5 text-violet-400" />
                        <span>Active</span>
                      </span>
                    ) : (
                      <button
                        type="button"
                        onClick={() => onSelectAndPredictModel(m.model_id)}
                        className="inline-flex items-center space-x-1 px-3 py-1 rounded-xl bg-slate-800 hover:bg-violet-600 text-slate-200 hover:text-white border border-slate-700 hover:border-violet-500 text-[11px] font-semibold transition cursor-pointer shadow"
                      >
                        <Zap className="w-3 h-3 text-emerald-400" />
                        <span>Predict Output</span>
                      </button>
                    )}
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
    </div>
  );
};
