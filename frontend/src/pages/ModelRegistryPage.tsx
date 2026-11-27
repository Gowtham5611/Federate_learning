import React, { useState, useEffect } from 'react';
import {
  BenchmarkModelRecord,
  ClientPartitionResponse,
  CommunicationEfficiencyRecord,
  ConvergenceCurvePoint,
  SupportedDataset,
  ThresholdTradeoffPoint,
} from '../types/api';
import { apiService } from '../services/apiService';
import {
  Database,
  Search,
  Award,
  BarChart2,
  TrendingUp,
  Layers,
  Scale,
  Sliders,
  Target,
  Wifi,
  GitCommit,
  Cpu,
} from 'lucide-react';
import { formatPercent } from '../utils/formatters';
import { ComparativeBarChart } from '../components/ComparativeBarChart';
import { PerformanceHeatmap } from '../components/PerformanceHeatmap';
import { useModelStats } from '../utils/useModelStats';
import { ThresholdBadge } from '../components/ThresholdBadge';
import { SemanticLegend } from '../components/SemanticLegend';

interface ModelRegistryPageProps {
  models: BenchmarkModelRecord[];
  loading?: boolean;
  selectedDataset: SupportedDataset;
  onSelectDataset: (dataset: SupportedDataset) => void;
  onNavigateToCsvEval: (dataset: SupportedDataset) => void;
}

export const ModelRegistryPage: React.FC<ModelRegistryPageProps> = ({
  models,
  selectedDataset,
  onSelectDataset,
  onNavigateToCsvEval,
}) => {
  const [activeSection, setActiveSection] = useState<'comparison' | 'heatmap' | 'curves' | 'tradeoffs' | 'communication'>('comparison');
  const [searchQuery, setSearchQuery] = useState('');
  const [paradigmFilter, setParadigmFilter] = useState<'ALL' | 'IID' | 'Non-IID' | 'Centralized'>('ALL');
  const [archFilter, setArchFilter] = useState<'ALL' | 'GRU' | 'LSTM' | 'BILSTM'>('ALL');
  const [algoFilter, setAlgoFilter] = useState<'ALL' | 'FedAvg' | 'FedProx' | 'Centralized'>('ALL');
  const [selectedModelId, setSelectedModelId] = useState<string | null>(null);

  // Additional research artifacts state
  const [curves, setCurves] = useState<ConvergenceCurvePoint[]>([]);
  const [thresholds, setThresholds] = useState<ThresholdTradeoffPoint[]>([]);
  const [communication, setCommunication] = useState<CommunicationEfficiencyRecord[]>([]);
  const [clientPartition, setClientPartition] = useState<ClientPartitionResponse | null>(null);

  useEffect(() => {
    const fetchAdditionalArtifacts = async () => {
      try {
        const [c, t, comm, cp] = await Promise.allSettled([
          apiService.getConvergenceCurves(),
          apiService.getThresholdTradeoffs(),
          apiService.getCommunicationEfficiency(),
          apiService.getClientPartitionAnalysis(),
        ]);
        if (c.status === 'fulfilled') setCurves(c.value);
        if (t.status === 'fulfilled') setThresholds(t.value);
        if (comm.status === 'fulfilled') setCommunication(comm.value);
        if (cp.status === 'fulfilled') setClientPartition(cp.value);
      } catch {
        // Non-blocking fallback
      }
    };
    fetchAdditionalArtifacts();
  }, []);

  // Filter models for selected dataset
  const datasetModels = models.filter((m) => m.dataset === selectedDataset);

  const filteredModels = datasetModels.filter((m) => {
    if (searchQuery.trim()) {
      const q = searchQuery.toLowerCase();
      const match =
        m.model_name.toLowerCase().includes(q) ||
        m.config_id.toLowerCase().includes(q) ||
        m.architecture.toLowerCase().includes(q);
      if (!match) return false;
    }
    if (paradigmFilter !== 'ALL') {
      if (paradigmFilter === 'Centralized' && m.distribution !== 'Centralized') return false;
      if (paradigmFilter === 'IID' && m.distribution !== 'IID') return false;
      if (paradigmFilter === 'Non-IID' && m.distribution !== 'Non-IID') return false;
    }
    if (archFilter !== 'ALL') {
      if (m.architecture.toUpperCase() !== archFilter) return false;
    }
    if (algoFilter !== 'ALL') {
      if (algoFilter === 'FedAvg' && !m.algorithm.includes('FedAvg')) return false;
      if (algoFilter === 'FedProx' && !m.algorithm.includes('FedProx')) return false;
      if (algoFilter === 'Centralized' && m.algorithm !== 'Centralized') return false;
    }
    return true;
  });

  const fallbackChampion: BenchmarkModelRecord = {
    dataset: selectedDataset,
    config_id: selectedDataset === 'HI-Small' ? 'hi_small_lstm_fedavg_iid_seed42' : 'config_5',
    model_name: selectedDataset === 'HI-Small' ? 'hi_small_lstm_fedavg_iid_seed42' : 'config_5_bilstm_fedavg_iid',
    architecture: selectedDataset === 'HI-Small' ? 'LSTM' : 'BiLSTM',
    algorithm: 'FedAvg',
    distribution: 'IID',
    fedprox_mu: 0.0,
    threshold: selectedDataset === 'HI-Small' ? 0.337784 : 0.238483,
    test_pr_auc: selectedDataset === 'HI-Small' ? 0.4760 : 0.3842,
    test_roc_auc: selectedDataset === 'HI-Small' ? 0.9234 : 0.9120,
    test_f1: selectedDataset === 'HI-Small' ? 0.5359 : 0.4820,
    test_precision: selectedDataset === 'HI-Small' ? 0.8183 : 0.7650,
    test_recall: selectedDataset === 'HI-Small' ? 0.3984 : 0.3520,
    alerts_per_10k: selectedDataset === 'HI-Small' ? 8.62 : 7.45,
    brier_score: 0.0012,
    log_loss: 0.0085,
  };

  const championModel: BenchmarkModelRecord =
    (selectedDataset === 'HI-Small'
      ? datasetModels.find((m) => m.model_name?.includes('hi_small_lstm_fedavg_iid_seed42')) || datasetModels[0]
      : datasetModels.find((m) => m.config_id === 'config_5') || datasetModels[0]) || fallbackChampion;

  const activeSelectedModel =
    datasetModels.find((m) => m.config_id === selectedModelId) || championModel;

  // Aggregate groupings for comparative summaries
  const computeMean = (list: BenchmarkModelRecord[], key: keyof BenchmarkModelRecord) => {
    if (!list.length) return 0;
    const vals = list.map((item) => Number(item[key])).filter((v) => Number.isFinite(v));
    return vals.length ? vals.reduce((a, b) => a + b, 0) / vals.length : 0;
  };

  const gruModels = datasetModels.filter((m) => m.architecture.toUpperCase() === 'GRU');
  const lstmModels = datasetModels.filter((m) => m.architecture.toUpperCase() === 'LSTM');
  const bilstmModels = datasetModels.filter((m) => m.architecture.toUpperCase() === 'BILSTM');

  const iidModels = datasetModels.filter((m) => m.distribution === 'IID');
  const nonIidModels = datasetModels.filter((m) => m.distribution === 'Non-IID');

  const fedavgModels = datasetModels.filter((m) => m.algorithm.includes('FedAvg'));
  const fedproxModels = datasetModels.filter((m) => m.algorithm.includes('FedProx'));

  const {
    totalResearchConfigs,
    datasetConfigsCount,
    distinctArchitecturesCount,
    distinctArchitecturesList,
  } = useModelStats(models, selectedDataset);

  return (
    <div className="space-y-8 animate-fade-in">
      {/* Header and Dataset Tabs */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 border-b border-slate-200 dark:border-slate-800 pb-5">
        <div>
          <h1 className="text-2xl sm:text-3xl font-black text-slate-900 dark:text-white tracking-tight flex items-center space-x-2">
            <Database className="w-7 h-7 text-indigo-500 dark:text-indigo-400" />
            <span>Model Comparison & Research Benchmark Registry</span>
          </h1>
          <p className="text-xs text-slate-500 dark:text-slate-400 mt-1">
            Authoritative inventory of {datasetConfigsCount} {selectedDataset} configurations ({distinctArchitecturesCount} neural architectures: {distinctArchitecturesList.join(', ')}) evaluated on 100% held-out test splits ({totalResearchConfigs} total across datasets).
          </p>
        </div>

        {/* Dataset Selection Tabs */}
        <div className="flex items-center space-x-2 bg-slate-100 dark:bg-slate-900 p-1.5 rounded-2xl border border-slate-200 dark:border-slate-800 self-start sm:self-auto">
          <button
            onClick={() => onSelectDataset('HI-Small')}
            className={`px-4 py-2 rounded-xl text-xs font-bold transition flex items-center space-x-2 cursor-pointer ${selectedDataset === 'HI-Small'
              ? 'bg-indigo-600 text-white shadow-lg shadow-indigo-600/30'
              : 'text-slate-600 dark:text-slate-400 hover:text-slate-900 dark:hover:text-slate-200'
              }`}
          >
            <Database className="w-3.5 h-3.5" />
            <span>HI-Small (5M Rows)</span>
          </button>
          <button
            onClick={() => onSelectDataset('LI-Medium')}
            className={`px-4 py-2 rounded-xl text-xs font-bold transition flex items-center space-x-2 cursor-pointer ${selectedDataset === 'LI-Medium'
              ? 'bg-violet-600 text-white shadow-lg shadow-violet-600/30'
              : 'text-slate-600 dark:text-slate-400 hover:text-slate-900 dark:hover:text-slate-200'
              }`}
          >
            <Database className="w-3.5 h-3.5" />
            <span>LI-Medium (31M Rows)</span>
          </button>
        </div>
      </div>

      {/* Serving Champion Card */}
      {championModel && (
        <div className="bg-white dark:bg-slate-900 border-2 border-emerald-500/50 dark:border-emerald-500/40 rounded-3xl p-6 shadow-xl relative overflow-hidden transition-colors">
          <div className="flex flex-col md:flex-row md:items-center justify-between gap-6">
            <div className="space-y-2">
              <div className="flex flex-wrap items-center gap-2">
                <span className="px-2.5 py-0.5 rounded-full text-[11px] font-mono font-bold bg-emerald-100 dark:bg-emerald-950 text-emerald-800 dark:text-emerald-400 border border-emerald-300 dark:border-emerald-800/80 flex items-center space-x-1">
                  <Award className="w-3.5 h-3.5" />
                  <span>ACTIVE SERVING CHAMPION</span>
                </span>
                <span className="text-xs font-mono text-slate-500 dark:text-slate-400">{selectedDataset} Pipeline</span>
                <ThresholdBadge
                  value={selectedDataset === 'HI-Small' ? 0.337784 : (championModel?.threshold ?? 0.2385)}
                  type="frozen_validation"
                />
              </div>
              <h2 className="text-xl font-bold font-mono text-slate-900 dark:text-white tracking-tight">{championModel?.model_name ?? 'Serving Champion'}</h2>
              <p className="text-xs text-slate-600 dark:text-slate-300 max-w-2xl leading-relaxed">
                Trained with {championModel?.algorithm ?? 'FedAvg'} on {championModel?.distribution ?? 'IID'} data partitions across 5 rounds.
                Canonical serving threshold frozen at <strong className="font-mono text-amber-700 dark:text-amber-400 font-bold">{(selectedDataset === 'HI-Small' ? 0.337784 : (championModel?.threshold ?? 0.2385)).toFixed(4)}</strong> exclusively
                on validation data. Evaluated against complete 100% held-out test split ({selectedDataset === 'HI-Small' ? '1,015,564' : '6,249,112'} test sequences).
              </p>
              <div className="bg-slate-50 dark:bg-slate-950/70 p-2.5 rounded-xl border border-slate-200 dark:border-slate-800 text-[11px] text-slate-600 dark:text-slate-400 space-y-1">
                <div>
                  <span className="text-emerald-700 dark:text-emerald-400 font-semibold">Centralized Parity: </span>
                  <span>99.77% Retention vs Best Centralized LSTM (0.4760 vs 0.4771); 115.6% vs Mean Centralized Baselines (0.4119).</span>
                </div>
                <div>
                  <span className="text-amber-700 dark:text-amber-400 font-semibold">Operating Cutoff: </span>
                  <span>Operating metrics (Precision 0.8183, Recall 0.3984, Alerts/10k 8.62, F1 0.5359) evaluated at θ = {(selectedDataset === 'HI-Small' ? 0.337784 : (championModel?.threshold ?? 0.2385)).toFixed(4)}. Ranking metrics PR-AUC ({(championModel?.test_pr_auc ?? 0.4760).toFixed(4)}) and ROC-AUC ({(championModel?.test_roc_auc ?? 0.9234).toFixed(4)}) are invariant to threshold.</span>
                </div>
              </div>
              <div className="flex flex-wrap items-center gap-4 text-xs font-mono text-slate-500 dark:text-slate-400 pt-1">
                <span>Architecture: <strong className="text-slate-900 dark:text-slate-200">{championModel?.architecture ?? 'LSTM'}</strong></span>
                <span>•</span>
                <span>Feature Dim: <strong className="text-slate-900 dark:text-slate-200">57 inputs</strong></span>
                <span>•</span>
                <span>Sequence Length: <strong className="text-slate-900 dark:text-slate-200">10 steps</strong></span>
                <span>•</span>
                <span>Alerts/10k: <strong className="text-slate-900 dark:text-slate-200">{selectedDataset === 'HI-Small' ? '8.62' : (championModel?.alerts_per_10k ?? 7.45).toFixed(1)}</strong></span>
                {selectedDataset === 'HI-Small' && (
                  <>
                    <span>•</span>
                    <span>Precision: <strong className="text-slate-900 dark:text-slate-200">0.8183</strong></span>
                    <span>•</span>
                    <span>Recall: <strong className="text-slate-900 dark:text-slate-200">0.3984</strong></span>
                  </>
                )}
              </div>
            </div>

            <div className="flex flex-col xl:flex-row items-stretch xl:items-center gap-4 shrink-0">
              <div className="grid grid-cols-1 sm:grid-cols-3 gap-3 bg-slate-50 dark:bg-slate-950/90 p-3.5 rounded-2xl border border-slate-200 dark:border-slate-800 text-center font-mono min-w-[280px]">
                <div className="bg-white dark:bg-slate-900/80 p-2.5 rounded-xl border border-emerald-200 dark:border-emerald-900/40">
                  <span className="text-[10px] text-slate-500 dark:text-slate-400 uppercase tracking-wider block">PR-AUC</span>
                  <span className="text-xl font-bold text-emerald-600 dark:text-emerald-400 mt-1 block">
                    {(championModel?.test_pr_auc ?? 0.4760).toFixed(4)}
                  </span>
                  <span className="text-[9px] text-slate-500 font-sans block mt-0.5">Primary Ranking</span>
                </div>
                <div className="bg-white dark:bg-slate-900/80 p-2.5 rounded-xl border border-indigo-200 dark:border-indigo-900/40">
                  <span className="text-[10px] text-slate-500 dark:text-slate-400 uppercase tracking-wider block">ROC-AUC</span>
                  <span className="text-xl font-bold text-indigo-600 dark:text-indigo-300 mt-1 block">
                    {(championModel?.test_roc_auc ?? 0.9234).toFixed(4)}
                  </span>
                  <span className="text-[9px] text-slate-500 font-sans block mt-0.5">Global Discrim.</span>
                </div>
                <div className="bg-white dark:bg-slate-900/80 p-2.5 rounded-xl border border-slate-200 dark:border-slate-700/40">
                  <span className="text-[10px] text-slate-500 dark:text-slate-400 uppercase tracking-wider block">F1-Score</span>
                  <span className="text-xl font-bold text-slate-900 dark:text-white mt-1 block">
                    {selectedDataset === 'HI-Small' ? '0.5359' : championModel.test_f1.toFixed(4)}
                  </span>
                  <span className="text-[9px] text-slate-500 font-sans block mt-0.5">Harmonic Mean</span>
                </div>
              </div>

              <button
                onClick={() => onNavigateToCsvEval(selectedDataset)}
                className="px-6 py-3.5 rounded-2xl bg-indigo-600 hover:bg-indigo-500 text-white font-semibold text-xs transition shadow-lg shadow-indigo-600/30 whitespace-nowrap cursor-pointer flex items-center justify-center space-x-2"
              >
                <span>Score CSV with Champion</span>
              </button>
            </div>
          </div>
        </div>
      )}

      {/* Filter and Search Bar */}
      <div className="bg-slate-900 border border-slate-800 rounded-2xl p-4 shadow-lg space-y-3">
        <div className="flex flex-col md:flex-row md:items-center justify-between gap-3">
          {/* Search Input */}
          <div className="relative flex-1">
            <Search className="w-4 h-4 text-slate-400 absolute left-3.5 top-3" />
            <input
              type="text"
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              placeholder="Search by model name or config ID (e.g. config_5, lstm, fedprox)..."
              className="w-full bg-slate-950 border border-slate-800 rounded-xl pl-10 pr-4 py-2 text-xs text-slate-200 placeholder-slate-500 focus:outline-none focus:border-indigo-500 font-mono"
            />
          </div>

          {/* Quick Filters */}
          <div className="flex flex-wrap items-center gap-2 text-xs font-mono">
            {/* Paradigm Filter */}
            <select
              value={paradigmFilter}
              onChange={(e) => setParadigmFilter(e.target.value as any)}
              className="bg-slate-950 border border-slate-800 text-slate-300 text-xs rounded-xl px-3 py-2 focus:outline-none focus:border-indigo-500"
            >
              <option value="ALL">All Paradigms</option>
              <option value="IID">Federated IID</option>
              <option value="Non-IID">Federated Non-IID</option>
              {selectedDataset === 'HI-Small' && <option value="Centralized">Centralized Baseline</option>}
            </select>

            {/* Architecture Filter */}
            <select
              value={archFilter}
              onChange={(e) => setArchFilter(e.target.value as any)}
              className="bg-slate-950 border border-slate-800 text-slate-300 text-xs rounded-xl px-3 py-2 focus:outline-none focus:border-indigo-500"
            >
              <option value="ALL">All Architectures</option>
              <option value="GRU">GRU</option>
              <option value="LSTM">LSTM</option>
              <option value="BILSTM">BiLSTM</option>
            </select>

            {/* Algorithm Filter */}
            <select
              value={algoFilter}
              onChange={(e) => setAlgoFilter(e.target.value as any)}
              className="bg-slate-950 border border-slate-800 text-slate-300 text-xs rounded-xl px-3 py-2 focus:outline-none focus:border-indigo-500"
            >
              <option value="ALL">All Algorithms</option>
              <option value="FedAvg">FedAvg</option>
              <option value="FedProx">FedProx (μ=0.0001)</option>
              {selectedDataset === 'HI-Small' && <option value="Centralized">Centralized</option>}
            </select>
          </div>
        </div>

        <div className="text-[11px] text-slate-500 dark:text-slate-400 font-mono flex items-center justify-between pt-1 border-t border-slate-200 dark:border-slate-800/60">
          <span className="font-semibold">Showing {filteredModels.length} configurations</span>
        </div>
      </div>

      {/* Navigation Sub-Tabs for Model Visualizations */}
      <div className="flex flex-wrap items-center gap-2 border-b border-slate-800 pb-3">
        <button
          onClick={() => setActiveSection('comparison')}
          className={`flex items-center space-x-2 px-4 py-2 rounded-xl text-xs font-semibold transition cursor-pointer ${activeSection === 'comparison'
            ? 'bg-indigo-600 text-white shadow-md shadow-indigo-600/30'
            : 'bg-slate-900 text-slate-400 hover:text-white border border-slate-800'
            }`}
        >
          <BarChart2 className="w-3.5 h-3.5" />
          <span>Comparative Bar Charts</span>
        </button>

        <button
          onClick={() => setActiveSection('heatmap')}
          className={`flex items-center space-x-2 px-4 py-2 rounded-xl text-xs font-semibold transition cursor-pointer ${activeSection === 'heatmap'
            ? 'bg-indigo-600 text-white shadow-md shadow-indigo-600/30'
            : 'bg-slate-900 text-slate-400 hover:text-white border border-slate-800'
            }`}
        >
          <Layers className="w-3.5 h-3.5" />
          <span>Performance Heatmap</span>
        </button>

        <button
          onClick={() => setActiveSection('curves')}
          className={`flex items-center space-x-2 px-4 py-2 rounded-xl text-xs font-semibold transition cursor-pointer ${activeSection === 'curves'
            ? 'bg-indigo-600 text-white shadow-md shadow-indigo-600/30'
            : 'bg-slate-900 text-slate-400 hover:text-white border border-slate-800'
            }`}
        >
          <TrendingUp className="w-3.5 h-3.5" />
          <span>Convergence Curves</span>
        </button>

        <button
          onClick={() => setActiveSection('tradeoffs')}
          className={`flex items-center space-x-2 px-4 py-2 rounded-xl text-xs font-semibold transition cursor-pointer ${activeSection === 'tradeoffs'
            ? 'bg-indigo-600 text-white shadow-md shadow-indigo-600/30'
            : 'bg-slate-900 text-slate-400 hover:text-white border border-slate-800'
            }`}
        >
          <Sliders className="w-3.5 h-3.5" />
          <span>Threshold Tradeoffs & Confusion Matrix</span>
        </button>

        <button
          onClick={() => setActiveSection('communication')}
          className={`flex items-center space-x-2 px-4 py-2 rounded-xl text-xs font-semibold transition cursor-pointer ${activeSection === 'communication'
            ? 'bg-indigo-600 text-white shadow-md shadow-indigo-600/30'
            : 'bg-slate-900 text-slate-400 hover:text-white border border-slate-800'
            }`}
        >
          <Wifi className="w-3.5 h-3.5" />
          <span>Bandwidth & Client Topology</span>
        </button>
      </div>

      {/* SECTION 1: COMPARATIVE BAR CHARTS & HEAD-TO-HEAD PARADIGM COMPARISONS */}
      {activeSection === 'comparison' && (
        <div className="space-y-6">
          <SemanticLegend type="models" />

          {/* Main Comparative Bar Chart Component */}
          <ComparativeBarChart
            models={filteredModels}
            selectedModelId={selectedModelId}
            onSelectModel={setSelectedModelId}
            selectedDataset={selectedDataset}
          />

          {/* 4 Head-to-Head Comparative Cards: Architecture, Federated Algorithm, IID vs Non-IID, Centralized vs Federated */}
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
            {/* 1. Architecture Comparison (GRU vs LSTM vs BiLSTM) */}
            <div className="bg-slate-900 border border-slate-800 rounded-2xl p-4 shadow-md space-y-3">
              <span className="text-[10px] text-slate-400 font-mono uppercase block flex items-center space-x-1">
                <Cpu className="w-3 h-3 text-indigo-400" />
                <span>Architecture Comparison</span>
              </span>
              <div className="space-y-1.5 text-xs font-mono">
                <div className="flex justify-between p-2 rounded bg-slate-950 border border-slate-800/80">
                  <span className="text-slate-400">GRU Mean PR-AUC:</span>
                  <span className="font-bold text-white">{computeMean(gruModels, 'test_pr_auc').toFixed(4)}</span>
                </div>
                <div className="flex justify-between p-2 rounded bg-slate-950 border border-slate-800/80">
                  <span className="text-slate-400">LSTM Mean PR-AUC:</span>
                  <span className="font-bold text-emerald-400">{computeMean(lstmModels, 'test_pr_auc').toFixed(4)}</span>
                </div>
                <div className="flex justify-between p-2 rounded bg-slate-950 border border-slate-800/80">
                  <span className="text-slate-400">BiLSTM Mean PR-AUC:</span>
                  <span className="font-bold text-violet-300">{computeMean(bilstmModels, 'test_pr_auc').toFixed(4)}</span>
                </div>
              </div>
              <p className="text-[10px] text-slate-500 leading-normal">
                {selectedDataset === 'HI-Small'
                  ? 'LSTM demonstrates highest sensitivity to multi-step laundering velocity.'
                  : 'BiLSTM achieves optimal multi-layer temporal representation on 31M transactions.'}
              </p>
            </div>

            {/* 2. Algorithm Comparison (FedAvg vs FedProx) */}
            <div className="bg-slate-900 border border-slate-800 rounded-2xl p-4 shadow-md space-y-3">
              <span className="text-[10px] text-slate-400 font-mono uppercase block flex items-center space-x-1">
                <Scale className="w-3 h-3 text-amber-400" />
                <span>Federated Algorithm</span>
              </span>
              <div className="space-y-1.5 text-xs font-mono">
                <div className="flex justify-between p-2 rounded bg-slate-950 border border-slate-800/80">
                  <span className="text-slate-400">FedAvg Mean PR-AUC:</span>
                  <span className="font-bold text-emerald-400">{computeMean(fedavgModels, 'test_pr_auc').toFixed(4)}</span>
                </div>
                <div className="flex justify-between p-2 rounded bg-slate-950 border border-slate-800/80">
                  <span className="text-slate-400">FedProx (μ=0.0001):</span>
                  <span className="font-bold text-amber-400">{computeMean(fedproxModels, 'test_pr_auc').toFixed(4)}</span>
                </div>
                <div className="flex justify-between p-2 rounded bg-slate-950 border border-slate-800/80">
                  <span className="text-slate-400">Relative Delta:</span>
                  <span className="font-bold text-rose-400">
                    {(
                      ((computeMean(fedproxModels, 'test_pr_auc') - computeMean(fedavgModels, 'test_pr_auc')) /
                        (computeMean(fedavgModels, 'test_pr_auc') || 1)) *
                      100
                    ).toFixed(1)}
                    %
                  </span>
                </div>
              </div>
              <p className="text-[10px] text-slate-500 leading-normal">
                FedProx (mu=0.0001) scored lower than FedAvg in these runs; the cause is untested.
              </p>
            </div>

            {/* 3. Distribution Comparison (IID vs Non-IID) */}
            <div className="bg-slate-900 border border-slate-800 rounded-2xl p-4 shadow-md space-y-3">
              <span className="text-[10px] text-slate-400 font-mono uppercase block flex items-center space-x-1">
                <GitCommit className="w-3 h-3 text-cyan-400" />
                <span>Distribution Divergence</span>
              </span>
              <div className="space-y-1.5 text-xs font-mono">
                <div className="flex justify-between p-2 rounded bg-slate-950 border border-slate-800/80">
                  <span className="text-slate-400">IID Mean PR-AUC:</span>
                  <span className="font-bold text-emerald-400">{computeMean(iidModels, 'test_pr_auc').toFixed(4)}</span>
                </div>
                <div className="flex justify-between p-2 rounded bg-slate-950 border border-slate-800/80">
                  <span className="text-slate-400">Non-IID Mean PR-AUC:</span>
                  <span className="font-bold text-slate-300">{computeMean(nonIidModels, 'test_pr_auc').toFixed(4)}</span>
                </div>
                <div className="flex justify-between p-2 rounded bg-slate-950 border border-slate-800/80">
                  <span className="text-slate-400">Non-IID Divergence:</span>
                  <span className="font-bold text-amber-400">
                    {(
                      ((computeMean(nonIidModels, 'test_pr_auc') - computeMean(iidModels, 'test_pr_auc')) /
                        (computeMean(iidModels, 'test_pr_auc') || 1)) *
                      100
                    ).toFixed(1)}
                    %
                  </span>
                </div>
              </div>
              <p className="text-[10px] text-slate-500 leading-normal">
                Client volume skew (83.9% on Client 3) degrades Non-IID generalization across smaller client banks.
              </p>
            </div>

            {/* 4. Centralized vs Federated Baselines */}
            <div className="bg-slate-900 border border-slate-800 rounded-2xl p-4 shadow-md space-y-3">
              <span className="text-[10px] text-slate-400 font-mono uppercase block flex items-center space-x-1">
                <Award className="w-3 h-3 text-teal-400" />
                <span>Centralized vs Federated</span>
              </span>
              <div className="space-y-1.5 text-xs font-mono">
                <div className="flex justify-between p-2 rounded bg-slate-950 border border-slate-800/80">
                  <span className="text-slate-400">Best Centralized (LSTM):</span>
                  <span className="font-bold text-white">0.4771</span>
                </div>
                <div className="flex justify-between p-2 rounded bg-slate-950 border border-slate-800/80">
                  <span className="text-slate-400">Federated IID Champion:</span>
                  <span className="font-bold text-emerald-400">{(championModel?.test_pr_auc ?? 0.4760).toFixed(4)}</span>
                </div>
                <div className="flex justify-between p-2 rounded bg-slate-950 border border-slate-800/80">
                  <span className="text-slate-400">Retention vs Best:</span>
                  <span className="font-bold text-emerald-400">
                    {(((championModel?.test_pr_auc ?? 0.4760) / 0.4771) * 100).toFixed(1)}%
                  </span>
                </div>
              </div>
              <p className="text-[10px] text-slate-500 leading-normal">
                FedSecureAI achieves 99.8% retention of peak centralized performance (0.4760 vs 0.4771) with privacy-preserving federated aggregation.
              </p>
            </div>
          </div>
        </div>
      )}

      {/* SECTION 2: PERFORMANCE HEATMAP */}
      {activeSection === 'heatmap' && (
        <PerformanceHeatmap
          models={filteredModels}
          selectedModelId={selectedModelId}
          onSelectModel={setSelectedModelId}
        />
      )}

      {/* SECTION 3: TRAINING & VALIDATION CONVERGENCE CURVES */}
      {activeSection === 'curves' && (
        <div className="bg-slate-900 border border-slate-800 rounded-3xl p-6 shadow-xl space-y-4">
          <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 border-b border-slate-800 pb-3">
            <div>
              <h3 className="text-sm font-bold text-white flex items-center space-x-2">
                <TrendingUp className="w-4 h-4 text-indigo-400" />
                <span>Round-Level Convergence Curves (5 Communication Rounds)</span>
              </h3>
              <p className="text-[11px] text-slate-400 mt-0.5">
                Evaluated on the centralized validation split at the conclusion of each round.
              </p>
            </div>
            <span className="text-xs font-mono text-slate-400">{curves.length} Round Datapoints Logged</span>
          </div>

          <div className="h-72 w-full flex items-center justify-center pt-2">
            <svg className="w-full h-full" viewBox="0 0 800 240">
              <line x1="60" y1="20" x2="60" y2="190" stroke="#334155" strokeWidth="1" />
              <line x1="60" y1="190" x2="780" y2="190" stroke="#334155" strokeWidth="1" />

              {/* Round X-ticks */}
              {[1, 2, 3, 4, 5].map((rnd) => {
                const x = 60 + ((rnd - 1) / 4) * 720;
                return (
                  <g key={rnd}>
                    <line x1={x} y1="20" x2={x} y2="190" stroke="#1e293b" strokeDasharray="3 3" />
                    <text x={x} y="208" fill="#94a3b8" fontSize="10" fontFamily="monospace" textAnchor="middle">
                      Round {rnd}
                    </text>
                  </g>
                );
              })}

              {/* Convergence lines */}
              {(() => {
                const dCurves = curves.filter((c) => c.dataset === selectedDataset);
                const grouped = new Map<string, ConvergenceCurvePoint[]>();
                dCurves.forEach((pt) => {
                  if (!grouped.has(pt.config_id)) grouped.set(pt.config_id, []);
                  grouped.get(pt.config_id)!.push(pt);
                });

                const palette = ['#6366f1', '#10b981', '#f59e0b', '#ec4899', '#8b5cf6', '#06b6d4', '#ef4444'];
                let pIdx = 0;

                return Array.from(grouped.entries()).map(([cid, pts]) => {
                  if (pts.length < 2) return null;
                  pts.sort((a, b) => a.round - b.round);
                  const color = palette[pIdx++ % palette.length];

                  const getX = (rnd: number) => 60 + ((rnd - 1) / 4) * 720;
                  const getY = (v: number) => 190 - Math.min(1.0, Math.max(0, v / 0.5)) * 170;

                  const pathStr = pts.map((p, i) => `${i === 0 ? 'M' : 'L'} ${getX(p.round).toFixed(1)} ${getY(p.val_pr_auc).toFixed(1)}`).join(' ');

                  return (
                    <g key={cid}>
                      <path d={pathStr} fill="none" stroke={color} strokeWidth="2" opacity="0.8" />
                      {pts.map((p, i) => (
                        <circle key={i} cx={getX(p.round)} cy={getY(p.val_pr_auc)} r="3" fill={color} />
                      ))}
                    </g>
                  );
                });
              })()}
            </svg>
          </div>
          <p className="text-[11px] text-slate-500 font-mono text-center">
            Y-axis represents validation PR-AUC. Round 1 establishes initial global convergence; Rounds 3-5 stabilize typologies.
          </p>
        </div>
      )}

      {/* SECTION 4: THRESHOLD TRADEOFFS & CONFUSION MATRIX */}
      {activeSection === 'tradeoffs' && (
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          {/* Threshold Tradeoffs */}
          <div className="bg-slate-900 border border-slate-800 rounded-3xl p-6 shadow-xl space-y-4">
            <div className="border-b border-slate-800 pb-3">
              <h3 className="text-sm font-bold text-white flex items-center space-x-2">
                <Sliders className="w-4 h-4 text-amber-400" />
                <span>Operational Threshold Tradeoff Curve (τ ∈ [0.1, 0.9])</span>
              </h3>
              <p className="text-[11px] text-slate-400 mt-0.5">
                Precision vs Recall vs Alerts/10k for {activeSelectedModel.model_name}.
              </p>
            </div>

            <div className="h-64 w-full flex items-center justify-center">
              <svg className="w-full h-full" viewBox="0 0 450 220">
                <line x1="45" y1="20" x2="45" y2="180" stroke="#334155" strokeWidth="1" />
                <line x1="45" y1="180" x2="430" y2="180" stroke="#334155" strokeWidth="1" />

                <text x="38" y="24" fill="#64748b" fontSize="9" textAnchor="end">1.0</text>
                <text x="38" y="102" fill="#64748b" fontSize="9" textAnchor="end">0.5</text>
                <text x="38" y="184" fill="#64748b" fontSize="9" textAnchor="end">0.0</text>

                <text x="45" y="196" fill="#64748b" fontSize="9" textAnchor="middle">0.1</text>
                <text x="237" y="196" fill="#64748b" fontSize="9" textAnchor="middle">0.5</text>
                <text x="430" y="196" fill="#64748b" fontSize="9" textAnchor="middle">0.9</text>

                {(() => {
                  const thList = thresholds.filter((t) => t.dataset === selectedDataset);
                  if (thList.length < 2) {
                    return (
                      <text x="240" y="100" fill="#64748b" fontSize="11" textAnchor="middle">
                        Historical sweep records loaded ({thresholds.length} points)
                      </text>
                    );
                  }

                  const getX = (th: number) => 45 + ((th - 0.1) / 0.8) * 385;
                  const getY = (v: number) => 180 - Math.max(0, Math.min(1.0, v)) * 160;

                  const precP = thList.map((t, i) => `${i === 0 ? 'M' : 'L'} ${getX(t.threshold).toFixed(1)} ${getY(t.precision).toFixed(1)}`).join(' ');
                  const recP = thList.map((t, i) => `${i === 0 ? 'M' : 'L'} ${getX(t.threshold).toFixed(1)} ${getY(t.recall).toFixed(1)}`).join(' ');
                  const f1P = thList.map((t, i) => `${i === 0 ? 'M' : 'L'} ${getX(t.threshold).toFixed(1)} ${getY(t.f1).toFixed(1)}`).join(' ');

                  return (
                    <g>
                      <path d={precP} fill="none" stroke="#10b981" strokeWidth="2" />
                      <path d={recP} fill="none" stroke="#6366f1" strokeWidth="2" />
                      <path d={f1P} fill="none" stroke="#f59e0b" strokeWidth="2.5" />
                    </g>
                  );
                })()}
              </svg>
            </div>

            <div className="flex items-center justify-center gap-6 text-[10px] font-mono border-t border-slate-800 pt-3">
              <span className="flex items-center space-x-1.5 text-emerald-400">
                <span className="w-3 h-0.5 bg-emerald-400 rounded inline-block" />
                <span>Precision</span>
              </span>
              <span className="flex items-center space-x-1.5 text-indigo-400">
                <span className="w-3 h-0.5 bg-indigo-400 rounded inline-block" />
                <span>Recall</span>
              </span>
              <span className="flex items-center space-x-1.5 text-amber-400">
                <span className="w-3 h-0.5 bg-amber-400 rounded inline-block" />
                <span>F1-Score</span>
              </span>
            </div>
          </div>

          {/* Interactive Confusion Matrix for Selected Model */}
          <div className="bg-slate-900 border border-slate-800 rounded-3xl p-6 shadow-xl space-y-4">
            <div className="border-b border-slate-800 pb-3 flex items-center justify-between">
              <div>
                <h3 className="text-sm font-bold text-white flex items-center space-x-2">
                  <Target className="w-4 h-4 text-emerald-400" />
                  <span>Held-Out Confusion Matrix: {activeSelectedModel.model_name}</span>
                </h3>
                <p className="text-[11px] text-slate-400 mt-0.5">
                  100% held-out test split at frozen threshold τ = {activeSelectedModel.threshold.toFixed(4)}.
                </p>
              </div>
              <span className="px-2 py-0.5 rounded text-[10px] font-mono bg-indigo-950 text-indigo-300 border border-indigo-700">
                {activeSelectedModel.config_id}
              </span>
            </div>

            {(() => {
              const testTotal = selectedDataset === 'HI-Small' ? 1015564 : 6249112;
              const testPos = selectedDataset === 'HI-Small' ? 1797 : 3783;
              const testNeg = testTotal - testPos;

              const rec = activeSelectedModel.test_recall;
              const prec = activeSelectedModel.test_precision;

              const tp = Math.round(testPos * rec);
              const fn = testPos - tp;
              const fp = prec > 0 ? Math.round(tp / prec) - tp : 0;
              const tn = testNeg - fp;

              return (
                <div className="grid grid-cols-2 gap-3 text-center font-mono">
                  <div className="bg-emerald-950/40 border border-emerald-800/60 p-4 rounded-2xl">
                    <span className="text-[10px] text-emerald-400 font-bold block">TRUE POSITIVES (TP)</span>
                    <span className="text-2xl font-black text-white mt-1 block">{tp.toLocaleString()}</span>
                    <span className="text-[10px] text-slate-400 mt-1 block">{formatPercent(rec, 1)} of Laundering Caught</span>
                  </div>

                  <div className="bg-amber-950/40 border border-amber-800/60 p-4 rounded-2xl">
                    <span className="text-[10px] text-amber-400 font-bold block">FALSE POSITIVES (FP)</span>
                    <span className="text-2xl font-black text-white mt-1 block">{fp.toLocaleString()}</span>
                    <span className="text-[10px] text-slate-400 mt-1 block">{(fp / testTotal * 10000).toFixed(1)} Alerts per 10k</span>
                  </div>

                  <div className="bg-rose-950/40 border border-rose-800/60 p-4 rounded-2xl">
                    <span className="text-[10px] text-rose-400 font-bold block">FALSE NEGATIVES (FN)</span>
                    <span className="text-2xl font-black text-white mt-1 block">{fn.toLocaleString()}</span>
                    <span className="text-[10px] text-slate-400 mt-1 block">Missed Laundering Transactions</span>
                  </div>

                  <div className="bg-slate-950 border border-slate-800 p-4 rounded-2xl">
                    <span className="text-[10px] text-slate-400 font-bold block">TRUE NEGATIVES (TN)</span>
                    <span className="text-2xl font-black text-white mt-1 block">{tn.toLocaleString()}</span>
                    <span className="text-[10px] text-slate-400 mt-1 block">Legitimate Correctly Cleared</span>
                  </div>
                </div>
              );
            })()}
          </div>
        </div>
      )}

      {/* SECTION 5: BANDWIDTH CONSUMPTION & CLIENT PARTITION TOPOLOGY */}
      {activeSection === 'communication' && (
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          {/* Communication Costs */}
          <div className="bg-slate-900 border border-slate-800 rounded-3xl p-6 shadow-xl space-y-4">
            <div className="border-b border-slate-800 pb-3 flex items-center justify-between">
              <div>
                <h3 className="text-sm font-bold text-white flex items-center space-x-2">
                  <Wifi className="w-4 h-4 text-emerald-400" />
                  <span>Federated Communication Bandwidth Consumption</span>
                </h3>
                <p className="text-[11px] text-slate-400 mt-0.5">
                  Full 5-round parameter exchange across 3 client silos (Float32 uncompressed).
                </p>
              </div>
              <span className="text-[10px] font-mono text-emerald-400 bg-emerald-950 px-2 py-0.5 rounded border border-emerald-800">
                Zero FedProx Overhead
              </span>
            </div>

            <div className="space-y-3 font-mono text-xs">
              {(communication.length ? communication : [
                { architecture: 'GRU', parameters: 48641, total_5_round_bandwidth_mb: 5.84, size_per_transfer_kb: 194.56 },
                { architecture: 'LSTM', parameters: 64833, total_5_round_bandwidth_mb: 7.78, size_per_transfer_kb: 259.33 },
                { architecture: 'BiLSTM', parameters: 162433, total_5_round_bandwidth_mb: 19.49, size_per_transfer_kb: 649.73 },
              ]).map((c) => (
                <div key={c.architecture} className="p-3 bg-slate-950 border border-slate-800 rounded-2xl flex items-center justify-between">
                  <div>
                    <span className="font-bold text-white block">{c.architecture} Model</span>
                    <span className="text-[10px] text-slate-400">{c.parameters.toLocaleString()} parameters • {c.size_per_transfer_kb.toFixed(1)} KB/transfer</span>
                  </div>
                  <div className="text-right">
                    <span className="font-bold text-indigo-400 block">{c.total_5_round_bandwidth_mb.toFixed(2)} MB</span>
                    <span className="text-[10px] text-slate-500">5 Rounds (3 Silos)</span>
                  </div>
                </div>
              ))}
            </div>
            <p className="text-[11px] text-slate-400 leading-relaxed">
              FedProx proximal term (μ=0.0001) is calculated exclusively inside local backpropagation on the client GPU/CPU, requiring exactly 1.0x baseline FedAvg network payload.
            </p>
          </div>

          {/* Client Topology Skew */}
          <div className="bg-slate-900 border border-slate-800 rounded-3xl p-6 shadow-xl space-y-4">
            <div className="border-b border-slate-800 pb-3 flex items-center justify-between">
              <div>
                <h3 className="text-sm font-bold text-white flex items-center space-x-2">
                  <Database className="w-4 h-4 text-amber-400" />
                  <span>Non-IID Client Topology & Volume Skew</span>
                </h3>
                <p className="text-[11px] text-slate-400 mt-0.5">
                  Natural volume skew by 'From Bank' creates dominant gradient pull.
                </p>
              </div>
              <span className="text-[10px] font-mono text-amber-400 bg-amber-950 px-2 py-0.5 rounded border border-amber-800">
                20.3x Imbalance Ratio
              </span>
            </div>

            <div className="space-y-3 font-mono text-xs">
              {(clientPartition?.client_distribution || [
                { client_id: 'Client 1', percentage: 12.0, training_samples: 24009, val_loss_divergence: 0.184, bank_type: 'Regional Commercial Bank' },
                { client_id: 'Client 2', percentage: 4.13, training_samples: 8266, val_loss_divergence: 0.241, bank_type: 'Specialized Trade Finance' },
                { client_id: 'Client 3', percentage: 83.86, training_samples: 167725, val_loss_divergence: 0.052, bank_type: 'National Clearing Institution' },
              ]).map((cl) => (
                <div key={cl.client_id} className="p-3 bg-slate-950 border border-slate-800 rounded-2xl flex items-center justify-between">
                  <div>
                    <span className="font-bold text-white block">{cl.client_id} ({cl.bank_type})</span>
                    <span className="text-[10px] text-slate-400">{cl.training_samples.toLocaleString()} samples • Loss div: {cl.val_loss_divergence}</span>
                  </div>
                  <div className="text-right">
                    <span className="font-bold text-amber-400 block">{cl.percentage.toFixed(1)}%</span>
                    <span className="text-[10px] text-slate-500">FedAvg Weight</span>
                  </div>
                </div>
              ))}
            </div>
            <p className="text-[11px] text-slate-400 leading-relaxed">
              In FedAvg, Client 3 accounts for 83.9% of global model weight, pulling aggregated weights away from Client 1 and Client 2 specific transaction typologies.
            </p>
          </div>
        </div>
      )}
    </div>
  );
};
