import React, { useState, useEffect } from 'react';
import { Header, DashboardTab } from './components/Header';
import { AmlEvaluationPage } from './pages/AmlEvaluationPage';
import { ResultsAnalyticsPage } from './pages/ResultsAnalyticsPage';
import { PredictionHistoryPage } from './pages/PredictionHistoryPage';
import { ModelRegistryPage } from './pages/ModelRegistryPage';
import { EnsembleAnalyticsPage } from './pages/EnsembleAnalyticsPage';
import { ErrorBoundary } from './components/ErrorBoundary';
import { apiService, ApiError } from './services/apiService';
import { BenchmarkModelRecord, CsvEvaluateResponse, HealthResponse, MultiModelEvaluateResponse, SupportedDataset } from './types/api';
import { AlertTriangle, RefreshCw } from 'lucide-react';

export const App: React.FC = () => {
  const [activeTab, setActiveTab] = useState<DashboardTab>('evaluation');
  const [health, setHealth] = useState<HealthResponse | null>(null);
  const [loadingHealth, setLoadingHealth] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);

  // Core CSV-driven evaluation result state
  const [currentResult, setCurrentResult] = useState<CsvEvaluateResponse | null>(null);
  const [multiModelResult, setMultiModelResult] = useState<MultiModelEvaluateResponse | null>(null);

  // Model comparison and ensemble state
  const [selectedDataset, setSelectedDataset] = useState<SupportedDataset>('HI-Small');
  const [researchModels, setResearchModels] = useState<BenchmarkModelRecord[]>([]);

  // Fetch initial health and model registry
  const checkHealth = async () => {
    setLoadingHealth(true);
    setError(null);

    try {
      const h = await apiService.getHealth();
      setHealth(h);
    } catch (err: unknown) {
      const msg = err instanceof ApiError ? err.message : err instanceof Error ? err.message : 'Backend connection error';
      setError(msg);
      setHealth(null);
    } finally {
      setLoadingHealth(false);
    }
  };

  const loadResearchModels = async () => {
    try {
      const models = await apiService.getResearchModels();
      setResearchModels(models);
    } catch {
      // Non-blocking fallback
    }
  };

  useEffect(() => {
    checkHealth();
    loadResearchModels();
  }, []);

  const handleEvaluationComplete = (evalResult: CsvEvaluateResponse) => {
    setCurrentResult(evalResult);
  };

  return (
    <div className="min-h-screen bg-slate-950 text-slate-100 flex flex-col font-sans antialiased selection:bg-indigo-500 selection:text-white">
      {/* Enterprise Platform Header */}
      <Header
        health={health}
        loadingHealth={loadingHealth}
        onRefreshHealth={checkHealth}
        activeTab={activeTab}
        onSelectTab={setActiveTab}
        hasEvaluatedData={Boolean(currentResult)}
      />

      {/* Main Content Area */}
      <main className="flex-1 max-w-7xl w-full mx-auto px-4 sm:px-6 lg:px-8 py-8">
        {/* Connection Notice if backend is offline */}
        {error && (
          <div className="mb-6 p-4 bg-rose-950/70 border border-rose-500/50 rounded-2xl flex items-center justify-between text-rose-200 text-xs shadow-lg">
            <div className="flex items-center space-x-3">
              <AlertTriangle className="w-5 h-5 text-rose-400 shrink-0" />
              <div>
                <span className="font-bold text-rose-300">Backend Communication Notice: </span>
                <span>{error}</span>
              </div>
            </div>
            <button
              onClick={checkHealth}
              className="px-3 py-1.5 rounded-lg bg-rose-900/60 hover:bg-rose-800 text-white font-mono text-[11px] transition flex items-center space-x-1 cursor-pointer"
            >
              <RefreshCw className="w-3 h-3" />
              <span>Retry</span>
            </button>
          </div>
        )}

        {/* Tab 1: AML Evaluation (Upload & Execute) */}
        <ErrorBoundary onReset={() => setActiveTab('evaluation')}>
          {activeTab === 'evaluation' && (
            <AmlEvaluationPage
              currentResult={currentResult}
              multiModelResult={multiModelResult}
              onEvaluationComplete={handleEvaluationComplete}
              onMultiModelEvaluationComplete={setMultiModelResult}
              onNavigateTab={setActiveTab}
            />
          )}

          {/* Tab 2: Results & Analytics (Metric Cards, Confusion Matrix, Curves, Multi-Model Comparison) */}
          {activeTab === 'analytics' && (
            <ResultsAnalyticsPage
              result={currentResult}
              multiModelResult={multiModelResult}
              onNavigateToUpload={() => setActiveTab('evaluation')}
            />
          )}

          {/* Tab 3: Model Comparison & Registry */}
          {activeTab === 'models' && (
            <ModelRegistryPage
              models={researchModels}
              selectedDataset={selectedDataset}
              onSelectDataset={setSelectedDataset}
              onNavigateToCsvEval={() => setActiveTab('evaluation')}
            />
          )}

          {/* Tab 4: Ensemble Analytics */}
          {activeTab === 'ensemble' && (
            <EnsembleAnalyticsPage
              selectedDataset={selectedDataset}
              onSelectDataset={setSelectedDataset}
            />
          )}

          {/* Tab 5: Prediction History (Searchable / Sortable Table) */}
          {activeTab === 'history' && (
            <PredictionHistoryPage
              result={currentResult}
              onNavigateToUpload={() => setActiveTab('evaluation')}
            />
          )}
        </ErrorBoundary>
      </main>

      {/* Footer */}
      <footer className="border-t border-slate-900 bg-slate-950 py-6 text-center text-xs text-slate-500 font-mono">
        <div className="max-w-7xl mx-auto px-4 flex flex-col sm:flex-row items-center justify-between gap-2">
          <span>FedSecureAI • Privacy-Preserving Anti-Money Laundering Detection</span>
          <span>Enterprise CSV Evaluation Engine • High-Throughput Stream Inference</span>
        </div>
      </footer>
    </div>
  );
};

export default App;
