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

const APP_BUILD_INFO = {
  gitSha: 'ab9c6df',
  buildTime: '2026-11-03 08:00 UTC',
};

export const App: React.FC = () => {
  const [activeTab, setActiveTab] = useState<DashboardTab>('evaluation');
  const [health, setHealth] = useState<HealthResponse | null>(null);
  const [loadingHealth, setLoadingHealth] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);

  // Theme state: dark / light
  const [theme, setTheme] = useState<'dark' | 'light'>(() => {
    try {
      const saved = localStorage.getItem('fedsecureai_theme');
      return saved === 'light' ? 'light' : 'dark';
    } catch {
      return 'dark';
    }
  });

  useEffect(() => {
    try {
      if (theme === 'light') {
        document.documentElement.classList.add('light');
        document.documentElement.classList.remove('dark');
      } else {
        document.documentElement.classList.add('dark');
        document.documentElement.classList.remove('light');
      }
      localStorage.setItem('fedsecureai_theme', theme);
    } catch {
      // Ignore
    }
  }, [theme]);

  const toggleTheme = () => {
    setTheme((prev) => (prev === 'dark' ? 'light' : 'dark'));
  };

  // Active evaluation run state: strictly null by default until user uploads & executes a CSV
  const [_activeRunId, setActiveRunId] = useState<string | null>(null);
  const [currentResult, setCurrentResult] = useState<CsvEvaluateResponse | null>(null);
  const [multiModelResult, setMultiModelResult] = useState<MultiModelEvaluateResponse | null>(null);

  // Model comparison and ensemble state
  const [selectedDataset, setSelectedDataset] = useState<SupportedDataset>('HI-Small');
  const [researchModels, setResearchModels] = useState<BenchmarkModelRecord[]>([]);

  // Clear any stale auto-loaded run ID on startup
  useEffect(() => {
    try {
      localStorage.removeItem('fedsecureai_active_run_id');
    } catch {
      // Ignore
    }
  }, []);

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
    const pid = evalResult.prediction_id || evalResult.download_id;
    if (pid) {
      setActiveRunId(pid);
      try {
        localStorage.setItem('fedsecureai_active_run_id', pid);
      } catch {
        // Quota exceeded fallback
      }
    }
  };

  const handleClearSavedResults = () => {
    setCurrentResult(null);
    setMultiModelResult(null);
    setActiveRunId(null);
    try {
      localStorage.removeItem('fedsecureai_active_run_id');
      apiService.clearPredictionHistory().catch(() => {});
    } catch {
      // Ignore
    }
  };

  const handleSelectHistoricalRun = async (runId: string) => {
    try {
      const run = await apiService.getPredictionRun(runId);
      setCurrentResult(run);
      setActiveRunId(runId);
      try {
        localStorage.setItem('fedsecureai_active_run_id', runId);
      } catch {
        // Ignore
      }
      setActiveTab('analytics');
    } catch (err) {
      setError('Failed to load historical run: ' + String(err));
    }
  };

  return (
    <div className="min-h-screen bg-slate-950 text-slate-100 flex flex-col font-sans antialiased selection:bg-indigo-500 selection:text-white overflow-x-hidden w-full max-w-full">
      {/* Enterprise Platform Header */}
      <Header
        health={health}
        loadingHealth={loadingHealth}
        onRefreshHealth={checkHealth}
        activeTab={activeTab}
        onSelectTab={setActiveTab}
        hasEvaluatedData={Boolean(currentResult)}
        theme={theme}
        onToggleTheme={toggleTheme}
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
              onSelectHistoricalRun={handleSelectHistoricalRun}
              onClearSavedResults={handleClearSavedResults}
            />
          )}
        </ErrorBoundary>
      </main>

      {/* Footer with Build / Version Stamp & Clear Results */}
      <footer className="border-t border-slate-900 bg-slate-950 py-6 text-center text-xs text-slate-500 font-mono">
        <div className="max-w-7xl mx-auto px-4 flex flex-col sm:flex-row items-center justify-between gap-3">
          <div className="flex items-center space-x-3">
            <span>FedSecureAI • Privacy-Preserving AML Detection</span>
            <span className="text-slate-600">|</span>
            <span className="px-2 py-0.5 rounded bg-slate-900 border border-slate-800 text-[11px] text-indigo-400">
              Build {APP_BUILD_INFO.gitSha} ({APP_BUILD_INFO.buildTime})
            </span>
          </div>

          <div className="flex items-center space-x-3">
            {currentResult && (
              <button
                type="button"
                onClick={handleClearSavedResults}
                className="text-xs text-slate-400 hover:text-rose-400 font-mono transition cursor-pointer"
              >
                Clear Active Evaluation ({currentResult.prediction_id})
              </button>
            )}
          </div>
        </div>
      </footer>
    </div>
  );
};

export default App;
