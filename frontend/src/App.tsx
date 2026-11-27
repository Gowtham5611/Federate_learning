import React, { useState, useEffect } from 'react';
import { Header, DashboardTab } from './components/Header';
import { AmlEvaluationPage } from './pages/AmlEvaluationPage';
import { ResultsAnalyticsPage } from './pages/ResultsAnalyticsPage';
import { PredictionHistoryPage } from './pages/PredictionHistoryPage';
import { ModelRegistryPage } from './pages/ModelRegistryPage';
import { EnsembleAnalyticsPage } from './pages/EnsembleAnalyticsPage';
import { ComparativeAnalysisPage } from './pages/ComparativeAnalysisPage';
import { TechnologyStackPage } from './pages/TechnologyStackPage';
import { ErrorBoundary } from './components/ErrorBoundary';
import { apiService, ApiError } from './services/apiService';
import { BenchmarkModelRecord, CsvEvaluateResponse, HealthResponse, MultiModelEvaluateResponse, SupportedDataset } from './types/api';
import { FALLBACK_BENCHMARK_MODELS } from './data/fallbackModels';
import { AlertTriangle, RefreshCw, FileSpreadsheet, BarChart2, Database, History, MoreHorizontal, Layers, BookOpen, Cpu, X } from 'lucide-react';

const TAB_TO_PATH: Record<DashboardTab, string> = {
  evaluation: '/',
  analytics: '/analytics',
  models: '/models',
  ensemble: '/ensemble',
  history: '/history',
  comparative: '/comparative-analysis',
  'tech-stack': '/tech-stack',
};

const pathToTab = (pathname: string): DashboardTab => {
  const clean = pathname.replace(/\/+$/, '') || '/';
  if (clean === '/comparative-analysis') return 'comparative';
  if (clean === '/tech-stack') return 'tech-stack';
  if (clean === '/analytics') return 'analytics';
  if (clean === '/models') return 'models';
  if (clean === '/ensemble') return 'ensemble';
  if (clean === '/history') return 'history';
  return 'evaluation';
};

export const App: React.FC = () => {
  const [activeTab, setActiveTabState] = useState<DashboardTab>(() =>
    typeof window !== 'undefined' ? pathToTab(window.location.pathname) : 'evaluation'
  );
  const [mobileMoreOpen, setMobileMoreOpen] = useState(false);

  const setActiveTab = (tab: DashboardTab) => {
    setActiveTabState(tab);
    if (typeof window !== 'undefined') {
      const targetPath = TAB_TO_PATH[tab] || '/';
      if (window.location.pathname !== targetPath) {
        window.history.pushState({ tab }, '', targetPath);
      }
    }
  };

  useEffect(() => {
    const onPopState = () => {
      setActiveTabState(pathToTab(window.location.pathname));
    };
    window.addEventListener('popstate', onPopState);
    return () => window.removeEventListener('popstate', onPopState);
  }, []);
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
  const [researchModels, setResearchModels] = useState<BenchmarkModelRecord[]>(FALLBACK_BENCHMARK_MODELS);

  // Fetch initial health and model registry
  const checkHealth = async () => {
    setLoadingHealth(true);
    setError(null);

    try {
      const h = await apiService.getHealth();
      setHealth(h);
      setError(null);
      // Asynchronously probe readiness
      apiService.getReady().then((r) => {
        if (r && r.status === 'degraded' && h) {
          setHealth({ ...h, readiness: 'initializing' });
        }
      }).catch(() => {});
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

    // Keep cloud server warm while dashboard is active (non-overlapping 3.5-min ping)
    let pingInFlight = false;
    const keepWarmInterval = setInterval(() => {
      if (pingInFlight) return;
      pingInFlight = true;
      apiService
        .probeHealth()
        .catch(() => {})
        .finally(() => {
          pingInFlight = false;
        });
    }, 3.5 * 60 * 1000);

    return () => clearInterval(keepWarmInterval);
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
    <div className="min-h-screen bg-slate-50 dark:bg-slate-950 text-slate-900 dark:text-slate-100 flex flex-col font-sans antialiased selection:bg-indigo-500 selection:text-white overflow-x-hidden w-full max-w-full transition-colors duration-150">
      {/* Enterprise Platform Header */}
      <Header
        health={health}
        loadingHealth={loadingHealth}
        serviceError={error}
        onRefreshHealth={checkHealth}
        activeTab={activeTab}
        onSelectTab={setActiveTab}
        hasEvaluatedData={Boolean(currentResult)}
        theme={theme}
        onToggleTheme={toggleTheme}
      />

      {/* Main Content Area */}
      <main className="flex-1 max-w-7xl w-full mx-auto px-4 sm:px-6 lg:px-8 pt-6 pb-24 md:pb-8">
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
              onClearEvaluation={handleClearSavedResults}
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

          {/* Tab 6: Comparative Analysis (opened via Explore dropdown or /comparative-analysis) */}
          {activeTab === 'comparative' && <ComparativeAnalysisPage />}

          {/* Tab 7: Technology Stack & Walkthrough (opened via Explore dropdown or /tech-stack) */}
          {activeTab === 'tech-stack' && <TechnologyStackPage />}
        </ErrorBoundary>
      </main>

      {/* Clean Footer (no build hash or pred_ IDs) */}
      <footer className="border-t border-slate-200 dark:border-slate-900 bg-white dark:bg-slate-950 py-5 pb-20 md:pb-5 text-center text-xs text-slate-500 transition-colors">
        <div className="max-w-7xl mx-auto px-4 flex flex-col sm:flex-row items-center justify-between gap-2">
          <span>FedSecureAI · Federated Anti-Money Laundering Detection</span>
          <span className="text-slate-400">Sequence Evaluation & Research Dashboard</span>
        </div>
      </footer>

      {/* Mobile Bottom Tab Bar (<768px): 4 Main Tabs + More Sheet */}
      <nav
        aria-label="Mobile Navigation"
        className="md:hidden fixed bottom-0 inset-x-0 z-50 bg-white/95 dark:bg-slate-900/95 backdrop-blur-md border-t border-slate-200 dark:border-slate-800 px-1 py-1 flex items-center justify-around"
      >
        {(
          [
            { id: 'evaluation', label: 'Evaluate', icon: <FileSpreadsheet className="w-5 h-5" /> },
            { id: 'analytics', label: 'Results', icon: <BarChart2 className="w-5 h-5" /> },
            { id: 'models', label: 'Models', icon: <Database className="w-5 h-5" /> },
            { id: 'history', label: 'History', icon: <History className="w-5 h-5" /> },
          ] as { id: DashboardTab; label: string; icon: React.ReactNode }[]
        ).map((item) => {
          const isActive = activeTab === item.id;
          return (
            <button
              key={item.id}
              type="button"
              onClick={() => {
                setMobileMoreOpen(false);
                setActiveTab(item.id);
              }}
              className={`flex flex-col items-center justify-center min-w-[60px] min-h-[44px] px-2 py-1 rounded-xl text-[11px] font-semibold transition cursor-pointer ${
                isActive ? 'text-indigo-600 dark:text-indigo-400' : 'text-slate-500 dark:text-slate-400'
              }`}
            >
              {item.icon}
              <span className="mt-0.5">{item.label}</span>
            </button>
          );
        })}
        <button
          type="button"
          onClick={() => setMobileMoreOpen((prev) => !prev)}
          aria-expanded={mobileMoreOpen}
          className={`flex flex-col items-center justify-center min-w-[60px] min-h-[44px] px-2 py-1 rounded-xl text-[11px] font-semibold transition cursor-pointer ${
            ['ensemble', 'comparative', 'tech-stack'].includes(activeTab) || mobileMoreOpen
              ? 'text-indigo-600 dark:text-indigo-400'
              : 'text-slate-500 dark:text-slate-400'
          }`}
        >
          <MoreHorizontal className="w-5 h-5" />
          <span className="mt-0.5">More</span>
        </button>
      </nav>

      {/* Mobile "More" Bottom Sheet */}
      {mobileMoreOpen && (
        <div
          className="md:hidden fixed inset-0 z-50 bg-black/60 backdrop-blur-sm flex items-end"
          onClick={() => setMobileMoreOpen(false)}
        >
          <div
            className="w-full bg-white dark:bg-slate-900 border-t border-slate-200 dark:border-slate-800 rounded-t-3xl p-5 space-y-2"
            onClick={(e) => e.stopPropagation()}
          >
            <div className="flex items-center justify-between pb-2 border-b border-slate-200 dark:border-slate-800">
              <span className="text-sm font-bold text-slate-900 dark:text-white">More Pages</span>
              <button
                type="button"
                onClick={() => setMobileMoreOpen(false)}
                aria-label="Close menu"
                className="p-2 rounded-xl text-slate-400 hover:text-white min-h-[44px] min-w-[44px] flex items-center justify-center"
              >
                <X className="w-5 h-5" />
              </button>
            </div>
            {(
              [
                { id: 'ensemble', label: 'Ensemble Analytics', icon: <Layers className="w-5 h-5 text-violet-400" /> },
                { id: 'comparative', label: 'Comparative Analysis (Literature)', icon: <BookOpen className="w-5 h-5 text-indigo-400" /> },
                { id: 'tech-stack', label: 'Technology Stack & Walkthrough', icon: <Cpu className="w-5 h-5 text-emerald-400" /> },
              ] as { id: DashboardTab; label: string; icon: React.ReactNode }[]
            ).map((m) => (
              <button
                key={m.id}
                type="button"
                onClick={() => {
                  setMobileMoreOpen(false);
                  setActiveTab(m.id);
                }}
                className={`w-full flex items-center space-x-3 px-4 py-3 rounded-2xl text-sm font-semibold min-h-[48px] transition cursor-pointer ${
                  activeTab === m.id
                    ? 'bg-indigo-600 text-white'
                    : 'bg-slate-100 dark:bg-slate-800/70 text-slate-800 dark:text-slate-200'
                }`}
              >
                {m.icon}
                <span>{m.label}</span>
              </button>
            ))}
          </div>
        </div>
      )}
    </div>
  );
};

export default App;
