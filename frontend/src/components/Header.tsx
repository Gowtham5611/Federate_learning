import React from 'react';
import { HealthResponse } from '../types/api';
import { Shield, FileSpreadsheet, BarChart2, History, Database, Layers, CheckCircle2, AlertCircle, RefreshCw, Sun, Moon } from 'lucide-react';

export type DashboardTab = 'evaluation' | 'analytics' | 'models' | 'ensemble' | 'history';

interface HeaderProps {
  health: HealthResponse | null;
  loadingHealth: boolean;
  serviceError?: string | null;
  onRefreshHealth: () => void;
  activeTab: DashboardTab;
  onSelectTab: (tab: DashboardTab) => void;
  hasEvaluatedData: boolean;
  theme?: 'dark' | 'light';
  onToggleTheme?: () => void;
}

export const Header: React.FC<HeaderProps> = ({
  health,
  loadingHealth,
  serviceError,
  onRefreshHealth,
  activeTab,
  onSelectTab,
  hasEvaluatedData,
  theme = 'dark',
  onToggleTheme,
}) => {
  // Determine exact health state: connected | starting | degraded | offline
  let statusBadge: {
    state: 'connected' | 'starting' | 'degraded' | 'offline';
    label: string;
    icon: React.ReactNode;
    colorClasses: string;
    tooltip: string;
  };

  if (loadingHealth && !health) {
    statusBadge = {
      state: 'starting',
      label: 'Starting...',
      icon: <span className="w-2 h-2 rounded-full bg-amber-500 animate-ping" />,
      colorClasses: 'bg-amber-50 dark:bg-amber-950/80 text-amber-700 dark:text-amber-300 border-amber-200 dark:border-amber-800/60',
      tooltip: 'FastAPI service is starting up on cloud infrastructure...',
    };
  } else if (!health || serviceError) {
    statusBadge = {
      state: 'offline',
      label: 'Offline',
      icon: <AlertCircle className="w-3.5 h-3.5 text-rose-500 dark:text-rose-400 shrink-0" />,
      colorClasses: 'bg-rose-50 dark:bg-rose-950/80 text-rose-700 dark:text-rose-300 border-rose-200 dark:border-rose-800/60',
      tooltip: serviceError ? `Backend communication notice: ${serviceError}` : 'Backend service offline or unreachable.',
    };
  } else if (health.status === 'degraded' || (health.loaded_models && health.loaded_models.length === 0)) {
    statusBadge = {
      state: 'degraded',
      label: 'Degraded',
      icon: <AlertCircle className="w-3.5 h-3.5 text-orange-500 dark:text-orange-400 shrink-0" />,
      colorClasses: 'bg-orange-50 dark:bg-orange-950/80 text-orange-700 dark:text-orange-300 border-orange-200 dark:border-orange-800/60',
      tooltip: 'Service is responding but some model checkpoints or metadata are incomplete.',
    };
  } else if (health.readiness === 'initializing') {
    statusBadge = {
      state: 'starting',
      label: 'Starting',
      icon: <span className="w-2 h-2 rounded-full bg-amber-500 animate-pulse" />,
      colorClasses: 'bg-amber-50 dark:bg-amber-950/80 text-amber-700 dark:text-amber-300 border-amber-200 dark:border-amber-800/60',
      tooltip: 'FastAPI process online; verifying champion models...',
    };
  } else {
    statusBadge = {
      state: 'connected',
      label: 'Connected',
      icon: <CheckCircle2 className="w-3.5 h-3.5 text-emerald-500 dark:text-emerald-400 shrink-0" />,
      colorClasses: 'bg-emerald-50 dark:bg-emerald-950/80 text-emerald-700 dark:text-emerald-300 border-emerald-200 dark:border-emerald-800/60',
      tooltip: `Connected to ${health.app_name} v${health.app_version} (${health.loaded_models?.length || 2} models verified on ${health.device || 'cpu'})`,
    };
  }

  const navItems: { id: DashboardTab; label: string; mediumLabel: string; shortLabel: string; icon: React.ReactNode; badge?: string }[] = [
    {
      id: 'evaluation',
      label: 'AML Evaluation',
      mediumLabel: 'Evaluation',
      shortLabel: 'Evaluate',
      icon: <FileSpreadsheet className="w-4 h-4 shrink-0" />,
    },
    {
      id: 'analytics',
      label: 'Results & Analytics',
      mediumLabel: 'Analytics',
      shortLabel: 'Analytics',
      icon: <BarChart2 className="w-4 h-4 shrink-0" />,
      badge: hasEvaluatedData ? 'Active' : undefined,
    },
    {
      id: 'models',
      label: 'Model Comparison',
      mediumLabel: 'Model Registry',
      shortLabel: 'Models',
      icon: <Database className="w-4 h-4 shrink-0" />,
    },
    {
      id: 'ensemble',
      label: 'Ensemble Analytics',
      mediumLabel: 'Ensemble',
      shortLabel: 'Ensemble',
      icon: <Layers className="w-4 h-4 shrink-0" />,
    },
    {
      id: 'history',
      label: 'Prediction History',
      mediumLabel: 'History & Triage',
      shortLabel: 'History',
      icon: <History className="w-4 h-4 shrink-0" />,
      badge: hasEvaluatedData ? 'Loaded' : undefined,
    },
  ];

  return (
    <header className="bg-white/95 dark:bg-slate-900/95 backdrop-blur-md border-b border-slate-200 dark:border-slate-800 sticky top-0 z-50 transition-colors duration-150">
      <div className="max-w-7xl mx-auto px-3 sm:px-6 lg:px-8">
        <div className="flex flex-col md:flex-row md:items-center justify-between py-2 md:h-16 gap-2.5 min-w-0 max-w-full">
          {/* Top Bar for Mobile / Left Brand for Desktop */}
          <div className="flex items-center justify-between shrink-0">
            {/* Logo & Platform Title */}
            <div
              className="flex items-center space-x-2.5 cursor-pointer focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-indigo-500 rounded-xl"
              onClick={() => onSelectTab('evaluation')}
              role="button"
              tabIndex={0}
              onKeyDown={(e) => {
                if (e.key === 'Enter' || e.key === ' ') onSelectTab('evaluation');
              }}
            >
              <div className="p-2 bg-gradient-to-tr from-indigo-600 to-violet-500 rounded-xl shadow-lg shadow-indigo-600/30 shrink-0">
                <Shield className="w-5 h-5 text-white" />
              </div>
              <div>
                <div className="flex items-center space-x-2">
                  <span className="font-black text-base sm:text-lg text-slate-900 dark:text-white tracking-tight">
                    FedSecureAI
                  </span>
                  <span className="px-2 py-0.5 text-[10px] font-mono rounded-md bg-emerald-50 dark:bg-emerald-950/80 text-emerald-700 dark:text-emerald-300 border border-emerald-200 dark:border-emerald-800/60 font-semibold flex items-center space-x-1 shrink-0">
                    <span className="w-1.5 h-1.5 rounded-full bg-emerald-500 dark:bg-emerald-400 animate-pulse" />
                    <span>AML DASHBOARD</span>
                  </span>
                </div>
                <p className="text-[10px] sm:text-[11px] text-slate-500 dark:text-slate-400 font-medium truncate">
                  Federated Anti-Money Laundering Architecture
                </p>
              </div>
            </div>

            {/* Mobile Controls: Status & Theme Toggle */}
            <div className="md:hidden flex items-center space-x-2">
              <div
                title={statusBadge.tooltip}
                className={`flex items-center space-x-1.5 px-2.5 py-1 rounded-xl border text-[11px] font-mono transition-all ${statusBadge.colorClasses}`}
              >
                {statusBadge.icon}
                <span className="font-semibold text-[10px]">{statusBadge.label}</span>
                <button
                  type="button"
                  onClick={onRefreshHealth}
                  disabled={loadingHealth}
                  title="Retry / Refresh connection"
                  className="p-0.5 rounded hover:bg-black/10 dark:hover:bg-white/10 transition cursor-pointer"
                >
                  <RefreshCw className={`w-2.5 h-2.5 ${loadingHealth ? 'animate-spin' : ''}`} />
                </button>
              </div>
              {onToggleTheme && (
                <button
                  type="button"
                  onClick={onToggleTheme}
                  aria-label={theme === 'dark' ? 'Switch to light theme' : 'Switch to dark theme'}
                  className="p-1.5 rounded-xl bg-slate-100 dark:bg-slate-950 border border-slate-200 dark:border-slate-800 text-slate-600 dark:text-slate-300 hover:text-slate-900 dark:hover:text-white transition"
                >
                  {theme === 'dark' ? <Sun className="w-3.5 h-3.5 text-amber-400" /> : <Moon className="w-3.5 h-3.5 text-indigo-500" />}
                </button>
              )}
            </div>
          </div>

          {/* Main Navigation (horizontally scrollable with snap on small screens, unclipped on desktop) */}
          <nav
            aria-label="Main Navigation"
            className="flex items-center space-x-1 bg-slate-100 dark:bg-slate-950 p-1 rounded-2xl border border-slate-200 dark:border-slate-800 overflow-x-auto no-scrollbar scroll-smooth snap-x snap-mandatory w-full md:w-auto max-w-full min-w-0 shrink"
          >
            {navItems.map((item) => {
              const isActive = activeTab === item.id;
              return (
                <button
                  key={item.id}
                  onClick={() => onSelectTab(item.id)}
                  className={`flex items-center space-x-1.5 sm:space-x-2 px-2.5 sm:px-3 py-1.5 sm:py-2 rounded-xl text-xs font-semibold transition shrink-0 cursor-pointer min-h-[38px] snap-start focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-indigo-500 ${
                    isActive
                      ? 'bg-indigo-600 text-white shadow-md shadow-indigo-600/30 font-bold'
                      : 'text-slate-600 dark:text-slate-400 hover:text-slate-900 dark:hover:text-slate-200 hover:bg-slate-200/60 dark:hover:bg-slate-900/80'
                  }`}
                >
                  {item.icon}
                  {/* Full label on wide desktops, medium label on standard desktops, short on mobile */}
                  <span className="hidden xl:inline whitespace-nowrap">{item.label}</span>
                  <span className="hidden md:inline xl:hidden whitespace-nowrap">{item.mediumLabel}</span>
                  <span className="md:hidden whitespace-nowrap">{item.shortLabel}</span>
                  {item.badge && !isActive && (
                    <span className="px-1.5 py-0.2 rounded-full text-[9px] font-mono bg-indigo-100 dark:bg-indigo-950 text-indigo-700 dark:text-indigo-300 border border-indigo-200 dark:border-indigo-800 shrink-0">
                      {item.badge}
                    </span>
                  )}
                </button>
              );
            })}
          </nav>

          {/* Desktop Backend Status Indicator & Theme Toggle */}
          <div className="hidden md:flex items-center space-x-2.5 shrink-0">
            <div
              title={statusBadge.tooltip}
              className={`flex items-center space-x-2 px-3 py-1.5 rounded-xl border text-xs font-mono transition-all ${statusBadge.colorClasses}`}
            >
              {statusBadge.icon}
              <span className="font-semibold">{statusBadge.label}</span>
              <button
                type="button"
                onClick={onRefreshHealth}
                disabled={loadingHealth}
                title="Retry / Refresh connection"
                className="ml-1 p-0.5 rounded hover:bg-black/10 dark:hover:bg-white/10 transition cursor-pointer"
              >
                <RefreshCw className={`w-3 h-3 ${loadingHealth ? 'animate-spin' : ''}`} />
              </button>
            </div>

            {onToggleTheme && (
              <button
                type="button"
                onClick={onToggleTheme}
                aria-label={theme === 'dark' ? 'Switch to light theme' : 'Switch to dark theme'}
                title={theme === 'dark' ? 'Switch to light theme' : 'Switch to dark theme'}
                className="p-2 rounded-xl bg-slate-100 dark:bg-slate-950 border border-slate-200 dark:border-slate-800 text-slate-600 dark:text-slate-300 hover:text-slate-900 dark:hover:text-white hover:border-slate-300 dark:hover:border-slate-700 transition cursor-pointer flex items-center justify-center focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-indigo-500"
              >
                {theme === 'dark' ? (
                  <Sun className="w-4 h-4 text-amber-400" />
                ) : (
                  <Moon className="w-4 h-4 text-indigo-500" />
                )}
              </button>
            )}
          </div>
        </div>
      </div>
    </header>
  );
};
export default Header;
