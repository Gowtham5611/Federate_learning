import React from 'react';
import { HealthResponse } from '../types/api';
import { Shield, FileSpreadsheet, BarChart2, History, Database, Layers, CheckCircle2, AlertCircle, Sun, Moon } from 'lucide-react';

export type DashboardTab = 'evaluation' | 'analytics' | 'models' | 'ensemble' | 'history';

interface HeaderProps {
  health: HealthResponse | null;
  loadingHealth: boolean;
  onRefreshHealth: () => void;
  activeTab: DashboardTab;
  onSelectTab: (tab: DashboardTab) => void;
  hasEvaluatedData: boolean;
  theme?: 'dark' | 'light';
  onToggleTheme?: () => void;
}

export const Header: React.FC<HeaderProps> = ({
  health,
  activeTab,
  onSelectTab,
  hasEvaluatedData,
  theme = 'dark',
  onToggleTheme,
}) => {
  const isReady = health?.readiness === 'ready';

  const navItems: { id: DashboardTab; label: string; shortLabel: string; icon: React.ReactNode; badge?: string }[] = [
    {
      id: 'evaluation',
      label: 'AML Evaluation',
      shortLabel: 'Evaluate',
      icon: <FileSpreadsheet className="w-4 h-4 shrink-0" />,
    },
    {
      id: 'analytics',
      label: 'Results & Analytics',
      shortLabel: 'Analytics',
      icon: <BarChart2 className="w-4 h-4 shrink-0" />,
      badge: hasEvaluatedData ? 'Active' : undefined,
    },
    {
      id: 'models',
      label: 'Model Comparison',
      shortLabel: 'Models',
      icon: <Database className="w-4 h-4 shrink-0" />,
    },
    {
      id: 'ensemble',
      label: 'Ensemble Analytics',
      shortLabel: 'Ensemble',
      icon: <Layers className="w-4 h-4 shrink-0" />,
    },
    {
      id: 'history',
      label: 'Prediction History',
      shortLabel: 'History',
      icon: <History className="w-4 h-4 shrink-0" />,
      badge: hasEvaluatedData ? 'Loaded' : undefined,
    },
  ];

  return (
    <header className="bg-slate-900/95 backdrop-blur-md border-b border-slate-800 sticky top-0 z-50">
      <div className="max-w-7xl mx-auto px-3 sm:px-6 lg:px-8">
        <div className="flex flex-col md:flex-row md:items-center justify-between py-2 md:h-16 gap-2 min-w-0 max-w-full">
          {/* Top Bar for Mobile: Brand & Status */}
          <div className="flex items-center justify-between">
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
                  <span className="font-black text-base sm:text-lg text-white tracking-tight">FedSecureAI</span>
                  <span className="px-2 py-0.5 text-[10px] font-mono rounded-md bg-emerald-950/80 text-emerald-300 border border-emerald-800/60 font-semibold flex items-center space-x-1 shrink-0">
                    <span className="w-1.5 h-1.5 rounded-full bg-emerald-400 animate-pulse"></span>
                    <span>AML DASHBOARD</span>
                  </span>
                </div>
                <p className="text-[10px] sm:text-[11px] text-slate-400 font-medium truncate">
                  Federated Anti-Money Laundering Architecture
                </p>
              </div>
            </div>

            {/* Mobile Controls: Status & Theme Toggle */}
            <div className="md:hidden flex items-center space-x-2">
              <div className="flex items-center space-x-1.5 px-2.5 py-1 rounded-xl bg-slate-950 border border-slate-800 text-[11px] font-mono">
                {isReady ? (
                  <>
                    <CheckCircle2 className="w-3 h-3 text-emerald-400 shrink-0" />
                    <span className="text-emerald-400 font-semibold text-[10px]">Online</span>
                  </>
                ) : (
                  <>
                    <AlertCircle className="w-3 h-3 text-amber-400 shrink-0" />
                    <span className="text-amber-400 font-semibold text-[10px]">Connecting</span>
                  </>
                )}
              </div>
              {onToggleTheme && (
                <button
                  type="button"
                  onClick={onToggleTheme}
                  aria-label={theme === 'dark' ? 'Switch to light theme' : 'Switch to dark theme'}
                  className="p-1.5 rounded-xl bg-slate-950 border border-slate-800 text-slate-300 hover:text-white transition"
                >
                  {theme === 'dark' ? <Sun className="w-3.5 h-3.5 text-amber-400" /> : <Moon className="w-3.5 h-3.5 text-indigo-400" />}
                </button>
              )}
            </div>
          </div>

          {/* Main Navigation (horizontally scrollable on small screens) */}
          <nav
            aria-label="Main Navigation"
            className="flex items-center space-x-1 bg-slate-950 p-1 rounded-2xl border border-slate-800 overflow-x-auto no-scrollbar scroll-smooth w-full md:w-auto max-w-full min-w-0 shrink"
          >
            {navItems.map((item) => {
              const isActive = activeTab === item.id;
              return (
                <button
                  key={item.id}
                  onClick={() => onSelectTab(item.id)}
                  className={`flex items-center space-x-1.5 sm:space-x-2 px-3 sm:px-3.5 py-2 rounded-xl text-xs font-semibold transition shrink-0 cursor-pointer min-h-[40px] focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-indigo-500 ${
                    isActive
                      ? 'bg-indigo-600 text-white shadow-md shadow-indigo-600/30'
                      : 'text-slate-400 hover:text-slate-200 hover:bg-slate-900/80'
                  }`}
                >
                  {item.icon}
                  <span className="hidden sm:inline whitespace-nowrap">{item.label}</span>
                  <span className="sm:hidden whitespace-nowrap">{item.shortLabel}</span>
                  {item.badge && !isActive && (
                    <span className="px-1.5 py-0.2 rounded-full text-[9px] font-mono bg-indigo-950 text-indigo-300 border border-indigo-800 shrink-0">
                      {item.badge}
                    </span>
                  )}
                </button>
              );
            })}
          </nav>

          {/* Desktop Backend Status Indicator & Theme Toggle */}
          <div className="hidden md:flex items-center space-x-2.5 shrink-0">
            <div className="flex items-center space-x-2 px-3 py-1.5 rounded-xl bg-slate-950 border border-slate-800 text-xs font-mono">
              {isReady ? (
                <>
                  <CheckCircle2 className="w-3.5 h-3.5 text-emerald-400 shrink-0" />
                  <span className="text-emerald-400 font-semibold">Service Online</span>
                </>
              ) : (
                <>
                  <AlertCircle className="w-3.5 h-3.5 text-amber-400 shrink-0" />
                  <span className="text-amber-400 font-semibold">Connecting...</span>
                </>
              )}
            </div>

            {onToggleTheme && (
              <button
                type="button"
                onClick={onToggleTheme}
                aria-label={theme === 'dark' ? 'Switch to light theme' : 'Switch to dark theme'}
                title={theme === 'dark' ? 'Switch to light theme' : 'Switch to dark theme'}
                className="p-2 rounded-xl bg-slate-950 border border-slate-800 text-slate-300 hover:text-white hover:border-slate-700 transition cursor-pointer flex items-center justify-center focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-indigo-500"
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
