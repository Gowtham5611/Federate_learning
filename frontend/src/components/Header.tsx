import React from 'react';
import { HealthResponse } from '../types/api';
import { Shield, RefreshCw, FileSpreadsheet, BarChart2, History, Database, Layers, CheckCircle2, AlertCircle } from 'lucide-react';

export type DashboardTab = 'evaluation' | 'analytics' | 'models' | 'ensemble' | 'history';

interface HeaderProps {
  health: HealthResponse | null;
  loadingHealth: boolean;
  onRefreshHealth: () => void;
  activeTab: DashboardTab;
  onSelectTab: (tab: DashboardTab) => void;
  hasEvaluatedData: boolean;
}

export const Header: React.FC<HeaderProps> = ({
  health,
  loadingHealth,
  onRefreshHealth,
  activeTab,
  onSelectTab,
  hasEvaluatedData,
}) => {
  const isReady = health?.readiness === 'ready';

  const navItems: { id: DashboardTab; label: string; icon: React.ReactNode; badge?: string }[] = [
    {
      id: 'evaluation',
      label: 'AML Evaluation',
      icon: <FileSpreadsheet className="w-4 h-4" />,
    },
    {
      id: 'analytics',
      label: 'Results & Analytics',
      icon: <BarChart2 className="w-4 h-4" />,
      badge: hasEvaluatedData ? 'Active' : undefined,
    },
    {
      id: 'models',
      label: 'Model Comparison',
      icon: <Database className="w-4 h-4" />,
    },
    {
      id: 'ensemble',
      label: 'Ensemble Analytics',
      icon: <Layers className="w-4 h-4" />,
    },
    {
      id: 'history',
      label: 'Prediction History',
      icon: <History className="w-4 h-4" />,
      badge: hasEvaluatedData ? 'Data Loaded' : undefined,
    },
  ];

  return (
    <header className="bg-slate-900/95 backdrop-blur-md border-b border-slate-800 sticky top-0 z-50">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex items-center justify-between h-16">
          {/* Logo & Platform Title */}
          <div
            className="flex items-center space-x-3 cursor-pointer"
            onClick={() => onSelectTab('evaluation')}
          >
            <div className="p-2 bg-gradient-to-tr from-indigo-600 to-violet-500 rounded-xl shadow-lg shadow-indigo-600/30">
              <Shield className="w-5 h-5 text-white" />
            </div>
            <div>
              <div className="flex items-center space-x-2">
                <span className="font-bold text-lg text-white tracking-tight">FedSecureAI</span>
                <span className="px-2 py-0.5 text-[10px] font-mono rounded-md bg-emerald-950/80 text-emerald-300 border border-emerald-800/60 font-semibold flex items-center space-x-1">
                  <span className="w-1.5 h-1.5 rounded-full bg-emerald-400 animate-pulse"></span>
                  <span>ENTERPRISE</span>
                </span>
              </div>
              <p className="text-[11px] text-slate-400 font-medium">AML Evaluation Platform</p>
            </div>
          </div>

          {/* Main Navigation (3 core tabs) */}
          <nav className="flex items-center space-x-1 bg-slate-950 p-1 rounded-2xl border border-slate-800">
            {navItems.map((item) => {
              const isActive = activeTab === item.id;
              return (
                <button
                  key={item.id}
                  onClick={() => onSelectTab(item.id)}
                  className={`flex items-center space-x-2 px-4 py-2 rounded-xl text-xs font-semibold transition ${
                    isActive
                      ? 'bg-indigo-600 text-white shadow-md shadow-indigo-600/30'
                      : 'text-slate-400 hover:text-slate-200 hover:bg-slate-900'
                  }`}
                >
                  {item.icon}
                  <span>{item.label}</span>
                  {item.badge && !isActive && (
                    <span className="px-1.5 py-0.2 rounded-full text-[9px] bg-indigo-950 text-indigo-300 border border-indigo-800">
                      {item.badge}
                    </span>
                  )}
                </button>
              );
            })}
          </nav>

          {/* Backend Status Indicator */}
          <div className="hidden sm:flex items-center space-x-3">
            <div className="flex items-center space-x-2 px-3 py-1.5 rounded-xl bg-slate-950 border border-slate-800 text-xs font-mono">
              {isReady ? (
                <>
                  <CheckCircle2 className="w-3.5 h-3.5 text-emerald-400" />
                  <span className="text-emerald-400 font-semibold">Service Online</span>
                </>
              ) : (
                <>
                  <AlertCircle className="w-3.5 h-3.5 text-amber-400" />
                  <span className="text-amber-400 font-semibold">Connecting...</span>
                </>
              )}
            </div>

            <button
              onClick={onRefreshHealth}
              disabled={loadingHealth}
              title="Refresh connection status"
              className="p-2 rounded-xl bg-slate-950 border border-slate-800 text-slate-400 hover:text-white hover:bg-slate-900 transition disabled:opacity-50"
            >
              <RefreshCw className={`w-3.5 h-3.5 ${loadingHealth ? 'animate-spin' : ''}`} />
            </button>
          </div>
        </div>
      </div>
    </header>
  );
};
