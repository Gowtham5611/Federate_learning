import React, { useState, useMemo } from 'react';
import {
  Layers,
  Table as TableIcon,
  HelpCircle,
  Eye,
  EyeOff,
  Info,
} from 'lucide-react';
import { BenchmarkModelRecord } from '../types/api';

export interface MetricDefinition {
  key: string;
  name: string;
  shortLabel: string;
  category: 'performance' | 'error' | 'workload';
  higherIsBetter: boolean;
  units: string;
  format: (val: number | null | undefined) => string;
  description: string;
}

export const HEATMAP_METRICS: MetricDefinition[] = [
  {
    key: 'test_pr_auc',
    name: 'Precision-Recall AUC (PR-AUC)',
    shortLabel: 'PR-AUC',
    category: 'performance',
    higherIsBetter: true,
    units: 'Ratio [0 - 1]',
    format: (v) => (v != null ? Number(v).toFixed(4) : 'N/A'),
    description: 'Primary ranking power invariant to majority-class size; highest scientific validity in imbalanced AML.',
  },
  {
    key: 'test_roc_auc',
    name: 'Receiver Operating Characteristic AUC (ROC-AUC)',
    shortLabel: 'ROC-AUC',
    category: 'performance',
    higherIsBetter: true,
    units: 'Ratio [0 - 1]',
    format: (v) => (v != null ? Number(v).toFixed(4) : 'N/A'),
    description: 'Probability that a randomly chosen laundering case is ranked above a legitimate transaction.',
  },
  {
    key: 'test_f1',
    name: 'F1-Score',
    shortLabel: 'F1',
    category: 'performance',
    higherIsBetter: true,
    units: 'Ratio [0 - 1]',
    format: (v) => (v != null ? Number(v).toFixed(4) : 'N/A'),
    description: 'Harmonic mean of precision and recall at serving threshold.',
  },
  {
    key: 'test_precision',
    name: 'Precision (Positive Predictive Value)',
    shortLabel: 'Precision',
    category: 'performance',
    higherIsBetter: true,
    units: 'Ratio [0 - 1]',
    format: (v) => (v != null ? Number(v).toFixed(4) : 'N/A'),
    description: 'Proportion of generated alerts that are confirmed money laundering cases.',
  },
  {
    key: 'test_recall',
    name: 'Recall (Sensitivity, Detection Rate)',
    shortLabel: 'Recall',
    category: 'performance',
    higherIsBetter: true,
    units: 'Ratio [0 - 1]',
    format: (v) => (v != null ? Number(v).toFixed(4) : 'N/A'),
    description: 'Proportion of all true laundering cases captured by the model.',
  },
  {
    key: 'alerts_per_10k',
    name: 'Alerts per 10,000 Transactions',
    shortLabel: 'Alerts/10k',
    category: 'workload',
    higherIsBetter: false,
    units: 'Alerts / 10k tx',
    format: (v) => (v != null ? Number(v).toFixed(1) : 'N/A'),
    description: 'Investigator workload volume density evaluated against capacity policies; reflects operational volume without intrinsic detection judgment.',
  },
  {
    key: 'brier_score',
    name: 'Brier Calibration Score',
    shortLabel: 'Brier',
    category: 'error',
    higherIsBetter: false,
    units: 'Squared Error',
    format: (v) => (v != null ? Number(v).toFixed(4) : 'N/A'),
    description: 'Mean squared error of probabilistic predictions (0.0 is perfect probability calibration).',
  },
];

interface PerformanceHeatmapProps {
  models: BenchmarkModelRecord[];
  selectedModelId?: string | null;
  onSelectModel?: (configId: string) => void;
  title?: string;
  subtitle?: string;
  className?: string;
}

export const PerformanceHeatmap: React.FC<PerformanceHeatmapProps> = ({
  models,
  selectedModelId,
  onSelectModel,
  title = 'Model Performance Heatmap',
  subtitle = 'Metric-aware color encoding across all evaluated model configurations. Hover or focus for exact values and ranks.',
  className = '',
}) => {
  const [viewMode, setViewMode] = useState<'heatmap' | 'table'>('heatmap');
  const [isVisible, setIsVisible] = useState<boolean>(true);
  const [hoveredCell, setHoveredCell] = useState<{
    configId: string;
    modelName: string;
    metric: MetricDefinition;
    value: number | null;
  } | null>(null);

  // Compute column statistics for continuous normalization
  const columnStats = useMemo(() => {
    const stats: Record<string, { min: number; max: number; count: number }> = {};

    HEATMAP_METRICS.forEach((m) => {
      const validValues = models
        .map((model) => ((model as any)[m.key] != null ? Number((model as any)[m.key]) : null))
        .filter((v): v is number => v !== null && !isNaN(v));

      if (validValues.length > 0) {
        stats[m.key] = {
          min: Math.min(...validValues),
          max: Math.max(...validValues),
          count: validValues.length,
        };
      } else {
        stats[m.key] = { min: 0, max: 1, count: 0 };
      }
    });

    return stats;
  }, [models]);

  // Continuous, colorblind-safe sequential scale per column (zero uncolored cells)
  const getCellStyling = (metric: MetricDefinition, val: number | null) => {
    if (val == null || isNaN(val)) {
      return {
        bgClass: 'bg-slate-100 dark:bg-slate-950/60',
        textClass: 'text-slate-400 italic',
        inlineStyle: {},
      };
    }

    const stat = columnStats[metric.key];
    if (!stat || stat.min === stat.max) {
      return {
        bgClass: 'bg-slate-100 dark:bg-slate-900',
        textClass: 'text-slate-800 dark:text-slate-200',
        inlineStyle: {},
      };
    }

    const rawRatio = Math.max(0.0, Math.min(1.0, (val - stat.min) / Math.max(0.000001, stat.max - stat.min)));
    const scoreRatio = metric.higherIsBetter ? rawRatio : (1.0 - rawRatio);

    // Continuous color scale with perceptually clear, colorblind-safe tints
    if (scoreRatio >= 0.75) {
      return {
        bgClass: 'bg-emerald-100/80 dark:bg-emerald-950/80',
        textClass: 'text-emerald-900 dark:text-emerald-300 font-bold',
        inlineStyle: { backgroundColor: `rgba(16, 185, 129, ${0.18 + scoreRatio * 0.45})` },
      };
    }
    if (scoreRatio >= 0.45) {
      return {
        bgClass: 'bg-teal-100/70 dark:bg-teal-950/70',
        textClass: 'text-teal-900 dark:text-teal-200 font-semibold',
        inlineStyle: { backgroundColor: `rgba(13, 148, 136, ${0.15 + scoreRatio * 0.35})` },
      };
    }
    if (scoreRatio >= 0.20) {
      return {
        bgClass: 'bg-indigo-100/60 dark:bg-indigo-950/60',
        textClass: 'text-indigo-900 dark:text-indigo-200',
        inlineStyle: { backgroundColor: `rgba(99, 102, 241, ${0.12 + scoreRatio * 0.28})` },
      };
    }
    // Lowest tier (0.0 to 0.20): muted slate-blue tint so config_1/config_6 are never uncolored
    return {
      bgClass: 'bg-slate-200/50 dark:bg-slate-900',
      textClass: 'text-slate-700 dark:text-slate-400',
      inlineStyle: { backgroundColor: `rgba(148, 163, 184, ${0.12 + scoreRatio * 0.16})` },
    };
  };

  return (
    <div className={`bg-white dark:bg-slate-900 border border-slate-200 dark:border-slate-800 rounded-3xl p-6 shadow-xl space-y-4 ${className}`}>
      {/* Header with Title and View Controls */}
      <div className="flex flex-col md:flex-row md:items-center justify-between gap-4 border-b border-slate-200 dark:border-slate-800 pb-4">
        <div>
          <div className="flex items-center space-x-2">
            <Layers className="w-5 h-5 text-indigo-500 dark:text-indigo-400" />
            <h3 className="text-base font-bold text-slate-900 dark:text-white tracking-tight">{title}</h3>
            <span className="px-2 py-0.5 rounded-full text-[10px] font-mono font-bold bg-slate-100 dark:bg-slate-800 text-slate-700 dark:text-slate-300 border border-slate-200 dark:border-slate-700">
              {models.length} Configurations
            </span>
          </div>
          <p className="text-xs text-slate-500 dark:text-slate-400 mt-1 max-w-2xl leading-relaxed">{subtitle}</p>
        </div>

        {/* View Mode & Visibility Controls */}
        <div className="flex flex-wrap items-center gap-2">
          <div className="inline-flex bg-slate-100 dark:bg-slate-950 p-1 rounded-xl border border-slate-200 dark:border-slate-800 text-xs font-semibold">
            <button
              onClick={() => {
                setViewMode('heatmap');
                setIsVisible(true);
              }}
              className={`flex items-center space-x-1.5 px-3 py-1.5 rounded-lg transition cursor-pointer ${
                viewMode === 'heatmap' && isVisible
                  ? 'bg-indigo-600 text-white shadow-md shadow-indigo-600/30 font-bold'
                  : 'text-slate-600 dark:text-slate-400 hover:text-slate-900 dark:hover:text-white'
              }`}
            >
              <Layers className="w-3.5 h-3.5" />
              <span>Heatmap</span>
            </button>
            <button
              onClick={() => {
                setViewMode('table');
                setIsVisible(true);
              }}
              className={`flex items-center space-x-1.5 px-3 py-1.5 rounded-lg transition cursor-pointer ${
                viewMode === 'table' && isVisible
                  ? 'bg-indigo-600 text-white shadow-md shadow-indigo-600/30 font-bold'
                  : 'text-slate-600 dark:text-slate-400 hover:text-slate-900 dark:hover:text-white'
              }`}
            >
              <TableIcon className="w-3.5 h-3.5" />
              <span>Accessible Table</span>
            </button>
          </div>

          <button
            onClick={() => setIsVisible(!isVisible)}
            className="flex items-center space-x-1.5 px-3 py-1.5 rounded-xl border border-slate-200 dark:border-slate-800 bg-slate-100 dark:bg-slate-950 text-slate-600 dark:text-slate-400 hover:text-slate-900 dark:hover:text-white text-xs transition cursor-pointer"
          >
            {isVisible ? <EyeOff className="w-3.5 h-3.5" /> : <Eye className="w-3.5 h-3.5" />}
            <span>{isVisible ? 'Collapse' : 'Expand'}</span>
          </button>
        </div>
      </div>

      {isVisible && (
        <div className="space-y-4">
          {/* Active Hover / Focus Detail Inspector */}
          <div className="p-3 bg-slate-50 dark:bg-slate-950/80 rounded-2xl border border-slate-200 dark:border-slate-800/80 min-h-[46px] flex items-center justify-between transition">
            {hoveredCell ? (
              <div className="flex flex-wrap items-center gap-x-4 gap-y-1 text-xs">
                <span className="font-mono font-bold text-indigo-600 dark:text-indigo-300">
                  {hoveredCell.configId}
                </span>
                <span className="text-slate-600 dark:text-slate-300 font-semibold">
                  {hoveredCell.metric.name}:
                </span>
                {hoveredCell.value != null ? (
                  <strong className="text-emerald-600 dark:text-emerald-400 font-mono text-sm">
                    {hoveredCell.metric.format(hoveredCell.value)} {hoveredCell.metric.units}
                  </strong>
                ) : (
                  <span className="text-slate-400 italic text-xs">
                    Value not recorded
                  </span>
                )}
                <span className="px-2 py-0.5 rounded-full text-[10px] bg-slate-200 dark:bg-slate-800 text-slate-700 dark:text-slate-300 border border-slate-300 dark:border-slate-700">
                  {hoveredCell.metric.higherIsBetter ? 'Higher is better' : 'Lower is better'}
                </span>
                <span className="text-slate-500 dark:text-slate-400 text-[11px] font-sans">
                  {hoveredCell.metric.description}
                </span>
              </div>
            ) : (
              <div className="text-slate-500 dark:text-slate-400 flex items-center space-x-2 text-[11px] font-sans">
                <HelpCircle className="w-3.5 h-3.5 text-slate-400 shrink-0" />
                <span>Hover or navigate with keyboard focus over any cell to inspect exact source values, units, and definitions.</span>
              </div>
            )}
          </div>

          {/* MAIN HEATMAP OR TABLE VIEW */}
          {models.length === 0 ? (
            <div className="p-8 text-center bg-slate-50 dark:bg-slate-950/40 rounded-2xl border border-slate-200 dark:border-slate-800 text-xs text-slate-400">
              No model comparison records match the active filter criteria.
            </div>
          ) : viewMode === 'heatmap' ? (
            /* HEATMAP VIEW */
            <div className="overflow-x-auto rounded-2xl border border-slate-200 dark:border-slate-800">
              <table className="w-full text-left border-collapse text-xs font-mono select-none">
                <thead>
                  <tr className="bg-slate-100 dark:bg-slate-950 border-b border-slate-200 dark:border-slate-800 text-slate-600 dark:text-slate-400 uppercase text-[10px]">
                    <th className="py-3 px-3.5 sticky left-0 bg-slate-100 dark:bg-slate-950 z-20 min-w-[130px] border-r border-slate-200 dark:border-slate-800">
                      Configuration
                    </th>
                    <th className="py-3 px-2.5 min-w-[70px]">Arch</th>
                    <th className="py-3 px-2.5 min-w-[70px]">Dist</th>
                    <th className="py-3 px-2.5 min-w-[80px]">Algo</th>
                    {/* Explicit Operating Threshold Column */}
                    <th className="py-3 px-3 text-center min-w-[95px] text-amber-700 dark:text-amber-400 font-bold whitespace-nowrap">
                      Operating θ
                    </th>
                    {HEATMAP_METRICS.map((m) => (
                      <th
                        key={m.key}
                        className="py-3 px-3 text-right min-w-[85px] cursor-help group"
                        title={`${m.name} (${m.higherIsBetter ? 'Higher is better' : 'Lower is better'})\nUnits: ${m.units}\n${m.description}`}
                      >
                        <div className="flex flex-col items-end">
                          <span>{m.shortLabel}</span>
                          <span className="text-[8px] text-slate-500 lowercase font-normal">
                            {m.higherIsBetter ? '▲ best' : '▼ best'}
                          </span>
                        </div>
                      </th>
                    ))}
                  </tr>
                </thead>
                <tbody className="divide-y divide-slate-200 dark:divide-slate-800/60">
                  {models.map((m) => {
                    const cfgId = m.config_id || (m as any).model_id || m.model_name || 'c_unknown';
                    const isSelected = selectedModelId === cfgId;
                    const isChampion =
                      String(m.model_name || '').includes('hi_small_lstm_fedavg_iid_seed42') ||
                      cfgId === 'config_5' ||
                      Boolean((m as any).is_champion);

                    return (
                      <tr
                        key={cfgId}
                        onClick={() => onSelectModel && onSelectModel(cfgId)}
                        className={`transition cursor-pointer ${
                          isSelected ? 'bg-indigo-50/60 dark:bg-indigo-950/40 ring-1 ring-indigo-500' : 'hover:bg-slate-50 dark:hover:bg-slate-800/40'
                        }`}
                      >
                        <td className="py-2.5 px-3.5 font-bold text-slate-900 dark:text-white sticky left-0 bg-white dark:bg-slate-900 z-10 border-r border-slate-200 dark:border-slate-800 truncate max-w-[150px]">
                          <div className="flex items-center space-x-1.5">
                            {isChampion && <span className="w-2 h-2 rounded-full bg-emerald-500 shrink-0" title="Serving Champion" />}
                            <span className="truncate">{cfgId}</span>
                          </div>
                        </td>
                        <td className="py-2.5 px-2.5 text-slate-700 dark:text-slate-300">{m.architecture || 'N/A'}</td>
                        <td className="py-2.5 px-2.5 text-slate-500 dark:text-slate-400">{m.distribution || 'N/A'}</td>
                        <td className="py-2.5 px-2.5 text-slate-500 dark:text-slate-400">{m.algorithm || 'N/A'}</td>
                        {/* Operating Threshold Cell */}
                        <td className="py-2.5 px-3 text-center whitespace-nowrap font-bold text-amber-700 dark:text-amber-400">
                          θ = {Number(m.threshold ?? 0.3378).toFixed(4)}
                        </td>

                        {HEATMAP_METRICS.map((metric) => {
                          const val = (m as any)[metric.key] != null ? Number((m as any)[metric.key]) : null;
                          const styling = getCellStyling(metric, val);

                          return (
                            <td
                              key={metric.key}
                              tabIndex={0}
                              onFocus={() =>
                                setHoveredCell({
                                  configId: cfgId,
                                  modelName: m.model_name || cfgId,
                                  metric,
                                  value: val,
                                })
                              }
                              onMouseEnter={() =>
                                setHoveredCell({
                                  configId: cfgId,
                                  modelName: m.model_name || cfgId,
                                  metric,
                                  value: val,
                                })
                              }
                              onMouseLeave={() => setHoveredCell(null)}
                              className={`py-2 px-3 text-right font-mono transition-all ${styling.bgClass} ${styling.textClass}`}
                              style={styling.inlineStyle}
                            >
                              {metric.format(val)}
                            </td>
                          );
                        })}
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
          ) : (
            /* ACCESSIBLE TABLE VIEW */
            <div className="overflow-x-auto rounded-2xl border border-slate-200 dark:border-slate-800">
              <table className="w-full text-left border-collapse text-xs font-mono">
                <thead>
                  <tr className="bg-slate-100 dark:bg-slate-950 border-b border-slate-200 dark:border-slate-800 text-slate-600 dark:text-slate-400 uppercase text-[10px]">
                    <th className="py-3 px-3.5">Config</th>
                    <th className="py-3 px-2.5">Arch</th>
                    <th className="py-3 px-2.5">Dist</th>
                    <th className="py-3 px-2.5">Algo</th>
                    <th className="py-3 px-3 text-center text-amber-700 dark:text-amber-400 font-bold">Operating θ</th>
                    {HEATMAP_METRICS.map((m) => (
                      <th key={m.key} className="py-3 px-3 text-right">
                        {m.shortLabel}
                      </th>
                    ))}
                  </tr>
                </thead>
                <tbody className="divide-y divide-slate-200 dark:divide-slate-800/60">
                  {models.map((m) => {
                    const cfgId = m.config_id || (m as any).model_id || 'c_unknown';
                    return (
                      <tr key={cfgId} className="hover:bg-slate-50 dark:hover:bg-slate-800/40">
                        <td className="py-2.5 px-3.5 font-bold text-slate-900 dark:text-white">{cfgId}</td>
                        <td className="py-2.5 px-2.5 text-slate-700 dark:text-slate-300">{m.architecture}</td>
                        <td className="py-2.5 px-2.5 text-slate-500 dark:text-slate-400">{m.distribution}</td>
                        <td className="py-2.5 px-2.5 text-slate-500 dark:text-slate-400">{m.algorithm}</td>
                        <td className="py-2.5 px-3 text-center font-bold text-amber-700 dark:text-amber-400">
                          θ = {Number(m.threshold ?? 0.3378).toFixed(4)}
                        </td>
                        {HEATMAP_METRICS.map((metric) => (
                          <td key={metric.key} className="py-2.5 px-3 text-right font-mono text-slate-800 dark:text-slate-200">
                            {metric.format((m as any)[metric.key] != null ? Number((m as any)[metric.key]) : null)}
                          </td>
                        ))}
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
          )}

          {/* Continuous Color Scale Legend & Discrepancy Note */}
          <div className="flex flex-col sm:flex-row items-center justify-between gap-3 pt-3 border-t border-slate-200 dark:border-slate-800 text-xs font-mono">
            <div className="flex items-center space-x-2">
              <span className="text-slate-500 dark:text-slate-400 text-[11px]">Continuous Scale:</span>
              <span className="text-[10px] text-slate-500 font-medium">Lowest rank</span>
              <div className="w-24 sm:w-32 h-2.5 rounded-full bg-gradient-to-r from-slate-400 via-teal-500 to-emerald-500" />
              <span className="text-[10px] text-emerald-600 dark:text-emerald-400 font-bold">Highest rank</span>
            </div>
            <div className="flex items-center space-x-1.5 text-slate-500 dark:text-slate-400 text-[11px] font-sans">
              <Info className="w-3.5 h-3.5 text-indigo-500 shrink-0" />
              <span>Operating metrics are calculated at each model's frozen validation cutoff θ. PR-AUC and ROC-AUC are ranking metrics invariant to threshold.</span>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};
