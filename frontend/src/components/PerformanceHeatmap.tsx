import React, { useState, useMemo } from 'react';
import {
  Layers,
  Table as TableIcon,
  HelpCircle,
  Eye,
  EyeOff,
} from 'lucide-react';

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
    key: 'test_f2',
    name: 'F2-Score (Recall-Weighted)',
    shortLabel: 'F2',
    category: 'performance',
    higherIsBetter: true,
    units: 'Ratio [0 - 1]',
    format: (v) => (v != null ? Number(v).toFixed(4) : 'N/A'),
    description: 'Recall-weighted harmonic mean prioritizing missed-money-laundering detection.',
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
    key: 'test_specificity',
    name: 'Specificity (True Negative Rate)',
    shortLabel: 'Specificity',
    category: 'performance',
    higherIsBetter: true,
    units: 'Ratio [0 - 1]',
    format: (v) => (v != null ? Number(v).toFixed(4) : 'N/A'),
    description: 'Proportion of legitimate transactions correctly cleared without an alert.',
  },
  {
    key: 'test_accuracy',
    name: 'Accuracy',
    shortLabel: 'Accuracy',
    category: 'performance',
    higherIsBetter: true,
    units: 'Ratio [0 - 1]',
    format: (v) => (v != null ? Number(v).toFixed(4) : 'N/A'),
    description: 'Overall classification accuracy (caution: uninformative under 0.18% class prevalence).',
  },
  {
    key: 'test_balanced_accuracy',
    name: 'Balanced Accuracy',
    shortLabel: 'Bal. Acc',
    category: 'performance',
    higherIsBetter: true,
    units: 'Ratio [0 - 1]',
    format: (v) => (v != null ? Number(v).toFixed(4) : 'N/A'),
    description: 'Unweighted macro-average of sensitivity and specificity.',
  },
  {
    key: 'test_mcc',
    name: "Matthews Correlation Coefficient (MCC)",
    shortLabel: 'MCC',
    category: 'performance',
    higherIsBetter: true,
    units: 'Ratio [-1 to +1]',
    format: (v) => (v != null ? Number(v).toFixed(4) : 'N/A'),
    description: 'Contingency matrix correlation coefficient accounting for TP, FP, FN, and TN.',
  },
  {
    key: 'test_cohen_kappa',
    name: "Cohen's Kappa (κ)",
    shortLabel: 'Kappa',
    category: 'performance',
    higherIsBetter: true,
    units: 'Ratio [-1 to +1]',
    format: (v) => (v != null ? Number(v).toFixed(4) : 'N/A'),
    description: 'Inter-rater agreement above chance between predictions and actual labels.',
  },
  {
    key: 'alerts_per_10k',
    name: 'Alerts per 10,000 Transactions',
    shortLabel: 'Alerts/10k',
    category: 'workload',
    higherIsBetter: false,
    units: 'Alerts / 10k tx',
    format: (v) => (v != null ? Number(v).toFixed(1) : 'N/A'),
    description: 'Investigator workload density evaluated against capacity policies; reflects operational volume without intrinsic detection judgment.',
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
  {
    key: 'test_ece',
    name: 'Expected Calibration Error (ECE)',
    shortLabel: 'ECE',
    category: 'error',
    higherIsBetter: false,
    units: 'Error [0 - 1]',
    format: (v) => (v != null ? Number(v).toFixed(4) : 'N/A'),
    description: 'Bin-weighted difference between confidence and empirical accuracy.',
  },
  {
    key: 'log_loss',
    name: 'Log Loss (Binary Cross-Entropy)',
    shortLabel: 'Log Loss',
    category: 'error',
    higherIsBetter: false,
    units: 'Nats',
    format: (v) => (v != null ? Number(v).toFixed(4) : 'N/A'),
    description: 'Cross-entropy penalizing confident incorrect classifications.',
  },
];

interface PerformanceHeatmapProps {
  models: Array<Record<string, any>>;
  title?: string;
  subtitle?: string;
  selectedModelId?: string | null;
  onSelectModel?: (configId: string) => void;
  className?: string;
}

export const PerformanceHeatmap: React.FC<PerformanceHeatmapProps> = ({
  models,
  title = 'Cross-Configuration Performance Heatmap',
  subtitle = 'Metric-aware color encoding across all evaluated model configurations. Hover or focus for exact values and ranks.',
  selectedModelId,
  onSelectModel,
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

  // Compute column-wise min and max across all provided models
  const columnStats = useMemo(() => {
    const stats: Record<
      string,
      {
        min: number;
        max: number;
        values: number[];
        availableCount: number;
      }
    > = {};

    HEATMAP_METRICS.forEach((m) => {
      const validEntries: Array<{ configId: string; val: number }> = [];

      models.forEach((row) => {
        const raw = row[m.key];
        if (raw != null && !isNaN(Number(raw))) {
          validEntries.push({ configId: row.config_id || row.model_id || row.model_name, val: Number(raw) });
        }
      });

      if (validEntries.length > 0) {
        const nums = validEntries.map((e) => e.val);
        stats[m.key] = {
          min: Math.min(...nums),
          max: Math.max(...nums),
          values: nums,
          availableCount: validEntries.length,
        };
      } else {
        stats[m.key] = {
          min: 0,
          max: 1,
          values: [],
          availableCount: 0,
        };
      }
    });

    return stats;
  }, [models]);

  // Color calculation: metric-aware color scale
  const getCellStyling = (metric: MetricDefinition, val: number | null, _configId: string) => {
    if (val == null || isNaN(val)) {
      return {
        bgClass: 'bg-slate-950/60',
        textClass: 'text-slate-600 italic',
        inlineStyle: {},
      };
    }

    const stat = columnStats[metric.key];
    if (!stat || stat.min === stat.max) {
      return {
        bgClass: 'bg-slate-900',
        textClass: 'text-slate-200',
        inlineStyle: {},
      };
    }

    // 1. Operational Workload Scale (Neutral Slate -> Cyan)
    if (metric.category === 'workload') {
      const ratio = Math.max(0.0, Math.min(1.0, (val - stat.min) / (stat.max - stat.min)));
      if (ratio >= 0.7) {
        return {
          bgClass: 'bg-sky-950/80 border-sky-600/40',
          textClass: 'text-sky-300 font-semibold',
          inlineStyle: { backgroundColor: `rgba(12, 74, 110, ${0.4 + ratio * 0.4})` },
        };
      }
      if (ratio >= 0.35) {
        return {
          bgClass: 'bg-cyan-950/60 border-cyan-700/30',
          textClass: 'text-cyan-200',
          inlineStyle: { backgroundColor: `rgba(22, 78, 99, ${0.35 + ratio * 0.35})` },
        };
      }
      return {
        bgClass: 'bg-slate-900 border-slate-700',
        textClass: 'text-slate-300',
        inlineStyle: { backgroundColor: 'rgba(30, 41, 59, 0.7)' },
      };
    }

    // 2. Calibration & Loss Error Metrics (Emerald = lowest error -> Amber -> Rose)
    if (metric.category === 'error') {
      const scoreRatio = Math.max(0.0, Math.min(1.0, 1.0 - (val - stat.min) / (stat.max - stat.min)));
      if (scoreRatio >= 0.8) {
        return {
          bgClass: 'bg-emerald-950/80 border-emerald-500/40',
          textClass: 'text-emerald-300 font-bold',
          inlineStyle: { backgroundColor: `rgba(6, 78, 59, ${0.4 + scoreRatio * 0.45})` },
        };
      }
      if (scoreRatio >= 0.45) {
        return {
          bgClass: 'bg-amber-950/60 border-amber-600/30',
          textClass: 'text-amber-200',
          inlineStyle: { backgroundColor: `rgba(120, 53, 15, ${0.35 + (1 - scoreRatio) * 0.35})` },
        };
      }
      return {
        bgClass: 'bg-rose-950/70 border-rose-600/40',
        textClass: 'text-rose-300',
        inlineStyle: { backgroundColor: `rgba(136, 19, 55, ${0.4 + (1 - scoreRatio) * 0.45})` },
      };
    }

    // 3. Detection Performance Metrics (Navy -> Indigo -> Emerald = highest)
    const scoreRatio = Math.max(0.0, Math.min(1.0, (val - stat.min) / (stat.max - stat.min)));
    if (scoreRatio >= 0.85) {
      return {
        bgClass: 'bg-emerald-950/80 border-emerald-500/40',
        textClass: 'text-emerald-300 font-bold',
        inlineStyle: { backgroundColor: `rgba(6, 78, 59, ${0.45 + scoreRatio * 0.45})` },
      };
    }
    if (scoreRatio >= 0.55) {
      return {
        bgClass: 'bg-indigo-950/70 border-indigo-500/30',
        textClass: 'text-indigo-200',
        inlineStyle: { backgroundColor: `rgba(49, 46, 129, ${0.35 + scoreRatio * 0.4})` },
      };
    }
    if (scoreRatio >= 0.3) {
      return {
        bgClass: 'bg-slate-900 border-slate-700',
        textClass: 'text-slate-300',
        inlineStyle: { backgroundColor: `rgba(30, 41, 59, ${0.4 + scoreRatio * 0.3})` },
      };
    }
    return {
      bgClass: 'bg-slate-950 border-slate-800',
      textClass: 'text-slate-400',
      inlineStyle: { backgroundColor: 'rgba(15, 23, 42, 0.85)' },
    };
  };

  return (
    <div className={`bg-slate-900 border border-slate-800 rounded-3xl p-6 shadow-xl space-y-4 ${className}`}>
      {/* Header with Title and View Controls */}
      <div className="flex flex-col md:flex-row md:items-center justify-between gap-4 border-b border-slate-800 pb-4">
        <div>
          <div className="flex items-center space-x-2">
            <Layers className="w-5 h-5 text-indigo-400" />
            <h3 className="text-base font-bold text-white tracking-tight">{title}</h3>
            <span className="px-2 py-0.5 rounded-full text-[10px] font-mono font-bold bg-slate-800 text-slate-300 border border-slate-700">
              {models.length} Configurations
            </span>
          </div>
          <p className="text-xs text-slate-400 mt-1 max-w-2xl leading-relaxed">{subtitle}</p>
        </div>

        {/* View Mode & Visibility Controls */}
        <div className="flex flex-wrap items-center gap-2">
          {/* Heatmap / Table Switcher */}
          <div className="inline-flex bg-slate-950 p-1 rounded-xl border border-slate-800 text-xs font-semibold">
            <button
              onClick={() => {
                setViewMode('heatmap');
                setIsVisible(true);
              }}
              className={`flex items-center space-x-1.5 px-3 py-1.5 rounded-lg transition cursor-pointer ${
                viewMode === 'heatmap' && isVisible
                  ? 'bg-indigo-600 text-white shadow-md shadow-indigo-600/30 font-bold'
                  : 'text-slate-400 hover:text-white'
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
                  : 'text-slate-400 hover:text-white'
              }`}
            >
              <TableIcon className="w-3.5 h-3.5" />
              <span>Accessible Table</span>
            </button>
          </div>

          {/* Show / Hide Toggle */}
          <button
            onClick={() => setIsVisible((prev) => !prev)}
            className="flex items-center space-x-1.5 px-3 py-1.5 rounded-xl bg-slate-950 hover:bg-slate-800 border border-slate-800 text-xs text-slate-300 transition cursor-pointer font-medium"
            title={isVisible ? 'Hide Heatmap section' : 'Show Heatmap section'}
          >
            {isVisible ? <EyeOff className="w-3.5 h-3.5 text-slate-400" /> : <Eye className="w-3.5 h-3.5 text-indigo-400" />}
            <span>{isVisible ? 'Hide' : 'Show'}</span>
          </button>
        </div>
      </div>

      {isVisible && (
        <div className="space-y-4 animate-fade-in">
          {/* Active Hover / Focus Detail Bar */}
          <div className="min-h-[44px] bg-slate-950/80 p-2.5 rounded-2xl border border-slate-800/80 flex flex-wrap items-center justify-between gap-2 text-xs font-mono">
            {hoveredCell ? (
              <div className="flex flex-wrap items-center gap-2 sm:gap-3">
                <span className="text-white font-bold bg-slate-800 px-2 py-0.5 rounded">
                  {hoveredCell.configId}
                </span>
                <span className="text-slate-300 font-semibold">
                  {hoveredCell.metric.name}:
                </span>
                {hoveredCell.value != null ? (
                  <strong className="text-emerald-400 text-sm">
                    {hoveredCell.metric.format(hoveredCell.value)} {hoveredCell.metric.units}
                  </strong>
                ) : (
                  <span className="text-slate-500 italic text-xs">
                    Value not recorded for this model configuration
                  </span>
                )}
                <span className="px-2 py-0.5 rounded-full text-[10px] bg-slate-800 text-slate-300 border border-slate-700">
                  {hoveredCell.metric.category === 'performance'
                    ? 'Performance: Higher is better'
                    : hoveredCell.metric.category === 'error'
                    ? 'Error: Lower is better'
                    : 'Workload: Volume density vs capacity policy'}
                </span>
                <span className="text-slate-400 text-[11px] font-sans">
                  {hoveredCell.metric.description}
                </span>
              </div>
            ) : (
              <div className="text-slate-500 flex items-center space-x-2 text-[11px] font-sans">
                <HelpCircle className="w-3.5 h-3.5 text-slate-400" />
                <span>Hover or navigate with keyboard focus over any metric cell to inspect exact source values, units, and operational definitions.</span>
              </div>
            )}
          </div>

          {/* MAIN HEATMAP OR TABLE VIEW */}
          {models.length === 0 ? (
            <div className="p-8 text-center bg-slate-950/40 rounded-2xl border border-slate-800 text-xs text-slate-400">
              No model comparison records match the active filter criteria.
            </div>
          ) : viewMode === 'heatmap' ? (
            /* HEATMAP VIEW */
            <div className="overflow-x-auto rounded-2xl border border-slate-800 scrollbar-thin">
              <table className="w-full text-left border-collapse text-xs font-mono select-none">
                <thead>
                  <tr className="bg-slate-950 border-b border-slate-800 text-slate-400 uppercase text-[10px]">
                    <th className="py-3 px-3.5 sticky left-0 bg-slate-950 z-20 min-w-[130px] border-r border-slate-800">
                      Configuration
                    </th>
                    <th className="py-3 px-2.5 min-w-[70px]">Arch</th>
                    <th className="py-3 px-2.5 min-w-[70px]">Dist</th>
                    <th className="py-3 px-2.5 min-w-[80px]">Algo</th>
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
                <tbody className="divide-y divide-slate-800/60">
                  {models.map((m) => {
                    const cfgId = m.config_id || m.model_id || m.model_name || 'c_unknown';
                    const isSelected = selectedModelId === cfgId;
                    const isChampion =
                      String(m.model_name || '').includes('hi_small_lstm_fedavg_iid_seed42') ||
                      cfgId === 'config_5' ||
                      m.is_champion;

                    return (
                      <tr
                        key={cfgId}
                        onClick={() => onSelectModel && onSelectModel(cfgId)}
                        className={`transition cursor-pointer ${
                          isSelected ? 'bg-indigo-950/40 ring-1 ring-indigo-500' : 'hover:bg-slate-800/40'
                        }`}
                      >
                        <td className="py-2.5 px-3.5 font-bold text-white sticky left-0 bg-slate-900 z-10 border-r border-slate-800 truncate max-w-[150px]">
                          <div className="flex items-center space-x-1.5">
                            {isChampion && <span className="w-2 h-2 rounded-full bg-emerald-400 shrink-0" title="Serving Champion" />}
                            <span className="truncate">{cfgId}</span>
                          </div>
                        </td>
                        <td className="py-2.5 px-2.5 text-slate-300">{m.architecture || 'N/A'}</td>
                        <td className="py-2.5 px-2.5 text-slate-400">{m.distribution || 'N/A'}</td>
                        <td className="py-2.5 px-2.5 text-slate-400">{m.algorithm || 'N/A'}</td>

                        {HEATMAP_METRICS.map((metric) => {
                          const val = m[metric.key] != null ? Number(m[metric.key]) : null;
                          const styling = getCellStyling(metric, val, cfgId);

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
                              className={`py-2 px-3 text-right transition border border-transparent font-mono ${styling.bgClass} ${styling.textClass}`}
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
            /* ACCESSIBLE DATA TABLE VIEW */
            <div className="overflow-x-auto rounded-2xl border border-slate-800">
              <table className="w-full text-left border-collapse text-xs font-mono">
                <thead>
                  <tr className="bg-slate-950 border-b border-slate-800 text-slate-300 uppercase text-[10px]">
                    <th className="py-3 px-3">Configuration</th>
                    <th className="py-3 px-3">Model Name</th>
                    <th className="py-3 px-3">Architecture</th>
                    <th className="py-3 px-3">Distribution</th>
                    <th className="py-3 px-3">Algorithm</th>
                    {HEATMAP_METRICS.map((m) => (
                      <th key={m.key} className="py-3 px-3 text-right">
                        {m.shortLabel}
                      </th>
                    ))}
                  </tr>
                </thead>
                <tbody className="divide-y divide-slate-800/80">
                  {models.map((m) => {
                    const cfgId = m.config_id || m.model_id || m.model_name || 'c_unknown';
                    return (
                      <tr key={cfgId} className="hover:bg-slate-800/30 transition">
                        <td className="py-2.5 px-3 font-bold text-white">{cfgId}</td>
                        <td className="py-2.5 px-3 text-slate-300 truncate max-w-[160px]">{m.model_name || cfgId}</td>
                        <td className="py-2.5 px-3 text-slate-400">{m.architecture || 'N/A'}</td>
                        <td className="py-2.5 px-3 text-slate-400">{m.distribution || 'N/A'}</td>
                        <td className="py-2.5 px-3 text-slate-400">{m.algorithm || 'N/A'}</td>
                        {HEATMAP_METRICS.map((metric) => (
                          <td key={metric.key} className="py-2.5 px-3 text-right text-slate-200">
                            {metric.format(m[metric.key])}
                          </td>
                        ))}
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
          )}

          {/* COLOR LEGEND & METRIC SCALE EXPLANATION */}
          <div className="p-4 bg-slate-950/70 rounded-2xl border border-slate-800 space-y-3 text-xs">
            <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3">
              <span className="font-bold text-slate-200 text-xs">Metric-Aware Color Legend & Directional Interpretation:</span>
              <span className="text-[11px] text-slate-400 font-sans">
                Normalized min-to-max within each column across loaded benchmark models.
              </span>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
              {/* Scale 1: Performance Metrics (Higher is Better) */}
              <div className="space-y-1.5">
                <div className="flex items-center justify-between text-[11px] text-slate-300 font-mono">
                  <span>Detection Quality:</span>
                  <span className="text-emerald-400 font-bold">Higher = Better</span>
                </div>
                <div className="flex items-center space-x-1.5 h-3 rounded-md overflow-hidden bg-gradient-to-r from-slate-900 via-indigo-900 to-emerald-600 border border-slate-700" />
                <div className="flex justify-between text-[10px] text-slate-500 font-mono">
                  <span>Lowest (Slate)</span>
                  <span>Median (Indigo)</span>
                  <span>Highest (Emerald)</span>
                </div>
                <p className="text-[10px] text-slate-400 font-sans leading-tight">
                  PR-AUC, ROC-AUC, F1, F2, Precision, Recall, Specificity, Bal. Acc, MCC, κ
                </p>
              </div>

              {/* Scale 2: Loss & Calibration Metrics (Lower is Better) */}
              <div className="space-y-1.5">
                <div className="flex items-center justify-between text-[11px] text-slate-300 font-mono">
                  <span>Calibration Error & Loss:</span>
                  <span className="text-emerald-400 font-bold">Lower = Better</span>
                </div>
                <div className="flex items-center space-x-1.5 h-3 rounded-md overflow-hidden bg-gradient-to-r from-emerald-600 via-amber-700 to-rose-700 border border-slate-700" />
                <div className="flex justify-between text-[10px] text-slate-500 font-mono">
                  <span>Lowest / Optimal (Emerald)</span>
                  <span>Moderate (Amber)</span>
                  <span>High Error (Rose)</span>
                </div>
                <p className="text-[10px] text-slate-400 font-sans leading-tight">
                  Brier Score, Expected Calibration Error (ECE), Log Loss
                </p>
              </div>

              {/* Scale 3: Investigator Workload Density (Neutral Policy Scale) */}
              <div className="space-y-1.5">
                <div className="flex items-center justify-between text-[11px] text-slate-300 font-mono">
                  <span>Investigator Workload:</span>
                  <span className="text-cyan-400 font-bold">Volume vs Policy</span>
                </div>
                <div className="flex items-center space-x-1.5 h-3 rounded-md overflow-hidden bg-gradient-to-r from-slate-900 via-cyan-900 to-sky-700 border border-slate-700" />
                <div className="flex justify-between text-[10px] text-slate-500 font-mono">
                  <span>Low Vol (Slate)</span>
                  <span>Medium (Cyan)</span>
                  <span>High Vol (Sky)</span>
                </div>
                <p className="text-[10px] text-slate-400 font-sans leading-tight">
                  Alerts/10k tx: evaluated against review capacity (e.g. 50/1k). Not intrinsically good/bad.
                </p>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};
