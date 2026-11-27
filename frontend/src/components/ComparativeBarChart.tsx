import React, { useState, useRef } from 'react';
import { BarChart2, Maximize2, Minimize2, Download, FileSpreadsheet } from 'lucide-react';
import { BenchmarkModelRecord } from '../types/api';
import { exportSvgToPng, exportDataToCsv } from '../utils/exportHelpers';

export type ChartMetricKey =
  | 'test_pr_auc'
  | 'test_roc_auc'
  | 'test_f1'
  | 'test_precision'
  | 'test_recall'
  | 'alerts_per_10k'
  | 'brier_score';

interface MetricOption {
  key: ChartMetricKey;
  label: string;
  fullName: string;
  units: string;
  higherIsBetter: boolean;
  maxAxis: number;
}

const METRIC_OPTIONS: MetricOption[] = [
  {
    key: 'test_pr_auc',
    label: 'PR-AUC',
    fullName: 'Precision-Recall Area Under Curve',
    units: 'Ratio [0 - 1]',
    higherIsBetter: true,
    maxAxis: 1.0,
  },
  {
    key: 'test_roc_auc',
    label: 'ROC-AUC',
    fullName: 'Receiver Operating Characteristic AUC',
    units: 'Ratio [0 - 1]',
    higherIsBetter: true,
    maxAxis: 1.0,
  },
  {
    key: 'test_f1',
    label: 'F1-Score',
    fullName: 'F1-Score (Harmonic Mean)',
    units: 'Ratio [0 - 1]',
    higherIsBetter: true,
    maxAxis: 1.0,
  },
  {
    key: 'test_precision',
    label: 'Precision',
    fullName: 'Precision (Positive Predictive Value)',
    units: 'Ratio [0 - 1]',
    higherIsBetter: true,
    maxAxis: 1.0,
  },
  {
    key: 'test_recall',
    label: 'Recall',
    fullName: 'Recall (Sensitivity, Detection Rate)',
    units: 'Ratio [0 - 1]',
    higherIsBetter: true,
    maxAxis: 1.0,
  },
  {
    key: 'alerts_per_10k',
    label: 'Alerts / 10k',
    fullName: 'Alerts per 10,000 Transactions',
    units: 'Rate / 10k tx',
    higherIsBetter: false,
    maxAxis: 35.0,
  },
  {
    key: 'brier_score',
    label: 'Brier Score',
    fullName: 'Brier Calibration Error',
    units: 'Squared Error',
    higherIsBetter: false,
    maxAxis: 0.005,
  },
];

interface ComparativeBarChartProps {
  models: BenchmarkModelRecord[];
  selectedModelId?: string | null;
  onSelectModel?: (configId: string) => void;
  selectedDataset?: string;
  className?: string;
}

export const ComparativeBarChart: React.FC<ComparativeBarChartProps> = ({
  models,
  selectedModelId,
  onSelectModel,
  selectedDataset = 'HI-Small',
  className = '',
}) => {
  const [selectedMetricKey, setSelectedMetricKey] = useState<ChartMetricKey>('test_pr_auc');
  const [autoScaleY, setAutoScaleY] = useState<boolean>(true);
  const svgRef = useRef<SVGSVGElement | null>(null);
  const [hoveredBar, setHoveredBar] = useState<{
    model: BenchmarkModelRecord;
    value: number;
    x: number;
    y: number;
  } | null>(null);

  const activeMetric = METRIC_OPTIONS.find((m) => m.key === selectedMetricKey) || METRIC_OPTIONS[0];

  const sortedModels = [...models].sort((a, b) => {
    const va = Number(a[selectedMetricKey]) || 0;
    const vb = Number(b[selectedMetricKey]) || 0;
    return activeMetric.higherIsBetter ? vb - va : va - vb;
  });

  // Dynamic axis maximum calculation with autoScale toggle
  const rawMax = Math.max(...sortedModels.map((m) => Number(m[selectedMetricKey]) || 0), 0.0001);
  const axisMax = autoScaleY
    ? Math.max(0.001, rawMax * 1.15)
    : activeMetric.maxAxis === 1.0
    ? 1.0
    : Math.max(activeMetric.maxAxis, rawMax * 1.15);

  const formatShortLabel = (m: BenchmarkModelRecord): string => {
    const arch = m.architecture || 'Model';
    const algo = m.algorithm?.includes('FedProx') ? 'FedProx' : m.algorithm || '';
    const dist = m.distribution || '';
    if (dist === 'Centralized' || algo === 'Centralized') return `${arch} · Cent`;
    return `${arch} · ${algo} · ${dist}`.replace(/\s+·\s+$/, '');
  };

  return (
    <div className={`bg-white dark:bg-slate-900 border border-slate-200 dark:border-slate-800 rounded-3xl p-6 shadow-xl space-y-4 ${className}`}>
      {/* Title & Metric Switcher */}
      <div className="flex flex-col lg:flex-row lg:items-center justify-between gap-3 border-b border-slate-200 dark:border-slate-800 pb-3">
        <div>
          <h3 className="text-sm font-bold text-slate-900 dark:text-white flex items-center space-x-2">
            <BarChart2 className="w-4 h-4 text-indigo-500 dark:text-indigo-400" />
            <span>Comparative Benchmark: {activeMetric.fullName}</span>
          </h3>
          <p className="text-[11px] text-slate-500 dark:text-slate-400 mt-0.5">
            Evaluating {models.length} configurations on 100% held-out test split. Units:{' '}
            <strong className="text-indigo-600 dark:text-indigo-300 font-mono">{activeMetric.units}</strong>
            {activeMetric.key === 'alerts_per_10k'
              ? ' (Workload volume evaluated against capacity policy).'
              : activeMetric.higherIsBetter
              ? ' (Higher is better).'
              : ' (Lower is better).'}
          </p>
        </div>

        <div className="flex flex-wrap items-center gap-2">
          {/* Y-Axis Scale Toggle */}
          <button
            type="button"
            onClick={() => setAutoScaleY((prev) => !prev)}
            title={autoScaleY ? 'Switch to Full scale (0 to 1.0)' : 'Switch to Auto-scale (0 to data max)'}
            className="flex items-center space-x-1 px-2.5 py-1 rounded-xl text-[11px] font-mono border border-slate-200 dark:border-slate-800 bg-slate-50 dark:bg-slate-950 text-slate-600 dark:text-slate-300 hover:text-slate-900 dark:hover:text-white transition cursor-pointer"
          >
            {autoScaleY ? <Minimize2 className="w-3 h-3 text-indigo-500" /> : <Maximize2 className="w-3 h-3 text-indigo-500" />}
            <span>{autoScaleY ? 'Scale: Auto' : 'Scale: Full'}</span>
          </button>

          {/* Metric Selector Buttons */}
          <div className="flex flex-wrap items-center gap-1 bg-slate-100 dark:bg-slate-950 p-1 rounded-xl border border-slate-200 dark:border-slate-800 text-xs">
            {METRIC_OPTIONS.map((opt) => (
              <button
                key={opt.key}
                onClick={() => setSelectedMetricKey(opt.key)}
                className={`px-2.5 py-1 rounded-lg font-mono text-[11px] transition cursor-pointer ${
                  selectedMetricKey === opt.key
                    ? 'bg-indigo-600 text-white font-bold shadow-sm shadow-indigo-600/40'
                    : 'text-slate-600 dark:text-slate-400 hover:text-slate-900 dark:hover:text-white'
                }`}
              >
                {opt.label}
              </button>
            ))}
          </div>
          {/* Export Actions */}
          <div className="flex items-center space-x-1.5">
            <button
              type="button"
              onClick={() => exportSvgToPng(svgRef.current, `fedsecureai_${selectedDataset}_${selectedMetricKey}`)}
              title="Export Bar Chart as High-DPI PNG"
              className="flex items-center space-x-1 px-2.5 py-1 rounded-xl text-[11px] font-mono border border-slate-200 dark:border-slate-800 bg-slate-50 dark:bg-slate-950 text-slate-600 dark:text-slate-300 hover:text-indigo-600 dark:hover:text-indigo-400 transition cursor-pointer"
            >
              <Download className="w-3 h-3" />
              <span>PNG</span>
            </button>
            <button
              type="button"
              onClick={() =>
                exportDataToCsv(
                  ['Config ID', 'Model Name', 'Architecture', 'Distribution', 'Algorithm', 'Threshold', activeMetric.label],
                  models.map((m) => [m.config_id, m.model_name, m.architecture, m.distribution, m.algorithm, m.threshold, m[selectedMetricKey] ?? 'N/A']),
                  `fedsecureai_${selectedDataset}_${selectedMetricKey}.csv`
                )
              }
              title="Export Visualization Data as CSV"
              className="flex items-center space-x-1 px-2.5 py-1 rounded-xl text-[11px] font-mono border border-slate-200 dark:border-slate-800 bg-slate-50 dark:bg-slate-950 text-slate-600 dark:text-slate-300 hover:text-indigo-600 dark:hover:text-indigo-400 transition cursor-pointer"
            >
              <FileSpreadsheet className="w-3 h-3" />
              <span>CSV</span>
            </button>
          </div>
        </div>
      </div>

      {/* Legend */}
      <div className="flex flex-wrap items-center gap-4 text-[11px] font-mono text-slate-500 dark:text-slate-400">
        <span className="inline-flex items-center gap-1.5">
          <span className="w-2.5 h-2.5 rounded-sm bg-emerald-500 inline-block" />
          Champion
        </span>
        <span className="inline-flex items-center gap-1.5">
          <span className="w-2.5 h-2.5 rounded-sm bg-violet-500 inline-block" />
          FedAvg IID
        </span>
        <span className="inline-flex items-center gap-1.5">
          <span className="w-2.5 h-2.5 rounded-sm bg-indigo-500 inline-block" />
          FedAvg Non-IID
        </span>
        <span className="inline-flex items-center gap-1.5">
          <span className="w-2.5 h-2.5 rounded-sm bg-amber-500 inline-block" />
          FedProx
        </span>
      </div>

      {/* Mobile Horizontal Bars (< 768px) */}
      <div className="md:hidden space-y-2.5 pt-1">
        {sortedModels.map((m) => {
          const val = Number(m[selectedMetricKey]) || 0;
          const pct = Math.max(4, Math.min(100, (val / axisMax) * 100));
          const isChamp =
            (selectedDataset === 'HI-Small' && m.model_name.includes('hi_small_lstm_fedavg_iid_seed42')) ||
            (selectedDataset === 'LI-Medium' && m.config_id === 'config_5');
          const barBg = isChamp
            ? 'bg-emerald-500'
            : m.algorithm.includes('FedProx')
            ? 'bg-amber-500'
            : m.distribution === 'Non-IID'
            ? 'bg-indigo-500'
            : 'bg-violet-500';
          return (
            <button
              key={`mob-bar-${m.config_id}`}
              type="button"
              onClick={() => onSelectModel && onSelectModel(m.config_id)}
              className="w-full text-left space-y-1 cursor-pointer"
            >
              <div className="flex items-center justify-between text-xs font-mono">
                <span className="text-slate-700 dark:text-slate-200 font-semibold truncate">{formatShortLabel(m)}</span>
                <span className="text-slate-900 dark:text-white font-bold ml-2 shrink-0">
                  {selectedMetricKey === 'alerts_per_10k' ? val.toFixed(1) : val.toFixed(4)}
                </span>
              </div>
              <div className="w-full h-2.5 rounded-full bg-slate-200 dark:bg-slate-800 overflow-hidden">
                <div className={`h-full rounded-full ${barBg}`} style={{ width: `${pct}%` }} />
              </div>
            </button>
          );
        })}
      </div>

      {/* SVG Bar Chart with Tooltip (Desktop/Tablet) */}
      <div className="hidden md:block relative w-full overflow-hidden pt-2">
        <svg
          ref={svgRef}
          className="w-full h-80 sm:h-96"
          viewBox="0 0 880 290"
          preserveAspectRatio="xMidYMid meet"
          aria-label={`Comparative Bar Chart of ${activeMetric.fullName}`}
        >
          {/* Grid lines and Y-axis scale */}
          <line x1="70" y1="20" x2="70" y2="200" stroke="#94a3b8" strokeWidth="1" strokeOpacity="0.4" />
          <line x1="70" y1="200" x2="860" y2="200" stroke="#94a3b8" strokeWidth="1" strokeOpacity="0.4" />
          <line x1="70" y1="110" x2="860" y2="110" stroke="#94a3b8" strokeDasharray="3 3" strokeOpacity="0.2" />
          <line x1="70" y1="20" x2="860" y2="20" stroke="#94a3b8" strokeDasharray="3 3" strokeOpacity="0.2" />

          {/* Y-axis ticks and labels (high contrast, >= 12px visual weight) */}
          <text x="62" y="24" fill="#64748b" className="dark:fill-slate-400" fontSize="10.5" fontFamily="monospace" textAnchor="end" fontWeight="600">
            {axisMax === 1.0 ? '1.000' : axisMax.toFixed(axisMax < 1 ? 4 : 1)}
          </text>
          <text x="62" y="114" fill="#64748b" className="dark:fill-slate-400" fontSize="10.5" fontFamily="monospace" textAnchor="end" fontWeight="600">
            {axisMax === 1.0 ? '0.500' : (axisMax / 2).toFixed(axisMax < 1 ? 4 : 1)}
          </text>
          <text x="62" y="204" fill="#64748b" className="dark:fill-slate-400" fontSize="10.5" fontFamily="monospace" textAnchor="end" fontWeight="600">
            0.000
          </text>

          {/* Y-axis Title */}
          <text
            x="-110"
            y="20"
            fill="#64748b"
            fontSize="9"
            fontFamily="monospace"
            textAnchor="middle"
            transform="rotate(-90)"
          >
            {activeMetric.label} ({activeMetric.units})
          </text>

          {/* Bars */}
          {(() => {
            if (!sortedModels.length) return null;
            const barSlotWidth = (860 - 80) / sortedModels.length;
            const barW = Math.min(36, barSlotWidth * 0.72);

            return sortedModels.map((m, idx) => {
              const x = 85 + idx * barSlotWidth;
              const val = Number(m[selectedMetricKey]) || 0;
              const clampedRatio = Math.max(0, Math.min(1.0, val / axisMax));
              const h = Math.max(4, clampedRatio * 180);
              const y = 200 - h;

              const isChamp =
                (selectedDataset === 'HI-Small' && m.model_name.includes('hi_small_lstm_fedavg_iid_seed42')) ||
                (selectedDataset === 'LI-Medium' && m.config_id === 'config_5');

              const isSelected = selectedModelId === m.config_id;

              const fill = isChamp
                ? '#10b981'
                : m.algorithm.includes('FedProx')
                ? '#f59e0b'
                : m.distribution === 'Non-IID'
                ? '#6366f1'
                : '#8b5cf6';

              const shortName = formatShortLabel(m);
              // Stagger labels when more than 8 models to prevent collision
              const staggerOffset = sortedModels.length > 8 && idx % 2 !== 0 ? 12 : 0;
              const labelY = 216 + staggerOffset;

              return (
                <g
                  key={m.config_id}
                  className="cursor-pointer group"
                  onClick={() => onSelectModel && onSelectModel(m.config_id)}
                  onMouseEnter={() => setHoveredBar({ model: m, value: val, x, y })}
                  onMouseLeave={() => setHoveredBar(null)}
                  tabIndex={0}
                  onFocus={() => setHoveredBar({ model: m, value: val, x, y })}
                  onBlur={() => setHoveredBar(null)}
                  aria-label={`${m.config_id}: ${val.toFixed(4)}`}
                >
                  {/* Bar rectangle with hover highlight */}
                  <rect
                    x={x}
                    y={y}
                    width={barW}
                    height={h}
                    rx="4"
                    fill={fill}
                    className="transition-all duration-200 group-hover:brightness-110"
                    stroke={isSelected ? '#38bdf8' : isChamp ? '#059669' : 'none'}
                    strokeWidth={isSelected ? 2.5 : isChamp ? 1.5 : 0}
                  />

                  {/* Value label on top of bar */}
                  <text
                    x={x + barW / 2}
                    y={y - 5}
                    fill="#334155"
                    className="dark:fill-slate-200"
                    fontSize="9"
                    fontFamily="monospace"
                    textAnchor="middle"
                    fontWeight="bold"
                  >
                    {val < 0.01 ? val.toFixed(4) : val.toFixed(2)}
                  </text>

                  {/* Rotated, staggered, collision-free X-axis label */}
                  <text
                    x={x + barW / 2}
                    y={labelY}
                    fill={isChamp ? '#059669' : isSelected ? '#0284c7' : '#64748b'}
                    className={isChamp ? 'dark:fill-emerald-400' : isSelected ? 'dark:fill-sky-400' : 'dark:fill-slate-400'}
                    fontSize="9.5"
                    fontFamily="monospace"
                    fontWeight={isChamp || isSelected ? 'bold' : 'normal'}
                    textAnchor="end"
                    transform={`rotate(-45 ${x + barW / 2} ${labelY})`}
                  >
                    {shortName}
                  </text>
                </g>
              );
            });
          })()}
        </svg>

        {/* Interactive Floating Hover Tooltip */}
        {hoveredBar && (
          <div
            className="absolute z-30 pointer-events-none bg-slate-900/95 dark:bg-slate-950/95 border border-indigo-500/70 p-3.5 rounded-2xl shadow-2xl text-xs font-mono text-white min-w-[240px] backdrop-blur-md animate-fade-in"
            style={{
              left: `${Math.min(72, Math.max(8, (hoveredBar.x / 880) * 100))}%`,
              top: '10px',
            }}
          >
            <div className="flex items-center justify-between pb-1.5 mb-1.5 border-b border-slate-700 text-[11px]">
              <span className="font-bold text-indigo-300">{hoveredBar.model.config_id}</span>
              <span className="px-2 py-0.5 rounded-full text-[9px] bg-slate-800 text-slate-300 border border-slate-700">
                {hoveredBar.model.architecture} • {hoveredBar.model.distribution}
              </span>
            </div>
            <div className="text-[10px] text-slate-300 font-mono truncate mb-1.5">
              {hoveredBar.model.model_name}
            </div>
            <div className="flex items-baseline justify-between py-1 bg-slate-800/60 px-2 rounded-lg mb-1.5">
              <span className="text-slate-300 text-[11px]">{activeMetric.label}:</span>
              <strong className="text-emerald-400 text-sm font-bold">
                {hoveredBar.value.toFixed(4)}
              </strong>
            </div>
            <div className="grid grid-cols-2 gap-x-2 gap-y-1 text-[10px] text-slate-300 pt-1 border-t border-slate-800">
              <div>Threshold: <strong className="text-amber-400">{hoveredBar.model?.threshold != null ? hoveredBar.model.threshold.toFixed(4) : 'N/A'}</strong></div>
              <div>PR-AUC: <strong className="text-white">{hoveredBar.model?.test_pr_auc != null ? hoveredBar.model.test_pr_auc.toFixed(4) : 'N/A'}</strong></div>
              <div>ROC-AUC: <strong className="text-white">{hoveredBar.model?.test_roc_auc != null ? hoveredBar.model.test_roc_auc.toFixed(4) : 'N/A'}</strong></div>
              <div>F1-Score: <strong className="text-white">{hoveredBar.model?.test_f1 != null ? hoveredBar.model.test_f1.toFixed(4) : 'N/A'}</strong></div>
              <div>Precision: <strong className="text-white">{hoveredBar.model?.test_precision != null ? hoveredBar.model.test_precision.toFixed(4) : 'N/A'}</strong></div>
              <div>Recall: <strong className="text-white">{hoveredBar.model?.test_recall != null ? hoveredBar.model.test_recall.toFixed(4) : 'N/A'}</strong></div>
            </div>
          </div>
        )}
      </div>

      {/* Legend with Color Semantics */}
      <div className="flex flex-wrap items-center justify-center gap-6 text-[11px] font-mono border-t border-slate-200 dark:border-slate-800 pt-3">
        <span className="flex items-center space-x-1.5 text-emerald-700 dark:text-emerald-400 font-bold">
          <span className="w-3 h-3 bg-emerald-500 rounded inline-block" />
          <span>Serving Champion</span>
        </span>
        <span className="flex items-center space-x-1.5 text-purple-700 dark:text-purple-400 font-medium">
          <span className="w-3 h-3 bg-purple-500 rounded inline-block" />
          <span>FedAvg (IID)</span>
        </span>
        <span className="flex items-center space-x-1.5 text-indigo-700 dark:text-indigo-400 font-medium">
          <span className="w-3 h-3 bg-indigo-500 rounded inline-block" />
          <span>FedAvg (Non-IID)</span>
        </span>
        <span className="flex items-center space-x-1.5 text-amber-700 dark:text-amber-400 font-medium">
          <span className="w-3 h-3 bg-amber-500 rounded inline-block" />
          <span>FedProx (μ=0.0001)</span>
        </span>
      </div>
    </div>
  );
};
