import React, { useState } from 'react';
import { PieChart as PieIcon } from 'lucide-react';

export type PieViewMode = 'review_status' | 'confusion' | 'actual' | 'predicted';

interface SliceData {
  id: string;
  label: string;
  count: number;
  color: string;
  hoverColor: string;
  badge: string;
  description: string;
  symbol: string;
}

interface EvaluationPieChartProps {
  tp: number;
  fp: number;
  fn: number;
  tn: number;
  totalRows?: number;
  hasLabels: boolean;
  activeModeName: string;
  threshold: number;
  className?: string;
}

export const EvaluationPieChart: React.FC<EvaluationPieChartProps> = ({
  tp,
  fp,
  fn,
  tn,
  totalRows,
  hasLabels,
  activeModeName,
  threshold,
  className = '',
}) => {
  const [activeMode, setActiveMode] = useState<PieViewMode>(hasLabels ? 'review_status' : 'predicted');
  const [hoveredSliceId, setHoveredSliceId] = useState<string | null>(null);

  const sumTotal = tp + fp + fn + tn;
  const total = totalRows && totalRows > 0 ? totalRows : sumTotal;

  // Build slices depending on active mode
  const getSlices = (): SliceData[] => {
    if (activeMode === 'review_status') {
      const flagged = tp + fp;
      const needsReview = fn;
      const cleared = tn;
      return [
        {
          id: 'review_flagged',
          label: 'Flagged (Model Alerts)',
          count: flagged,
          color: '#e11d48',
          hoverColor: '#f43f5e',
          badge: 'Model Trigger',
          description: `Transactions scoring ≥ ${threshold.toFixed(4)} triggering an operational alert`,
          symbol: 'Flagged',
        },
        {
          id: 'review_needs_review',
          label: 'Needs Review (False Negatives)',
          count: needsReview,
          color: '#f59e0b',
          hoverColor: '#fbbf24',
          badge: 'Ground-Truth Laundering',
          description: `Transactions labeled laundering in CSV that scored below active cutoff threshold θ = ${threshold.toFixed(4)}`,
          symbol: 'Needs Review',
        },
        {
          id: 'review_cleared',
          label: 'Legitimate Cleared (True Negatives)',
          count: cleared,
          color: '#059669',
          hoverColor: '#10b981',
          badge: 'Legitimate Cleared',
          description: `Normal legitimate transactions correctly scoring below cutoff threshold`,
          symbol: 'Cleared',
        },
      ];
    }

    if (activeMode === 'confusion') {
      return [
        {
          id: 'tp',
          label: 'True Positives (TP)',
          count: tp,
          color: '#10b981',
          hoverColor: '#34d399',
          badge: 'Detected Laundering',
          description: 'Confirmed laundering transactions correctly identified by model',
          symbol: 'TP',
        },
        {
          id: 'fp',
          label: 'False Positives (FP)',
          count: fp,
          color: '#f43f5e',
          hoverColor: '#fb7185',
          badge: 'False Alarms',
          description: 'Legitimate transactions incorrectly flagged for compliance review',
          symbol: 'FP',
        },
        {
          id: 'fn',
          label: 'False Negatives (FN)',
          count: fn,
          color: '#f59e0b',
          hoverColor: '#fbbf24',
          badge: 'Missed Laundering',
          description: 'Confirmed laundering transactions that scored below the cutoff threshold',
          symbol: 'FN',
        },
        {
          id: 'tn',
          label: 'True Negatives (TN)',
          count: tn,
          color: '#64748b',
          hoverColor: '#94a3b8',
          badge: 'Correctly Cleared',
          description: 'Legitimate transactions correctly allowed through without alert',
          symbol: 'TN',
        },
      ];
    }

    if (activeMode === 'actual') {
      const posCount = tp + fn;
      const negCount = fp + tn;
      return [
        {
          id: 'actual_pos',
          label: 'Confirmed Laundering',
          count: posCount,
          color: '#f43f5e',
          hoverColor: '#fb7185',
          badge: 'Ground Truth Positive',
          description: 'True laundering transactions present in the uploaded evaluation batch',
          symbol: 'Positives',
        },
        {
          id: 'actual_neg',
          label: 'Confirmed Legitimate',
          count: negCount,
          color: '#10b981',
          hoverColor: '#34d399',
          badge: 'Ground Truth Negative',
          description: 'Legitimate normal transactions in the uploaded evaluation batch',
          symbol: 'Negatives',
        },
      ];
    }

    // predicted
    const flagged = tp + fp;
    const cleared = fn + tn;
    return [
      {
        id: 'pred_flagged',
        label: 'Flagged for Review (Alerts)',
        count: flagged,
        color: '#e11d48',
        hoverColor: '#f43f5e',
        badge: 'Model Alerts',
        description: `Transactions scoring ≥ ${threshold.toFixed(4)} routed to compliance analysts`,
        symbol: 'Alerts',
      },
      {
        id: 'pred_cleared',
        label: 'Cleared as Legitimate',
        count: cleared,
        color: '#059669',
        hoverColor: '#10b981',
        badge: 'Passed Triage',
        description: `Transactions scoring < ${threshold.toFixed(4)} cleared without investigation`,
        symbol: 'Cleared',
      },
    ];
  };

  const slices = getSlices();
  const currentTotal = slices.reduce((acc, s) => acc + s.count, 0);

  // SVG Geometry Calculation (Donut chart)
  const cx = 110;
  const cy = 110;
  const outerRadius = 85;
  const innerRadius = 45;

  let cumulativeAngle = 0;
  const arcPaths = slices.map((slice) => {
    const fraction = currentTotal > 0 ? slice.count / currentTotal : 0;
    const sweepAngle = fraction * 360;
    const startAngle = cumulativeAngle;
    const endAngle = cumulativeAngle + sweepAngle;
    cumulativeAngle += sweepAngle;

    // Convert degrees to radians (start from 12 o'clock = -90 deg)
    const startRad = ((startAngle - 90) * Math.PI) / 180;
    const endRad = ((endAngle - 90) * Math.PI) / 180;

    const x1 = cx + outerRadius * Math.cos(startRad);
    const y1 = cy + outerRadius * Math.sin(startRad);
    const x2 = cx + outerRadius * Math.cos(endRad);
    const y2 = cy + outerRadius * Math.sin(endRad);

    const xi1 = cx + innerRadius * Math.cos(endRad);
    const yi1 = cy + innerRadius * Math.sin(endRad);
    const xi2 = cx + innerRadius * Math.cos(startRad);
    const yi2 = cy + innerRadius * Math.sin(startRad);

    const largeArc = sweepAngle > 180 ? 1 : 0;

    // Path command for donut sector
    const path =
      fraction >= 0.9999
        ? `M ${cx} ${cy - outerRadius} A ${outerRadius} ${outerRadius} 0 1 1 ${cx} ${cy + outerRadius} A ${outerRadius} ${outerRadius} 0 1 1 ${cx} ${cy - outerRadius} M ${cx} ${cy - innerRadius} A ${innerRadius} ${innerRadius} 0 1 0 ${cx} ${cy + innerRadius} A ${innerRadius} ${innerRadius} 0 1 0 ${cx} ${cy - innerRadius} Z`
        : `M ${x1} ${y1} A ${outerRadius} ${outerRadius} 0 ${largeArc} 1 ${x2} ${y2} L ${xi1} ${yi1} A ${innerRadius} ${innerRadius} 0 ${largeArc} 0 ${xi2} ${yi2} Z`;

    return {
      ...slice,
      fraction,
      path,
      sweepAngle,
    };
  });

  const activeHoveredSlice = slices.find((s) => s.id === hoveredSliceId);

  return (
    <div className={`bg-slate-900 border border-slate-800 rounded-3xl p-6 shadow-xl space-y-5 ${className}`}>
      {/* Header & 3-Way Mode Toggle Controls */}
      <div className="flex flex-col md:flex-row md:items-center justify-between gap-4 border-b border-slate-800 pb-4">
        <div>
          <h3 className="text-sm font-bold text-white flex items-center space-x-2">
            <PieIcon className="w-4 h-4 text-indigo-400" />
            <span>Authoritative Evaluation Distribution Pie Charts</span>
          </h3>
          <p className="text-xs text-slate-400 mt-0.5">
            Switch between Confusion Matrix outcomes, Ground-Truth class balance, and Predicted triage decisions. Each view accounts for all {total.toLocaleString()} transactions.
          </p>
        </div>

        {/* Accessible Mode Toggles */}
        <div
          role="tablist"
          aria-label="Evaluation distribution views"
          className="flex flex-wrap items-center p-1 bg-slate-950 border border-slate-800 rounded-xl gap-1 shrink-0"
        >
          <button
            role="tab"
            aria-selected={activeMode === 'review_status'}
            tabIndex={0}
            onClick={() => setActiveMode('review_status')}
            onKeyDown={(e) => {
              if (e.key === 'Enter' || e.key === ' ') {
                e.preventDefault();
                setActiveMode('review_status');
              }
            }}
            disabled={!hasLabels}
            className={`px-3 py-1.5 rounded-lg text-xs font-semibold transition cursor-pointer ${
              activeMode === 'review_status'
                ? 'bg-indigo-600 text-white font-bold shadow-md shadow-indigo-600/30'
                : 'text-slate-400 hover:text-white disabled:opacity-40 disabled:cursor-not-allowed'
            }`}
          >
            Review Status
          </button>

          <button
            role="tab"
            aria-selected={activeMode === 'confusion'}
            tabIndex={0}
            onClick={() => setActiveMode('confusion')}
            onKeyDown={(e) => {
              if (e.key === 'Enter' || e.key === ' ') {
                e.preventDefault();
                setActiveMode('confusion');
              }
            }}
            disabled={!hasLabels}
            className={`px-3 py-1.5 rounded-lg text-xs font-semibold transition cursor-pointer ${
              activeMode === 'confusion'
                ? 'bg-indigo-600 text-white font-bold shadow-md shadow-indigo-600/30'
                : 'text-slate-400 hover:text-white disabled:opacity-40 disabled:cursor-not-allowed'
            }`}
          >
            Confusion Outcomes
          </button>

          <button
            role="tab"
            aria-selected={activeMode === 'actual'}
            tabIndex={0}
            onClick={() => setActiveMode('actual')}
            onKeyDown={(e) => {
              if (e.key === 'Enter' || e.key === ' ') {
                e.preventDefault();
                setActiveMode('actual');
              }
            }}
            disabled={!hasLabels}
            className={`px-3 py-1.5 rounded-lg text-xs font-semibold transition cursor-pointer ${
              activeMode === 'actual'
                ? 'bg-indigo-600 text-white font-bold shadow-md shadow-indigo-600/30'
                : 'text-slate-400 hover:text-white disabled:opacity-40 disabled:cursor-not-allowed'
            }`}
          >
            Actual Labels
          </button>

          <button
            role="tab"
            aria-selected={activeMode === 'predicted'}
            tabIndex={0}
            onClick={() => setActiveMode('predicted')}
            onKeyDown={(e) => {
              if (e.key === 'Enter' || e.key === ' ') {
                e.preventDefault();
                setActiveMode('predicted');
              }
            }}
            className={`px-3 py-1.5 rounded-lg text-xs font-semibold transition cursor-pointer ${
              activeMode === 'predicted'
                ? 'bg-indigo-600 text-white font-bold shadow-md shadow-indigo-600/30'
                : 'text-slate-400 hover:text-white'
            }`}
          >
            Predicted Decisions
          </button>
        </div>
      </div>

      {/* Main Chart Body: Donut Visual on Left + Rich Accessible Legend on Right */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6 items-center">
        {/* SVG Donut */}
        <div className="lg:col-span-5 flex flex-col items-center justify-center relative">
          <div className="relative w-56 h-56 flex items-center justify-center">
            <svg
              className="w-full h-full overflow-visible"
              viewBox="0 0 220 220"
              role="img"
              aria-label={`Donut chart showing ${activeMode} distribution of ${currentTotal.toLocaleString()} transactions`}
            >
              {arcPaths.map((slice) => {
                const isHovered = hoveredSliceId === slice.id;
                return (
                  <path
                    key={slice.id}
                    d={slice.path}
                    fill={isHovered ? slice.hoverColor : slice.color}
                    stroke="#0f172a"
                    strokeWidth={isHovered ? 2.5 : 1.5}
                    className="transition-all duration-150 cursor-pointer"
                    onMouseEnter={() => setHoveredSliceId(slice.id)}
                    onMouseLeave={() => setHoveredSliceId(null)}
                    tabIndex={0}
                    role="graphics-symbol"
                    aria-label={`${slice.label}: ${slice.count.toLocaleString()} transactions (${(slice.fraction * 100).toFixed(2)}%)`}
                  />
                );
              })}
            </svg>

            {/* Center Callout */}
            <div className="absolute inset-0 flex flex-col items-center justify-center pointer-events-none text-center">
              <span className="text-[10px] uppercase tracking-wider font-mono text-slate-400">
                {activeHoveredSlice ? activeHoveredSlice.symbol : 'Total Evaluated'}
              </span>
              <span className="text-xl font-black font-mono text-white">
                {activeHoveredSlice
                  ? activeHoveredSlice.count.toLocaleString()
                  : currentTotal.toLocaleString()}
              </span>
              <span className="text-[10px] font-mono text-indigo-400">
                {activeHoveredSlice
                  ? `${((activeHoveredSlice.count / Math.max(1, currentTotal)) * 100).toFixed(2)}%`
                  : '100.0%'}
              </span>
            </div>
          </div>

          <span className="text-[11px] text-slate-500 font-mono mt-2">
            Hover over any sector or legend card to inspect details
          </span>
        </div>

        {/* Right Column: Prominent Legend Cards with Clear Counts, Percentages & Tooltips */}
        <div className="lg:col-span-7 space-y-2.5">
          {slices.map((slice) => {
            const pct = currentTotal > 0 ? (slice.count / currentTotal) * 100 : 0;
            const isHovered = hoveredSliceId === slice.id;

            return (
              <div
                key={slice.id}
                onMouseEnter={() => setHoveredSliceId(slice.id)}
                onMouseLeave={() => setHoveredSliceId(null)}
                className={`p-3 rounded-2xl border transition-all cursor-pointer ${
                  isHovered
                    ? 'bg-indigo-50/90 dark:bg-slate-800/90 border-indigo-500 shadow-md scale-[1.01]'
                    : 'bg-white dark:bg-slate-950/70 border-slate-200 dark:border-slate-800/80 hover:bg-slate-50 dark:hover:bg-slate-900/90'
                }`}
              >
                <div className="flex items-center justify-between gap-2">
                  <div className="flex items-center space-x-2.5 min-w-0">
                    <span
                      className="w-3.5 h-3.5 rounded-full shrink-0 shadow-sm"
                      style={{ backgroundColor: slice.color }}
                    />
                    <div className="truncate">
                      <div className="flex items-center space-x-2">
                        <span className="text-xs font-bold text-slate-900 dark:text-white truncate">{slice.label}</span>
                        <span
                          className="px-1.5 py-0.2 text-[9px] font-mono rounded border uppercase font-semibold"
                          style={{
                            color: slice.color === '#f59e0b' ? '#b45309' : slice.color,
                            borderColor: `${slice.color}40`,
                            backgroundColor: `${slice.color}15`,
                          }}
                        >
                          {slice.badge}
                        </span>
                      </div>
                      <p className="text-[10.5px] text-slate-600 dark:text-slate-400 truncate mt-0.5">{slice.description}</p>
                    </div>
                  </div>

                  <div className="text-right shrink-0">
                    <span className="text-sm font-black font-mono text-slate-900 dark:text-white block">
                      {slice.count.toLocaleString()}
                    </span>
                    <span className="text-[10px] font-mono text-slate-400">
                      {pct >= 0.1 ? `${pct.toFixed(2)}%` : `${pct.toFixed(3)}%`}
                    </span>
                  </div>
                </div>
              </div>
            );
          })}

          {/* Mathematical Totals Reconciliation Bar */}
          <div className="flex items-center justify-between px-3 py-2 bg-slate-950 rounded-xl border border-slate-800/60 text-[11px] font-mono text-slate-400 mt-1">
            <span>
              Sum: <strong className="text-white">{currentTotal.toLocaleString()} transactions</strong>
            </span>
            <span>
              Target: <strong className="text-emerald-400">{total.toLocaleString()} (100.0%)</strong>
            </span>
            <span className="hidden sm:inline text-indigo-400 font-semibold">{activeModeName}</span>
          </div>
        </div>
      </div>
    </div>
  );
};
