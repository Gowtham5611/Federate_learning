import React, { useState, useMemo } from 'react';
import { TransactionPredictionItem } from '../types/api';
import {
  Grid,
  Table as TableIcon,
  ChevronDown,
  ChevronUp,
  Filter,
  X,
  Info,
  Layers,
  BarChart2,
  Calendar,
} from 'lucide-react';

export type HeatmapMeasure = 'count' | 'mean_prob' | 'flagged_rate' | 'laundering_rate';

const STANDARD_MONTHS = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'];
const MONTH_KEY_MAP: Record<string, string> = {
  '01': 'Jan', '02': 'Feb', '03': 'Mar', '04': 'Apr', '05': 'May', '06': 'Jun',
  '07': 'Jul', '08': 'Aug', '09': 'Sep', '10': 'Oct', '11': 'Nov', '12': 'Dec',
  '1': 'Jan', '2': 'Feb', '3': 'Mar', '4': 'Apr', '5': 'May', '6': 'Jun',
  '7': 'Jul', '8': 'Aug', '9': 'Sep',
};

export function getFastMonth(ts?: string | null): string {
  if (!ts) return 'Jan';
  const str = String(ts).trim();
  // Formats like "2022/09/01 08:14" or "2022-09-01"
  if (str.length >= 7 && (str[4] === '/' || str[4] === '-')) {
    const mm = str.slice(5, 7);
    if (MONTH_KEY_MAP[mm]) return MONTH_KEY_MAP[mm];
  }
  // Formats like "09/01/2022"
  if (str.length >= 5 && (str[2] === '/' || str[2] === '-')) {
    const mm = str.slice(0, 2);
    if (MONTH_KEY_MAP[mm]) return MONTH_KEY_MAP[mm];
  }
  return 'Jan';
}

interface TransactionRiskHeatmapProps {
  predictions: TransactionPredictionItem[];
  activeThreshold: number;
  hasLabels: boolean;
  selectedCell?: { timeBucket: string; riskBand: string } | null;
  onSelectCell?: (cell: { timeBucket: string; riskBand: string; minScore: number; maxScore: number } | null) => void;
  className?: string;
}

interface CellData {
  timeBucket: string;
  riskBand: string;
  minScore: number;
  maxScore: number;
  count: number;
  sumProb: number;
  meanProb: number;
  flaggedCount: number;
  flaggedRate: number;
  launderingCount: number;
  launderingRate: number;
}

export const TransactionRiskHeatmap: React.FC<TransactionRiskHeatmapProps> = ({
  predictions,
  activeThreshold,
  hasLabels,
  selectedCell,
  onSelectCell,
  className = '',
}) => {
  const [isVisible, setIsVisible] = useState<boolean>(true);
  const [viewMode, setViewMode] = useState<'grid' | 'table'>('grid');
  const [measure, setMeasure] = useState<HeatmapMeasure>('count');
  const [hoveredCell, setHoveredCell] = useState<CellData | null>(null);

  // Define 5 Risk Bands
  const riskBands = useMemo(
    () => [
      { label: '0.8 - 1.0 (Critical)', min: 0.8, max: 1.0001, desc: 'Severe predicted risk' },
      { label: '0.6 - 0.8 (High)', min: 0.6, max: 0.8, desc: 'High predicted risk' },
      { label: '0.4 - 0.6 (Elevated)', min: 0.4, max: 0.6, desc: 'Elevated predicted risk' },
      { label: '0.2 - 0.4 (Moderate)', min: 0.2, max: 0.4, desc: 'Includes cutoff threshold' },
      { label: '0.0 - 0.2 (Minimal)', min: 0.0, max: 0.2, desc: 'Baseline normal volume' },
    ],
    []
  );

  // Check if usable timestamp exists
  const hasTimestamps = useMemo(() => {
    return predictions.some((p) => p.timestamp && String(p.timestamp).trim().length >= 4);
  }, [predictions]);

  // Extract chronological time buckets and compute cell statistics in a single high-performance O(N) pass
  const { timeBuckets, gridData, maxValues } = useMemo(() => {
    if (!predictions || predictions.length === 0 || !hasTimestamps) {
      return { timeBuckets: [], gridData: {}, maxValues: { count: 1, mean_prob: 1, flagged_rate: 1, laundering_rate: 1 } };
    }

    const bucketSet = new Set<string>();
    for (let i = 0; i < predictions.length; i++) {
      const ts = predictions[i].timestamp;
      if (ts) bucketSet.add(getFastMonth(ts));
    }

    const buckets = STANDARD_MONTHS.filter((m) => bucketSet.has(m));
    if (buckets.length === 0) {
      Array.from(bucketSet).forEach((b) => buckets.push(b));
    }

    // Initialize grid data matrix: grid[riskBandLabel][timeBucket] = CellData
    const grid: Record<string, Record<string, CellData>> = {};
    riskBands.forEach((band) => {
      grid[band.label] = {};
      buckets.forEach((tb) => {
        grid[band.label][tb] = {
          timeBucket: tb,
          riskBand: band.label,
          minScore: band.min,
          maxScore: band.max,
          count: 0,
          sumProb: 0,
          meanProb: 0,
          flaggedCount: 0,
          flaggedRate: 0,
          launderingCount: 0,
          launderingRate: 0,
        };
      });
    });

    // Populate matrix in single ultrafast pass (O(1) month lookup & O(1) index binning)
    for (let i = 0; i < predictions.length; i++) {
      const item = predictions[i];
      const prob = Number(item.predicted_probability ?? 0);
      const tbKey = getFastMonth(item.timestamp);
      if (!grid[riskBands[0].label][tbKey]) continue;

      const bandIdx = prob >= 0.8 ? 0 : prob >= 0.6 ? 1 : prob >= 0.4 ? 2 : prob >= 0.2 ? 3 : 4;
      const band = riskBands[bandIdx];
      const cell = grid[band.label][tbKey];

      cell.count += 1;
      cell.sumProb += prob;
      if (prob >= activeThreshold) cell.flaggedCount += 1;

      const isLaundering =
        item.ground_truth_label === 1 ||
        item.actual_label === 'SUSPICIOUS' ||
        (item as any)['Is Laundering'] === 1 ||
        (item as any)['is_laundering'] === 1;
      if (isLaundering) cell.launderingCount += 1;
    }

    // Compute aggregated measures and find maximums
    let maxCount = 0;
    riskBands.forEach((band) => {
      buckets.forEach((tb) => {
        const cell = grid[band.label][tb];
        if (cell.count > 0) {
          cell.meanProb = cell.sumProb / cell.count;
          cell.flaggedRate = cell.flaggedCount / cell.count;
          cell.launderingRate = cell.launderingCount / cell.count;
          if (cell.count > maxCount) maxCount = cell.count;
        }
      });
    });

    return {
      timeBuckets: buckets,
      gridData: grid,
      maxValues: {
        count: Math.max(1, maxCount),
        mean_prob: 1.0,
        flagged_rate: 1.0,
        laundering_rate: 1.0,
      },
    };
  }, [predictions, riskBands, activeThreshold, hasTimestamps]);

  // Color interpolation for heatmap cells
  const getCellColor = (val: number, maxVal: number, measureType: HeatmapMeasure): string => {
    if (val === 0) return 'bg-slate-950/60 border-slate-900 text-slate-600';

    const ratio = Math.min(1, Math.max(0, val / maxVal));

    if (measureType === 'count') {
      // Warm amber to bright flame scale with log scaling for visibility
      const logRatio = Math.log1p(val) / Math.log1p(maxVal);
      if (logRatio < 0.2) return 'bg-amber-950/40 border-amber-900/40 text-amber-300';
      if (logRatio < 0.4) return 'bg-amber-900/60 border-amber-700/60 text-amber-200';
      if (logRatio < 0.65) return 'bg-amber-700/70 border-amber-600/70 text-amber-100 font-semibold';
      if (logRatio < 0.85) return 'bg-orange-600/80 border-orange-500/80 text-white font-bold';
      return 'bg-rose-600 border-rose-400 text-white font-black shadow-inner shadow-rose-900';
    }

    if (measureType === 'mean_prob' || measureType === 'flagged_rate') {
      // Risk alert gradient
      if (ratio < 0.2) return 'bg-emerald-950/40 border-emerald-900/40 text-emerald-300';
      if (ratio < 0.4) return 'bg-teal-900/60 border-teal-700/60 text-teal-200';
      if (ratio < 0.6) return 'bg-amber-900/70 border-amber-700/70 text-amber-200';
      if (ratio < 0.8) return 'bg-orange-700/80 border-orange-500/80 text-white font-bold';
      return 'bg-rose-600 border-rose-400 text-white font-black';
    }

    // laundering_rate (ground truth label presence)
    if (ratio < 0.1) return 'bg-slate-900 border-slate-800 text-slate-400';
    if (ratio < 0.3) return 'bg-purple-950/50 border-purple-800/50 text-purple-300';
    if (ratio < 0.6) return 'bg-purple-900/70 border-purple-700/70 text-purple-200 font-semibold';
    return 'bg-purple-600 border-purple-400 text-white font-bold';
  };

  const formatCellValue = (cell: CellData, measureType: HeatmapMeasure): string => {
    if (cell.count === 0) return '—';
    if (measureType === 'count') return cell.count.toLocaleString();
    if (measureType === 'mean_prob') return cell.meanProb.toFixed(3);
    if (measureType === 'flagged_rate') return `${(cell.flaggedRate * 100).toFixed(1)}%`;
    return `${(cell.launderingRate * 100).toFixed(1)}%`;
  };

  if (!hasTimestamps || timeBuckets.length === 0) {
    return (
      <div className={`bg-slate-900 border border-slate-800 rounded-3xl p-6 shadow-xl space-y-3 ${className}`}>
        <div className="flex items-center space-x-2 text-slate-300 border-b border-slate-800 pb-3">
          <Calendar className="w-4 h-4 text-amber-400" />
          <h3 className="text-sm font-bold text-white">Temporal Transaction Risk Heatmap</h3>
        </div>
        <div className="p-4 rounded-2xl bg-slate-950 border border-slate-800 text-xs text-slate-400 flex items-start space-x-2.5">
          <Info className="w-4 h-4 text-amber-400 shrink-0 mt-0.5" />
          <div>
            <span className="font-bold text-slate-300 block">Temporal Heatmap Unavailable</span>
            <span>
              The uploaded evaluation dataset does not contain chronological timestamp fields or standard temporal records. Transaction risk is evaluated across cross-sectional feature dimensions.
            </span>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className={`bg-slate-900 border border-slate-800 rounded-3xl p-6 shadow-xl space-y-4 ${className}`}>
      {/* Header with Title & Action Controls */}
      <div className="flex flex-col md:flex-row md:items-center justify-between gap-3 border-b border-slate-800 pb-4">
        <div>
          <div className="flex items-center space-x-2">
            <Layers className="w-4 h-4 text-indigo-400" />
            <h3 className="text-sm font-bold text-white">
              Uploaded Batch Risk Distribution Heatmap
            </h3>
            <span className="px-2 py-0.5 rounded text-[10px] font-mono bg-indigo-950 text-indigo-300 border border-indigo-700/60">
              {predictions.length.toLocaleString()} txs
            </span>
          </div>
          <p className="text-xs text-slate-400 mt-0.5">
            Two-dimensional risk distribution mapping chronological time intervals against model predicted probability tiers.
          </p>
        </div>

        <div className="flex items-center space-x-2 flex-wrap gap-2">
          {/* Active Cell Filter Pill */}
          {selectedCell && (
            <div className="flex items-center space-x-1.5 px-2.5 py-1 rounded-xl bg-indigo-950 border border-indigo-700/70 text-indigo-300 text-xs font-mono">
              <Filter className="w-3 h-3 text-indigo-400" />
              <span>
                {selectedCell.timeBucket} • {selectedCell.riskBand}
              </span>
              <button
                onClick={() => onSelectCell && onSelectCell(null)}
                className="hover:text-white transition ml-1 p-0.5 cursor-pointer"
                title="Clear heatmap filter"
                aria-label="Clear heatmap filter"
              >
                <X className="w-3 h-3" />
              </button>
            </div>
          )}

          {/* View Switch: Grid vs Table */}
          <div
            role="tablist"
            aria-label="Heatmap view mode"
            className="flex items-center p-1 bg-slate-950 border border-slate-800 rounded-xl space-x-1"
          >
            <button
              role="tab"
              aria-selected={viewMode === 'grid'}
              tabIndex={0}
              onClick={() => setViewMode('grid')}
              onKeyDown={(e) => {
                if (e.key === 'Enter' || e.key === ' ') {
                  e.preventDefault();
                  setViewMode('grid');
                }
              }}
              className={`p-1.5 rounded-lg text-xs font-semibold flex items-center space-x-1 transition cursor-pointer ${
                viewMode === 'grid'
                  ? 'bg-indigo-600 text-white font-bold shadow-md shadow-indigo-600/30'
                  : 'text-slate-400 hover:text-white'
              }`}
              title="Interactive Heatmap Grid"
            >
              <Grid className="w-3.5 h-3.5" />
              <span className="hidden sm:inline">Grid</span>
            </button>

            <button
              role="tab"
              aria-selected={viewMode === 'table'}
              tabIndex={0}
              onClick={() => setViewMode('table')}
              onKeyDown={(e) => {
                if (e.key === 'Enter' || e.key === ' ') {
                  e.preventDefault();
                  setViewMode('table');
                }
              }}
              className={`p-1.5 rounded-lg text-xs font-semibold flex items-center space-x-1 transition cursor-pointer ${
                viewMode === 'table'
                  ? 'bg-indigo-600 text-white font-bold shadow-md shadow-indigo-600/30'
                  : 'text-slate-400 hover:text-white'
              }`}
              title="Tabular Matrix View"
            >
              <TableIcon className="w-3.5 h-3.5" />
              <span className="hidden sm:inline">Table</span>
            </button>
          </div>

          {/* Show / Hide Toggle */}
          <button
            onClick={() => setIsVisible(!isVisible)}
            className="p-1.5 px-2.5 rounded-xl bg-slate-950 hover:bg-slate-800 border border-slate-800 text-slate-300 text-xs font-semibold flex items-center space-x-1 transition cursor-pointer"
            aria-expanded={isVisible}
          >
            {isVisible ? (
              <>
                <ChevronUp className="w-3.5 h-3.5 text-slate-400" />
                <span className="hidden sm:inline">Hide</span>
              </>
            ) : (
              <>
                <ChevronDown className="w-3.5 h-3.5 text-slate-400" />
                <span className="hidden sm:inline">Show</span>
              </>
            )}
          </button>
        </div>
      </div>

      {isVisible && (
        <div className="space-y-4 animate-fade-in">
          {/* Measure Selector Tabs */}
          <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 bg-slate-950/70 p-2.5 rounded-2xl border border-slate-800/80">
            <div className="flex items-center space-x-1.5 flex-wrap gap-1">
              <span className="text-[11px] font-mono text-slate-400 mr-2 flex items-center space-x-1">
                <BarChart2 className="w-3 h-3 text-indigo-400" />
                <span>Cell Metric:</span>
              </span>

              <button
                onClick={() => setMeasure('count')}
                className={`px-2.5 py-1 rounded-lg text-xs font-mono transition cursor-pointer ${
                  measure === 'count'
                    ? 'bg-indigo-600 text-white font-bold shadow-md shadow-indigo-600/30'
                    : 'bg-slate-900 text-slate-300 hover:bg-slate-800 border border-slate-800'
                }`}
              >
                Transaction Count
              </button>

              <button
                onClick={() => setMeasure('mean_prob')}
                className={`px-2.5 py-1 rounded-lg text-xs font-mono transition cursor-pointer ${
                  measure === 'mean_prob'
                    ? 'bg-indigo-600 text-white font-bold shadow-md shadow-indigo-600/30'
                    : 'bg-slate-900 text-slate-300 hover:bg-slate-800 border border-slate-800'
                }`}
              >
                Mean Risk Probability
              </button>

              <button
                onClick={() => setMeasure('flagged_rate')}
                className={`px-2.5 py-1 rounded-lg text-xs font-mono transition cursor-pointer ${
                  measure === 'flagged_rate'
                    ? 'bg-indigo-600 text-white font-bold shadow-md shadow-indigo-600/30'
                    : 'bg-slate-900 text-slate-300 hover:bg-slate-800 border border-slate-800'
                }`}
              >
                Predicted Flagged Rate (≥ θ)
              </button>

              {hasLabels && (
                <button
                  onClick={() => setMeasure('laundering_rate')}
                  className={`px-2.5 py-1 rounded-lg text-xs font-mono transition cursor-pointer ${
                    measure === 'laundering_rate'
                      ? 'bg-purple-600 text-white font-bold shadow-md shadow-purple-600/30'
                      : 'bg-slate-900 text-slate-300 hover:bg-slate-800 border border-slate-800'
                  }`}
                >
                  CSV Laundering Rate
                </button>
              )}
            </div>

            <div className="text-[10px] text-slate-400 font-mono">
              Click cell to filter transactions table
            </div>
          </div>

          {/* View 1: Heatmap Grid Mode */}
          {viewMode === 'grid' ? (
            <div className="overflow-x-auto pb-2">
              <div className="min-w-[620px]">
                {/* Grid Header (Time Buckets) */}
                <div className="grid grid-cols-12 gap-1.5 mb-1.5 ml-36">
                  {timeBuckets.map((tb) => (
                    <div
                      key={tb}
                      className="text-center text-[11px] font-mono font-bold text-slate-300 uppercase py-1 bg-slate-950/80 rounded-lg border border-slate-800/60"
                    >
                      {tb}
                    </div>
                  ))}
                </div>

                {/* Grid Rows (Risk Bands) */}
                <div className="space-y-1.5">
                  {riskBands.map((band) => (
                    <div key={band.label} className="flex items-center space-x-1.5">
                      {/* Row Label (Y-axis) */}
                      <div className="w-36 shrink-0 text-right pr-2">
                        <span className="text-xs font-mono font-bold text-slate-200 block truncate">
                          {band.label}
                        </span>
                        <span className="text-[9.5px] text-slate-500 block truncate">
                          {band.desc}
                        </span>
                      </div>

                      {/* Cells */}
                      <div className="grid grid-cols-12 gap-1.5 flex-1">
                        {timeBuckets.map((tb) => {
                          const cell = gridData[band.label]?.[tb];
                          if (!cell) return null;

                          const isSelected =
                            selectedCell?.timeBucket === tb && selectedCell?.riskBand === band.label;
                          const cellVal =
                            measure === 'count'
                              ? cell.count
                              : measure === 'mean_prob'
                              ? cell.meanProb
                              : measure === 'flagged_rate'
                              ? cell.flaggedRate
                              : cell.launderingRate;

                          const maxVal = maxValues[measure];
                          const colorClasses = getCellColor(cellVal, maxVal, measure);

                          return (
                            <button
                              key={tb}
                              type="button"
                              onClick={() => {
                                if (onSelectCell) {
                                  if (isSelected) {
                                    onSelectCell(null);
                                  } else {
                                    onSelectCell({
                                      timeBucket: tb,
                                      riskBand: band.label,
                                      minScore: band.min,
                                      maxScore: band.max,
                                    });
                                  }
                                }
                              }}
                              onMouseEnter={() => setHoveredCell(cell)}
                              onMouseLeave={() => setHoveredCell(null)}
                              onFocus={() => setHoveredCell(cell)}
                              onBlur={() => setHoveredCell(null)}
                              className={`h-11 rounded-xl border flex flex-col items-center justify-center transition-all cursor-pointer select-none relative ${colorClasses} ${
                                isSelected
                                  ? 'ring-2 ring-indigo-400 border-white scale-105 z-10'
                                  : 'hover:scale-[1.03] hover:z-10 hover:border-slate-300'
                              }`}
                              tabIndex={0}
                              role="button"
                              aria-label={`${band.label} in ${tb}: ${formatCellValue(cell, measure)}`}
                            >
                              <span className="text-xs font-mono font-bold leading-none">
                                {formatCellValue(cell, measure)}
                              </span>
                            </button>
                          );
                        })}
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            </div>
          ) : (
            /* View 2: Accessible Tabular Matrix Mode */
            <div className="overflow-x-auto rounded-2xl border border-slate-800">
              <table className="w-full text-left border-collapse text-xs font-mono">
                <thead>
                  <tr className="bg-slate-950 border-b border-slate-800 text-slate-400 uppercase text-[10px]">
                    <th className="py-2.5 px-3 min-w-[140px]">Risk Band</th>
                    {timeBuckets.map((tb) => (
                      <th key={tb} className="py-2.5 px-2 text-center min-w-[55px]">
                        {tb}
                      </th>
                    ))}
                    <th className="py-2.5 px-3 text-right">Band Total</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-slate-800/60">
                  {riskBands.map((band) => {
                    const rowCells = timeBuckets.map((tb) => gridData[band.label]?.[tb]);
                    const rowTotal = rowCells.reduce((acc, c) => acc + (c?.count || 0), 0);

                    return (
                      <tr key={band.label} className="hover:bg-slate-800/40 transition">
                        <td className="py-2.5 px-3 font-bold text-white whitespace-nowrap">
                          <div>
                            <span>{band.label}</span>
                            <span className="text-[10px] text-slate-400 block font-normal">
                              {band.desc}
                            </span>
                          </div>
                        </td>
                        {timeBuckets.map((tb) => {
                          const cell = gridData[band.label]?.[tb];
                          if (!cell) return <td key={tb}>-</td>;
                          const isSelected =
                            selectedCell?.timeBucket === tb && selectedCell?.riskBand === band.label;
                          return (
                            <td
                              key={tb}
                              onClick={() => {
                                if (onSelectCell) {
                                  onSelectCell(
                                    isSelected
                                      ? null
                                      : {
                                          timeBucket: tb,
                                          riskBand: band.label,
                                          minScore: band.min,
                                          maxScore: band.max,
                                        }
                                  );
                                }
                              }}
                              className={`py-2 px-2 text-center cursor-pointer transition ${
                                isSelected
                                  ? 'bg-indigo-900/60 font-bold text-indigo-300'
                                  : 'hover:bg-slate-800 text-slate-300'
                              }`}
                            >
                              {formatCellValue(cell, measure)}
                            </td>
                          );
                        })}
                        <td className="py-2.5 px-3 text-right font-bold text-white">
                          {rowTotal.toLocaleString()}
                        </td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
          )}

          {/* Interactive Inspection Tooltip / Summary Panel */}
          {hoveredCell && (
            <div className="p-3 bg-slate-950 rounded-2xl border border-indigo-500/40 text-xs font-mono text-slate-300 grid grid-cols-2 sm:grid-cols-4 gap-3 animate-fade-in">
              <div>
                <span className="text-[10px] text-slate-500 uppercase block">Selected Interval</span>
                <span className="font-bold text-white">
                  {hoveredCell.timeBucket} • {hoveredCell.riskBand}
                </span>
              </div>
              <div>
                <span className="text-[10px] text-slate-500 uppercase block">Total Volume</span>
                <span className="font-bold text-white">
                  {hoveredCell.count.toLocaleString()} transactions
                </span>
              </div>
              <div>
                <span className="text-[10px] text-slate-500 uppercase block">Mean Risk Score</span>
                <span className="font-bold text-teal-300">{hoveredCell.meanProb.toFixed(4)}</span>
              </div>
              <div>
                <span className="text-[10px] text-slate-500 uppercase block">Triage Breakdown</span>
                <span className="font-bold text-rose-300">
                  {hoveredCell.flaggedCount} alerts ({((hoveredCell.flaggedRate) * 100).toFixed(1)}%)
                  {hasLabels && (
                    <span className="text-purple-300 ml-1">
                      • {hoveredCell.launderingCount} GT
                    </span>
                  )}
                </span>
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  );
};
