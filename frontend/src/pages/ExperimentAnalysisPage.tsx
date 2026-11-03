import React, { useState } from 'react';
import {
  AblationSummaryResponse,
  ConvergenceCurvePoint,
  SupportedDataset,
  ThresholdTradeoffPoint,
} from '../types/api';
import {
  TrendingUp,
  GitCompare,
  Sliders,
  CheckCircle2,
  AlertTriangle,
  Scale,
} from 'lucide-react';

interface ExperimentAnalysisPageProps {
  curves: ConvergenceCurvePoint[];
  thresholds: ThresholdTradeoffPoint[];
  ablation: AblationSummaryResponse | null;
  selectedDataset: SupportedDataset;
  onSelectDataset: (dataset: SupportedDataset) => void;
}

export const ExperimentAnalysisPage: React.FC<ExperimentAnalysisPageProps> = ({
  curves,
  thresholds,
  ablation,
  selectedDataset,
  onSelectDataset,
}) => {
  const [metricView, setMetricView] = useState<'val_pr_auc' | 'val_loss'>('val_pr_auc');
  const [selectedArch, setSelectedArch] = useState<'ALL' | 'LSTM' | 'BiLSTM' | 'GRU'>('ALL');

  // Filter curves for current dataset
  const datasetCurves = curves.filter((c) => c.dataset === selectedDataset);

  // Group curves by config_id
  const configsMap = new Map<string, ConvergenceCurvePoint[]>();
  datasetCurves.forEach((pt) => {
    if (selectedArch !== 'ALL' && pt.architecture.toUpperCase() !== selectedArch.toUpperCase()) {
      return;
    }
    if (!configsMap.has(pt.config_id)) {
      configsMap.set(pt.config_id, []);
    }
    configsMap.get(pt.config_id)!.push(pt);
  });

  // Sort each config points by round
  configsMap.forEach((pts) => pts.sort((a, b) => a.round - b.round));

  // Determine chart scales
  let maxMetric = 0.5;
  let minMetric = 0.0;
  if (metricView === 'val_pr_auc') {
    maxMetric = selectedDataset === 'HI-Small' ? 0.35 : 0.25;
    minMetric = 0.0;
  } else {
    maxMetric = selectedDataset === 'HI-Small' ? 0.1 : 0.005;
    minMetric = 0.0;
  }

  const svgWidth = 800;
  const svgHeight = 280;
  const padLeft = 60;
  const padRight = 40;
  const padTop = 30;
  const padBottom = 40;

  const chartW = svgWidth - padLeft - padRight;
  const chartH = svgHeight - padTop - padBottom;

  const getX = (round: number) => padLeft + ((round - 1) / 4) * chartW;
  const getY = (val: number) => {
    const clamped = Math.max(minMetric, Math.min(maxMetric, val));
    const ratio = (clamped - minMetric) / (maxMetric - minMetric || 1);
    return padTop + (1 - ratio) * chartH;
  };

  const colors = [
    '#6366f1', '#10b981', '#f59e0b', '#ec4899', '#8b5cf6',
    '#06b6d4', '#ef4444', '#14b8a6', '#f97316', '#3b82f6',
  ];

  return (
    <div className="space-y-8 animate-fade-in">
      {/* Header and Controls */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 border-b border-slate-800 pb-5">
        <div>
          <h1 className="text-2xl sm:text-3xl font-black text-white tracking-tight">Training & Experiment Analysis</h1>
          <p className="text-xs text-slate-400 mt-1">
            Empirical convergence curves, ablation studies, and operational threshold tradeoffs from authoritative runs.
          </p>
        </div>

        {/* Dataset Toggle */}
        <div className="flex items-center space-x-2 bg-slate-900 p-1.5 rounded-2xl border border-slate-800">
          <button
            onClick={() => onSelectDataset('HI-Small')}
            className={`px-3.5 py-1.5 rounded-xl text-xs font-bold transition ${
              selectedDataset === 'HI-Small'
                ? 'bg-indigo-600 text-white shadow-lg shadow-indigo-600/30'
                : 'text-slate-400 hover:text-slate-200'
            }`}
          >
            HI-Small (5M)
          </button>
          <button
            onClick={() => onSelectDataset('LI-Medium')}
            className={`px-3.5 py-1.5 rounded-xl text-xs font-bold transition ${
              selectedDataset === 'LI-Medium'
                ? 'bg-violet-600 text-white shadow-lg shadow-violet-600/30'
                : 'text-slate-400 hover:text-slate-200'
            }`}
          >
            LI-Medium (31M)
          </button>
        </div>
      </div>

      {/* Convergence Curves Card */}
      <div className="bg-slate-900 border border-slate-800 rounded-3xl p-6 shadow-xl space-y-4">
        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3">
          <div>
            <h2 className="text-lg font-bold text-white flex items-center space-x-2">
              <TrendingUp className="w-5 h-5 text-indigo-400" />
              <span>Federated Convergence Curves (5 Communication Rounds)</span>
            </h2>
            <p className="text-xs text-slate-400">
              Evaluated on the validation split at the conclusion of each federated round.
            </p>
          </div>

          <div className="flex items-center space-x-2 text-xs">
            {/* Metric Toggle */}
            <div className="flex items-center bg-slate-950 p-1 rounded-xl border border-slate-800">
              <button
                onClick={() => setMetricView('val_pr_auc')}
                className={`px-3 py-1 rounded-lg text-xs font-semibold transition ${
                  metricView === 'val_pr_auc' ? 'bg-indigo-600 text-white' : 'text-slate-400'
                }`}
              >
                Validation PR-AUC
              </button>
              <button
                onClick={() => setMetricView('val_loss')}
                className={`px-3 py-1 rounded-lg text-xs font-semibold transition ${
                  metricView === 'val_loss' ? 'bg-indigo-600 text-white' : 'text-slate-400'
                }`}
              >
                Validation Loss
              </button>
            </div>

            {/* Architecture Filter */}
            <select
              value={selectedArch}
              onChange={(e) => setSelectedArch(e.target.value as any)}
              className="bg-slate-950 border border-slate-800 text-slate-300 text-xs rounded-xl px-3 py-1.5 focus:outline-none"
            >
              <option value="ALL">All Architectures</option>
              <option value="LSTM">LSTM Only</option>
              <option value="BiLSTM">BiLSTM Only</option>
              <option value="GRU">GRU Only</option>
            </select>
          </div>
        </div>

        {/* SVG Chart */}
        <div className="bg-slate-950/70 rounded-2xl border border-slate-800/80 p-4 overflow-x-auto">
          <svg viewBox={`0 0 ${svgWidth} ${svgHeight}`} className="w-full h-auto min-w-[650px]">
            {/* Horizontal Grid lines */}
            {[0, 0.25, 0.5, 0.75, 1.0].map((frac, i) => {
              const yVal = minMetric + frac * (maxMetric - minMetric);
              const yPos = getY(yVal);
              return (
                <g key={i}>
                  <line
                    x1={padLeft}
                    y1={yPos}
                    x2={svgWidth - padRight}
                    y2={yPos}
                    stroke="#1e293b"
                    strokeDasharray="4 4"
                    strokeWidth={1}
                  />
                  <text
                    x={padLeft - 10}
                    y={yPos + 3}
                    textAnchor="end"
                    fill="#64748b"
                    fontSize={10}
                    fontFamily="monospace"
                  >
                    {yVal.toFixed(metricView === 'val_pr_auc' ? 3 : 4)}
                  </text>
                </g>
              );
            })}

            {/* Vertical Round Grid lines */}
            {[1, 2, 3, 4, 5].map((round) => {
              const xPos = getX(round);
              return (
                <g key={round}>
                  <line
                    x1={xPos}
                    y1={padTop}
                    x2={xPos}
                    y2={svgHeight - padBottom}
                    stroke="#1e293b"
                    strokeWidth={1}
                  />
                  <text
                    x={xPos}
                    y={svgHeight - padBottom + 18}
                    textAnchor="middle"
                    fill="#94a3b8"
                    fontSize={11}
                    fontFamily="monospace"
                  >
                    Round {round}
                  </text>
                </g>
              );
            })}

            {/* Plot Lines */}
            {Array.from(configsMap.entries()).map(([configId, pts], idx) => {
              if (pts.length < 2) return null;
              const color = colors[idx % colors.length];

              const pathD = pts
                .map((p, i) => {
                  const val = metricView === 'val_pr_auc' ? p.val_pr_auc : p.val_loss;
                  return `${i === 0 ? 'M' : 'L'} ${getX(p.round)} ${getY(val)}`;
                })
                .join(' ');

              return (
                <g key={configId}>
                  <path
                    d={pathD}
                    fill="none"
                    stroke={color}
                    strokeWidth={configId.includes('seed42') || configId === 'config_5' ? 2.5 : 1.5}
                    opacity={0.85}
                  />
                  {pts.map((p, i) => {
                    const val = metricView === 'val_pr_auc' ? p.val_pr_auc : p.val_loss;
                    return (
                      <circle
                        key={i}
                        cx={getX(p.round)}
                        cy={getY(val)}
                        r={3}
                        fill={color}
                      />
                    );
                  })}
                </g>
              );
            })}
          </svg>
        </div>

        {/* Legend */}
        <div className="flex flex-wrap items-center gap-3 pt-2">
          {Array.from(configsMap.keys()).map((configId, idx) => {
            const color = colors[idx % colors.length];
            return (
              <div key={configId} className="flex items-center space-x-1.5 text-[11px] font-mono text-slate-300">
                <span className="w-2.5 h-2.5 rounded-full" style={{ backgroundColor: color }} />
                <span className="truncate max-w-[200px]" title={configId}>{configId}</span>
              </div>
            );
          })}
        </div>
      </div>

      {/* Empirical Ablation Findings Cards */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        {/* FedProx vs FedAvg Imbalance Collapse */}
        <div className="bg-slate-900 border border-slate-800 rounded-2xl p-6 shadow-xl space-y-3">
          <div className="flex items-center space-x-2 text-rose-400">
            <AlertTriangle className="w-5 h-5 shrink-0" />
            <h3 className="font-bold text-white text-base">FedProx Proximal Term Suppression</h3>
          </div>
          <p className="text-xs text-slate-300 leading-relaxed">
            In standard federated benchmarks (e.g. MNIST/CIFAR), FedProx's proximal penalty parameter (mu=0.0001)
            stabilizes client drift. However, under extreme AML class imbalance (0.06% - 0.18%), the proximal term pulls
            updates toward the global majority class (legitimate), severely contracting minority positive gradients.
          </p>
          <div className="bg-slate-950 p-3 rounded-xl border border-slate-800 font-mono text-xs space-y-1">
            <div className="text-slate-400">HI-Small LSTM: <span className="text-rose-400 font-bold">-40.66% PR-AUC drop</span> (0.4760 → 0.2824)</div>
            <div className="text-slate-400">LI-Medium BiLSTM: <span className="text-rose-400 font-bold">-65.63% PR-AUC drop</span> (0.2534 → 0.0871)</div>
          </div>
        </div>

        {/* Non-IID Skew Degradation */}
        <div className="bg-slate-900 border border-slate-800 rounded-2xl p-6 shadow-xl space-y-3">
          <div className="flex items-center space-x-2 text-amber-400">
            <GitCompare className="w-5 h-5 shrink-0" />
            <h3 className="font-bold text-white text-base">Non-IID Bank Typology Divergence</h3>
          </div>
          <p className="text-xs text-slate-300 leading-relaxed">
            When banks represent specialized business domains (retail branch vs corporate vs crypto exchange),
            transaction patterns diverge sharply. On 5M rows (HI-Small), Non-IID drop was manageable (-5.35%).
            At 31M rows (LI-Medium), Non-IID partitioning induced a catastrophic -90.54% collapse.
          </p>
          <div className="bg-slate-950 p-3 rounded-xl border border-slate-800 font-mono text-xs space-y-1">
            <div className="text-slate-400">HI-Small LSTM Non-IID: <span className="text-amber-400 font-bold">-5.35% PR-AUC drop</span> (0.4760 → 0.4505)</div>
            <div className="text-slate-400">LI-Medium BiLSTM Non-IID: <span className="text-rose-400 font-bold">-90.54% collapse</span> (0.2534 → 0.0240)</div>
          </div>
        </div>

        {/* Hybrid Ensembling */}
        <div className="bg-slate-900 border border-slate-800 rounded-2xl p-6 shadow-xl space-y-3">
          <div className="flex items-center space-x-2 text-indigo-400">
            <Sliders className="w-5 h-5 shrink-0" />
            <h3 className="font-bold text-white text-base">Phase 7 Hybrid Ensembling (Alpha Grid)</h3>
          </div>
          <p className="text-xs text-slate-300 leading-relaxed">
            Formulation: P_blend(x) = alpha * P_IID(x) + (1 - alpha) * P_NonIID(x).
            Rigorous grid search on validation splits revealed optimal alpha* = 1.00 for both HI-Small and LI-Medium,
            protecting production inference by rejecting corrupted non-IID weights.
          </p>
          <div className="bg-slate-950 p-3 rounded-xl border border-slate-800 font-mono text-xs text-slate-300">
            Optimal alpha*: <strong className="text-emerald-400">1.00</strong> (Selected serving models are 100% IID stream)
          </div>
        </div>

        {/* Post-Hoc Probability Calibration */}
        <div className="bg-slate-900 border border-slate-800 rounded-2xl p-6 shadow-xl space-y-3">
          <div className="flex items-center space-x-2 text-emerald-400">
            <CheckCircle2 className="w-5 h-5 shrink-0" />
            <h3 className="font-bold text-white text-base">Phase 10 Probability Calibration</h3>
          </div>
          <p className="text-xs text-slate-300 leading-relaxed">
            Tested on HI-Small reference model across 1,015,564 test sequences. Isotonic Regression achieved a
            56.18% reduction in Expected Calibration Error (ECE) without modifying discrimination rankings (F1 unchanged).
          </p>
          <div className="bg-slate-950 p-3 rounded-xl border border-slate-800 font-mono text-xs space-y-1">
            <div className="text-slate-400">Uncalibrated Baseline ECE: <span className="text-slate-200">{ablation?.phase10_probability_calibration.uncalibrated_ece ?? 0.000494}</span></div>
            <div className="text-slate-400">Isotonic Regression ECE: <span className="text-emerald-400 font-bold">{ablation?.phase10_probability_calibration.isotonic_ece ?? 0.000217} (-56.18%)</span></div>
          </div>
        </div>
      </div>

      {/* Operational Threshold Tradeoff Table */}
      {thresholds.length > 0 && (
        <div className="bg-slate-900 border border-slate-800 rounded-3xl p-6 shadow-xl space-y-4">
          <div className="flex items-center space-x-2 text-amber-400">
            <Scale className="w-5 h-5 shrink-0" />
            <h3 className="font-bold text-white text-base">Operational Decision Threshold Tradeoffs</h3>
          </div>
          <p className="text-xs text-slate-400">
            Precision, Recall, F1, and alert volume evaluated across operating thresholds for the selected model.
          </p>
          <div className="overflow-x-auto border border-slate-800 rounded-2xl">
            <table className="w-full text-left border-collapse text-xs font-mono">
              <thead>
                <tr className="bg-slate-950 border-b border-slate-800 text-slate-400 text-[10px] uppercase">
                  <th className="py-2.5 px-3">Operating Threshold</th>
                  <th className="py-2.5 px-3 text-right">Precision</th>
                  <th className="py-2.5 px-3 text-right">Recall</th>
                  <th className="py-2.5 px-3 text-right">F1-Score</th>
                  <th className="py-2.5 px-3 text-right">Alerts per 10k</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-800/60">
                {thresholds.slice(0, 8).map((t, i) => (
                  <tr key={i} className="hover:bg-slate-800/40">
                    <td className="py-2.5 px-3 text-amber-400 font-bold">{t.threshold.toFixed(4)}</td>
                    <td className="py-2.5 px-3 text-right text-slate-200">{t.precision.toFixed(4)}</td>
                    <td className="py-2.5 px-3 text-right text-slate-200">{t.recall.toFixed(4)}</td>
                    <td className="py-2.5 px-3 text-right text-white font-bold">{t.f1.toFixed(4)}</td>
                    <td className="py-2.5 px-3 text-right text-slate-400">{t.alerts_per_10k.toFixed(1)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}
    </div>
  );
};
