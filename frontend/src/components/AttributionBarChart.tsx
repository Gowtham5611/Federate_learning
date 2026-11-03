import React from 'react';
import { ExplainResponse } from '../types/api';
import { formatFeatureName } from '../utils/formatters';
import { HelpCircle, AlertCircle, CheckCircle } from 'lucide-react';

interface AttributionBarChartProps {
  explanation: ExplainResponse;
}

export const AttributionBarChart: React.FC<AttributionBarChartProps> = ({ explanation }) => {
  const {
    method,
    top_positive_drivers,
    top_negative_drivers,
    completeness_residual,
    steps,
    baseline_type,
    explanation_time_ms,
  } = explanation;

  // Find max absolute attribution for proportionate scaling
  const allDrivers = [...top_positive_drivers, ...top_negative_drivers];
  const maxAbsScore = Math.max(...allDrivers.map((d) => Math.abs(d.attribution)), 0.001);

  const axiomSatisfied = completeness_residual < 0.05;

  return (
    <div className="bg-slate-900 border border-slate-800 rounded-2xl p-6 shadow-xl">
      {/* Header */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between pb-4 border-b border-slate-800 gap-2">
        <div>
          <div className="flex items-center space-x-2">
            <span className="text-xs font-mono uppercase tracking-wider text-indigo-400 font-semibold">
              Attribution Algorithm:
            </span>
            <span className="px-2.5 py-0.5 rounded-full text-xs font-mono bg-indigo-950 text-indigo-300 border border-indigo-500/30 font-semibold">
              {method}
            </span>
          </div>
          <p className="text-xs text-slate-400 mt-1">
            Path integration along straight-line trajectory from baseline (m={steps} Riemann steps).
          </p>
        </div>

        {/* Completeness Residual Pill */}
        <div className="flex items-center space-x-2 px-3 py-1.5 rounded-xl bg-slate-950 border border-slate-800 text-xs">
          <div className="text-slate-400">Completeness Residual:</div>
          <div className="font-mono text-slate-200">
            |∑IG - ΔF| = {completeness_residual.toExponential(4)}
          </div>
          {axiomSatisfied ? (
            <span title="Axiom 1 (Completeness) Satisfied (< 0.05)" className="text-emerald-400 flex items-center">
              <CheckCircle className="w-3.5 h-3.5" />
            </span>
          ) : (
            <span title="Residual exceeds standard 0.05 tolerance" className="text-amber-400 flex items-center">
              <AlertCircle className="w-3.5 h-3.5" />
            </span>
          )}
        </div>
      </div>

      {/* History Warning Banner */}
      {explanation.history_warning && (
        <div className="mt-4 p-3.5 bg-amber-950/50 border border-amber-600/70 rounded-xl text-xs text-amber-200 flex items-start space-x-2.5">
          <AlertCircle className="w-4 h-4 text-amber-400 shrink-0 mt-0.5" />
          <div className="space-y-0.5">
            <span className="font-bold text-amber-300">INSUFFICIENT SEQUENCE CONTEXT WARNING:</span>
            <p className="text-amber-200/90 leading-relaxed font-sans">{explanation.history_warning}</p>
          </div>
        </div>
      )}

      {/* Driver Analysis Grid */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-8 mt-6">
        {/* Positive Drivers (Pushes towards Laundering Alert) */}
        <div>
          <div className="flex items-center justify-between mb-3">
            <div className="flex items-center space-x-2">
              <div className="w-2.5 h-2.5 rounded-full bg-rose-500 shadow-sm shadow-rose-500/50" />
              <h4 className="text-xs font-semibold uppercase tracking-wider text-rose-400">
                Top Risk Drivers (Pushing Toward Alert)
              </h4>
            </div>
            <span className="text-xs text-slate-500 font-mono">Sign: +</span>
          </div>

          {top_positive_drivers.length === 0 ? (
            <div className="p-4 bg-slate-950/40 rounded-xl border border-slate-800/80 text-center text-xs text-slate-500">
              No positive risk drivers detected for this sequence.
            </div>
          ) : (
            <div className="space-y-3">
              {top_positive_drivers.map((driver, idx) => {
                const pct = Math.min((driver.attribution / maxAbsScore) * 100, 100);
                return (
                  <div key={idx} className="p-3 bg-slate-950/60 rounded-xl border border-slate-800/80 hover:border-slate-700 transition">
                    <div className="flex items-center justify-between text-xs mb-1.5">
                      <div className="flex items-center space-x-2 truncate">
                        <span className="font-medium text-slate-200 truncate" title={driver.feature_name}>
                          {formatFeatureName(driver.feature_name)}
                        </span>
                        <span className="px-1.5 py-0.5 rounded text-[10px] bg-slate-800 text-slate-400">
                          {driver.feature_group}
                        </span>
                      </div>
                      <span className="font-mono text-rose-400 font-semibold ml-2">
                        +{driver.attribution.toFixed(4)}
                      </span>
                    </div>

                    {/* Progress Bar */}
                    <div className="w-full bg-slate-800/80 rounded-full h-2 overflow-hidden">
                      <div
                        className="bg-gradient-to-r from-rose-600 to-rose-400 h-2 rounded-full transition-all duration-500"
                        style={{ width: `${Math.max(pct, 4)}%` }}
                      />
                    </div>
                  </div>
                );
              })}
            </div>
          )}
        </div>

        {/* Negative Drivers (Pushes towards Legitimate Clearance) */}
        <div>
          <div className="flex items-center justify-between mb-3">
            <div className="flex items-center space-x-2">
              <div className="w-2.5 h-2.5 rounded-full bg-emerald-500 shadow-sm shadow-emerald-500/50" />
              <h4 className="text-xs font-semibold uppercase tracking-wider text-emerald-400">
                Top Mitigating Drivers (Dampening Risk)
              </h4>
            </div>
            <span className="text-xs text-slate-500 font-mono">Sign: -</span>
          </div>

          {top_negative_drivers.length === 0 ? (
            <div className="p-4 bg-slate-950/40 rounded-xl border border-slate-800/80 text-center text-xs text-slate-500">
              No mitigating drivers detected for this sequence.
            </div>
          ) : (
            <div className="space-y-3">
              {top_negative_drivers.map((driver, idx) => {
                const pct = Math.min((Math.abs(driver.attribution) / maxAbsScore) * 100, 100);
                return (
                  <div key={idx} className="p-3 bg-slate-950/60 rounded-xl border border-slate-800/80 hover:border-slate-700 transition">
                    <div className="flex items-center justify-between text-xs mb-1.5">
                      <div className="flex items-center space-x-2 truncate">
                        <span className="font-medium text-slate-200 truncate" title={driver.feature_name}>
                          {formatFeatureName(driver.feature_name)}
                        </span>
                        <span className="px-1.5 py-0.5 rounded text-[10px] bg-slate-800 text-slate-400">
                          {driver.feature_group}
                        </span>
                      </div>
                      <span className="font-mono text-emerald-400 font-semibold ml-2">
                        {driver.attribution.toFixed(4)}
                      </span>
                    </div>

                    {/* Progress Bar */}
                    <div className="w-full bg-slate-800/80 rounded-full h-2 overflow-hidden">
                      <div
                        className="bg-gradient-to-r from-emerald-600 to-emerald-400 h-2 rounded-full transition-all duration-500"
                        style={{ width: `${Math.max(pct, 4)}%` }}
                      />
                    </div>
                  </div>
                );
              })}
            </div>
          )}
        </div>
      </div>

      {/* Footer Details */}
      <div className="mt-6 pt-4 border-t border-slate-800/80 flex flex-wrap items-center justify-between text-[11px] text-slate-400 gap-2">
        <div className="flex items-center space-x-1.5">
          <HelpCircle className="w-3.5 h-3.5 text-slate-500" />
          <span>
            Baseline reference: <strong className="text-slate-300 font-mono">{baseline_type}</strong>
          </span>
        </div>
        <div className="text-slate-500 font-mono">
          Attribution calculation completed in {explanation_time_ms.toFixed(2)} ms
        </div>
      </div>
    </div>
  );
};

