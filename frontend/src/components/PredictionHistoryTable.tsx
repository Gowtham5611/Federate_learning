import React from 'react';
import { PredictionHistoryEntry } from '../types/api';
import { formatProbability, formatThreshold, formatTimestamp, getRiskLevel } from '../utils/formatters';
import { Clock, ShieldAlert, Trash2 } from 'lucide-react';

interface PredictionHistoryTableProps {
  entries: PredictionHistoryEntry[];
  onClearHistory: () => void;
}

export const PredictionHistoryTable: React.FC<PredictionHistoryTableProps> = ({
  entries,
  onClearHistory,
}) => {
  return (
    <div className="bg-slate-900 border border-slate-800 rounded-2xl p-6 shadow-xl">
      <div className="flex items-center justify-between pb-4 border-b border-slate-800">
        <div>
          <div className="flex items-center space-x-2">
            <Clock className="w-5 h-5 text-indigo-400" />
            <h3 className="font-semibold text-white text-base">Session Prediction Audit Trail</h3>
            <span className="px-2 py-0.5 rounded-full text-xs font-mono bg-slate-800 text-slate-300">
              {entries.length} queries
            </span>
          </div>
          <p className="text-xs text-slate-400 mt-1">
            In-memory ephemeral audit trail. Zero customer account numbers or raw transaction sequences are persisted.
          </p>
        </div>

        {entries.length > 0 && (
          <button
            onClick={onClearHistory}
            className="flex items-center space-x-1.5 px-3 py-1.5 rounded-lg bg-slate-800 hover:bg-rose-950/60 hover:text-rose-300 text-slate-300 text-xs transition border border-slate-700 hover:border-rose-500/40"
          >
            <Trash2 className="w-3.5 h-3.5" />
            <span>Clear Session Log</span>
          </button>
        )}
      </div>

      {entries.length === 0 ? (
        <div className="text-center py-12 text-slate-500 text-sm">
          No prediction requests have been executed in this session yet.
        </div>
      ) : (
        <div className="overflow-x-auto mt-4">
          <table className="w-full text-left text-xs">
            <thead className="bg-slate-950/80 text-slate-400 font-mono uppercase tracking-wider text-[11px] border-b border-slate-800">
              <tr>
                <th className="py-3 px-4">Request / Sample ID</th>
                <th className="py-3 px-4">Dataset</th>
                <th className="py-3 px-4">Timestamp</th>
                <th className="py-3 px-4">Laundering Probability</th>
                <th className="py-3 px-4">Decision Threshold</th>
                <th className="py-3 px-4">Authoritative Decision</th>
                <th className="py-3 px-4">Latency</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-800/60 font-sans">
              {entries.map((entry) => {
                const prob = entry.predicted_probability ?? entry.predictedProbability ?? 0;
                const thresh = entry.decision_threshold ?? entry.decisionThreshold ?? 0;
                const latency = entry.inference_time_ms ?? entry.inferenceTimeMs ?? 0;
                const risk = getRiskLevel(prob, thresh);
                return (
                  <tr key={entry.id} className="hover:bg-slate-800/40 transition">
                    <td className="py-3 px-4 font-mono text-slate-300">
                      {entry.sample_id || entry.sampleId || entry.id}
                    </td>
                    <td className="py-3 px-4 font-semibold text-slate-200">
                      {entry.dataset}
                    </td>
                    <td className="py-3 px-4 text-slate-400 font-mono">
                      {formatTimestamp(entry.timestamp)}
                    </td>
                    <td className="py-3 px-4 font-mono font-semibold text-slate-100">
                      {formatProbability(prob)}
                    </td>
                    <td className="py-3 px-4 font-mono text-amber-400">
                      {formatThreshold(thresh)}
                    </td>
                    <td className="py-3 px-4">
                      <span
                        className={`inline-flex items-center px-2 py-0.5 rounded-full text-[11px] font-semibold border ${risk.badgeClass}`}
                      >
                        {risk.label}
                      </span>
                    </td>
                    <td className="py-3 px-4 font-mono text-slate-400">
                      {latency.toFixed(2)} ms
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      )}

      <div className="mt-4 pt-3 border-t border-slate-800/60 flex items-center justify-between text-[11px] text-slate-500">
        <div className="flex items-center space-x-1.5">
          <ShieldAlert className="w-3.5 h-3.5 text-indigo-400" />
          <span>Privacy Guaranteed: Memory-only storage; data flushed upon browser tab close.</span>
        </div>
        <span>FedSecureAI Phase 14B Compliance</span>
      </div>
    </div>
  );
};

