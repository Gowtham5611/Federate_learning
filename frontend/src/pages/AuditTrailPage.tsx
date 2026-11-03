import React from 'react';
import { PredictionHistoryEntry } from '../types/api';
import { formatProbability, formatThreshold } from '../utils/formatters';
import { History, ShieldAlert, CheckCircle2, Download, Trash2, Clock } from 'lucide-react';

interface AuditTrailPageProps {
  history: PredictionHistoryEntry[];
  onClearHistory: () => void;
}

export const AuditTrailPage: React.FC<AuditTrailPageProps> = ({ history, onClearHistory }) => {
  const handleDownloadLog = () => {
    if (!history.length) return;

    const headers = ['ID', 'Timestamp', 'Dataset', 'Model Name', 'Probability', 'Threshold', 'Decision', 'Latency (ms)', 'Sample ID'];
    const rows = history.map((e) => [
      e.id,
      e.timestamp,
      e.dataset,
      e.model_name || e.modelName || 'Champion',
      e.predicted_probability ?? e.predictedProbability ?? 0,
      e.decision_threshold ?? e.decisionThreshold ?? 0,
      e.decision,
      e.inference_time_ms ?? e.inferenceTimeMs ?? 0,
      e.sample_id || e.sampleId || 'N/A',
    ]);

    const csvContent = [headers.join(','), ...rows.map((r) => r.map((c) => `"${c}"`).join(','))].join('\n');
    const blob = new Blob([csvContent], { type: 'text/csv;charset=utf-8;' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `fedsecureai_audit_log_${Date.now()}.csv`;
    a.click();
    URL.revokeObjectURL(url);
  };

  return (
    <div className="space-y-6 animate-fade-in">
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 border-b border-slate-800 pb-5">
        <div>
          <h1 className="text-2xl sm:text-3xl font-black text-white tracking-tight flex items-center space-x-2">
            <History className="w-7 h-7 text-indigo-400" />
            <span>Session Audit Trail & Governance Log</span>
          </h1>
          <p className="text-xs text-slate-400 mt-1">
            Privacy-preserving audit trail of all scoring transactions executed during the active analyst session.
          </p>
        </div>

        <div className="flex items-center space-x-2">
          {history.length > 0 && (
            <>
              <button
                onClick={handleDownloadLog}
                className="flex items-center space-x-1.5 px-3.5 py-2 rounded-xl bg-slate-800 hover:bg-slate-700 text-slate-200 text-xs font-semibold border border-slate-700 transition"
              >
                <Download className="w-3.5 h-3.5 text-indigo-400" />
                <span>Export Audit Log</span>
              </button>
              <button
                onClick={onClearHistory}
                className="flex items-center space-x-1.5 px-3.5 py-2 rounded-xl bg-rose-950/40 hover:bg-rose-900/60 text-rose-300 text-xs font-semibold border border-rose-800/60 transition"
              >
                <Trash2 className="w-3.5 h-3.5" />
                <span>Clear</span>
              </button>
            </>
          )}
        </div>
      </div>

      {history.length === 0 ? (
        <div className="bg-slate-900 border border-slate-800 rounded-3xl p-12 text-center shadow-xl space-y-3">
          <div className="w-12 h-12 rounded-2xl bg-slate-950 border border-slate-800 flex items-center justify-center mx-auto text-slate-500">
            <Clock className="w-6 h-6" />
          </div>
          <h3 className="text-base font-bold text-white">No Scoring Events Logged</h3>
          <p className="text-xs text-slate-400 max-w-sm mx-auto">
            Transactions evaluated via the Single Inference tab or individual inspection will append tamper-evident audit records here.
          </p>
        </div>
      ) : (
        <div className="bg-slate-900 border border-slate-800 rounded-2xl overflow-hidden shadow-xl">
          <div className="overflow-x-auto">
            <table className="w-full text-left border-collapse text-xs">
              <thead>
                <tr className="bg-slate-950 border-b border-slate-800 text-slate-400 font-mono uppercase text-[10px]">
                  <th className="py-3 px-4">Event ID</th>
                  <th className="py-3 px-4">Timestamp</th>
                  <th className="py-3 px-4">Dataset</th>
                  <th className="py-3 px-4">Model</th>
                  <th className="py-3 px-4 text-right">Probability</th>
                  <th className="py-3 px-4 text-right">Threshold</th>
                  <th className="py-3 px-4 text-center">Decision</th>
                  <th className="py-3 px-4 text-right">Latency</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-800/60 font-mono">
                {history.map((e) => {
                  const isFlagged = e.decision === 'FLAGGED_LAUNDERING';
                  return (
                    <tr key={e.id} className="hover:bg-slate-800/40 transition">
                      <td className="py-3 px-4 text-slate-400">{e.id}</td>
                      <td className="py-3 px-4 text-slate-300 font-sans">{new Date(e.timestamp).toLocaleTimeString()}</td>
                      <td className="py-3 px-4">
                        <span
                          className={`px-2 py-0.5 rounded text-[10px] font-bold ${
                            e.dataset === 'HI-Small'
                              ? 'bg-indigo-950 text-indigo-300 border border-indigo-800/60'
                              : 'bg-violet-950 text-violet-300 border border-violet-800/60'
                          }`}
                        >
                          {e.dataset}
                        </span>
                      </td>
                      <td className="py-3 px-4 text-slate-300 truncate max-w-xs">{e.model_name || e.modelName || 'Champion'}</td>
                      <td className="py-3 px-4 text-right font-bold text-white">
                        {formatProbability(e.predicted_probability ?? e.predictedProbability ?? 0)}
                      </td>
                      <td className="py-3 px-4 text-right text-amber-400">{formatThreshold(e.decision_threshold ?? e.decisionThreshold ?? 0)}</td>
                      <td className="py-3 px-4 text-center">
                        <span
                          className={`px-2.5 py-1 rounded-full text-[10px] font-bold inline-flex items-center space-x-1 ${
                            isFlagged
                              ? 'bg-rose-950 text-rose-300 border border-rose-800'
                              : 'bg-emerald-950 text-emerald-300 border border-emerald-800'
                          }`}
                        >
                          {isFlagged ? <ShieldAlert className="w-3 h-3 text-rose-400" /> : <CheckCircle2 className="w-3 h-3 text-emerald-400" />}
                          <span>{isFlagged ? 'FLAGGED' : 'CLEARED'}</span>
                        </span>
                      </td>
                      <td className="py-3 px-4 text-right text-slate-400">{(e.inference_time_ms ?? e.inferenceTimeMs ?? 0).toFixed(1)} ms</td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        </div>
      )}
    </div>
  );
};
