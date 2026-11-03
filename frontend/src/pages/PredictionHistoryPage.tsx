import React, { useState } from 'react';
import { CsvEvaluateResponse, TransactionPredictionItem } from '../types/api';
import {
  History,
  Download,
  Search,
  Eye,
  Upload,
  ShieldAlert,
  ShieldCheck,
  X,
} from 'lucide-react';
import { apiService } from '../services/apiService';
import { formatMetric } from '../utils/formatters';

interface PredictionHistoryPageProps {
  result: CsvEvaluateResponse | null;
  onNavigateToUpload: () => void;
}

export const PredictionHistoryPage: React.FC<PredictionHistoryPageProps> = ({
  result,
  onNavigateToUpload,
}) => {
  const [statusFilter, setStatusFilter] = useState<'ALL' | 'SUSPICIOUS' | 'LEGITIMATE'>('ALL');
  const [searchQuery, setSearchQuery] = useState<string>('');
  const [currentPage, setCurrentPage] = useState<number>(1);
  const pageSize = 20;

  // Explainability inspection modal
  const [inspectingItem, setInspectingItem] = useState<TransactionPredictionItem | null>(null);
  const [explaining, setExplaining] = useState<boolean>(false);
  const [explanationData, setExplanationData] = useState<{
    positiveDrivers: { feature_name: string; attribution_score: number }[];
    negativeDrivers: { feature_name: string; attribution_score: number }[];
  } | null>(null);
  const [explainError, setExplainError] = useState<string | null>(null);

  if (!result || !result.predictions || result.predictions.length === 0) {
    return (
      <div className="bg-slate-900 border border-slate-800 rounded-3xl p-12 text-center space-y-4 max-w-2xl mx-auto my-12 shadow-xl animate-fade-in">
        <div className="w-16 h-16 rounded-2xl bg-indigo-950/80 text-indigo-400 flex items-center justify-center mx-auto border border-indigo-700/50">
          <History className="w-8 h-8" />
        </div>
        <h2 className="text-xl font-bold text-white">No Predictions Available</h2>
        <p className="text-xs text-slate-400 max-w-md mx-auto">
          Please upload and evaluate a banking transaction CSV in the AML Evaluation tab to inspect individual transaction scores, actual labels, and predicted statuses.
        </p>
        <button
          onClick={onNavigateToUpload}
          className="inline-flex items-center space-x-2 px-5 py-2.5 rounded-xl bg-indigo-600 hover:bg-indigo-500 text-white text-xs font-semibold shadow-lg shadow-indigo-600/30 transition"
        >
          <Upload className="w-4 h-4" />
          <span>Go to AML Evaluation</span>
        </button>
      </div>
    );
  }

  // Handle Export CSV
  const handleExportCsv = () => {
    if (result.download_url) {
      window.open(result.download_url, '_blank');
      return;
    }
    if (result.download_id) {
      window.open(apiService.getPredictionsDownloadUrl(result.download_id), '_blank');
      return;
    }

    const headers = [
      'Transaction ID',
      'Row ID',
      'Timestamp',
      'Sender Account',
      'Receiver Account',
      'Amount',
      'Laundering Probability',
      'Predicted Label',
      'Actual Label',
      'Risk Level',
      'Final Status',
    ];

    const rows = result.predictions.map((p) => {
      const predLabel = p.predicted_label || (p.decision === 'FLAGGED_LAUNDERING' || p.decision === 'SUSPICIOUS' ? 'SUSPICIOUS' : 'LEGITIMATE');
      let actLabel = 'UNLABELED';
      if (p.ground_truth_label !== null) {
        actLabel = p.ground_truth_label === 1 ? 'SUSPICIOUS' : 'LEGITIMATE';
      }
      const riskLevel = (p as any).risk_level || (p.predicted_probability >= 0.7 ? 'High Risk' : p.predicted_probability >= 0.3 ? 'Medium Risk' : 'Low Risk');

      return [
        p.transaction_id || `TX_${p.row_id.toString().padStart(5, '0')}`,
        p.row_id,
        p.timestamp,
        p.from_account,
        p.to_account,
        p.amount,
        p.predicted_probability,
        predLabel,
        actLabel,
        riskLevel,
        predLabel,
      ];
    });

    const csvStr = [
      headers.join(','),
      ...rows.map((r) => r.map((c) => `"${String(c).replace(/"/g, '""')}"`).join(',')),
    ].join('\n');

    const blob = new Blob([csvStr], { type: 'text/csv;charset=utf-8;' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `aml_predictions_${Date.now()}.csv`;
    a.click();
    URL.revokeObjectURL(url);
  };

  // Filter items
  const filtered = result.predictions.filter((item) => {
    const isSuspicious =
      item.predicted_label === 'SUSPICIOUS' ||
      item.status === 'SUSPICIOUS' ||
      item.decision === 'SUSPICIOUS' ||
      item.decision === 'FLAGGED_LAUNDERING';

    if (statusFilter === 'SUSPICIOUS' && !isSuspicious) return false;
    if (statusFilter === 'LEGITIMATE' && isSuspicious) return false;

    if (searchQuery.trim()) {
      const q = searchQuery.toLowerCase();
      const match =
        item.from_account.toLowerCase().includes(q) ||
        item.to_account.toLowerCase().includes(q) ||
        item.timestamp.toLowerCase().includes(q) ||
        (item.transaction_id && item.transaction_id.toLowerCase().includes(q)) ||
        String(item.row_id) === q;
      if (!match) return false;
    }

    return true;
  });

  const totalPages = Math.ceil(filtered.length / pageSize) || 1;
  const paginated = filtered.slice((currentPage - 1) * pageSize, currentPage * pageSize);

  // Handle Inspect Feature Attributions
  const handleInspect = async (item: TransactionPredictionItem) => {
    setInspectingItem(item);
    setExplanationData(null);
    setExplainError(null);
    setExplaining(true);

    try {
      const resp = await apiService.explain({
        dataset: result.dataset || 'HI-Small',
        sequence: {
          features: item.sequence_features,
          length: item.sequence_length,
          sample_id: item.transaction_id || `TX_${item.row_id}`,
        },
        steps: 32,
      });

      setExplanationData({
        positiveDrivers: resp.top_positive_drivers.map((d) => ({
          feature_name: d.feature_name,
          attribution_score: d.attribution,
        })),
        negativeDrivers: resp.top_negative_drivers.map((d) => ({
          feature_name: d.feature_name,
          attribution_score: d.attribution,
        })),
      });
    } catch (err: unknown) {
      setExplainError(err instanceof Error ? err.message : 'Explainability generation failed');
    } finally {
      setExplaining(false);
    }
  };

  return (
    <div className="space-y-8 animate-fade-in">
      {/* Header */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 border-b border-slate-800 pb-5">
        <div>
          <h1 className="text-2xl sm:text-3xl font-black text-white tracking-tight flex items-center space-x-2">
            <History className="w-7 h-7 text-indigo-400" />
            <span>Transaction Prediction Results</span>
          </h1>
          <p className="text-xs text-slate-400 mt-1">
            Displaying {result.predictions.length} scored transactions with predicted laundering probability, actual label, and final status.
          </p>
        </div>

        <button
          onClick={handleExportCsv}
          className="flex items-center space-x-1.5 px-4 py-2 rounded-xl bg-indigo-600 hover:bg-indigo-500 text-white text-xs font-semibold shadow-lg shadow-indigo-600/30 transition cursor-pointer"
        >
          <Download className="w-3.5 h-3.5" />
          <span>Download Results CSV</span>
        </button>
      </div>

      {/* Main Table Card */}
      <div className="bg-slate-900 border border-slate-800 rounded-3xl p-6 shadow-xl space-y-4">
        {/* Search & Filter Bar */}
        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3">
          <div className="flex items-center space-x-2 bg-slate-950 p-1 rounded-xl border border-slate-800 text-xs">
            <button
              onClick={() => {
                setStatusFilter('ALL');
                setCurrentPage(1);
              }}
              className={`px-3 py-1.5 rounded-lg transition font-medium cursor-pointer ${
                statusFilter === 'ALL' ? 'bg-indigo-600 text-white' : 'text-slate-400 hover:text-slate-200'
              }`}
            >
              All ({result.predictions.length})
            </button>
            <button
              onClick={() => {
                setStatusFilter('SUSPICIOUS');
                setCurrentPage(1);
              }}
              className={`px-3 py-1.5 rounded-lg transition font-medium cursor-pointer ${
                statusFilter === 'SUSPICIOUS' ? 'bg-rose-600 text-white' : 'text-slate-400 hover:text-slate-200'
              }`}
            >
              Suspicious ({result.flagged_laundering_count})
            </button>
            <button
              onClick={() => {
                setStatusFilter('LEGITIMATE');
                setCurrentPage(1);
              }}
              className={`px-3 py-1.5 rounded-lg transition font-medium cursor-pointer ${
                statusFilter === 'LEGITIMATE' ? 'bg-emerald-600 text-white' : 'text-slate-400 hover:text-slate-200'
              }`}
            >
              Legitimate ({result.cleared_count})
            </button>
          </div>

          <div className="relative w-full sm:w-72">
            <Search className="w-3.5 h-3.5 text-slate-400 absolute left-3 top-3" />
            <input
              type="text"
              value={searchQuery}
              onChange={(e) => {
                setSearchQuery(e.target.value);
                setCurrentPage(1);
              }}
              placeholder="Search by transaction ID, account..."
              className="w-full bg-slate-950 border border-slate-800 rounded-xl pl-9 pr-3 py-2 text-xs text-slate-200 placeholder-slate-500 focus:outline-none"
            />
          </div>
        </div>

        {/* Table */}
        <div className="overflow-x-auto border border-slate-800 rounded-2xl">
          <table className="w-full text-left border-collapse text-xs">
            <thead>
              <tr className="bg-slate-950 border-b border-slate-800 text-slate-400 font-mono uppercase text-[10px]">
                <th className="py-3 px-3">Transaction ID</th>
                <th className="py-3 px-3">Timestamp</th>
                <th className="py-3 px-3">Sender Account</th>
                <th className="py-3 px-3">Receiver Account</th>
                <th className="py-3 px-3 text-right">Amount</th>
                <th className="py-3 px-3">Laundering Probability</th>
                <th className="py-3 px-3 text-center">Actual Label</th>
                <th className="py-3 px-3 text-center">Predicted Label</th>
                <th className="py-3 px-3 text-center">Final Status</th>
                <th className="py-3 px-3 text-center">Model Applied</th>
                <th className="py-3 px-3 text-right">Threshold</th>
                <th className="py-3 px-3 text-center">Explain</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-800/60 font-mono">
              {paginated.map((item) => {
                const txId = item.transaction_id || `TX_${item.row_id.toString().padStart(5, '0')}`;
                const isSuspicious =
                  item.predicted_label === 'SUSPICIOUS' ||
                  item.status === 'SUSPICIOUS' ||
                  item.decision === 'SUSPICIOUS' ||
                  item.decision === 'FLAGGED_LAUNDERING';
                const probPct = formatMetric((item.predicted_probability ?? 0) * 100, 2, '0.00');

                let actualBadge = <span className="text-slate-500">Unlabeled</span>;
                if (item.ground_truth_label !== null) {
                  if (item.ground_truth_label === 1) {
                    actualBadge = (
                      <span className="px-2 py-0.5 rounded text-[10px] font-bold bg-rose-950 text-rose-300 border border-rose-800">
                        SUSPICIOUS
                      </span>
                    );
                  } else {
                    actualBadge = (
                      <span className="px-2 py-0.5 rounded text-[10px] font-semibold bg-slate-950 text-slate-300 border border-slate-800">
                        LEGITIMATE
                      </span>
                    );
                  }
                }

                return (
                  <tr key={item.row_id} className="hover:bg-slate-800/40 transition">
                    <td className="py-2.5 px-3 text-indigo-400 font-bold">{txId}</td>
                    <td className="py-2.5 px-3 text-slate-300 font-sans">{item.timestamp}</td>
                    <td className="py-2.5 px-3 text-slate-300 font-bold truncate max-w-[110px]" title={item.from_account}>
                      {item.from_account}
                    </td>
                    <td className="py-2.5 px-3 text-slate-400 truncate max-w-[110px]" title={item.to_account}>
                      {item.to_account}
                    </td>
                    <td className="py-2.5 px-3 text-right text-slate-200">
                      ${item.amount.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}
                    </td>
                    <td className="py-2.5 px-3">
                      <div className="flex items-center space-x-2">
                        <span className={`w-12 font-bold ${isSuspicious ? 'text-rose-400' : 'text-slate-300'}`}>
                          {probPct}%
                        </span>
                        <div className="w-16 bg-slate-800 rounded-full h-1.5 overflow-hidden">
                          <div
                            className={`h-full rounded-full ${isSuspicious ? 'bg-rose-500' : 'bg-emerald-500'}`}
                            style={{ width: `${Math.min(100, Math.max(2, item.predicted_probability * 100))}%` }}
                          />
                        </div>
                      </div>
                    </td>
                    <td className="py-2.5 px-3 text-center">{actualBadge}</td>
                    <td className="py-2.5 px-3 text-center">
                      <span
                        className={`px-2 py-0.5 rounded text-[10px] font-bold ${
                          isSuspicious
                            ? 'bg-rose-950/80 text-rose-300 border border-rose-800/80'
                            : 'bg-emerald-950/60 text-emerald-300 border border-emerald-800/60'
                        }`}
                      >
                        {isSuspicious ? 'SUSPICIOUS' : 'LEGITIMATE'}
                      </span>
                    </td>
                    <td className="py-2.5 px-3 text-center">
                      <span
                        className={`inline-flex items-center space-x-1 px-2 py-0.5 rounded text-[10px] font-bold ${
                          isSuspicious
                            ? 'bg-rose-900/60 text-rose-200 border border-rose-700'
                            : 'bg-emerald-900/60 text-emerald-200 border border-emerald-700'
                        }`}
                      >
                        {isSuspicious ? <ShieldAlert className="w-3 h-3" /> : <ShieldCheck className="w-3 h-3" />}
                        <span>{isSuspicious ? 'SUSPICIOUS' : 'LEGITIMATE'}</span>
                      </span>
                    </td>
                    <td className="py-2.5 px-3 text-center text-slate-400 text-[10px] truncate max-w-[120px]" title={result.model_name}>
                      {result.model_name.length > 16 ? result.model_name.slice(0, 16) + '…' : result.model_name}
                    </td>
                    <td className="py-2.5 px-3 text-right text-amber-400 font-bold">
                      {(item.decision_threshold || result.decision_threshold).toFixed(4)}
                    </td>
                    <td className="py-2.5 px-3 text-center">
                      <button
                        onClick={() => handleInspect(item)}
                        className="px-2 py-1 rounded-lg bg-indigo-600/20 hover:bg-indigo-600/40 text-indigo-300 border border-indigo-500/40 text-[10px] font-semibold transition inline-flex items-center space-x-1 cursor-pointer"
                      >
                        <Eye className="w-3 h-3" />
                        <span>Inspect</span>
                      </button>
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>

        {/* Pagination */}
        {totalPages > 1 && (
          <div className="flex items-center justify-between text-xs text-slate-400 font-mono pt-2">
            <span>
              Page {currentPage} of {totalPages} ({filtered.length} total matched)
            </span>
            <div className="flex items-center space-x-1">
              <button
                onClick={() => setCurrentPage((p) => Math.max(1, p - 1))}
                disabled={currentPage === 1}
                className="px-2.5 py-1 rounded bg-slate-800 hover:bg-slate-700 disabled:opacity-40 cursor-pointer"
              >
                Previous
              </button>
              <button
                onClick={() => setCurrentPage((p) => Math.min(totalPages, p + 1))}
                disabled={currentPage === totalPages}
                className="px-2.5 py-1 rounded bg-slate-800 hover:bg-slate-700 disabled:opacity-40 cursor-pointer"
              >
                Next
              </button>
            </div>
          </div>
        )}
      </div>

      {/* Feature Attribution Inspection Modal */}
      {inspectingItem && (
        <div className="fixed inset-0 bg-slate-950/80 backdrop-blur-sm z-50 flex items-center justify-center p-4">
          <div className="bg-slate-900 border border-slate-800 rounded-3xl max-w-lg w-full p-6 space-y-4 shadow-2xl relative">
            <button
              onClick={() => setInspectingItem(null)}
              className="absolute right-4 top-4 text-slate-400 hover:text-white p-1 rounded-lg hover:bg-slate-800 transition"
            >
              <X className="w-5 h-5" />
            </button>

            <div>
              <h3 className="text-base font-bold text-white flex items-center space-x-2">
                <Eye className="w-5 h-5 text-indigo-400" />
                <span>Feature Attribution: {inspectingItem.transaction_id || `TX_${inspectingItem.row_id}`}</span>
              </h3>
              <p className="text-xs text-slate-400 mt-0.5">
                Integrated Gradients attribution showing top drivers influencing the prediction score.
              </p>
            </div>

            {explaining && (
              <div className="py-8 text-center space-y-2">
                <div className="w-8 h-8 border-2 border-indigo-500 border-t-transparent rounded-full animate-spin mx-auto"></div>
                <div className="text-xs text-slate-400 font-mono">Computing axiomatic feature attributions...</div>
              </div>
            )}

            {explainError && (
              <div className="p-3 bg-rose-950/60 border border-rose-800/80 rounded-xl text-xs text-rose-300">
                {explainError}
              </div>
            )}

            {explanationData && (
              <div className="space-y-4 pt-1">
                {/* Positive Drivers */}
                <div className="space-y-2">
                  <div className="text-xs font-mono font-bold text-rose-400 uppercase">
                    Risk-Increasing Features (Pushed toward Suspicious)
                  </div>
                  <div className="space-y-1.5">
                    {explanationData.positiveDrivers.slice(0, 4).map((d, i) => (
                      <div
                        key={i}
                        className="flex items-center justify-between p-2 rounded-xl bg-slate-950 border border-slate-800 text-xs font-mono"
                      >
                        <span className="text-slate-300">{d.feature_name}</span>
                        <span className="font-bold text-rose-400">+{formatMetric(d.attribution_score, 4)}</span>
                      </div>
                    ))}
                  </div>
                </div>

                {/* Negative Drivers */}
                <div className="space-y-2">
                  <div className="text-xs font-mono font-bold text-emerald-400 uppercase">
                    Risk-Decreasing Features (Pushed toward Legitimate)
                  </div>
                  <div className="space-y-1.5">
                    {explanationData.negativeDrivers.slice(0, 4).map((d, i) => (
                      <div
                        key={i}
                        className="flex items-center justify-between p-2 rounded-xl bg-slate-950 border border-slate-800 text-xs font-mono"
                      >
                        <span className="text-slate-300">{d.feature_name}</span>
                        <span className="font-bold text-emerald-400">{formatMetric(d.attribution_score, 4)}</span>
                      </div>
                    ))}
                  </div>
                </div>
              </div>
            )}

            <div className="pt-2 flex justify-end">
              <button
                onClick={() => setInspectingItem(null)}
                className="px-4 py-2 rounded-xl bg-slate-800 hover:bg-slate-700 text-slate-200 text-xs font-semibold"
              >
                Close
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};
