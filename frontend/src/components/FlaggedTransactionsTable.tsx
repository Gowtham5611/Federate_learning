import React, { useState, useMemo } from 'react';
import { TransactionPredictionItem, SupportedDataset } from '../types/api';
import { getFastMonth } from './TransactionRiskHeatmap';
import {
  Search,
  ArrowUpDown,
  ShieldAlert,
  ShieldCheck,
  AlertTriangle,
  Info,
  ChevronLeft,
  ChevronRight,
  Eye,
  X,
  Filter,
} from 'lucide-react';

export type TableFilterMode = 'all' | 'flagged' | 'cleared' | 'needs_review_fn' | 'laundering' | 'legitimate';

interface HeatmapFilterCriteria {
  timeBucket: string;
  riskBand: string;
  minScore: number;
  maxScore: number;
}

interface FlaggedTransactionsTableProps {
  predictions: TransactionPredictionItem[];
  activeThreshold: number;
  activeModeName: string;
  hasLabels: boolean;
  dataset?: SupportedDataset;
  modelName?: string;
  heatmapFilter?: HeatmapFilterCriteria | null;
  onClearHeatmapFilter?: () => void;
  onExplainTransaction?: (prediction: TransactionPredictionItem) => void;
  onFilterChange?: (filterMode: TableFilterMode) => void;
  className?: string;
}

export const FlaggedTransactionsTable: React.FC<FlaggedTransactionsTableProps> = ({
  predictions,
  activeThreshold,
  activeModeName,
  hasLabels,
  dataset: _dataset,
  modelName = 'Federated Neural Classifier',
  heatmapFilter,
  onClearHeatmapFilter,
  onExplainTransaction: _onExplainTransaction,
  onFilterChange,
  className = '',
}) => {
  const [searchQuery, setSearchQuery] = useState('');
  const [filterMode, setFilterMode] = useState<TableFilterMode>('flagged');

  const handleSetFilterMode = (mode: TableFilterMode) => {
    setFilterMode(mode);
    setCurrentPage(1);
    onFilterChange?.(mode);
  };
  const [sortField, setSortField] = useState<'predicted_probability' | 'amount' | 'row_id' | 'timestamp'>('predicted_probability');
  const [sortDirection, setSortDirection] = useState<'asc' | 'desc'>('desc');
  const [currentPage, setCurrentPage] = useState<number>(1);
  const [inspectItem, setInspectItem] = useState<TransactionPredictionItem | null>(null);
  const pageSize = 15;

  // Calculate counts under active threshold
  const { flaggedCount, clearedCount, fnCount } = useMemo(() => {
    let flagged = 0;
    let cleared = 0;
    let fn = 0;
    predictions.forEach((item) => {
      const prob = Number(item.predicted_probability ?? 0);
      const isFlagged = prob >= activeThreshold;
      if (isFlagged) flagged++;
      else cleared++;
      if (hasLabels) {
        const isLaunderingLabel =
          item.ground_truth_label === 1 ||
          item.actual_label === 'SUSPICIOUS' ||
          (item as any)['Is Laundering'] === 1 ||
          (item as any)['is_laundering'] === 1;
        if (isLaunderingLabel && !isFlagged) fn++;
      }
    });
    return { flaggedCount: flagged, clearedCount: cleared, fnCount: fn };
  }, [predictions, activeThreshold, hasLabels]);

  // Filtered predictions
  const filteredRows = useMemo(() => {
    return predictions.filter((item) => {
      const prob = Number(item.predicted_probability ?? 0);
      const isFlagged = prob >= activeThreshold;
      const isLaunderingLabel =
        item.ground_truth_label === 1 ||
        item.actual_label === 'SUSPICIOUS' ||
        (item as any)['Is Laundering'] === 1 ||
        (item as any)['is_laundering'] === 1;
      const isLegitLabel =
        item.ground_truth_label === 0 ||
        item.actual_label === 'LEGITIMATE' ||
        (item as any)['Is Laundering'] === 0 ||
        (item as any)['is_laundering'] === 0;

      // Mode filter
      if (filterMode === 'flagged' && !isFlagged) return false;
      if (filterMode === 'cleared' && isFlagged) return false;
      if (filterMode === 'needs_review_fn' && !(isLaunderingLabel && !isFlagged)) return false;
      if (filterMode === 'laundering' && !isLaunderingLabel) return false;
      if (filterMode === 'legitimate' && !isLegitLabel) return false;

      // Heatmap 2D cell filter if active (ultrafast O(1) month extraction)
      if (heatmapFilter) {
        if (prob < heatmapFilter.minScore || prob >= heatmapFilter.maxScore) return false;
        const monthKey = getFastMonth(item.timestamp);
        if (monthKey && monthKey !== heatmapFilter.timeBucket) return false;
      }

      // Text search: matches row reference (TX_00435 or 435), accounts, name, amount
      if (searchQuery.trim()) {
        const q = searchQuery.toLowerCase().trim();
        const txId = String(item.transaction_id || `TX_${String(item.row_id).padStart(5, '0')}`).toLowerCase();
        const rowIdStr = String(item.row_id || '');
        const fromAcc = String(item.from_account || '').toLowerCase();
        const toAcc = String(item.to_account || '').toLowerCase();
        const senderName = String(item.sender_name || item.transaction_name || '').toLowerCase();
        const amountStr = String(item.amount || '');

        const matches =
          txId.includes(q) ||
          rowIdStr === q ||
          fromAcc.includes(q) ||
          toAcc.includes(q) ||
          senderName.includes(q) ||
          amountStr.includes(q);

        if (!matches) return false;
      }

      return true;
    });
  }, [predictions, activeThreshold, filterMode, heatmapFilter, searchQuery]);

  // High-performance sort with zero Date allocations
  const sortedRows = useMemo(() => {
    if (filteredRows.length <= 1) return filteredRows;
    const isDesc = sortDirection === 'desc';
    const field = sortField;

    return [...filteredRows].sort((a, b) => {
      if (field === 'timestamp') {
        const strA = String(a.timestamp || '');
        const strB = String(b.timestamp || '');
        return isDesc ? (strA < strB ? 1 : strA > strB ? -1 : 0) : (strA < strB ? -1 : strA > strB ? 1 : 0);
      }
      const valA = Number(a[field]) || 0;
      const valB = Number(b[field]) || 0;
      return isDesc ? valB - valA : valA - valB;
    });
  }, [filteredRows, sortField, sortDirection]);

  // Pagination
  const totalPages = Math.max(1, Math.ceil(sortedRows.length / pageSize));
  const paginatedRows = useMemo(() => {
    const start = (currentPage - 1) * pageSize;
    return sortedRows.slice(start, start + pageSize);
  }, [sortedRows, currentPage, pageSize]);

  const handleSort = (field: 'predicted_probability' | 'amount' | 'row_id' | 'timestamp') => {
    if (sortField === field) {
      setSortDirection((prev) => (prev === 'asc' ? 'desc' : 'asc'));
    } else {
      setSortField(field);
      setSortDirection('desc');
    }
    setCurrentPage(1);
  };

  return (
    <div className={`bg-slate-900 border border-slate-800 rounded-3xl p-6 shadow-xl space-y-4 ${className}`}>
      {/* Title & Controls Bar */}
      <div className="flex flex-col md:flex-row md:items-center justify-between gap-4 border-b border-slate-800 pb-4">
        <div>
          <div className="flex items-center space-x-2">
            <ShieldAlert className="w-5 h-5 text-rose-500 dark:text-rose-400" />
            <h3 className="text-base font-bold text-slate-900 dark:text-white tracking-tight">
              Transaction Evaluation Records & Triage
            </h3>
            <span className="px-2 py-0.5 rounded-full text-[10px] font-mono font-bold bg-slate-100 dark:bg-slate-800 text-slate-700 dark:text-slate-300 border border-slate-300 dark:border-slate-700">
              {filteredRows.length.toLocaleString()} Matches
            </span>
          </div>
          <p className="text-xs text-slate-600 dark:text-slate-400 mt-1 max-w-2xl leading-relaxed">
            Transaction details showing model risk predictions, generated row references, account routing, and ground-truth CSV labels under{' '}
            <strong className="text-amber-700 dark:text-amber-300 font-mono">{activeModeName}</strong> (cutoff <code className="text-amber-700 dark:text-amber-300 font-bold">θ = {activeThreshold.toFixed(4)}</code>).
          </p>
        </div>

        {/* Filter Pills */}
        <div className="flex flex-wrap items-center gap-1.5 bg-slate-100 dark:bg-slate-950 p-1 rounded-xl border border-slate-300 dark:border-slate-800 text-xs">
          <button
            onClick={() => handleSetFilterMode('flagged')}
            className={`px-3 py-1.5 rounded-lg transition font-medium cursor-pointer flex items-center space-x-1.5 ${
              filterMode === 'flagged' ? 'bg-rose-600 text-white font-bold shadow-sm' : 'text-slate-600 dark:text-slate-400 hover:text-slate-900 dark:hover:text-white'
            }`}
          >
            <span>Flagged (Model Alerts)</span>
            <span className={`px-1.5 py-0.2 rounded-full text-[10px] font-mono ${
              filterMode === 'flagged' ? 'bg-rose-800 text-white' : 'bg-slate-200 dark:bg-slate-800 text-slate-800 dark:text-slate-300'
            }`}>
              {flaggedCount.toLocaleString()}
            </span>
          </button>

          {hasLabels && (
            <button
              onClick={() => handleSetFilterMode('needs_review_fn')}
              className={`px-3 py-1.5 rounded-lg transition font-medium cursor-pointer flex items-center space-x-1.5 ${
                filterMode === 'needs_review_fn' ? 'bg-amber-600 text-white font-bold shadow-sm' : 'text-amber-700 dark:text-amber-500 hover:text-amber-800 dark:hover:text-amber-400 font-semibold'
              }`}
            >
              <span>Needs Review (False Negatives)</span>
              <span className={`px-1.5 py-0.2 rounded-full text-[10px] font-mono ${
                filterMode === 'needs_review_fn' ? 'bg-amber-800 text-white' : 'bg-slate-200 dark:bg-slate-800 text-slate-800 dark:text-slate-300'
              }`}>
                {fnCount.toLocaleString()}
              </span>
            </button>
          )}

          <button
            onClick={() => handleSetFilterMode('cleared')}
            className={`px-3 py-1.5 rounded-lg transition font-medium cursor-pointer flex items-center space-x-1.5 ${
              filterMode === 'cleared' ? 'bg-emerald-600 text-white font-bold shadow-sm' : 'text-slate-600 dark:text-slate-400 hover:text-slate-900 dark:hover:text-white'
            }`}
          >
            <span>Legitimate Cleared</span>
            <span className={`px-1.5 py-0.2 rounded-full text-[10px] font-mono ${
              filterMode === 'cleared' ? 'bg-emerald-800 text-white' : 'bg-slate-200 dark:bg-slate-800 text-slate-800 dark:text-slate-300'
            }`}>
              {clearedCount.toLocaleString()}
            </span>
          </button>

          <button
            onClick={() => handleSetFilterMode('all')}
            className={`px-3 py-1.5 rounded-lg transition font-medium cursor-pointer flex items-center space-x-1.5 ${
              filterMode === 'all' ? 'bg-indigo-600 text-white font-bold shadow-sm' : 'text-slate-600 dark:text-slate-400 hover:text-slate-900 dark:hover:text-white'
            }`}
          >
            <span>All Transactions</span>
            <span className={`px-1.5 py-0.2 rounded-full text-[10px] font-mono ${
              filterMode === 'all' ? 'bg-indigo-800 text-white' : 'bg-slate-200 dark:bg-slate-800 text-slate-800 dark:text-slate-300'
            }`}>
              {predictions.length.toLocaleString()}
            </span>
          </button>
        </div>
      </div>

      {/* Active Heatmap Filter Indicator if selected */}
      {heatmapFilter && (
        <div className="flex items-center justify-between p-2.5 rounded-xl bg-indigo-950/80 border border-indigo-700/60 text-xs font-mono text-indigo-200">
          <div className="flex items-center space-x-2">
            <Filter className="w-3.5 h-3.5 text-indigo-400" />
            <span>
              <strong>Heatmap Filter Active:</strong> Interval: <u>{heatmapFilter.timeBucket}</u> • Score Band: <u>{heatmapFilter.riskBand}</u> ({filteredRows.length} transactions match)
            </span>
          </div>
          {onClearHeatmapFilter && (
            <button
              onClick={onClearHeatmapFilter}
              className="flex items-center space-x-1 px-2 py-0.5 rounded bg-indigo-900 hover:bg-indigo-800 text-white transition cursor-pointer"
            >
              <span>Clear Filter</span>
              <X className="w-3 h-3" />
            </button>
          )}
        </div>
      )}

      {/* Needs Review — False Negatives Informative Banner */}
      {filterMode === 'needs_review_fn' && (
        <div className="p-4 rounded-2xl bg-amber-950/40 border border-amber-600/50 text-amber-200 text-xs space-y-2.5">
          <div className="flex items-center space-x-2 text-amber-300 font-bold">
            <AlertTriangle className="w-4 h-4 text-amber-400 shrink-0" />
            <span>Needs Review — False Negative Triage Queue</span>
            <span className="px-2 py-0.5 rounded-full text-[10px] bg-amber-900/80 text-amber-200 font-mono border border-amber-700/60">
              {filteredRows.length.toLocaleString()} Under-Review Transactions
            </span>
          </div>
          <div className="grid grid-cols-2 sm:grid-cols-4 gap-2.5 text-[11px] pt-1 border-t border-amber-800/40 font-mono">
            <div className="p-2 rounded-xl bg-slate-950/70 border border-amber-900/40">
              <span className="text-amber-400/70 block uppercase text-[9.5px]">Model Prediction</span>
              <strong className="text-emerald-400">Cleared (Score &lt; θ*)</strong>
            </div>
            <div className="p-2 rounded-xl bg-slate-950/70 border border-amber-900/40">
              <span className="text-amber-400/70 block uppercase text-[9.5px]">CSV Ground Truth</span>
              <strong className="text-purple-300">Laundering (1)</strong>
            </div>
            <div className="p-2 rounded-xl bg-slate-950/70 border border-amber-900/40">
              <span className="text-amber-400/70 block uppercase text-[9.5px]">Evaluation Outcome</span>
              <strong className="text-amber-300">False negative at active cutoff (θ = {activeThreshold.toFixed(4)})</strong>
            </div>
            <div className="p-2 rounded-xl bg-slate-950/70 border border-amber-900/40">
              <span className="text-amber-400/70 block uppercase text-[9.5px]">Suggested Handling</span>
              <strong className="text-white">Needs Analyst Review</strong>
            </div>
          </div>
          <p className="text-[11px] text-amber-300/80 leading-relaxed font-sans pt-1">
            <strong>Important Notice:</strong> These transactions are cleared by the model under the active serving threshold and are <strong>not</strong> counted as model alerts. Later operational data or expanded sequence history could change a future re-score, but under-review transactions are not promised or guaranteed to become flagged.
          </p>
        </div>
      )}

      {/* Search & Audit Note */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3">
        <div className="relative flex-1 max-w-md">
          <Search className="w-4 h-4 text-slate-400 absolute left-3.5 top-3" />
          <input
            type="text"
            value={searchQuery}
            onChange={(e) => {
              setSearchQuery(e.target.value);
              setCurrentPage(1);
            }}
            placeholder="Search by row ref (TX_00435 or 435), account, amount..."
            className="w-full pl-10 pr-4 py-2 bg-slate-950 border border-slate-800 rounded-xl text-xs text-white placeholder-slate-500 focus:outline-none focus:border-indigo-500 transition font-mono"
          />
        </div>

        <div className="flex items-center space-x-2 text-[11px] text-slate-400 font-sans">
          <Info className="w-3.5 h-3.5 text-indigo-400 shrink-0" />
          <span>
            Transaction IDs are generated from source row positions. Synthetic CSV labels are ground-truth testing markers, not confirmed real-world findings.
          </span>
        </div>
      </div>

      {/* Main Table */}
      <div className="overflow-x-auto rounded-2xl border border-slate-800">
        <table className="w-full text-left border-collapse text-xs font-mono">
          <thead>
            <tr className="bg-slate-950 border-b border-slate-800 text-slate-400 uppercase text-[10px]">
              <th
                onClick={() => handleSort('row_id')}
                className="py-3 px-3 cursor-pointer hover:text-white select-none min-w-[110px]"
              >
                <div className="flex items-center space-x-1">
                  <span>Row Ref (Generated)</span>
                  <ArrowUpDown className="w-3 h-3" />
                </div>
              </th>
              <th className="py-3 px-3 min-w-[120px]">Account / Entity</th>
              <th className="py-3 px-3 min-w-[140px]">Routing (From → To)</th>
              <th
                onClick={() => handleSort('timestamp')}
                className="py-3 px-3 cursor-pointer hover:text-white select-none min-w-[110px]"
              >
                <div className="flex items-center space-x-1">
                  <span>Timestamp</span>
                  <ArrowUpDown className="w-3 h-3" />
                </div>
              </th>
              <th
                onClick={() => handleSort('amount')}
                className="py-3 px-3 text-right cursor-pointer hover:text-white select-none min-w-[100px]"
              >
                <div className="flex items-center justify-end space-x-1">
                  <span>Amount</span>
                  <ArrowUpDown className="w-3 h-3" />
                </div>
              </th>
              <th
                onClick={() => handleSort('predicted_probability')}
                className="py-3 px-3 text-right cursor-pointer hover:text-white select-none min-w-[95px]"
              >
                <div className="flex items-center justify-end space-x-1">
                  <span>Risk Score</span>
                  <ArrowUpDown className="w-3 h-3" />
                </div>
              </th>
              <th className="py-3 px-3 text-center min-w-[125px]">Model Verdict</th>
              <th className="py-3 px-3 text-center min-w-[135px]">CSV Ground Truth</th>
              <th className="py-3 px-3 min-w-[170px]">Sequence Context</th>
              <th className="py-3 px-2 text-center min-w-[70px]">Inspect</th>
            </tr>
          </thead>
          <tbody className="divide-y divide-slate-800/60">
            {paginatedRows.length === 0 ? (
              <tr>
                <td colSpan={10} className="py-8 text-center text-slate-500 text-xs">
                  No transaction records match the current filters.
                </td>
              </tr>
            ) : (
              paginatedRows.map((item) => {
                const prob = Number(item.predicted_probability ?? 0);
                const isFlagged = prob >= activeThreshold;
                const origRowIdx = Number(item.row_id || 1);
                const rowRefStr = item.transaction_id || `TX_${String(origRowIdx).padStart(5, '0')}`;
                const nameDisplay = item.sender_name || item.transaction_name || item.customer_name || 'Name not provided';

                const isLaunderingLabel =
                  item.ground_truth_label === 1 ||
                  item.actual_label === 'SUSPICIOUS' ||
                  (item as any)['Is Laundering'] === 1 ||
                  (item as any)['is_laundering'] === 1;
                const isLegitLabel =
                  item.ground_truth_label === 0 ||
                  item.actual_label === 'LEGITIMATE' ||
                  (item as any)['Is Laundering'] === 0 ||
                  (item as any)['is_laundering'] === 0;

                const currStr = (item as any).currency || (item as any)['Payment Currency'] || '';
                const amtNum = Number(item.amount || (item as any)['Amount Paid'] || 0);

                // Sequence / context explanation
                const note = item.usable_history_note;
                const isUsable = item.is_history_sufficient !== false;

                let explBadge = (
                  <span className="text-slate-400">
                    Temporal history evaluated.
                  </span>
                );

                if (!isUsable || note) {
                  explBadge = (
                    <span className="text-amber-400 flex items-center space-x-1">
                      <AlertTriangle className="w-3 h-3 shrink-0" />
                      <span className="truncate" title={note || 'Single-transaction cold start'}>
                        Cold-start: velocity = 0.0
                      </span>
                    </span>
                  );
                } else if (isFlagged) {
                  explBadge = (
                    <span className="text-rose-300 font-sans leading-tight block">
                      Score ({prob.toFixed(4)}) ≥ θ* ({activeThreshold.toFixed(4)})
                    </span>
                  );
                } else if (hasLabels && isLaunderingLabel) {
                  explBadge = (
                    <span className="text-amber-400 font-sans leading-tight block">
                      <strong className="block text-amber-300">Needs Review (FN)</strong>
                      Score ({prob.toFixed(4)}) &lt; θ* ({activeThreshold.toFixed(4)})
                    </span>
                  );
                }

                return (
                  <tr key={item.row_id} className="hover:bg-slate-800/40 transition">
                    <td className="py-2.5 px-3 font-bold text-white whitespace-nowrap">
                      <div>
                        <span>{rowRefStr}</span>
                        <span className="text-[9.5px] text-slate-500 block font-normal">
                          Row {origRowIdx} (generated)
                        </span>
                      </div>
                    </td>
                    <td className="py-2.5 px-3">
                      <span
                        className={`text-xs ${
                          nameDisplay === 'Name not provided' ? 'text-slate-500 italic' : 'text-slate-200 font-medium'
                        }`}
                      >
                        {nameDisplay}
                      </span>
                    </td>
                    <td className="py-2.5 px-3 text-slate-300">
                      <div className="truncate max-w-[150px]">
                        <span className="text-slate-400">{item.from_account}</span>
                        <span className="text-slate-500 mx-1">→</span>
                        <span className="text-slate-300">{item.to_account}</span>
                      </div>
                    </td>
                    <td className="py-2.5 px-3 text-slate-400 whitespace-nowrap">
                      {String(item.timestamp || '').slice(0, 16)}
                    </td>
                    <td className="py-2.5 px-3 text-right font-bold text-white whitespace-nowrap">
                      {amtNum.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}{' '}
                      <span className="text-[10px] text-slate-400 font-normal">{currStr}</span>
                    </td>
                    <td className="py-2.5 px-3 text-right whitespace-nowrap">
                      <span className={`font-bold ${isFlagged ? 'text-rose-400' : 'text-emerald-400'}`}>
                        {prob.toFixed(4)}
                      </span>
                    </td>
                    <td className="py-2.5 px-3 text-center whitespace-nowrap">
                      {isFlagged ? (
                        <span className="inline-flex items-center space-x-1 px-2 py-0.5 rounded-full text-[10px] font-bold bg-rose-950/80 text-rose-300 border border-rose-700/60">
                          <ShieldAlert className="w-3 h-3 text-rose-400" />
                          <span>FLAGGED ALERT</span>
                        </span>
                      ) : (
                        <span className="inline-flex items-center space-x-1 px-2 py-0.5 rounded-full text-[10px] font-bold bg-emerald-950/80 text-emerald-300 border border-emerald-700/60">
                          <ShieldCheck className="w-3 h-3 text-emerald-400" />
                          <span>CLEARED</span>
                        </span>
                      )}
                    </td>
                    <td className="py-2.5 px-3 text-center whitespace-nowrap">
                      {hasLabels ? (
                        isLaunderingLabel ? (
                          <span className="px-2 py-0.5 rounded text-[10px] font-bold bg-purple-950 text-purple-300 border border-purple-800">
                            CSV Laundering (1)
                          </span>
                        ) : isLegitLabel ? (
                          <span className="px-2 py-0.5 rounded text-[10px] font-bold bg-slate-800 text-slate-300 border border-slate-700">
                            CSV Legitimate (0)
                          </span>
                        ) : (
                          <span className="text-slate-500 text-[10px]">Unknown</span>
                        )
                      ) : (
                        <span className="text-slate-500 text-[10px] italic">Unlabeled</span>
                      )}
                    </td>
                    <td className="py-2.5 px-3 text-[11px] max-w-[180px] leading-relaxed">
                      {explBadge}
                    </td>
                    <td className="py-2.5 px-2 text-center">
                      <button
                        onClick={() => setInspectItem(item)}
                        className="px-2 py-1 rounded-lg bg-indigo-950 hover:bg-indigo-900 border border-indigo-700/50 text-indigo-300 text-[10px] font-semibold transition cursor-pointer flex items-center space-x-1 mx-auto"
                        title="Inspect full transaction details"
                      >
                        <Eye className="w-3 h-3" />
                        <span>Inspect</span>
                      </button>
                    </td>
                  </tr>
                );
              })
            )}
          </tbody>
        </table>
      </div>

      {/* Pagination Bar */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 pt-2 text-xs font-mono text-slate-400">
        <div>
          Showing page <strong className="text-white">{currentPage}</strong> of{' '}
          <strong className="text-white">{totalPages}</strong> ({sortedRows.length.toLocaleString()} total filtered records)
        </div>
        <div className="flex items-center space-x-2">
          <button
            onClick={() => setCurrentPage((p) => Math.max(1, p - 1))}
            disabled={currentPage <= 1}
            className="p-1.5 rounded-lg bg-slate-950 border border-slate-800 text-slate-300 disabled:opacity-40 disabled:cursor-not-allowed hover:bg-slate-800 transition cursor-pointer"
          >
            <ChevronLeft className="w-4 h-4" />
          </button>
          <span>
            {currentPage} / {totalPages}
          </span>
          <button
            onClick={() => setCurrentPage((p) => Math.min(totalPages, p + 1))}
            disabled={currentPage >= totalPages}
            className="p-1.5 rounded-lg bg-slate-950 border border-slate-800 text-slate-300 disabled:opacity-40 disabled:cursor-not-allowed hover:bg-slate-800 transition cursor-pointer"
          >
            <ChevronRight className="w-4 h-4" />
          </button>
        </div>
      </div>

      {/* Inspect Transaction Modal */}
      {inspectItem && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/70 backdrop-blur-sm p-4 animate-fade-in">
          <div className="bg-slate-900 border border-indigo-500/50 rounded-3xl p-6 max-w-xl w-full shadow-2xl space-y-4 font-mono text-xs">
            <div className="flex items-center justify-between border-b border-slate-800 pb-3">
              <div className="flex items-center space-x-2">
                <ShieldAlert className="w-4 h-4 text-indigo-400" />
                <h4 className="text-sm font-bold text-white">
                  Transaction Audit Inspector: #{inspectItem.row_id}
                </h4>
              </div>
              <button
                onClick={() => setInspectItem(null)}
                className="p-1 rounded-lg text-slate-400 hover:text-white hover:bg-slate-800 transition"
              >
                <X className="w-4 h-4" />
              </button>
            </div>

            <div className="grid grid-cols-2 gap-3 text-xs">
              <div className="p-2.5 rounded-xl bg-slate-950 border border-slate-800">
                <span className="text-[10px] text-slate-500 block uppercase">Row Reference</span>
                <span className="font-bold text-white">
                  {inspectItem.transaction_id || `TX_${String(inspectItem.row_id).padStart(5, '0')}`}
                </span>
                <span className="text-[9px] text-slate-500 block font-normal">Generated from row position {inspectItem.row_id}</span>
              </div>

              <div className="p-2.5 rounded-xl bg-slate-950 border border-slate-800">
                <span className="text-[10px] text-slate-500 block uppercase">Timestamp</span>
                <span className="font-bold text-white">{String(inspectItem.timestamp || 'N/A')}</span>
              </div>

              <div className="p-2.5 rounded-xl bg-slate-950 border border-slate-800">
                <span className="text-[10px] text-slate-500 block uppercase">Sender Account</span>
                <span className="font-bold text-slate-300">{inspectItem.from_account || 'N/A'}</span>
              </div>

              <div className="p-2.5 rounded-xl bg-slate-950 border border-slate-800">
                <span className="text-[10px] text-slate-500 block uppercase">Receiver Account</span>
                <span className="font-bold text-slate-300">{inspectItem.to_account || 'N/A'}</span>
              </div>

              <div className="p-2.5 rounded-xl bg-slate-950 border border-slate-800">
                <span className="text-[10px] text-slate-500 block uppercase">Amount & Currency</span>
                <span className="font-bold text-white">
                  {Number(inspectItem.amount || (inspectItem as any)['Amount Paid'] || 0).toLocaleString(undefined, { minimumFractionDigits: 2 })}{' '}
                  {(inspectItem as any).currency || (inspectItem as any)['Payment Currency'] || ''}
                </span>
              </div>

              <div className="p-2.5 rounded-xl bg-slate-950 border border-slate-800">
                <span className="text-[10px] text-slate-500 block uppercase">Risk Probability</span>
                <span className="font-bold text-teal-300">
                  {Number(inspectItem.predicted_probability ?? 0).toFixed(6)}
                </span>
              </div>

              <div className="p-2.5 rounded-xl bg-slate-950 border border-slate-800">
                <span className="text-[10px] text-slate-500 block uppercase">Model Verdict</span>
                <span className={`font-bold ${Number(inspectItem.predicted_probability ?? 0) >= activeThreshold ? 'text-rose-400' : 'text-emerald-400'}`}>
                  {Number(inspectItem.predicted_probability ?? 0) >= activeThreshold ? 'FLAGGED (Alert)' : 'CLEARED'}
                </span>
                <span className="text-[9px] text-slate-500 block font-normal">Cutoff threshold θ* = {activeThreshold.toFixed(4)}</span>
              </div>

              <div className="p-2.5 rounded-xl bg-slate-950 border border-slate-800">
                <span className="text-[10px] text-slate-500 block uppercase">CSV Ground Truth</span>
                <span className="font-bold text-purple-300">
                  {hasLabels
                    ? inspectItem.ground_truth_label === 1 || (inspectItem as any)['Is Laundering'] === 1
                      ? 'CSV Laundering Label (1)'
                      : 'CSV Legitimate Label (0)'
                    : 'Unlabeled Testing Data'}
                </span>
                <span className="text-[9px] text-slate-500 block font-normal">Testing label from uploaded CSV</span>
              </div>

              <div className="p-2.5 rounded-xl bg-slate-950 border border-slate-800 col-span-2">
                <span className="text-[10px] text-slate-500 block uppercase">Evaluation Outcome</span>
                <span className={`font-bold ${
                  hasLabels && Number(inspectItem.predicted_probability ?? 0) < activeThreshold && (inspectItem.ground_truth_label === 1 || (inspectItem as any)['Is Laundering'] === 1)
                    ? 'text-amber-400'
                    : Number(inspectItem.predicted_probability ?? 0) >= activeThreshold
                    ? 'text-rose-400'
                    : 'text-emerald-400'
                }`}>
                  {hasLabels
                    ? (inspectItem.ground_truth_label === 1 || (inspectItem as any)['Is Laundering'] === 1)
                      ? Number(inspectItem.predicted_probability ?? 0) >= activeThreshold
                        ? 'TRUE POSITIVE'
                        : 'FALSE NEGATIVE (Needs Analyst Review)'
                      : Number(inspectItem.predicted_probability ?? 0) >= activeThreshold
                      ? 'FALSE POSITIVE'
                      : 'TRUE NEGATIVE'
                    : Number(inspectItem.predicted_probability ?? 0) >= activeThreshold
                    ? 'FLAGGED (Alert)'
                    : 'CLEARED'}
                </span>
                <span className="text-[9px] text-slate-500 block font-normal">
                  {hasLabels
                    ? Number(inspectItem.predicted_probability ?? 0) < activeThreshold && (inspectItem.ground_truth_label === 1 || (inspectItem as any)['Is Laundering'] === 1)
                      ? 'Ground truth = Laundering, Model Score < θ*'
                      : 'Ground truth vs. active threshold'
                    : 'Unlabeled dataset'}
                </span>
              </div>
            </div>

            {hasLabels && Number(inspectItem.predicted_probability ?? 0) < activeThreshold && (inspectItem.ground_truth_label === 1 || (inspectItem as any)['Is Laundering'] === 1) && (
              <div className="p-3.5 rounded-xl bg-amber-950/40 border border-amber-600/50 text-amber-200 space-y-1.5">
                <div className="flex items-center space-x-2 font-bold text-amber-300">
                  <AlertTriangle className="w-4 h-4 text-amber-400 shrink-0" />
                  <span>Suggested Handling: Needs Analyst Review</span>
                </div>
                <div className="grid grid-cols-2 gap-2 text-[10.5px] font-mono pt-1">
                  <div>
                    <span className="text-amber-400/70 block uppercase text-[9px]">Model Prediction</span>
                    <strong className="text-emerald-400">Cleared</strong>
                  </div>
                  <div>
                    <span className="text-amber-400/70 block uppercase text-[9px]">CSV Ground Truth</span>
                    <strong className="text-purple-300">Laundering</strong>
                  </div>
                </div>
                <p className="text-[11px] text-amber-300/80 leading-relaxed font-sans pt-1">
                  <strong>Notice:</strong> This transaction was cleared by the model under the active serving cutoff and is <strong>not</strong> counted as a model alert. Later data could change a future re-score, but under-review transactions are not promised to become flagged.
                </p>
              </div>
            )}

            <div className="p-3 rounded-xl bg-slate-100 dark:bg-slate-950 border border-slate-300 dark:border-slate-800 space-y-1">
              <span className="text-[10px] text-slate-500 block uppercase font-semibold">Applied Model & Execution Context</span>
              <p className="text-[11px] text-slate-700 dark:text-slate-300 leading-relaxed">
                Model: <strong>{(inspectItem as any)?.model_name || modelName}</strong>.{' '}
                {((inspectItem as any)?.model_name || modelName).toLowerCase().includes('fallback') || ((inspectItem as any)?.model_name || modelName).toLowerCase().includes('lightgbm')
                  ? ' Evaluated via Tabular Fallback (LightGBM). Single-transaction evaluation mode without sequence history.'
                  : ` Evaluated via Federated Sequence Neural Network (${(inspectItem as any)?.sequence_length || 10}-step temporal sequence context). Full account transaction history utilized.`}
              </p>
            </div>

            <div className="flex justify-end pt-1">
              <button
                onClick={() => setInspectItem(null)}
                className="px-4 py-1.5 rounded-xl bg-slate-200 hover:bg-slate-300 dark:bg-slate-800 dark:hover:bg-slate-700 text-slate-900 dark:text-white font-semibold text-xs transition cursor-pointer"
              >
                Close Inspector
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};
