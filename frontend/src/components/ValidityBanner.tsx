import React, { useState } from 'react';
import { ValidityBanner as ValidityBannerType } from '../types/api';
import { AlertTriangle, CheckCircle2, Info, X } from 'lucide-react';

interface ValidityBannerProps {
  banner?: ValidityBannerType | null;
}

export const ValidityBanner: React.FC<ValidityBannerProps> = ({ banner }) => {
  const [dismissed, setDismissed] = useState(false);
  const [showDetails, setShowDetails] = useState(false);

  if (!banner || dismissed) return null;

  const isInvalid = banner.status === 'INVALID';
  const isValid = banner.status === 'VALID';

  const containerClasses = isValid
    ? 'bg-emerald-950/30 border-emerald-500/40 text-emerald-200'
    : 'bg-amber-950/30 border-amber-500/40 text-amber-200';

  const summaryText = isValid
    ? 'Representative sequence benchmark'
    : banner.reasons?.[0] ||
      (isInvalid
        ? 'Low sequence depth or distribution shift detected; treat metrics as diagnostic.'
        : 'Partial sequence history detected on this dataset.');

  return (
    <div className={`px-4 py-3 rounded-xl border ${containerClasses} space-y-2`}>
      <div className="flex items-center justify-between gap-3">
        <div className="flex items-center gap-2.5 min-w-0">
          {isValid ? (
            <CheckCircle2 className="w-4 h-4 text-emerald-400 shrink-0" />
          ) : (
            <AlertTriangle className="w-4 h-4 text-amber-400 shrink-0" />
          )}
          <span className="text-sm font-medium truncate">{summaryText}</span>
        </div>

        <div className="flex items-center gap-1 shrink-0">
          <button
            type="button"
            onClick={() => setShowDetails((prev) => !prev)}
            title="View dataset diagnostics"
            aria-label="Toggle dataset diagnostics"
            className="p-2 rounded-lg text-slate-400 hover:text-white hover:bg-slate-800/70 transition cursor-pointer"
          >
            <Info className="w-4 h-4" />
          </button>
          <button
            type="button"
            onClick={() => setDismissed(true)}
            title="Dismiss notice"
            aria-label="Dismiss notice"
            className="p-2 rounded-lg text-slate-400 hover:text-white hover:bg-slate-800/70 transition cursor-pointer"
          >
            <X className="w-4 h-4" />
          </button>
        </div>
      </div>

      {showDetails && (
        <div className="pt-2 border-t border-slate-800/70 space-y-2 text-xs text-slate-300">
          {banner.reasons && banner.reasons.length > 0 && (
            <ul className="list-disc list-inside space-y-0.5 text-slate-300">
              {banner.reasons.slice(0, 3).map((r, i) => (
                <li key={i}>{r}</li>
              ))}
            </ul>
          )}
          {banner.diagnostics && (
            <div className="flex flex-wrap gap-x-4 gap-y-1 text-[11px] font-mono text-slate-400">
              <span>Rows: <strong className="text-white">{banner.diagnostics.n_rows}</strong></span>
              {banner.diagnostics.prevalence !== null && (
                <span>Prevalence: <strong className="text-white">{(banner.diagnostics.prevalence * 100).toFixed(2)}%</strong></span>
              )}
              <span>Unique Senders: <strong className="text-white">{(banner.diagnostics.unique_sender_ratio * 100).toFixed(1)}%</strong></span>
              <span>Mean History: <strong className="text-white">{banner.diagnostics.mean_history_len}/10</strong></span>
              <span>&ge;3 Prior Tx: <strong className="text-white">{banner.diagnostics.pct_rows_with_ge_3_prior_tx}%</strong></span>
              <span>Chronological: <strong className={banner.diagnostics.is_chronological ? 'text-emerald-400' : 'text-rose-400'}>{banner.diagnostics.is_chronological ? 'Yes' : 'No'}</strong></span>
            </div>
          )}
        </div>
      )}
    </div>
  );
};
