import React from 'react';
import { ValidityBanner as ValidityBannerType } from '../types/api';
import { AlertTriangle, CheckCircle2 } from 'lucide-react';

interface ValidityBannerProps {
  banner?: ValidityBannerType | null;
}

export const ValidityBanner: React.FC<ValidityBannerProps> = ({ banner }) => {
  if (!banner) return null;

  const isInvalid = banner.status === 'INVALID';
  const isValid = banner.status === 'VALID';

  const containerClasses = isValid
    ? 'bg-emerald-950/40 border-emerald-500/60 text-emerald-200'
    : isInvalid
    ? 'bg-rose-950/40 border-rose-500/70 text-rose-200'
    : 'bg-amber-950/40 border-amber-500/60 text-amber-200';

  const pillClasses = isValid
    ? 'bg-emerald-900/80 text-emerald-200 border-emerald-600'
    : isInvalid
    ? 'bg-rose-900/80 text-rose-200 border-rose-600'
    : 'bg-amber-900/80 text-amber-200 border-amber-600';

  return (
    <div className={`p-5 rounded-2xl border shadow-xl flex items-start space-x-4 ${containerClasses}`}>
      {isValid ? (
        <CheckCircle2 className="w-6 h-6 text-emerald-400 shrink-0 mt-0.5" />
      ) : (
        <AlertTriangle className="w-6 h-6 text-rose-400 shrink-0 mt-0.5" />
      )}
      <div className="space-y-2 flex-1">
        <div className="flex flex-wrap items-center justify-between gap-2 border-b border-slate-800/60 pb-2">
          <div className="flex items-center space-x-2">
            <span className={`px-2.5 py-0.5 rounded text-xs font-mono font-bold tracking-wide uppercase border ${pillClasses}`}>
              {isValid
                ? 'VALID EVALUATION DATASET'
                : isInvalid
                ? 'INVALID / NON-REPRESENTATIVE BENCHMARK'
                : 'CAUTION: STATISTICAL SAMPLE NOTICE'}
            </span>
            <span className="text-xs text-slate-300 font-sans">
              {isValid
                ? 'Data matches chronological sequence requirements for AML sequence models.'
                : 'This dataset is not representative of genuine model performance.'}
            </span>
          </div>
          {banner.flags && banner.flags.length > 0 && (
            <div className="flex flex-wrap gap-1 font-mono text-[10px]">
              {banner.flags.map((f) => (
                <span key={f} className="px-2 py-0.5 rounded bg-slate-900/80 text-slate-300 border border-slate-700">
                  {f}
                </span>
              ))}
            </div>
          )}
        </div>

        {/* Plain-language reasons */}
        {banner.reasons && banner.reasons.length > 0 && (
          <div className="space-y-1.5 pt-1">
            {banner.reasons.map((reason, idx) => (
              <div key={idx} className="flex items-start space-x-2 text-xs font-sans text-slate-200">
                <span className="text-amber-400 font-bold">•</span>
                <span>{reason}</span>
              </div>
            ))}
          </div>
        )}

        {/* Plain-language disclaimer */}
        {banner.disclaimer && (
          <p className="text-xs text-slate-300 italic font-sans pt-1">
            {banner.disclaimer}
          </p>
        )}

        {/* Diagnostics summary bar */}
        {banner.diagnostics && (
          <div className="pt-2 text-[11px] font-mono flex flex-wrap gap-x-4 gap-y-1 border-t border-slate-800/80 text-slate-400">
            <span>Rows: <strong className="text-white">{banner.diagnostics.n_rows}</strong></span>
            {banner.diagnostics.prevalence !== null && (
              <span>Prevalence: <strong className="text-white">{(banner.diagnostics.prevalence * 100).toFixed(2)}%</strong></span>
            )}
            {banner.diagnostics.positives_count !== null && (
              <span>Positives: <strong className="text-white">{banner.diagnostics.positives_count}</strong></span>
            )}
            <span>Unique Senders: <strong className="text-white">{(banner.diagnostics.unique_sender_ratio * 100).toFixed(1)}%</strong></span>
            <span>Mean History: <strong className="text-white">{banner.diagnostics.mean_history_len}/10</strong></span>
            <span>&gt;=3 Prior Tx: <strong className="text-white">{banner.diagnostics.pct_rows_with_ge_3_prior_tx}%</strong></span>
            <span>Chronological: <strong className={banner.diagnostics.is_chronological ? 'text-emerald-400' : 'text-rose-400'}>{banner.diagnostics.is_chronological ? 'Yes' : 'No'}</strong></span>
            <span>Time Span: <strong className="text-white">{banner.diagnostics.time_span}</strong></span>
          </div>
        )}
      </div>
    </div>
  );
};

