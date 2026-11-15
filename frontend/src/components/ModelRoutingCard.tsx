import React from 'react';
import { GitBranch, AlertTriangle, CheckCircle2, ArrowRight, Cpu, Layers } from 'lucide-react';
import { UniversalCsvInspectionResponse } from '../types/api';

interface ModelRoutingCardProps {
  inspection: UniversalCsvInspectionResponse | null;
  selectedModelId: string;
  onSelectModel: (modelId: string) => void;
}

export const ModelRoutingCard: React.FC<ModelRoutingCardProps> = ({
  inspection,
  selectedModelId,
  onSelectModel,
}) => {
  if (!inspection || !inspection.routing_decision) {
    return null;
  }

  const decision = inspection.routing_decision;
  const recommendedId = inspection.recommended_model_id || 'hi_small_lstm_fedavg_iid_seed42';
  const recommendedName = inspection.recommended_model_name || 'HI-Small Champion (Federated LSTM FedAvg IID)';
  const reason = inspection.routing_reason || '';
  const uniqueRatio = inspection.unique_sender_ratio !== undefined ? inspection.unique_sender_ratio * 100 : null;
  const avgDepth = inspection.average_sequence_depth !== undefined ? inspection.average_sequence_depth : null;

  const isTabularRecommended = decision === 'RECOMMEND_TABULAR_FALLBACK';
  const isMissingFeatures = decision === 'REQUEST_MISSING_FEATURES';

  const isCurrentModelRecommended = selectedModelId === recommendedId;
  const isRecurrentSelectedOnColdStart = isTabularRecommended && selectedModelId !== 'lightgbm_tabular_fallback' && selectedModelId !== 'tabular';

  return (
    <div className={`p-4 rounded-2xl border transition shadow-lg space-y-3 ${isMissingFeatures
        ? 'bg-rose-950/30 border-rose-500/40'
        : isTabularRecommended
          ? 'bg-amber-950/30 border-amber-500/40'
          : 'bg-indigo-950/30 border-indigo-500/40'
      }`}>
      <div className="flex items-center justify-between">
        <div className="flex items-center space-x-2">
          <GitBranch className={`w-4 h-4 ${isMissingFeatures ? 'text-rose-400' : isTabularRecommended ? 'text-amber-400' : 'text-indigo-400'
            }`} />
          <h4 className="text-xs font-bold font-mono uppercase tracking-wide text-white">
            Automatic Model Routing Strategy
          </h4>
        </div>
        <span className={`text-[10px] font-mono px-2 py-0.5 rounded-full font-bold uppercase ${isMissingFeatures
            ? 'bg-rose-500/20 text-rose-300 border border-rose-500/30'
            : isTabularRecommended
              ? 'bg-amber-500/20 text-amber-300 border border-amber-500/30'
              : 'bg-emerald-500/20 text-emerald-300 border border-emerald-500/30'
          }`}>
          {isMissingFeatures
            ? 'Missing Required Features'
            : isTabularRecommended
              ? 'Tabular Fallback Recommended'
              : 'Recurrent Models Compatible'}
        </span>
      </div>

      {/* Rationale explanation */}
      <p className="text-xs text-slate-300 leading-relaxed font-sans">
        {reason}
      </p>

      {/* Context telemetry pills */}
      <div className="flex items-center space-x-2 flex-wrap gap-y-1 font-mono text-[10px]">
        {uniqueRatio !== null && (
          <span className="bg-slate-900 border border-slate-800 px-2 py-1 rounded-lg text-slate-300">
            Senders Unique: <strong className="text-white">{uniqueRatio.toFixed(1)}%</strong>
          </span>
        )}
        {avgDepth !== null && (
          <span className="bg-slate-900 border border-slate-800 px-2 py-1 rounded-lg text-slate-300">
            Avg Sequence Depth: <strong className="text-white">{avgDepth.toFixed(1)} tx/sender</strong>
          </span>
        )}
        <span className="bg-slate-900 border border-slate-800 px-2 py-1 rounded-lg text-slate-300">
          Target Labels: <strong className={inspection.has_ground_truth ? 'text-emerald-400' : 'text-amber-400'}>
            {inspection.has_ground_truth ? 'Present (Supervised)' : 'Absent (Score-Only)'}
          </strong>
        </span>
      </div>

      {/* Recommendation and Selection Action */}
      <div className="pt-2 border-t border-slate-800/80 flex flex-col sm:flex-row sm:items-center justify-between gap-2">
        <div className="text-xs">
          <span className="text-slate-400 font-mono text-[10px] uppercase block">Recommended Model</span>
          <span className="text-white font-semibold font-mono text-xs flex items-center space-x-1.5 mt-0.5">
            {isTabularRecommended ? <Layers className="w-3.5 h-3.5 text-amber-400" /> : <Cpu className="w-3.5 h-3.5 text-indigo-400" />}
            <span>{recommendedName}</span>
          </span>
        </div>

        {!isMissingFeatures && !isCurrentModelRecommended && (
          <button
            type="button"
            onClick={() => onSelectModel(recommendedId)}
            className="px-3 py-1.5 rounded-xl bg-indigo-600 hover:bg-indigo-500 text-white font-mono text-xs font-semibold flex items-center space-x-1.5 self-start sm:self-auto transition shadow-md cursor-pointer"
          >
            <span>Apply Recommended Model</span>
            <ArrowRight className="w-3.5 h-3.5" />
          </button>
        )}

        {isCurrentModelRecommended && (
          <span className="text-[11px] font-mono text-emerald-400 flex items-center space-x-1">
            <CheckCircle2 className="w-3.5 h-3.5" />
            <span>Recommended model is active</span>
          </span>
        )}
      </div>

      {/* Warning if recurrent model is selected on cold-start dataset */}
      {isRecurrentSelectedOnColdStart && (
        <div className="p-2.5 rounded-xl bg-amber-900/40 border border-amber-500/50 text-amber-200 text-xs font-mono flex items-start space-x-2">
          <AlertTriangle className="w-4 h-4 text-amber-400 shrink-0 mt-0.5" />
          <div className="space-y-0.5 flex-1">
            <span className="font-bold block">Cold-Start Model Warning</span>
            <span className="text-[11px] text-amber-300 font-sans block">
              You currently have a recurrent sequence model selected, but this file contains {uniqueRatio?.toFixed(1)}% single-transaction accounts. The model will evaluate zero-padded sequences and output conservative baseline probabilities. Consider switching to the Tabular Fallback model.
            </span>
          </div>
        </div>
      )}
    </div>
  );
};
