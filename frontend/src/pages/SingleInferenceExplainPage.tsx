import React, { useState, useEffect } from 'react';
import {
  ExplainResponse,
  ModelMetadataResponse,
  PredictionHistoryEntry,
  PredictResponse,
  SequenceItem,
  SupportedDataset,
  TransactionPredictionItem,
} from '../types/api';
import { apiService } from '../services/apiService';
import { DEMO_SAMPLES } from '../demo/demoSamples';
import { AttributionBarChart } from '../components/AttributionBarChart';
import {
  formatProbability,
  formatThreshold,
} from '../utils/formatters';
import {
  ShieldAlert,
  Send,
  Sparkles,
  AlertTriangle,
  CheckCircle2,
  Clock,
  Layers,
  Eye,
  RefreshCw,
  Info,
} from 'lucide-react';

interface SingleInferenceExplainPageProps {
  selectedDataset: SupportedDataset;
  onSelectDataset: (dataset: SupportedDataset) => void;
  models: ModelMetadataResponse[];
  onAddHistory: (entry: PredictionHistoryEntry) => void;
  passedTransactionItem: TransactionPredictionItem | null;
  onClearPassedTransaction: () => void;
}

export const SingleInferenceExplainPage: React.FC<SingleInferenceExplainPageProps> = ({
  selectedDataset,
  onSelectDataset,
  models,
  onAddHistory,
  passedTransactionItem,
  onClearPassedTransaction,
}) => {
  const [selectedDemoId, setSelectedDemoId] = useState<string>('');
  const [sequence, setSequence] = useState<SequenceItem | null>(null);
  const [sampleDescription, setSampleDescription] = useState<string>('');

  // Execution states
  const [predicting, setPredicting] = useState<boolean>(false);
  const [explaining, setExplaining] = useState<boolean>(false);
  const [prediction, setPrediction] = useState<PredictResponse | null>(null);
  const [explanation, setExplanation] = useState<ExplainResponse | null>(null);
  const [error, setError] = useState<string | null>(null);

  // Load passed-in transaction from CSV evaluator if provided
  useEffect(() => {
    if (passedTransactionItem) {
      const seq: SequenceItem = {
        features: passedTransactionItem.sequence_features,
        length: passedTransactionItem.sequence_length,
        sample_id: `CSV_ROW_${passedTransactionItem.row_id}`,
      };
      setSequence(seq);
      setSampleDescription(
        `Selected transaction #${passedTransactionItem.row_id} from CSV Evaluation (${passedTransactionItem.timestamp}, Account: ${passedTransactionItem.from_account}).`
      );
      setSelectedDemoId('');
      setPrediction(null);
      setExplanation(null);
      setError(null);
    }
  }, [passedTransactionItem]);

  // Demo Sample selection
  const handleSelectDemo = (demoId: string) => {
    setSelectedDemoId(demoId);
    onClearPassedTransaction();
    if (!demoId) {
      setSequence(null);
      setSampleDescription('');
      setPrediction(null);
      setExplanation(null);
      return;
    }

    const found = DEMO_SAMPLES.find((s) => s.id === demoId);
    if (found) {
      onSelectDataset(found.dataset);
      setSequence(found.sequence);
      setSampleDescription(found.description);
      setPrediction(null);
      setExplanation(null);
      setError(null);
    }
  };

  // Run Prediction
  const handleRunPredict = async () => {
    if (!sequence) {
      setError('Please select or configure a sequence before running prediction.');
      return;
    }

    setPredicting(true);
    setError(null);
    setPrediction(null);

    try {
      const resp = await apiService.predict({
        dataset: selectedDataset,
        sequence: sequence,
      });

      setPrediction(resp);

      // Add to session history
      onAddHistory({
        id: resp.request_id,
        timestamp: new Date().toISOString(),
        dataset: selectedDataset,
        modelName: resp.model_name,
        predictedProbability: resp.predicted_probability,
        decisionThreshold: resp.decision_threshold,
        decision: resp.decision,
        inferenceTimeMs: resp.inference_time_ms,
        sampleId: resp.sample_id,
      });
    } catch (err: unknown) {
      const msg = err instanceof Error ? err.message : 'Inference request failed';
      setError(msg);
    } finally {
      setPredicting(false);
    }
  };

  // Run Integrated Gradients Explanation
  const handleRunExplain = async () => {
    if (!sequence) {
      setError('No sequence configured to explain.');
      return;
    }

    setExplaining(true);
    setError(null);

    try {
      const resp = await apiService.explain({
        dataset: selectedDataset,
        sequence: sequence,
        steps: 50,
        top_k: 8,
      });

      setExplanation(resp);
    } catch (err: unknown) {
      const msg = err instanceof Error ? err.message : 'Explanation request failed';
      setError(msg);
    } finally {
      setExplaining(false);
    }
  };

  const activeModel = models.find((m) => m.dataset === selectedDataset);
  const isFlagged = prediction?.decision === 'FLAGGED_LAUNDERING';

  return (
    <div className="space-y-8 animate-fade-in">
      {/* Header */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 border-b border-slate-800 pb-5">
        <div>
          <h1 className="text-2xl sm:text-3xl font-black text-white tracking-tight flex items-center space-x-2">
            <Eye className="w-7 h-7 text-indigo-400" />
            <span>Single Sequence Inference & Integrated Gradients</span>
          </h1>
          <p className="text-xs text-slate-400 mt-1">
            Real-time inference and axiomatic attribution analysis for transaction sequences.
          </p>
        </div>

        {/* Dataset Pipeline Switcher */}
        <div className="flex items-center space-x-2 bg-slate-900 p-1.5 rounded-2xl border border-slate-800">
          <button
            onClick={() => onSelectDataset('HI-Small')}
            className={`px-3.5 py-1.5 rounded-xl text-xs font-bold transition ${
              selectedDataset === 'HI-Small'
                ? 'bg-indigo-600 text-white shadow-lg shadow-indigo-600/30'
                : 'text-slate-400 hover:text-slate-200'
            }`}
          >
            HI-Small (LSTM)
          </button>
          <button
            onClick={() => onSelectDataset('LI-Medium')}
            className={`px-3.5 py-1.5 rounded-xl text-xs font-bold transition ${
              selectedDataset === 'LI-Medium'
                ? 'bg-violet-600 text-white shadow-lg shadow-violet-600/30'
                : 'text-slate-400 hover:text-slate-200'
            }`}
          >
            LI-Medium (BiLSTM)
          </button>
        </div>
      </div>

      {/* Target Model Spec Strip */}
      <div className="bg-slate-900 border border-slate-800 rounded-2xl p-4 shadow-lg flex flex-wrap items-center justify-between gap-4 font-mono text-xs">
        <div className="flex items-center space-x-2">
          <Layers className="w-4 h-4 text-indigo-400" />
          <span className="text-slate-400">Serving Champion:</span>
          <strong className="text-white">{activeModel?.model_name || 'Champion Model'}</strong>
        </div>
        <div className="flex items-center space-x-4 text-slate-400">
          <span>Arch: <strong className="text-slate-200">{activeModel?.architecture || 'LSTM'}</strong></span>
          <span>•</span>
          <span>Threshold: <strong className="text-amber-400">{activeModel ? formatThreshold(activeModel.frozen_threshold) : '0.3378'}</strong></span>
          <span>•</span>
          <span>Input Shape: <strong className="text-slate-200">(1, 10, 57)</strong></span>
        </div>
      </div>

      {/* Error state */}
      {error && (
        <div className="p-4 bg-rose-950/60 border border-rose-500/50 rounded-2xl flex items-start space-x-3 text-rose-200 text-sm">
          <AlertTriangle className="w-5 h-5 text-rose-400 shrink-0 mt-0.5" />
          <div>
            <h4 className="font-semibold text-rose-300">Execution Error</h4>
            <p className="text-xs text-rose-200/90 mt-1">{error}</p>
          </div>
        </div>
      )}

      {/* Sequence Selection & Execution Card */}
      <div className="bg-slate-900 border border-slate-800 rounded-3xl p-6 shadow-xl space-y-6">
        <div className="space-y-4">
          <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3">
            <label className="text-xs font-mono uppercase text-slate-400 font-semibold">
              Select Sample Sequence or Uploaded Transaction
            </label>
            <div className="text-xs font-mono text-slate-500">
              {sequence ? `Configured (Length: ${sequence.length || 10} steps)` : 'No sequence active'}
            </div>
          </div>

          <select
            value={selectedDemoId}
            onChange={(e) => handleSelectDemo(e.target.value)}
            className="w-full bg-slate-950 border border-slate-800 rounded-xl px-4 py-2.5 text-xs text-slate-200 focus:outline-none focus:border-indigo-500"
          >
            <option value="">-- Choose verified demonstration sequence --</option>
            {DEMO_SAMPLES.map((s) => (
              <option key={s.id} value={s.id}>
                [{s.dataset}] {s.label} ({s.groundTruthLabel})
              </option>
            ))}
          </select>

          {sampleDescription && (
            <div className="p-3.5 bg-slate-950 rounded-2xl border border-slate-800 text-xs text-slate-300 flex items-start space-x-2">
              <Info className="w-4 h-4 text-indigo-400 shrink-0 mt-0.5" />
              <p>{sampleDescription}</p>
            </div>
          )}

          {/* Action Buttons */}
          <div className="flex flex-wrap items-center gap-3 pt-2">
            <button
              onClick={handleRunPredict}
              disabled={predicting || !sequence}
              className="flex items-center space-x-2 px-5 py-2.5 rounded-xl bg-indigo-600 hover:bg-indigo-500 text-white font-medium text-xs transition shadow-lg shadow-indigo-600/30 disabled:opacity-50"
            >
              {predicting ? (
                <>
                  <RefreshCw className="w-3.5 h-3.5 animate-spin" />
                  <span>Evaluating Sequence...</span>
                </>
              ) : (
                <>
                  <Send className="w-3.5 h-3.5" />
                  <span>Run Model Inference</span>
                </>
              )}
            </button>

            <button
              onClick={handleRunExplain}
              disabled={explaining || !sequence}
              className="flex items-center space-x-2 px-5 py-2.5 rounded-xl bg-slate-800 hover:bg-slate-700 text-indigo-300 font-medium text-xs border border-indigo-500/40 transition disabled:opacity-50"
            >
              {explaining ? (
                <>
                  <RefreshCw className="w-3.5 h-3.5 animate-spin" />
                  <span>Computing Integrated Gradients...</span>
                </>
              ) : (
                <>
                  <Sparkles className="w-3.5 h-3.5 text-indigo-400" />
                  <span>Generate Integrated Gradients Explanation</span>
                </>
              )}
            </button>
          </div>
        </div>
      </div>

      {/* Prediction Output Card */}
      {prediction && (
        <div className="bg-slate-900 border border-slate-800 rounded-3xl p-6 shadow-xl space-y-6">
          <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 border-b border-slate-800 pb-4">
            <div>
              <span className="text-[10px] text-slate-500 font-mono block uppercase">Request ID: {prediction.request_id}</span>
              <h3 className="text-xl font-bold text-white font-mono mt-0.5">Model Scoring Outcome</h3>
            </div>
            <div className="flex items-center space-x-2 text-xs font-mono text-slate-400">
              <Clock className="w-4 h-4 text-indigo-400" />
              <span>Inference: {prediction.inference_time_ms.toFixed(2)} ms</span>
            </div>
          </div>

          <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
            {/* Probability Card */}
            <div className="bg-slate-950 p-5 rounded-2xl border border-slate-800 space-y-2">
              <span className="text-xs text-slate-400 font-mono uppercase block">Predicted Probability</span>
              <span className={`text-3xl font-black font-mono block ${isFlagged ? 'text-rose-400' : 'text-slate-200'}`}>
                {formatProbability(prediction.predicted_probability)}
              </span>
              <div className="w-full bg-slate-800 h-2 rounded-full overflow-hidden">
                <div
                  className={`h-full ${isFlagged ? 'bg-rose-500' : 'bg-emerald-500'}`}
                  style={{ width: `${Math.min(100, Math.max(3, prediction.predicted_probability * 100))}%` }}
                />
              </div>
            </div>

            {/* Frozen Decision Threshold */}
            <div className="bg-slate-950 p-5 rounded-2xl border border-slate-800 space-y-2">
              <span className="text-xs text-slate-400 font-mono uppercase block">Frozen Decision Threshold</span>
              <span className="text-3xl font-black font-mono text-amber-400 block">
                {prediction.decision_threshold.toFixed(4)}
              </span>
              <span className="text-[11px] text-slate-500 font-mono block">Phase 7 Authoritative Lock</span>
            </div>

            {/* Final Action / Decision */}
            <div className="bg-slate-950 p-5 rounded-2xl border border-slate-800 space-y-2 flex flex-col justify-between">
              <span className="text-xs text-slate-400 font-mono uppercase block">Compliance Decision</span>
              <div className="pt-1">
                <span
                  className={`px-3 py-1.5 rounded-xl text-xs font-bold font-mono tracking-wider inline-flex items-center space-x-2 ${
                    isFlagged
                      ? 'bg-rose-950/80 text-rose-300 border border-rose-800/80 shadow-lg shadow-rose-950/50'
                      : 'bg-emerald-950/80 text-emerald-300 border border-emerald-800/80 shadow-lg shadow-emerald-950/50'
                  }`}
                >
                  {isFlagged ? <ShieldAlert className="w-4 h-4 text-rose-400" /> : <CheckCircle2 className="w-4 h-4 text-emerald-400" />}
                  <span>{isFlagged ? 'FLAGGED FOR INVESTIGATION' : 'CLEARED AS LEGITIMATE'}</span>
                </span>
              </div>
              <span className="text-[11px] text-slate-500 font-mono block">
                Rule: Probability {'>='} Threshold
              </span>
            </div>
          </div>
        </div>
      )}

      {/* Integrated Gradients Attribution Waterfall Chart */}
      {explanation && (
        <div className="bg-slate-900 border border-slate-800 rounded-3xl p-6 shadow-xl space-y-6">
          <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 border-b border-slate-800 pb-4">
            <div>
              <div className="flex items-center space-x-2">
                <Sparkles className="w-5 h-5 text-indigo-400" />
                <h3 className="text-lg font-bold text-white tracking-tight">
                  PyTorch-Native Integrated Gradients Attribution
                </h3>
              </div>
              <p className="text-xs text-slate-400 mt-0.5">
                Axiomatic attribution (Sundararajan et al., 2017) over 50 Riemann midpoint integration steps. Target: pre-sigmoid logit.
              </p>
            </div>
            <div className="text-right text-xs font-mono text-slate-400">
              <span>Completeness Residual: </span>
              <strong className="text-emerald-400">{explanation.completeness_residual.toExponential(3)}</strong>
            </div>
          </div>

          {/* Waterfall Chart */}
          <AttributionBarChart
            explanation={explanation}
          />
        </div>
      )}
    </div>
  );
};
