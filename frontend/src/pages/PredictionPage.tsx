import React, { useState } from 'react';
import {
  ExplainResponse,
  ModelMetadataResponse,
  PredictionHistoryEntry,
  PredictResponse,
  SequenceItem,
  SupportedDataset,
} from '../types/api';
import { apiService } from '../services/apiService';
import { DEMO_SAMPLES } from '../demo/demoSamples';
import { AttributionBarChart } from '../components/AttributionBarChart';
import {
  formatProbability,
  formatScore,
  formatThreshold,
  getRiskLevel,
} from '../utils/formatters';
import {
  ShieldAlert,
  Send,
  Sparkles,
  AlertTriangle,
  CheckCircle2,
  FileCode2,
  Clock,
  Layers,
} from 'lucide-react';

interface PredictionPageProps {
  selectedDataset: SupportedDataset;
  onSelectDataset: (dataset: SupportedDataset) => void;
  models: ModelMetadataResponse[];
  onAddHistory: (entry: PredictionHistoryEntry) => void;
}

export const PredictionPage: React.FC<PredictionPageProps> = ({
  selectedDataset,
  onSelectDataset,
  models,
  onAddHistory,
}) => {
  // Current sequence state
  const [selectedDemoId, setSelectedDemoId] = useState<string>('');
  const [sequence, setSequence] = useState<SequenceItem | null>(null);
  const [sampleDescription, setSampleDescription] = useState<string>('');

  // Request & Execution states
  const [predicting, setPredicting] = useState<boolean>(false);
  const [explaining, setExplaining] = useState<boolean>(false);
  const [prediction, setPrediction] = useState<PredictResponse | null>(null);
  const [explanation, setExplanation] = useState<ExplainResponse | null>(null);
  const [error, setError] = useState<string | null>(null);

  // Active model metadata
  const activeModel = models.find((m) => m.dataset === selectedDataset);

  // Handle Demo Sample Selection
  const handleSelectDemo = (demoId: string) => {
    setSelectedDemoId(demoId);
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

  // Run Inference (/predict)
  const handleRunPredict = async () => {
    if (!sequence) {
      setError('Please select or configure a sequence before running prediction.');
      return;
    }

    setPredicting(true);
    setError(null);
    setPrediction(null);
    setExplanation(null);

    try {
      const resp = await apiService.predict({
        dataset: selectedDataset,
        sequence: sequence,
      });

      setPrediction(resp);

      // Append to session history
      onAddHistory({
        id: resp.request_id,
        dataset: resp.dataset,
        timestamp: new Date().toISOString(),
        predicted_probability: resp.predicted_probability,
        predictedProbability: resp.predicted_probability,
        decision_threshold: resp.decision_threshold,
        decisionThreshold: resp.decision_threshold,
        decision: resp.decision,
        sample_id: resp.sample_id,
        sampleId: resp.sample_id,
        sequence_length: resp.sequence_length,
        inference_time_ms: resp.inference_time_ms,
        inferenceTimeMs: resp.inference_time_ms,
      });
    } catch (err: unknown) {
      const msg = err instanceof Error ? err.message : 'Unknown prediction error occurred';
      setError(msg);
    } finally {
      setPredicting(false);
    }
  };

  // Run Explainability (/explain)
  const handleRunExplain = async () => {
    if (!sequence) {
      setError('Please select a sequence before running attribution.');
      return;
    }

    setExplaining(true);
    setError(null);

    try {
      const resp = await apiService.explain({
        dataset: selectedDataset,
        sequence: sequence,
        steps: 25,
        top_k: 5,
      });

      setExplanation(resp);
    } catch (err: unknown) {
      const msg = err instanceof Error ? err.message : 'Unknown attribution error occurred';
      setError(msg);
    } finally {
      setExplaining(false);
    }
  };

  // Filter demo samples by selected dataset
  const filteredDemos = DEMO_SAMPLES.filter((s) => s.dataset === selectedDataset);

  return (
    <div className="space-y-8">
      {/* Top Controls Card */}
      <div className="bg-slate-900 border border-slate-800 rounded-3xl p-6 shadow-xl">
        <div className="grid grid-cols-1 md:grid-cols-3 gap-6 items-end">
          {/* Dataset Selector */}
          <div>
            <label className="block text-xs font-mono uppercase tracking-wider text-slate-400 mb-2">
              1. Target AML Dataset
            </label>
            <div className="flex rounded-xl bg-slate-950 p-1 border border-slate-800">
              <button
                type="button"
                onClick={() => {
                  onSelectDataset('HI-Small');
                  setSelectedDemoId('');
                  setSequence(null);
                  setPrediction(null);
                  setExplanation(null);
                }}
                className={`flex-1 py-2 px-3 rounded-lg text-xs font-semibold font-mono transition ${
                  selectedDataset === 'HI-Small'
                    ? 'bg-indigo-600 text-white shadow'
                    : 'text-slate-400 hover:text-white'
                }`}
              >
                IBM AML HI-Small
              </button>
              <button
                type="button"
                onClick={() => {
                  onSelectDataset('LI-Medium');
                  setSelectedDemoId('');
                  setSequence(null);
                  setPrediction(null);
                  setExplanation(null);
                }}
                className={`flex-1 py-2 px-3 rounded-lg text-xs font-semibold font-mono transition ${
                  selectedDataset === 'LI-Medium'
                    ? 'bg-indigo-600 text-white shadow'
                    : 'text-slate-400 hover:text-white'
                }`}
              >
                IBM AML LI-Medium
              </button>
            </div>
            {activeModel && (
              <div className="text-[11px] text-slate-400 font-mono mt-1.5 truncate">
                Champion: {activeModel.model_name} (threshold: {formatThreshold(activeModel.frozen_threshold)})
              </div>
            )}
          </div>

          {/* Demonstration Sample Selector */}
          <div>
            <label className="block text-xs font-mono uppercase tracking-wider text-slate-400 mb-2">
              2. Demonstration Validation Sample
            </label>
            <select
              value={selectedDemoId}
              onChange={(e) => handleSelectDemo(e.target.value)}
              className="w-full bg-slate-950 border border-slate-800 text-slate-200 text-xs rounded-xl p-2.5 focus:border-indigo-500 focus:outline-none"
            >
              <option value="">-- Choose Authoritative Validation Sample --</option>
              {filteredDemos.map((demo) => (
                <option key={demo.id} value={demo.id}>
                  {demo.label}
                </option>
              ))}
            </select>
          </div>

          {/* Action Buttons */}
          <div className="flex space-x-3">
            <button
              onClick={handleRunPredict}
              disabled={predicting || !sequence}
              className="flex-1 flex items-center justify-center space-x-2 py-2.5 px-4 rounded-xl bg-indigo-600 hover:bg-indigo-500 text-white font-medium text-xs transition shadow-lg shadow-indigo-600/20 disabled:opacity-50 disabled:cursor-not-allowed"
            >
              <Send className={`w-3.5 h-3.5 ${predicting ? 'animate-pulse' : ''}`} />
              <span>{predicting ? 'Evaluating...' : 'Predict Laundering Risk'}</span>
            </button>
            <button
              onClick={handleRunExplain}
              disabled={explaining || !sequence}
              className="flex items-center space-x-1.5 py-2.5 px-4 rounded-xl bg-slate-800 hover:bg-slate-700 text-slate-200 text-xs font-medium transition border border-slate-700 disabled:opacity-50 disabled:cursor-not-allowed"
              title="Compute Integrated Gradients Attributions"
            >
              <Sparkles className={`w-3.5 h-3.5 text-amber-400 ${explaining ? 'animate-spin' : ''}`} />
              <span>{explaining ? 'Calculating...' : 'Explain (IG)'}</span>
            </button>
          </div>
        </div>

        {/* Selected Sample Information Banner */}
        {sampleDescription && (
          <div className="mt-4 pt-4 border-t border-slate-800/80 flex items-start space-x-3 text-xs text-slate-300">
            <FileCode2 className="w-4 h-4 text-indigo-400 shrink-0 mt-0.5" />
            <div>
              <span className="font-semibold text-slate-200">Validation Sequence Provenance: </span>
              <span className="text-slate-400">{sampleDescription}</span>
              <span className="ml-2 font-mono text-[11px] text-indigo-300">
                (Length: {sequence?.length || 10} steps × 57 float32 features)
              </span>
            </div>
          </div>
        )}
      </div>

      {/* Error Banner */}
      {error && (
        <div className="p-4 bg-rose-950/60 border border-rose-500/50 rounded-2xl flex items-start space-x-3 text-rose-200 text-sm">
          <AlertTriangle className="w-5 h-5 text-rose-400 shrink-0 mt-0.5" />
          <div>
            <h4 className="font-semibold text-rose-300">API Execution Error</h4>
            <p className="text-xs text-rose-200/80 mt-1">{error}</p>
          </div>
        </div>
      )}

      {/* Authoritative Prediction Result Card */}
      {prediction && (
        <div className="bg-slate-900 border border-slate-800 rounded-3xl p-8 shadow-2xl relative overflow-hidden">
          {/* Background Ambient Glow */}
          <div
            className={`absolute top-0 right-0 w-96 h-96 rounded-full blur-3xl opacity-10 pointer-events-none ${
              prediction.decision === 'FLAGGED_LAUNDERING' ? 'bg-rose-500' : 'bg-emerald-500'
            }`}
          />

          <div className="relative z-10">
            <div className="flex flex-col sm:flex-row sm:items-center justify-between pb-6 border-b border-slate-800 gap-4">
              <div>
                <span className="text-xs font-mono uppercase tracking-wider text-slate-400">
                  Backend Evaluation Result
                </span>
                <h3 className="text-xl font-bold text-white mt-1 flex items-center space-x-2">
                  <span>Authoritative AML Risk Classification</span>
                  <span className="text-xs font-mono font-normal text-slate-500">
                    (ID: {prediction.request_id})
                  </span>
                </h3>
              </div>

              {/* Status Badge */}
              <div>
                {(() => {
                  const risk = getRiskLevel(prediction.predicted_probability, prediction.decision_threshold);
                  return (
                    <div
                      className={`inline-flex items-center space-x-2 px-4 py-2 rounded-2xl text-xs font-bold uppercase tracking-wider border shadow-lg ${risk.badgeClass}`}
                    >
                      {prediction.decision === 'FLAGGED_LAUNDERING' ? (
                        <ShieldAlert className="w-4 h-4 text-rose-400" />
                      ) : (
                        <CheckCircle2 className="w-4 h-4 text-emerald-400" />
                      )}
                      <span>{risk.label}</span>
                    </div>
                  );
                })()}
              </div>
            </div>

            {/* Metrics Breakdown Grid */}
            <div className="grid grid-cols-1 sm:grid-cols-3 gap-6 mt-6">
              {/* Laundering Probability */}
              <div className="p-5 bg-slate-950/60 rounded-2xl border border-slate-800">
                <div className="text-xs text-slate-400 font-medium">Estimated Laundering Probability</div>
                <div className="mt-2 text-3xl font-extrabold font-mono text-white">
                  {formatProbability(prediction.predicted_probability)}
                </div>
                <div className="text-[11px] font-mono text-slate-500 mt-1">
                  Raw logit sigmoid: {formatScore(prediction.predicted_probability)}
                </div>
              </div>

              {/* Frozen Decision Threshold */}
              <div className="p-5 bg-slate-950/60 rounded-2xl border border-slate-800">
                <div className="text-xs text-slate-400 font-medium">Locked Operational Threshold</div>
                <div className="mt-2 text-3xl font-extrabold font-mono text-amber-400">
                  {formatThreshold(prediction.decision_threshold)}
                </div>
                <div className="text-[11px] text-slate-400 mt-1">
                  Frozen in Phase 7 audit lock
                </div>
              </div>

              {/* Latency & Causal Steps */}
              <div className="p-5 bg-slate-950/60 rounded-2xl border border-slate-800">
                <div className="text-xs text-slate-400 font-medium">Inference Execution Latency</div>
                <div className="mt-2 text-3xl font-extrabold font-mono text-slate-200 flex items-center space-x-2">
                  <Clock className="w-6 h-6 text-indigo-400" />
                  <span>{prediction.inference_time_ms.toFixed(2)} ms</span>
                </div>
                <div className="text-[11px] text-slate-400 mt-1">
                  Evaluated {prediction.sequence_length} causal transaction steps
                </div>
              </div>
            </div>

            {/* Explanation Note on Decision Authority */}
            <div className="mt-6 p-4 bg-slate-950/80 rounded-2xl border border-slate-800/80 flex items-start space-x-3 text-xs text-slate-300">
              <CheckCircle2 className="w-4 h-4 text-emerald-400 shrink-0 mt-0.5" />
              <div>
                <p>
                  <strong>Authoritative Backend Determination:</strong> Classification decision{' '}
                  <code className="text-indigo-300 font-mono font-semibold">{prediction.decision}</code>{' '}
                  is derived strictly from the condition{' '}
                  <code className="font-mono text-slate-200">
                    P({formatScore(prediction.predicted_probability)}) &gt;= τ({formatThreshold(prediction.decision_threshold)})
                  </code>.
                  The frontend never computes, alters, or overrides the backend decision threshold.
                </p>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Integrated Gradients Attribution Panel */}
      {explanation && (
        <div className="space-y-4">
          <div className="flex items-center space-x-2">
            <Sparkles className="w-5 h-5 text-amber-400" />
            <h3 className="text-lg font-bold text-white">Axiomatic Feature Explanations</h3>
          </div>
          <AttributionBarChart explanation={explanation} />
        </div>
      )}

      {/* Empty Initial State Prompt */}
      {!prediction && !predicting && (
        <div className="border border-dashed border-slate-800 rounded-3xl p-12 text-center bg-slate-900/20">
          <Layers className="w-10 h-10 text-slate-600 mx-auto mb-3" />
          <h4 className="font-medium text-slate-300 text-sm">No Active Prediction Loaded</h4>
          <p className="text-xs text-slate-500 max-w-md mx-auto mt-1">
            Select an authoritative demonstration sample from the dropdown above, then click{' '}
            <strong className="text-indigo-400">Predict Laundering Risk</strong> to run sequence classification.
          </p>
        </div>
      )}
    </div>
  );
};
