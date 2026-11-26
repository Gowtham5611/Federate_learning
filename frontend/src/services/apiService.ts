/**
 * API client service for FedSecureAI FastAPI backend.
 * Uses configurable base URL and provides structured error handling.
 */

import {
  AblationSummaryResponse,
  BenchmarkModelRecord,
  ClientPartitionResponse,
  CommunicationEfficiencyRecord,
  ConvergenceCurvePoint,
  CsvEvaluateResponse,
  CustomModelTrainResponse,
  DatasetMetadataResponse,
  ExplainRequest,
  ExplainResponse,
  HealthResponse,
  ReadinessResponse,
  ModelCompatibilityResult,
  ModelMetadataResponse,
  PredictRequest,
  PredictResponse,
  RegisteredModelItem,
  ResearchOverviewResponse,
  SchemaValidationResponse,
  SupportedDataset,
  ThresholdTradeoffPoint,
  UniversalCsvInspectionResponse,
} from '../types/api';

import { FALLBACK_BENCHMARK_MODELS } from '../data/fallbackModels';

const resolveApiBase = (): string => {
  // In the browser:
  if (typeof window !== 'undefined') {
    // 1. Local development: connect directly to local FastAPI server
    if (window.location.hostname === 'localhost' || window.location.hostname === '127.0.0.1') {
      return 'http://127.0.0.1:8000';
    }
    // 2. Explicit environment variable override (ignore localhost in cloud production)
    const envBase = import.meta.env.VITE_API_URL || import.meta.env.VITE_API_BASE_URL;
    if (
      envBase &&
      typeof envBase === 'string' &&
      envBase.trim() !== '' &&
      !envBase.includes('localhost') &&
      !envBase.includes('127.0.0.1')
    ) {
      return envBase.trim().replace(/\/+$/, '');
    }
    // 3. In production: direct connection to live Render backend
    // Bypasses Vercel edge/serverless proxy 10s request limits while utilizing verified CORS
    return 'https://federate-learning.onrender.com';
  }
  const envBase = import.meta.env.VITE_API_URL || import.meta.env.VITE_API_BASE_URL;
  if (
    envBase &&
    typeof envBase === 'string' &&
    envBase.trim() !== '' &&
    !envBase.includes('localhost') &&
    !envBase.includes('127.0.0.1')
  ) {
    return envBase.trim().replace(/\/+$/, '');
  }
  return 'https://federate-learning.onrender.com';
};

const API_BASE = resolveApiBase();

let _hasHealthSucceeded = false;

/**
 * Builds a clean canonical URL for API requests, avoiding double '/api/api' segments.
 */
const buildUrl = (endpoint: string): string => {
  const ep = endpoint.startsWith('/') ? endpoint : `/${endpoint}`;
  if (!API_BASE) {
    return ep;
  }
  if (API_BASE.endsWith('/api') && ep.startsWith('/api/')) {
    return `${API_BASE.slice(0, -4)}${ep}`;
  }
  return `${API_BASE}${ep}`;
};

export class ApiError extends Error {
  constructor(
    public status: number,
    message: string,
    public details?: unknown,
    public requestId?: string,
    public isColdStart: boolean = false
  ) {
    super(message);
    this.name = 'ApiError';
  }
}

/**
 * Executes a JSON HTTP request with automatic retry for cold-start resiliency.
 */
async function request<T>(
  endpoint: string,
  options: RequestInit = {},
  retries: number = 3,
  timeoutMs: number = 90000
): Promise<T> {
  const url = buildUrl(endpoint);
  const headers = {
    'Content-Type': 'application/json',
    ...(options.headers || {}),
  };

  let lastError: unknown = null;

  for (let attempt = 0; attempt < retries; attempt++) {
    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(), timeoutMs);
    try {
      const response = await fetch(url, { ...options, headers, signal: controller.signal });
      clearTimeout(timer);

      if (!response.ok) {
        // Render free-tier cold-start recovery (502 / 503 / 504 during wakeup)
        if ([502, 503, 504].includes(response.status) && attempt < retries - 1) {
          const delayMs = 2500 * (attempt + 1);
          console.warn(`[apiService] Cloud service warming up (HTTP ${response.status}). Retrying in ${delayMs}ms (attempt ${attempt + 1}/${retries})...`);
          await new Promise((res) => setTimeout(res, delayMs));
          continue;
        }

        let errorMessage = `HTTP ${response.status} (${response.statusText})`;
        let errorDetails: unknown = null;
        try {
          const errorData = await response.json();
          errorDetails = errorData;
          if (errorData.detail) {
            errorMessage =
              typeof errorData.detail === 'string'
                ? errorData.detail
                : JSON.stringify(errorData.detail);
          }
        } catch {
          // Response was not JSON (e.g. gateway HTML/plaintext 502)
        }

        const rndrId =
          response.headers.get('rndr-id') ||
          response.headers.get('x-render-id') ||
          response.headers.get('x-request-id') ||
          '';

        const isCold = [502, 503, 504].includes(response.status);

        if (response.status === 502) {
          errorMessage = `Gateway Error (HTTP 502 Bad Gateway): The hosting reverse proxy failed to reach the backend instance (at ${url}). The backend container may be cold-starting or restarting after memory limits. ${rndrId ? `[Trace: ${rndrId}]` : ''} Please check service readiness at /ready and retry.`;
        } else if (response.status === 504) {
          errorMessage = `Gateway Timeout (HTTP 504 Gateway Timeout): The hosting gateway timed out waiting for sequence evaluation to complete. Try evaluating with a smaller row slice or using the background job flow.`;
        } else if (response.status === 503) {
          errorMessage = `Service Unavailable (HTTP 503): Backend model service is initializing or verifying model checkpoints: ${errorMessage}`;
        } else if (response.status === 404) {
          errorMessage = `API route not found at ${url}. Ensure the FastAPI server is running with all route handlers registered.`;
        }

        throw new ApiError(response.status, errorMessage, errorDetails, rndrId || undefined, isCold);
      }

      return (await response.json()) as T;
    } catch (err) {
      clearTimeout(timer);
      if (err instanceof ApiError) {
        throw err;
      }
      if (err instanceof DOMException && err.name === 'AbortError') {
        throw new ApiError(
          504,
          `Request timed out after ${Math.round(timeoutMs / 1000)}s. Server may be waking up, please wait and retry.`,
          err,
          undefined,
          true
        );
      }
      lastError = err;
      // Network drop / socket timeout while cloud instance is waking up
      if (attempt < retries - 1) {
        const delayMs = 3000 * (attempt + 1);
        console.warn(`[apiService] Network blip or cold-start at ${url}. Retrying in ${delayMs}ms (attempt ${attempt + 1}/${retries})...`);
        await new Promise((res) => setTimeout(res, delayMs));
        continue;
      }
    }
  }

  throw new ApiError(
    0,
    `Backend connection error at ${url}. The cloud server may be waking up from an idle state.`,
    lastError,
    undefined,
    true
  );
}

export const apiService = {
  /**
   * Returns true once the backend /health or /ready endpoint has responded successfully in this session.
   */
  hasBackendWokenUp(): boolean {
    return _hasHealthSucceeded;
  },

  /**
   * Check API liveness (process is running and responsive).
   * Canonical endpoint: /health with automatic /api/health and /api/v1/health fallbacks.
   */
  async getHealth(): Promise<HealthResponse> {
    try {
      const h = await request<HealthResponse>('/health');
      _hasHealthSucceeded = true;
      return h;
    } catch (e1) {
      try {
        const h = await request<HealthResponse>('/api/health');
        _hasHealthSucceeded = true;
        return h;
      } catch {
        try {
          const h = await request<HealthResponse>('/api/v1/health');
          _hasHealthSucceeded = true;
          return h;
        } catch {
          throw e1;
        }
      }
    }
  },

  /**
   * Single-shot non-cascading health probe for background keep-alive pings.
   */
  async probeHealth(): Promise<HealthResponse> {
    const h = await request<HealthResponse>('/health', {}, 1, 15000);
    _hasHealthSucceeded = true;
    return h;
  },

  /**
   * Check detailed service operational readiness (models in memory and verified, datasets ready).
   */
  async getReady(): Promise<ReadinessResponse> {
    try {
      return await request<ReadinessResponse>('/ready');
    } catch {
      try {
        return await request<ReadinessResponse>('/api/ready');
      } catch {
        return await request<ReadinessResponse>('/api/v1/ready');
      }
    }
  },

  /**
   * Single-shot readiness probe for bounded cold-start polling (no retry cascade).
   */
  async probeReady(): Promise<ReadinessResponse> {
    return request<ReadinessResponse>('/ready', {}, 1);
  },

  /**
   * List verified model catalog with parameter counts and thresholds.
   */
  async getModels(): Promise<ModelMetadataResponse[]> {
    const res = await request<unknown>('/models');
    if (Array.isArray(res)) return res as ModelMetadataResponse[];
    if (res && typeof res === 'object' && 'models' in res && Array.isArray((res as { models: unknown }).models)) {
      return (res as { models: ModelMetadataResponse[] }).models;
    }
    return [];
  },

  /**
   * Fetch 57-feature metadata and frozen threshold for a dataset.
   */
  async getDatasetMetadata(dataset: SupportedDataset): Promise<DatasetMetadataResponse> {
    return request<DatasetMetadataResponse>(`/metadata/${dataset}`);
  },

  /**
   * Run AML prediction for a single transaction sequence.
   */
  async predict(payload: PredictRequest): Promise<PredictResponse> {
    return request<PredictResponse>('/predict', {
      method: 'POST',
      body: JSON.stringify(payload),
    });
  },

  /**
   * Run Integrated Gradients feature attribution on a sequence.
   */
  async explain(payload: ExplainRequest): Promise<ExplainResponse> {
    try {
      return await request<ExplainResponse>('/api/v1/explain', {
        method: 'POST',
        body: JSON.stringify(payload),
      });
    } catch (err) {
      if (err instanceof ApiError && err.status === 404) {
        return await request<ExplainResponse>('/explain', {
          method: 'POST',
          body: JSON.stringify(payload),
        });
      }
      throw err;
    }
  },

  /**
   * Phase 17: Upload and evaluate a transaction CSV batch.
   * Tries primary /evaluate/csv and falls back to /api/v1/evaluate/csv if 404.
   */
  async evaluateCsv(payload: {
    csv_content: string;
    dataset?: SupportedDataset;
    filename?: string;
    row_limit?: number;
  }): Promise<CsvEvaluateResponse> {
    try {
      return await request<CsvEvaluateResponse>('/evaluate/csv', {
        method: 'POST',
        body: JSON.stringify(payload),
      });
    } catch (err) {
      if (err instanceof ApiError && err.status === 404) {
        return await request<CsvEvaluateResponse>('/api/v1/evaluate/csv', {
          method: 'POST',
          body: JSON.stringify(payload),
        });
      }
      throw err;
    }
  },

  /**
   * Inspect and adapt CSV schema without running heavy model inference.
   */
  async validateCsvSchema(payload: {
    csv_content: string;
    dataset?: SupportedDataset;
    filename?: string;
  }): Promise<SchemaValidationResponse> {
    try {
      return await request<SchemaValidationResponse>('/evaluate/validate-schema', {
        method: 'POST',
        body: JSON.stringify(payload),
      });
    } catch (err) {
      if (err instanceof ApiError && err.status === 404) {
        return await request<SchemaValidationResponse>('/api/v1/evaluate/validate-schema', {
          method: 'POST',
          body: JSON.stringify(payload),
        });
      }
      throw err;
    }
  },

  /**
   * Phase 17: Fetch verified high-level research overview.
   */
  async getResearchOverview(): Promise<ResearchOverviewResponse> {
    return request<ResearchOverviewResponse>('/research/overview');
  },

  /**
   * Phase 17: Fetch master benchmark matrix across all 27 models.
   */
  async getResearchModels(): Promise<BenchmarkModelRecord[]> {
    try {
      const res = await request<BenchmarkModelRecord[]>('/research/models');
      if (Array.isArray(res) && res.length > 0) {
        return res;
      }
    } catch (e) {
      console.warn('Network request for research models failed, using authoritative fallback data:', e);
    }
    return FALLBACK_BENCHMARK_MODELS;
  },

  /**
   * Phase 17: Fetch 120 round-by-round convergence datapoints.
   */
  async getConvergenceCurves(): Promise<ConvergenceCurvePoint[]> {
    return request<ConvergenceCurvePoint[]>('/research/curves');
  },

  /**
   * Phase 17: Fetch operational threshold tradeoffs.
   */
  async getThresholdTradeoffs(): Promise<ThresholdTradeoffPoint[]> {
    return request<ThresholdTradeoffPoint[]>('/research/thresholds');
  },

  /**
   * Phase 17: Fetch empirical ablation summary.
   */
  async getAblationSummary(): Promise<AblationSummaryResponse> {
    return request<AblationSummaryResponse>('/research/ablation');
  },

  /**
   * Phase H: Fetch verified communication costs and bandwidth profiles.
   */
  async getCommunicationEfficiency(): Promise<CommunicationEfficiencyRecord[]> {
    return request<CommunicationEfficiencyRecord[]>('/research/communication');
  },

  /**
   * Phase H: Fetch Non-IID client volume distribution and bank typology analysis.
   */
  async getClientPartitionAnalysis(): Promise<ClientPartitionResponse> {
    return request<ClientPartitionResponse>('/research/clients');
  },

  /**
   * Direct download URL for sample CSV.
   */
  getSampleCsvUrl(dataset: SupportedDataset): string {
    return `${API_BASE}/research/sample-csv/${dataset}`;
  },

  /**
   * Inspect CSV structure, delimiter, encoding, semantic fields and sequence constructibility.
   * Directly targets canonical /api/v1/csv/inspect without duplicate retries on failure.
   */
  async inspectCsv(payload: {
    csv_content: string;
    filename?: string;
  }): Promise<UniversalCsvInspectionResponse> {
    return request<UniversalCsvInspectionResponse>('/api/v1/csv/inspect', {
      method: 'POST',
      body: JSON.stringify(payload),
    });
  },

  /**
   * Check feature compatibility of an uploaded CSV against available models.
   */
  async checkModelCompatibility(payload: {
    csv_content: string;
    filename?: string;
    model_id?: string;
  }): Promise<{ results: ModelCompatibilityResult[] }> {
    return request<{ results: ModelCompatibilityResult[] }>('/api/v1/models/compatibility', {
      method: 'POST',
      body: JSON.stringify(payload),
    });
  },

  /**
   * Predict on a CSV batch using a specific or automatically selected model.
   * Utilizes background job execution when available to eliminate gateway timeouts (HTTP 502/504).
   */
  async predictCsv(
    payload: {
      csv_content: string;
      model_id?: string;
      filename?: string;
      column_mapping?: Record<string, string>;
      row_limit?: number;
      threshold?: number;
      fallback_model?: string;
    },
    onProgress?: (progress: number, phase: string) => void
  ): Promise<CsvEvaluateResponse> {
    // 1. Try background job flow
    let submitRes: {
      status: string;
      job_id: string;
      run_id: string;
      job_status: string;
      progress: number;
      phase: string;
    } | null = null;

    try {
      submitRes = await request<{
        status: string;
        job_id: string;
        run_id: string;
        job_status: string;
        progress: number;
        phase: string;
      }>('/api/v1/predict/jobs', {
        method: 'POST',
        body: JSON.stringify(payload),
      });
    } catch (err) {
      // Only fall back to synchronous /predict/csv if the /predict/jobs route itself is 404
      if (err instanceof ApiError && err.status === 404) {
        submitRes = null;
      } else {
        throw err;
      }
    }

    if (submitRes && submitRes.job_id) {
      const jobId = submitRes.job_id;
      if (onProgress) onProgress(submitRes.progress || 0.1, submitRes.phase || 'queued');

      // Poll job status as long as it is queued/running (no fixed 90s cap for large CSVs)
      let consecutivePollErrors = 0;
      while (true) {
        await new Promise((res) => setTimeout(res, 1500));
        let job: {
          job_id: string;
          status: 'queued' | 'running' | 'complete' | 'failed';
          progress: number;
          phase: string;
          result?: CsvEvaluateResponse;
          error?: string;
          error_phase?: string;
        };
        try {
          job = await request(`/api/v1/predict/jobs/${jobId}`, {}, 2, 20000);
          consecutivePollErrors = 0;
        } catch (pollErr) {
          if (pollErr instanceof ApiError && pollErr.status === 404) {
            throw new ApiError(
              404,
              'Evaluation job was lost because the server restarted or ran out of memory. Please click Retry Now.',
              { jobId }
            );
          }
          consecutivePollErrors += 1;
          if (consecutivePollErrors >= 5) {
            throw pollErr;
          }
          continue;
        }

        if (onProgress) onProgress(job.progress, job.phase);

        if (job.status === 'complete' && job.result) {
          return job.result;
        }
        if (job.status === 'failed') {
          throw new ApiError(
            400,
            job.error || 'Evaluation job failed during execution.',
            { phase: job.error_phase, jobId }
          );
        }
      }
    }

    // Fallback: direct synchronous prediction
    try {
      return await request<CsvEvaluateResponse>('/api/v1/predict/csv', {
        method: 'POST',
        body: JSON.stringify(payload),
      });
    } catch (err) {
      if (err instanceof ApiError && err.status === 404) {
        return await request<CsvEvaluateResponse>('/predict/csv', {
          method: 'POST',
          body: JSON.stringify(payload),
        });
      }
      throw err;
    }
  },

  /**
   * Run multi-model evaluation across all selected models on identical preprocessed sequences.
   */
  async evaluateMultiModel(payload: {
    csv_content: string;
    model_ids?: string[];
    filename?: string;
    row_limit?: number;
    threshold?: number;
    column_mapping?: Record<string, string>;
  }): Promise<import('../types/api').MultiModelEvaluateResponse> {
    try {
      return await request<import('../types/api').MultiModelEvaluateResponse>('/api/v1/evaluate/multi-model', {
        method: 'POST',
        body: JSON.stringify(payload),
      });
    } catch (err) {
      if (err instanceof ApiError && err.status === 404) {
        return await request<import('../types/api').MultiModelEvaluateResponse>('/evaluate/multi-model', {
          method: 'POST',
          body: JSON.stringify(payload),
        });
      }
      throw err;
    }
  },

  /**
   * Train a dynamic custom AML classifier on a user-uploaded labeled CSV (Mode B).
   */
  async trainCustomModel(payload: {
    csv_content: string;
    model_name: string;
    target_column?: string;
    model_type?: string;
    filename?: string;
  }): Promise<CustomModelTrainResponse> {
    try {
      return await request<CustomModelTrainResponse>('/api/v1/train/csv', {
        method: 'POST',
        body: JSON.stringify(payload),
      });
    } catch (err) {
      if (err instanceof ApiError && err.status === 404) {
        return await request<CustomModelTrainResponse>('/train/csv', {
          method: 'POST',
          body: JSON.stringify(payload),
        });
      }
      throw err;
    }
  },

  /**
   * List all registered models in the registry (built-ins + custom trained models).
   */
  async getRegisteredModels(): Promise<RegisteredModelItem[]> {
    let raw: unknown;
    try {
      raw = await request<unknown>('/api/v1/models');
    } catch (err) {
      if (err instanceof ApiError && err.status === 404) {
        raw = await request<unknown>('/models');
      } else {
        throw err;
      }
    }

    let items: unknown[] = [];
    if (Array.isArray(raw)) {
      items = raw;
    } else if (raw && typeof raw === 'object' && 'models' in raw && Array.isArray((raw as { models: unknown[] }).models)) {
      items = (raw as { models: unknown[] }).models;
    }

    return items.map((m: unknown): RegisteredModelItem => {
      const obj = (m && typeof m === 'object' ? m : {}) as Record<string, unknown>;
      const modelId = String(obj.model_id || obj.dataset || 'unknown_model');
      const modelName = String(obj.model_name || obj.dataset || modelId);
      const architecture = String(obj.architecture || 'Neural Network');
      const threshold = typeof obj.decision_threshold === 'number'
        ? obj.decision_threshold
        : typeof obj.frozen_threshold === 'number'
        ? obj.frozen_threshold
        : 0.5;

      return {
        model_id: modelId,
        model_name: modelName,
        architecture,
        feature_columns: Array.isArray(obj.feature_columns) ? (obj.feature_columns as string[]) : [],
        numerical_columns: Array.isArray(obj.numerical_columns) ? (obj.numerical_columns as string[]) : [],
        categorical_columns: Array.isArray(obj.categorical_columns) ? (obj.categorical_columns as string[]) : [],
        version: String(obj.version || '1.0.0'),
        decision_threshold: threshold,
        created_at: String(obj.created_at || ''),
        frozen: Boolean(obj.frozen ?? true),
        is_custom: Boolean(obj.is_custom ?? false),
        metrics: (obj.metrics && typeof obj.metrics === 'object' ? obj.metrics : {}) as Record<string, unknown>,
      };
    });
  },

  /**
   * Get direct prediction CSV download URL.
   */
  getPredictionsDownloadUrl(id: string): string {
    return `${API_BASE}/api/v1/predictions/${id}/download`;
  },

  /**
   * Get direct Flagged Alerts CSV download URL.
   */
  getFlaggedCsvDownloadUrl(id: string, threshold?: number): string {
    const q = threshold !== undefined ? `?threshold=${threshold}` : '';
    return `${API_BASE}/api/v1/predict/download/${id}/flagged${q}`;
  },

  /**
   * Get direct Needs Review (False Negatives) CSV download URL.
   */
  getNeedsReviewCsvDownloadUrl(id: string, threshold?: number): string {
    const q = threshold !== undefined ? `?threshold=${threshold}` : '';
    return `${API_BASE}/api/v1/predict/download/${id}/needs-review${q}`;
  },

  /**
   * Get direct Legitimate (Non-Flagged / Cleared) CSV download URL.
   */
  getLegitimateCsvDownloadUrl(id: string, threshold?: number): string {
    const q = threshold !== undefined ? `?threshold=${threshold}` : '';
    return `${API_BASE}/api/v1/predict/download/${id}/legitimate${q}`;
  },

  /**
   * Get direct evaluation PDF report download URL with selected operating mode and filter.
   */
  getPdfReportUrl(id: string, operatingPoint?: string, filter?: string): string {
    const params = new URLSearchParams();
    if (operatingPoint) params.set('operating_point', operatingPoint);
    if (filter) params.set('filter', filter);
    const q = params.toString();
    return `${API_BASE}/api/v1/predict/report/${id}.pdf${q ? `?${q}` : ''}`;
  },

  /**
   * Run validation-only IID + Non-IID ensemble grid search and single-pass test evaluation.
   */
  async evaluateEnsemble(payload: {
    dataset?: SupportedDataset;
    alpha_step?: number;
    objective?: string;
    cost_fp?: number;
    cost_fn?: number;
    min_recall?: number;
  }): Promise<any> {
    try {
      return await request<any>('/api/v1/ensemble/evaluate', {
        method: 'POST',
        body: JSON.stringify(payload),
      });
    } catch (err) {
      if (err instanceof ApiError && err.status === 404) {
        return await request<any>('/ensemble/evaluate', {
          method: 'POST',
          body: JSON.stringify(payload),
        });
      }
      throw err;
    }
  },

  /**
   * Get cached ensemble summary and audit metrics for a dataset.
   */
  async getEnsembleSummary(dataset: SupportedDataset = 'HI-Small'): Promise<any> {
    try {
      return await request<any>(`/api/v1/ensemble/summary?dataset=${dataset}`);
    } catch (err) {
      if (err instanceof ApiError && err.status === 404) {
        return await request<any>(`/ensemble/summary?dataset=${dataset}`);
      }
      throw err;
    }
  },

  /**
   * Fetch list of saved historical prediction runs from backend artifacts.
   */
  async getPredictionHistory(): Promise<any[]> {
    try {
      return await request<any[]>('/api/v1/predictions/history');
    } catch (err) {
      if (err instanceof ApiError && err.status === 404) {
        return await request<any[]>('/predictions/history');
      }
      throw err;
    }
  },

  /**
   * Fetch a saved prediction run by its run ID.
   */
  async getPredictionRun(predictionId: string): Promise<CsvEvaluateResponse> {
    try {
      return await request<CsvEvaluateResponse>(`/api/v1/predictions/${predictionId}`);
    } catch (err) {
      if (err instanceof ApiError && err.status === 404) {
        return await request<CsvEvaluateResponse>(`/predictions/${predictionId}`);
      }
      throw err;
    }
  },

  /**
   * Clear in-memory prediction cache.
   */
  async clearPredictionHistory(): Promise<{ status: string; message: string }> {
    try {
      return await request<{ status: string; message: string }>('/api/v1/predictions/clear-history', {
        method: 'POST',
      });
    } catch (err) {
      if (err instanceof ApiError && err.status === 404) {
        return await request<{ status: string; message: string }>('/predictions/clear-history', {
          method: 'POST',
        });
      }
      throw err;
    }
  },
};
