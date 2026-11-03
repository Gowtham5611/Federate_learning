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

const resolveApiBase = (): string => {
  if (import.meta.env.VITE_API_BASE_URL) {
    return import.meta.env.VITE_API_BASE_URL.replace(/\/+$/, '');
  }
  if (typeof window !== 'undefined' && (window.location.hostname === 'localhost' || window.location.hostname === '127.0.0.1')) {
    return 'http://127.0.0.1:8000';
  }
  return '/api';
};

const API_BASE = resolveApiBase();

export class ApiError extends Error {
  constructor(
    public status: number,
    message: string,
    public details?: unknown
  ) {
    super(message);
    this.name = 'ApiError';
  }
}

async function request<T>(endpoint: string, options: RequestInit = {}): Promise<T> {
  const cleanEndpoint = endpoint.startsWith('/') ? endpoint : `/${endpoint}`;
  const url = `${API_BASE}${cleanEndpoint}`;
  const headers = {
    'Content-Type': 'application/json',
    ...(options.headers || {}),
  };

  try {
    const response = await fetch(url, { ...options, headers });

    if (!response.ok) {
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
        // Response was not JSON
      }

      if (response.status === 404) {
        errorMessage = `API route not found at ${url}. Ensure the FastAPI server is running with all route handlers registered.`;
      } else if (response.status === 503) {
        errorMessage = `Backend model service unavailable: ${errorMessage}`;
      }

      throw new ApiError(response.status, errorMessage, errorDetails);
    }

    return (await response.json()) as T;
  } catch (err) {
    if (err instanceof ApiError) {
      throw err;
    }
    throw new ApiError(
      0,
      `Backend connection error at ${url}. Ensure the FastAPI server is running on http://127.0.0.1:8000.`,
      err
    );
  }
}

export const apiService = {
  /**
   * Check API liveness and model loading state.
   */
  async getHealth(): Promise<HealthResponse> {
    return request<HealthResponse>('/health');
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
    return request<ExplainResponse>('/explain', {
      method: 'POST',
      body: JSON.stringify(payload),
    });
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
    return request<BenchmarkModelRecord[]>('/research/models');
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
   */
  async inspectCsv(payload: {
    csv_content: string;
    filename?: string;
  }): Promise<UniversalCsvInspectionResponse> {
    try {
      return await request<UniversalCsvInspectionResponse>('/api/v1/csv/inspect', {
        method: 'POST',
        body: JSON.stringify(payload),
      });
    } catch (err) {
      if (err instanceof ApiError && err.status === 404) {
        return await request<UniversalCsvInspectionResponse>('/csv/inspect', {
          method: 'POST',
          body: JSON.stringify(payload),
        });
      }
      throw err;
    }
  },

  /**
   * Check feature compatibility of an uploaded CSV against available models.
   */
  async checkModelCompatibility(payload: {
    csv_content: string;
    filename?: string;
    model_id?: string;
  }): Promise<{ results: ModelCompatibilityResult[] }> {
    try {
      return await request<{ results: ModelCompatibilityResult[] }>('/api/v1/models/compatibility', {
        method: 'POST',
        body: JSON.stringify(payload),
      });
    } catch (err) {
      if (err instanceof ApiError && err.status === 404) {
        return await request<{ results: ModelCompatibilityResult[] }>('/models/compatibility', {
          method: 'POST',
          body: JSON.stringify(payload),
        });
      }
      throw err;
    }
  },

  /**
   * Predict on a CSV batch using a specific or automatically selected model.
   */
  async predictCsv(payload: {
    csv_content: string;
    model_id?: string;
    filename?: string;
    column_mapping?: Record<string, string>;
    row_limit?: number;
    threshold?: number;
  }): Promise<CsvEvaluateResponse> {
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
};
