"""Universal CSV inspection, schema compatibility, prediction, training, and evaluation routes."""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException, Response, status
from pydantic import BaseModel, Field

try:
    from interfaces.api.services.csv_evaluation_service import CsvEvaluationService
    from infrastructure.models.registry.universal_model_registry import UniversalModelRegistry
    from infrastructure.data.validators.csv_data_quality_auditor import CsvDataQualityAuditor
    from infrastructure.aml.investigation_report_generator import InvestigationReportGenerator
    from application.services.ensemble_experiment_service import EnsembleExperimentService
except ModuleNotFoundError:
    from backend.interfaces.api.services.csv_evaluation_service import CsvEvaluationService
    from backend.infrastructure.models.registry.universal_model_registry import UniversalModelRegistry
    from backend.infrastructure.data.validators.csv_data_quality_auditor import CsvDataQualityAuditor
    from backend.infrastructure.aml.investigation_report_generator import InvestigationReportGenerator
    from backend.application.services.ensemble_experiment_service import EnsembleExperimentService

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Universal CSV & AML Model Engine"])


# ==============================================================================
# Pydantic Request Models
# ==============================================================================


class CsvInspectRequest(BaseModel):
    """Request schema for inspecting an uploaded CSV file."""

    csv_content: str = Field(..., description="Raw CSV string content (UTF-8, Latin-1, with or without BOM)")
    filename: Optional[str] = Field("upload.csv", description="Origin filename for logging and traceability")


class ModelCompatibilityRequest(BaseModel):
    """Request schema for checking schema compatibility against a model."""

    csv_content: str = Field(..., description="Raw CSV string content")
    model_id: Optional[str] = Field("auto", description="Target model identifier (e.g. 'HI-Small', 'LI-Medium', or 'auto')")
    filename: Optional[str] = Field("upload.csv", description="Filename")
    user_mappings: Optional[Dict[str, str]] = Field(None, description="Optional custom column mappings")
    column_mapping: Optional[Dict[str, str]] = Field(None, description="Alias for user_mappings")


class CsvPredictRequest(BaseModel):
    """Request schema for executing prediction on a CSV file."""

    csv_content: str = Field(..., description="Raw CSV string content")
    model_id: Optional[str] = Field("auto", description="Target model to use for prediction (or 'auto')")
    dataset: Optional[str] = Field(None, description="Legacy dataset pipeline parameter")
    filename: Optional[str] = Field("upload.csv", description="Origin filename")
    row_limit: Optional[int] = Field(5000, ge=1, le=1000000, description="Max rows to process (up to 1,000,000)")
    user_mappings: Optional[Dict[str, str]] = Field(None, description="Optional field overrides")
    column_mapping: Optional[Dict[str, str]] = Field(None, description="Alias for user_mappings")
    tenant_id: Optional[str] = Field("default_tenant", description="Tenant or session ID for isolated persistent account history")
    enable_persistent_history: Optional[bool] = Field(False, description="Whether to maintain causal account history across multiple uploads")
    threshold: Optional[float] = Field(None, ge=0.01, le=0.99, description="Optional custom decision threshold override")
    fallback_model: Optional[str] = Field(None, description="Optional opt-in fallback model ('tabular') for zero-history rows")


class MultiModelEvaluateRequest(BaseModel):
    """Request schema for evaluating an uploaded CSV against multiple models simultaneously."""

    csv_content: str = Field(..., description="Raw CSV string content")
    model_ids: Optional[List[str]] = Field(None, description="Optional subset of models to compare")
    filename: Optional[str] = Field("upload.csv", description="Origin filename")
    row_limit: Optional[int] = Field(10000, ge=1, le=1000000, description="Max rows to evaluate")
    threshold: Optional[float] = Field(None, ge=0.01, le=0.99, description="Custom classification threshold override")
    user_mappings: Optional[Dict[str, str]] = Field(None, description="Optional field overrides")
    column_mapping: Optional[Dict[str, str]] = Field(None, description="Alias for user_mappings")


class ThresholdAnalysisRequest(BaseModel):
    """Request schema for validation-only threshold tuning."""

    dataset: Optional[str] = Field("HI-Small", description="Dataset identifier ('HI-Small' or 'LI-Medium')")
    objective: Optional[str] = Field("max_f1", description="Optimization objective: 'max_f1', 'max_f2', 'target_recall', 'cost_sensitive'")
    target_recall: Optional[float] = Field(0.80, description="Target recall when objective is 'target_recall'")
    cost_fp: Optional[float] = Field(1.0, description="Cost of false positive when objective is 'cost_sensitive'")
    cost_fn: Optional[float] = Field(10.0, description="Cost of false negative when objective is 'cost_sensitive'")


class EnsembleEvaluateRequest(BaseModel):
    """Request schema for executing Phase F/G IID + Non-IID ensemble experiment."""

    dataset: Optional[str] = Field("HI-Small", description="Dataset identifier: 'HI-Small' or 'LI-Medium'")
    alpha_step: Optional[float] = Field(0.05, ge=0.01, le=0.5, description="Grid step size for alpha exploration")
    objective: Optional[str] = Field("max_pr_auc", description="Objective: 'max_pr_auc', 'max_f1', 'max_f2', 'cost_sensitive', 'recall_constrained'")
    cost_fp: Optional[float] = Field(1.0, description="Cost of false positive in cost-sensitive objective")
    cost_fn: Optional[float] = Field(10.0, description="Cost of false negative in cost-sensitive objective")
    min_recall: Optional[float] = Field(None, description="Minimum recall constraint when objective is recall_constrained")


class CsvTrainRequest(BaseModel):
    """Request schema for training a custom model from a labeled CSV file."""

    csv_content: str = Field(..., description="Raw CSV string containing training data with binary labels")
    model_name: str = Field(..., description="User-specified name for the new model")
    target_column: Optional[str] = Field(None, description="Target label column (auto-detected if omitted)")
    feature_columns: Optional[List[str]] = Field(None, description="Feature columns to use")
    model_type: Optional[str] = Field("HistGradientBoosting", description="Model architecture type ('HistGradientBoosting' or 'RandomForest')")
    filename: Optional[str] = Field("train.csv", description="Uploaded training filename")


# Legacy schema for existing tests
class CsvEvaluateRequest(BaseModel):
    """Request schema for legacy CSV transaction batch evaluation."""

    csv_content: str = Field(..., description="Raw CSV string content (UTF-8)")
    dataset: Optional[str] = Field(None, description="Optional target pipeline ('HI-Small' or 'LI-Medium'). Auto-inferred if omitted.")
    filename: Optional[str] = Field("upload.csv", description="Origin filename")
    row_limit: int = Field(5000, ge=1, le=50000, description="Max rows to process")


class CsvDataQualityRequest(BaseModel):
    """Request schema for comprehensive CSV data quality profiling."""

    csv_content: str = Field(..., description="Raw CSV string content")
    filename: Optional[str] = Field("upload.csv", description="Uploaded filename")
    target_column: Optional[str] = Field(None, description="Optional target label column name")


class InvestigationReportRequest(BaseModel):
    """Request schema for SAR-style draft investigation report export."""

    transactions: List[Dict[str, Any]] = Field(..., description="List of evaluated transactions to include")
    analyst_id: Optional[str] = Field("analyst_internal", description="ID of analyst generating report")
    model_version: Optional[str] = Field("HI-Small_LSTM_v1.0", description="Model version used for scoring")
    decision_threshold: Optional[float] = Field(0.337784, description="Decision threshold applied")
    analyst_notes: Optional[str] = Field("Draft investigative triage report.", description="Analyst narrative notes")
    format: Optional[str] = Field("json", description="Export format: 'json' or 'csv'")


# ==============================================================================
# 1. CSV Inspection & Schema Detection
# ==============================================================================


@router.post("/csv/inspect")
@router.post("/api/csv/inspect")
@router.post("/api/v1/csv/inspect")
def inspect_csv(request: CsvInspectRequest) -> Dict[str, Any]:
    """Inspect CSV encoding, delimiter, columns, statistical types, and financial semantics."""
    service = CsvEvaluationService()
    fname = request.filename or "upload.csv"
    try:
        csv_bytes = request.csv_content.encode("utf-8")
        result = service.inspect_csv_bytes(csv_bytes, filename=fname)
        return result
    except Exception as e:
        logger.error("Error during CSV inspection: %s", e, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Failed to inspect CSV: {str(e)}",
        )


# ==============================================================================
# 2. Model Compatibility Checker
# ==============================================================================


@router.post("/models/compatibility")
@router.post("/api/models/compatibility")
@router.post("/api/v1/models/compatibility")
def check_compatibility(request: ModelCompatibilityRequest) -> Dict[str, Any]:
    """Verify feature compatibility between uploaded CSV and selected model."""
    service = CsvEvaluationService()
    fname = request.filename or "upload.csv"
    try:
        csv_bytes = request.csv_content.encode("utf-8")
        mappings = request.user_mappings or request.column_mapping
        result = service.check_compatibility(
            csv_bytes=csv_bytes,
            model_id=request.model_id or "auto",
            filename=fname,
            user_mappings=mappings,
        )
        return result
    except Exception as e:
        logger.error("Error during compatibility check: %s", e, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Compatibility check failed: {str(e)}",
        )


# ==============================================================================
# 3. Model Registry Listing
# ==============================================================================


@router.get("/models")
@router.get("/api/models")
@router.get("/api/v1/models")
def list_available_models() -> Dict[str, Any]:
    """List all registered AML models (both built-in federated models and custom trained models)."""
    registry = UniversalModelRegistry()
    models = registry.list_models()
    return {
        "status": "success",
        "total_models": len(models),
        "models": models,
    }


# ==============================================================================
# 4. Universal Prediction on CSV
# ==============================================================================


@router.post("/predict/csv")
@router.post("/api/predict/csv")
@router.post("/api/v1/predict/csv")
def predict_csv(request: CsvPredictRequest) -> Dict[str, Any]:
    """Generate AML risk predictions on an uploaded CSV file using selected model."""
    service = CsvEvaluationService()
    fname = request.filename or "upload.csv"

    if not request.csv_content or len(request.csv_content.strip()) == 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded CSV content is empty.",
        )

    try:
        csv_bytes = request.csv_content.encode("utf-8")
        mappings = request.user_mappings or request.column_mapping
        result = service.evaluate_csv_bytes(
            csv_bytes=csv_bytes,
            dataset=request.dataset,
            filename=fname,
            row_limit=request.row_limit,
            model_id=request.model_id or request.dataset or "auto",
            user_mappings=mappings,
            tenant_id=request.tenant_id or "default_tenant",
            enable_persistent_history=bool(request.enable_persistent_history),
            threshold=request.threshold,
            fallback_model=request.fallback_model,
        )
        return result
    except ValueError as ve:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(ve),
        )
    except Exception as e:
        logger.error("Error during CSV prediction: %s", e, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Prediction failed: {str(e)}",
        )


@router.post("/evaluate/multi-model")
@router.post("/api/evaluate/multi-model")
@router.post("/api/v1/evaluate/multi-model")
def evaluate_multi_model(request: MultiModelEvaluateRequest) -> Dict[str, Any]:
    """Evaluate an uploaded CSV across all selected models on identical preprocessed sequences."""
    service = CsvEvaluationService()
    fname = request.filename or "upload.csv"

    if not request.csv_content or len(request.csv_content.strip()) == 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded CSV content is empty.",
        )

    try:
        csv_bytes = request.csv_content.encode("utf-8")
        mappings = request.user_mappings or request.column_mapping
        return service.evaluate_multi_model_csv(
            csv_bytes=csv_bytes,
            model_ids=request.model_ids,
            filename=fname,
            threshold=request.threshold,
            row_limit=request.row_limit,
            user_mappings=mappings,
        )
    except ValueError as ve:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(ve),
        )
    except Exception as e:
        logger.error("Error during multi-model evaluation: %s", e, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Multi-model evaluation failed: {str(e)}",
        )


# ==============================================================================
# 5. Dynamic Custom Model Training (Mode B)
# ==============================================================================


@router.post("/train/csv")
@router.post("/api/train/csv")
@router.post("/api/v1/train/csv")
def train_model_from_csv(request: CsvTrainRequest) -> Dict[str, Any]:
    """Train and register a new AML classifier from an uploaded labeled CSV file."""
    service = CsvEvaluationService()
    fname = request.filename or "train.csv"

    if not request.csv_content or len(request.csv_content.strip()) == 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded training CSV content is empty.",
        )

    try:
        csv_bytes = request.csv_content.encode("utf-8")
        result = service.train_custom_model_from_csv(
            csv_bytes=csv_bytes,
            model_name=request.model_name,
            target_column=request.target_column,
            feature_columns=request.feature_columns,
            model_type=request.model_type or "HistGradientBoosting",
            filename=fname,
        )
        return result
    except ValueError as ve:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(ve),
        )
    except Exception as e:
        logger.error("Error during custom model training: %s", e, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Model training failed: {str(e)}",
        )


# ==============================================================================
# 6. Download Prediction Results as CSV
# ==============================================================================


@router.get("/predictions/{prediction_id}/download")
@router.get("/predict/download/{prediction_id}")
@router.get("/predict/{prediction_id}/download")
@router.get("/api/predictions/{prediction_id}/download")
@router.get("/api/predict/download/{prediction_id}")
@router.get("/api/predict/{prediction_id}/download")
@router.get("/api/v1/predictions/{prediction_id}/download")
@router.get("/api/v1/predict/download/{prediction_id}")
@router.get("/api/v1/predict/{prediction_id}/download")
def download_prediction_csv(prediction_id: str) -> Response:
    """Download itemized prediction records as a standard CSV file."""
    service = CsvEvaluationService()
    res = service.get_prediction_download(prediction_id)
    if not res:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Prediction session '{prediction_id}' not found.",
        )

    if res[0] == "EXPIRED":
        raise HTTPException(
            status_code=status.HTTP_410_GONE,
            detail=f"Prediction session '{prediction_id}' has deliberately expired or was removed.",
        )

    csv_content, filename = res
    return Response(
        content=csv_content,
        media_type="text/csv",
        headers={"Content-Disposition": f"attachment; filename={filename}"},
    )


@router.post("/predictions/{prediction_id}/expire")
@router.post("/api/v1/predictions/{prediction_id}/expire")
def expire_prediction_csv(prediction_id: str) -> Dict[str, Any]:
    """Deliberately expire a prediction download file to test HTTP 410 semantics."""
    service = CsvEvaluationService()
    success = service.expire_prediction_download(prediction_id)
    if not success:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Prediction session '{prediction_id}' not found.",
        )
    return {
        "status": "success",
        "prediction_id": prediction_id,
        "message": f"Prediction session '{prediction_id}' has been deliberately marked as expired.",
    }


# ==============================================================================
# 7. Legacy Evaluation Endpoints (100% Backward Compatibility)
# ==============================================================================


@router.post("/evaluate/validate-schema")
@router.post("/evaluate/validate-schema/")
@router.post("/api/evaluate/validate-schema")
@router.post("/api/v1/evaluate/validate-schema")
def validate_schema(request: CsvEvaluateRequest) -> Dict[str, Any]:
    """Inspect and adapt CSV schema without running heavy model inference."""
    fname = request.filename or "upload.csv"
    if not request.csv_content or len(request.csv_content.strip()) == 0:
        return {
            "status": "UNSUPPORTED",
            "user_message": "Uploaded file is empty (0 bytes). Please upload a valid CSV file.",
            "format_category": "UNKNOWN",
            "target_pipeline": "HI-Small",
            "decision_threshold": 0.33778378,
            "has_labels": False,
            "field_mappings": {},
            "missing_required_fields": [],
            "conflicting_fields": {},
            "excluded_leakage_fields": [],
            "audit_notes": [],
        }

    service = CsvEvaluationService()
    try:
        csv_bytes = request.csv_content.encode("utf-8")
        result = service.validate_csv_bytes(
            csv_bytes=csv_bytes,
            filename=fname,
            dataset=request.dataset,
        )
        return result
    except Exception as e:
        logger.error("Error during schema validation: %s", e, exc_info=True)
        return {
            "status": "UNSUPPORTED",
            "user_message": f"This CSV cannot be evaluated: {str(e)}",
            "format_category": "UNKNOWN",
            "target_pipeline": "HI-Small",
            "decision_threshold": 0.33778378,
            "has_labels": False,
            "field_mappings": {},
            "missing_required_fields": [],
            "conflicting_fields": {},
            "excluded_leakage_fields": [],
            "audit_notes": [],
        }


@router.post("/evaluate/csv")
@router.post("/evaluate/csv/")
@router.post("/api/evaluate/csv")
@router.post("/api/v1/evaluate/csv")
def evaluate_csv(request: CsvEvaluateRequest) -> Dict[str, Any]:
    """Evaluate a transaction CSV against authoritative federated AML models."""
    fname = request.filename or "upload.csv"
    if not fname.lower().endswith(".csv") and not fname.lower().endswith(".txt"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid file type. Only standard CSV files (.csv) are accepted.",
        )

    if not request.csv_content or len(request.csv_content.strip()) == 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded CSV content is empty.",
        )

    if request.dataset and request.dataset not in ("HI-Small", "LI-Medium"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unknown dataset '{request.dataset}'. Supported pipelines are 'HI-Small' and 'LI-Medium'.",
        )

    service = CsvEvaluationService()
    try:
        csv_bytes = request.csv_content.encode("utf-8")
        result = service.evaluate_csv_bytes(
            csv_bytes=csv_bytes,
            dataset=request.dataset,
            filename=fname,
            row_limit=request.row_limit,
        )
        return result
    except ValueError as ve:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(ve),
        )
    except Exception as e:
        logger.error("Error during CSV evaluation: %s", e, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"CSV processing failed: {str(e)}",
        )


@router.get("/evaluate/csv")
@router.get("/api/v1/evaluate/csv")
def evaluate_csv_info() -> Dict[str, Any]:
    """Information endpoint for the CSV evaluation service."""
    return {
        "status": "ready",
        "description": "FedSecureAI Universal Adaptive CSV Evaluation Engine",
        "supported_methods": ["POST", "GET"],
        "endpoints": [
            "/api/v1/csv/inspect",
            "/api/v1/models/compatibility",
            "/api/v1/predict/csv",
            "/api/v1/train/csv",
            "/api/v1/evaluate/threshold-analysis",
            "/api/v1/evaluate/validate-schema",
            "/api/v1/evaluate/csv",
            "/api/v1/models",
            "/api/v1/predictions/{id}/download",
        ],
        "supported_formats": [
            "IBM AML Raw Format (11 columns)",
            "Processed Feature Format (26+ features)",
            "Synthetic / Mobile Banking Formats (PaySim, generic)",
            "Aliased Transaction Formats (custom column headers)",
            "Custom Tabular AML Datasets (arbitrary columns)",
        ],
        "row_limit_bounds": {"min": 1, "max": 50000, "default": 5000},
    }


@router.post("/evaluate/threshold-analysis")
@router.post("/api/evaluate/threshold-analysis")
@router.post("/api/v1/evaluate/threshold-analysis")
def analyze_thresholds_endpoint(request: ThresholdAnalysisRequest) -> Dict[str, Any]:
    """Perform validation-only threshold analysis across candidate thresholds."""
    service = CsvEvaluationService()
    try:
        return service.analyze_validation_thresholds(
            dataset=request.dataset or "HI-Small",
            objective=request.objective or "max_f1",
            target_recall=request.target_recall or 0.80,
            cost_fp=request.cost_fp or 1.0,
            cost_fn=request.cost_fn or 10.0,
        )
    except Exception as e:
        logger.error("Error during threshold analysis: %s", e, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Threshold analysis failed: {str(e)}",
        )


# ==============================================================================
# 7. CSV Data Quality Report (Phase B)
# ==============================================================================


@router.post("/csv/data-quality")
@router.post("/api/v1/csv/data-quality")
def audit_csv_data_quality(request: CsvDataQualityRequest) -> Dict[str, Any]:
    """Execute deep data quality profiling on an uploaded transaction CSV."""
    if not request.csv_content or len(request.csv_content.strip()) == 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="CSV content is empty.",
        )

    service = CsvEvaluationService()
    try:
        csv_bytes = request.csv_content.encode("utf-8")
        df, _ = service.reader.read(csv_bytes)
        auditor = CsvDataQualityAuditor()
        report = auditor.audit_dataframe(
            df=df,
            filename=request.filename or "uploaded_transactions.csv",
            target_column=request.target_column,
        )
        return {
            "filename": report.filename,
            "total_rows": report.total_rows,
            "total_columns": report.total_columns,
            "duplicate_rows_count": report.duplicate_rows_count,
            "invalid_timestamps_count": report.invalid_timestamps_count,
            "invalid_numerical_values_count": report.invalid_numerical_values_count,
            "total_missing_cells": report.total_missing_cells,
            "overall_missing_pct": report.overall_missing_pct,
            "schema_status": report.schema_status,
            "detected_target_column": report.detected_target_column,
            "has_labels": report.has_labels,
            "label_distribution": report.label_distribution,
            "preprocessing_actions": report.preprocessing_actions,
            "rejected_rows_count": report.rejected_rows_count,
            "rejection_reasons": report.rejection_reasons,
            "is_ready_for_inference": report.is_ready_for_inference,
            "audit_timestamp": report.audit_timestamp,
        }
    except Exception as e:
        logger.error("Error during CSV data quality audit: %s", e, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Data quality audit failed: {str(e)}",
        )


# ==============================================================================
# 8. SAR-Style Draft Investigation Report Export (Phase P)
# ==============================================================================


@router.post("/investigation/report")
@router.post("/api/v1/investigation/report")
def export_investigation_report(request: InvestigationReportRequest):
    """Generate internal Suspicious Activity Investigation Report — Draft in JSON or CSV format."""
    generator = InvestigationReportGenerator()
    report_data = generator.generate_report(
        transactions=request.transactions,
        analyst_id=request.analyst_id or "analyst_internal",
        model_version=request.model_version or "HI-Small_LSTM_v1.0",
        decision_threshold=request.decision_threshold or 0.337784,
        analyst_notes=request.analyst_notes or "Draft triage for compliance review.",
    )

    if (request.format or "json").lower() == "csv":
        csv_text = generator.export_csv_format(report_data)
        report_id = report_data["summary"]["report_id"]
        return Response(
            content=csv_text,
            media_type="text/csv",
            headers={"Content-Disposition": f'attachment; filename="{report_id}.csv"'},
        )

    return report_data


# ==============================================================================
# 9. Real IID + Non-IID Hybrid Ensemble Optimization & Audit (Phase F)
# ==============================================================================


@router.post("/ensemble/evaluate")
@router.post("/api/v1/ensemble/evaluate")
def run_ensemble_evaluation(request: EnsembleEvaluateRequest) -> Dict[str, Any]:
    """Execute validation-only alpha grid search and single-pass test evaluation."""
    service = EnsembleExperimentService()
    try:
        outcome = service.run_ensemble_experiment(
            dataset=request.dataset or "HI-Small",
            alpha_step=request.alpha_step or 0.05,
            objective=request.objective or "max_pr_auc",
            cost_fp=request.cost_fp or 1.0,
            cost_fn=request.cost_fn or 10.0,
            min_recall=request.min_recall,
        )
        return {
            "status": "success",
            "dataset": outcome.dataset,
            "objective": outcome.objective,
            "alpha_step": outcome.alpha_step,
            "best_alpha": outcome.best_alpha,
            "frozen_threshold": outcome.frozen_threshold,
            "ensemble_improved": outcome.ensemble_improved,
            "scientific_rationale": outcome.scientific_rationale,
            "delta_vs_iid": outcome.delta_vs_iid,
            "validation_metrics": outcome.validation_metrics,
            "test_metrics": outcome.test_metrics,
            "iid_standalone_test_metrics": outcome.iid_standalone_test_metrics,
            "non_iid_standalone_test_metrics": outcome.non_iid_standalone_test_metrics,
            "alpha_candidates": [
                {
                    "alpha": c.alpha,
                    "val_pr_auc": round(c.val_pr_auc, 6),
                    "val_roc_auc": round(c.val_roc_auc, 6),
                    "val_f1": round(c.val_f1, 6),
                    "val_f2": round(c.val_f2, 6),
                    "val_precision": round(c.val_precision, 6),
                    "val_recall": round(c.val_recall, 6),
                    "val_threshold": round(c.val_threshold, 6),
                    "val_brier_score": round(c.val_brier_score, 6),
                    "val_ece": round(c.val_ece, 6),
                    "val_cost": round(c.val_cost, 2),
                    "selected": c.selected,
                }
                for c in outcome.alpha_candidates
            ],
        }
    except Exception as e:
        logger.error("Ensemble experiment execution failed: %s", e, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Ensemble evaluation failed: {str(e)}",
        )


@router.get("/ensemble/summary")
@router.get("/api/v1/ensemble/summary")
def get_ensemble_summary(dataset: str = "HI-Small") -> Dict[str, Any]:
    """Retrieve verified frozen ensemble configuration and audit metrics."""
    service = EnsembleExperimentService()
    try:
        outcome = service.run_ensemble_experiment(
            dataset=dataset,
            alpha_step=0.1,
            objective="max_pr_auc",
            subsample_limit=10000,
        )
        return {
            "status": "success",
            "dataset": outcome.dataset,
            "best_alpha": outcome.best_alpha,
            "frozen_threshold": outcome.frozen_threshold,
            "ensemble_improved": outcome.ensemble_improved,
            "scientific_rationale": outcome.scientific_rationale,
            "delta_vs_iid": outcome.delta_vs_iid,
            "test_metrics": outcome.test_metrics,
            "iid_standalone_test_metrics": outcome.iid_standalone_test_metrics,
            "non_iid_standalone_test_metrics": outcome.non_iid_standalone_test_metrics,
        }
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Failed to retrieve ensemble summary: {str(e)}",
        )


