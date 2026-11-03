"""Inference and explainability routes for single and batch predictions."""

from __future__ import annotations

import uuid
from fastapi import APIRouter, HTTPException, status

try:
    from interfaces.api.schemas.predict import (
        BatchPredictRequest,
        BatchPredictResponse,
        PredictRequest,
        PredictResponse,
    )
    from interfaces.api.schemas.explain import ExplainRequest, ExplainResponse
    from interfaces.api.services.model_service import ModelService, ModelIntegrityError
    from interfaces.api.services.explain_service import ExplanationService
except ModuleNotFoundError:
    from backend.interfaces.api.schemas.predict import (
        BatchPredictRequest,
        BatchPredictResponse,
        PredictRequest,
        PredictResponse,
    )
    from backend.interfaces.api.schemas.explain import ExplainRequest, ExplainResponse
    from backend.interfaces.api.services.model_service import ModelService, ModelIntegrityError
    from backend.interfaces.api.services.explain_service import ExplanationService

router = APIRouter(tags=["Inference & Explainability"])


@router.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest) -> PredictResponse:
    """Predict laundering risk for a single transaction sequence."""
    model_service = ModelService()
    try:
        prob, threshold, decision, latency_ms = model_service.predict_sequence(
            dataset=request.dataset, sequence=request.sequence
        )
    except ModelIntegrityError as e:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(e))
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Inference evaluation failed: {str(e)}",
        )

    req_id = f"pred_{uuid.uuid4().hex[:12]}"
    seq_len = request.sequence.length or len(request.sequence.features)

    return PredictResponse(
        request_id=req_id,
        dataset=request.dataset,
        model_name=request.model_name or "champion",
        predicted_probability=prob,
        decision_threshold=threshold,
        decision=decision,  # type: ignore[arg-type]
        sample_id=request.sequence.sample_id,
        sequence_length=seq_len,
        inference_time_ms=latency_ms,
    )


@router.post("/predict/batch", response_model=BatchPredictResponse)
def predict_batch(request: BatchPredictRequest) -> BatchPredictResponse:
    """Evaluate a bounded batch of sequences (max 64 items)."""
    model_service = ModelService()
    try:
        item_results, batch_latency_ms = model_service.predict_batch(
            dataset=request.dataset, items=request.items
        )
    except ModelIntegrityError as e:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(e))
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Batch execution failed: {str(e)}",
        )

    successful = sum(1 for r in item_results if r.success)
    failed = len(item_results) - successful

    return BatchPredictResponse(
        request_id=f"batch_{uuid.uuid4().hex[:12]}",
        dataset=request.dataset,
        model_name=request.model_name or "champion",
        total_items=len(item_results),
        successful_items=successful,
        failed_items=failed,
        predictions=item_results,
        batch_inference_time_ms=batch_latency_ms,
    )


@router.post("/explain", response_model=ExplainResponse)
def explain(request: ExplainRequest) -> ExplainResponse:
    """Generate Integrated Gradients feature attributions for a transaction sequence."""
    explain_service = ExplanationService()
    try:
        response = explain_service.explain_sequence(request)
        return response
    except ModelIntegrityError as e:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(e))
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Explanation calculation failed: {str(e)}",
        )
