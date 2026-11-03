"""Read-only routes for service health, models catalog, and dataset metadata."""

from __future__ import annotations

import datetime
import os
from pathlib import Path
from typing import List

from fastapi import APIRouter, HTTPException, status

try:
    from infrastructure.data.sequences.processed_feature_metadata import ProcessedFeatureMetadata
    from interfaces.api.schemas.common import (
        FROZEN_THRESHOLDS,
        SUPPORTED_MODELS_BY_DATASET,
        SupportedDataset,
    )
    from interfaces.api.schemas.health import (
        DatasetMetadataResponse,
        HealthResponse,
        ModelMetadataResponse,
    )
    from interfaces.api.services.model_service import ModelService
except ModuleNotFoundError:
    from backend.infrastructure.data.sequences.processed_feature_metadata import ProcessedFeatureMetadata
    from backend.interfaces.api.schemas.common import (
        FROZEN_THRESHOLDS,
        SUPPORTED_MODELS_BY_DATASET,
        SupportedDataset,
    )
    from backend.interfaces.api.schemas.health import (
        DatasetMetadataResponse,
        HealthResponse,
        ModelMetadataResponse,
    )
    from backend.interfaces.api.services.model_service import ModelService

router = APIRouter(tags=["Health & Metadata"])

BASE_DIR = Path(__file__).resolve().parents[4]


def _get_expanded_feature_names(dataset: str) -> List[str]:
    """Retrieve canonical expanded 57-feature names from processed metadata."""
    data_dir_env = os.getenv("FEDSECUREAI_DATA_DIR")
    if data_dir_env:
        data_dir = Path(data_dir_env) / f"processed/ibm_aml/{dataset}"
        if not data_dir.exists():
            data_dir = Path(data_dir_env) / dataset
    else:
        data_dir = BASE_DIR / f"data/processed/ibm_aml/{dataset}"

    if not data_dir.exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Dataset metadata directory for '{dataset}' not found.",
        )
    meta = ProcessedFeatureMetadata.read(data_dir)
    expansions = {nominal.column: nominal for nominal in meta.nominal_columns}
    names = []
    for col in meta.feature_columns:
        if col in expansions:
            nom = expansions[col]
            for i in range(nom.vocabulary_size):
                names.append(f"{col}__cat_{i}")
        else:
            names.append(col)
    return names


@router.get("/health", response_model=HealthResponse)
def get_health() -> HealthResponse:
    """Check service liveness and model readiness state.

    Distinguishes process liveness (HTTP 200) from whether models are loaded and ready.
    """
    model_service = ModelService()
    is_ready = model_service.is_ready()
    loaded_models = model_service.get_loaded_models()

    return HealthResponse(
        status="ok",
        app_name="FedSecureAI Inference Service",
        app_version="1.0.0",
        loaded_models=loaded_models,
        device="cpu",
        readiness="ready" if is_ready else "initializing",
        timestamp=datetime.datetime.now(datetime.timezone.utc).isoformat(),
    )


@router.get("/models", response_model=List[ModelMetadataResponse])
def get_models() -> List[ModelMetadataResponse]:
    """List all verified, supported federated AML models."""
    model_service = ModelService()
    models_list: List[ModelMetadataResponse] = []

    for dataset, model_name in SUPPORTED_MODELS_BY_DATASET.items():
        try:
            meta = model_service.get_model_metadata(dataset)
            models_list.append(
                ModelMetadataResponse(
                    dataset=dataset,
                    model_name=meta["model_name"],
                    architecture=meta["architecture"],
                    parameter_count=meta["parameter_count"],
                    sequence_length=meta["sequence_length"],
                    feature_dimension=meta["feature_dimension"],
                    frozen_threshold=meta["frozen_threshold"],
                    checkpoint_verified=meta["checkpoint_verified"],
                    supported_explanation_methods=["integrated_gradients"],
                )
            )
        except Exception:
            # Model not yet loaded; omit or report unverified
            pass

    return models_list


@router.get("/metadata/{dataset}", response_model=DatasetMetadataResponse)
def get_dataset_metadata(dataset: str) -> DatasetMetadataResponse:
    """Retrieve feature schema and frozen threshold for a dataset."""
    if dataset not in SUPPORTED_MODELS_BY_DATASET:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Unsupported dataset '{dataset}'. Valid datasets are {list(SUPPORTED_MODELS_BY_DATASET.keys())}.",
        )

    feature_names = _get_expanded_feature_names(dataset)
    threshold = FROZEN_THRESHOLDS[dataset]
    active_model = SUPPORTED_MODELS_BY_DATASET[dataset]

    return DatasetMetadataResponse(
        dataset=dataset,
        feature_columns=feature_names,
        raw_feature_count=23,
        expanded_feature_count=len(feature_names),
        sequence_length=10,
        frozen_threshold=threshold,
        active_model_name=active_model,
        preprocessing_version="1.0.0",
    )
