"""Model registry module for built-in and dynamic models."""

from infrastructure.models.registry.universal_model_registry import (
    BUILTIN_MODELS,
    ModelSchemaMetadata,
    UniversalModelRegistry,
)

__all__ = ["UniversalModelRegistry", "ModelSchemaMetadata", "BUILTIN_MODELS"]

