"""Tests for Phase 14A secure deployment packaging, configuration, and constraints."""

from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import patch

import pytest
import yaml

try:
    from interfaces.api.services.model_service import ModelService, MODEL_REGISTRY, ModelIntegrityError
    from interfaces.api.routes.health import _get_expanded_feature_names
except ModuleNotFoundError:
    from backend.interfaces.api.services.model_service import ModelService, MODEL_REGISTRY, ModelIntegrityError
    from backend.interfaces.api.routes.health import _get_expanded_feature_names

BASE_DIR = Path(__file__).resolve().parents[1]


# 1. Checkpoint Path Environment Override Tests
def test_checkpoint_path_default_resolution():
    ms = ModelService()
    default_path = MODEL_REGISTRY["HI-Small"]["checkpoint_path"]
    with patch.dict(os.environ, {}, clear=True):
        resolved = ms._resolve_checkpoint_path("HI-Small", default_path)
        assert resolved == default_path


def test_checkpoint_path_explicit_env_override():
    ms = ModelService()
    default_path = MODEL_REGISTRY["HI-Small"]["checkpoint_path"]
    custom_target = Path("/custom/checkpoints/hi_small_champion.pt")
    with patch.dict(os.environ, {"FEDSECUREAI_HI_SMALL_CHECKPOINT": str(custom_target)}):
        resolved = ms._resolve_checkpoint_path("HI-Small", default_path)
        assert resolved == custom_target


def test_checkpoint_dir_env_resolution(tmp_path):
    ms = ModelService()
    default_path = MODEL_REGISTRY["HI-Small"]["checkpoint_path"]
    # Create fake checkpoint file in temp directory with same basename
    temp_ckpt = tmp_path / default_path.name
    temp_ckpt.write_bytes(b"dummy_data")

    with patch.dict(os.environ, {"FEDSECUREAI_CHECKPOINT_DIR": str(tmp_path)}):
        resolved = ms._resolve_checkpoint_path("HI-Small", default_path)
        assert resolved == temp_ckpt


# 2. Data Directory Environment Override Tests
def test_metadata_dir_default_resolution():
    names = _get_expanded_feature_names("HI-Small")
    assert len(names) == 57


def test_metadata_dir_custom_env_resolution(tmp_path):
    # Copy metadata to tmp_path
    source_meta_dir = BASE_DIR / "data/processed/ibm_aml/HI-Small/metadata"
    dest_dir = tmp_path / "HI-Small/metadata"
    dest_dir.mkdir(parents=True)
    for f in source_meta_dir.glob("*.json"):
        dest_dir.joinpath(f.name).write_bytes(f.read_bytes())

    with patch.dict(os.environ, {"FEDSECUREAI_DATA_DIR": str(tmp_path)}):
        names = _get_expanded_feature_names("HI-Small")
        assert len(names) == 57


# 3. Requirements Specification Audit
def test_inference_requirements_strictness():
    req_path = BASE_DIR / "requirements/inference.txt"
    assert req_path.exists(), "requirements/inference.txt must exist"
    lines = [
        l.strip().lower() for l in req_path.read_text(encoding="utf-8").splitlines()
        if l.strip() and not l.strip().startswith("#")
    ]
    raw_joined = " ".join(lines)

    # Must contain essential inference tools
    assert "fastapi" in raw_joined
    assert "uvicorn" in raw_joined
    assert "pydantic" in raw_joined
    assert "torch" in raw_joined

    # Must NOT contain training, dataset downloader, or test tools
    assert not any("kaggle" in l for l in lines)
    assert not any("pytest" in l for l in lines)
    assert not any("flwr" in l for l in lines)
    assert not any("streamlit" in l for l in lines)


# 4. Containerfile Security & Hardening Audit
def test_dockerfile_security_directives():
    df_path = BASE_DIR / "docker/Dockerfile.inference"
    assert df_path.exists(), "Dockerfile.inference must exist"
    content = df_path.read_text(encoding="utf-8")

    # Non-root user directive
    assert "USER 10001:10001" in content or "USER fedsecure" in content

    # Standard healthcheck directive
    assert "HEALTHCHECK" in content
    assert "/health" in content

    # Port exposure
    assert "EXPOSE 8000" in content

    # Ensure sensitive credentials or large data are not directly ADDed
    assert "data/raw" not in content
    assert "train" not in content
    assert "kaggle.json" not in content


# 5. Docker Compose Hardened Configuration Audit
def test_docker_compose_security_and_network():
    compose_path = BASE_DIR / "docker-compose.yml"
    assert compose_path.exists(), "docker-compose.yml must exist"
    with open(compose_path, "r", encoding="utf-8") as f:
        compose = yaml.safe_load(f)

    svc = compose["services"]["fedsecure-inference"]

    # 1. Loopback binding verification (restricts access to 127.0.0.1)
    ports = svc.get("ports", [])
    assert any("127.0.0.1:8000" in str(p) for p in ports), "Ports must bind strictly to 127.0.0.1"

    # 2. Read-only filesystem verification
    assert svc.get("read_only") is True, "read_only must be true"

    # 3. Memory tmpfs volume verification
    assert "tmpfs" in svc, "tmpfs must be configured for temporary scratch space"

    # 4. Volume mount read-only checks
    volumes = svc.get("volumes", [])
    for vol in volumes:
        if isinstance(vol, dict):
            assert vol.get("read_only") is True, f"Volume {vol} must be read_only"
        elif isinstance(vol, str):
            assert vol.endswith(":ro"), f"Volume string {vol} must be marked :ro"

    # 5. Healthcheck configured
    assert "healthcheck" in svc, "Service must define a healthcheck"


# 6. Dockerignore Completeness Audit
def test_dockerignore_rules():
    ignore_path = BASE_DIR / ".dockerignore"
    assert ignore_path.exists(), ".dockerignore must exist"
    content = ignore_path.read_text(encoding="utf-8")

    assert ".env" in content
    assert "kaggle.json" in content
    assert "data/raw" in content
    assert ".venv" in content
    assert "results/phase7" in content or "results/authoritative_results" in content
