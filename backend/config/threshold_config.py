"""Configuration loader for canonical decision thresholds."""

from pathlib import Path
from typing import Dict, Any, Optional
import yaml

CONFIG_PATH = Path(__file__).resolve().parent / "thresholds.yaml"

_DEFAULT_THRESHOLDS: Dict[str, float] = {
    "HI-Small": 0.337784,
    "LI-Medium": 0.238483,
}

_CACHED_CONFIG: Optional[Dict[str, Any]] = None


def load_threshold_config() -> Dict[str, Any]:
    """Load threshold configuration from thresholds.yaml with memory caching."""
    global _CACHED_CONFIG
    if _CACHED_CONFIG is not None:
        return _CACHED_CONFIG

    if CONFIG_PATH.exists():
        try:
            with open(CONFIG_PATH, "r", encoding="utf-8") as f:
                _CACHED_CONFIG = yaml.safe_load(f)
                return _CACHED_CONFIG
        except Exception:
            pass

    _CACHED_CONFIG = {
        "thresholds": _DEFAULT_THRESHOLDS,
        "default_dataset": "HI-Small",
        "default_threshold": 0.337784,
    }
    return _CACHED_CONFIG


def get_canonical_threshold(dataset_or_model: str) -> float:
    """Retrieve canonical frozen decision threshold for a dataset or model name."""
    cfg = load_threshold_config()
    thresholds = cfg.get("thresholds", _DEFAULT_THRESHOLDS)

    # Check direct match
    if dataset_or_model in thresholds:
        return float(thresholds[dataset_or_model])

    name_lower = dataset_or_model.lower()
    if "li_medium" in name_lower or "li-medium" in name_lower:
        return float(thresholds.get("LI-Medium", 0.238483))

    # Default to HI-Small canonical threshold
    return float(thresholds.get("HI-Small", 0.337784))

