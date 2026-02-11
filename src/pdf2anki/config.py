"""Configuration management for pdf2anki.

Loads config from YAML file with environment variable overrides.
Priority: env vars > YAML > defaults.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field

DEFAULT_MODEL = "claude-sonnet-4-5-20250929"

_DEFAULT_CARD_TYPES = [
    "qa",
    "term_definition",
    "summary_point",
    "reversible",
    "sequence",
    "compare_contrast",
]

# Mapping of env var names to (config field, type converter)
_ENV_OVERRIDES: dict[str, tuple[str, type]] = {
    "PDF2ANKI_MODEL": ("model", str),
    "PDF2ANKI_BUDGET_LIMIT": ("cost_budget_limit", float),
}


class AppConfig(BaseModel, frozen=True):
    """Application configuration. Immutable."""

    # Claude API
    model: str = DEFAULT_MODEL
    model_overridden: bool = Field(default=False, exclude=True)  # CLI-only runtime flag
    max_tokens: int = Field(default=8192, gt=0)

    # Quality pipeline
    quality_confidence_threshold: float = Field(default=0.90, ge=0.0, le=1.0)
    quality_enable_critique: bool = True
    quality_max_critique_rounds: int = Field(default=2, ge=0)

    # Output
    output_format: str = "tsv"
    output_encoding: str = "utf-8"

    # Cards
    cards_max_cards: int = Field(default=50, gt=0)
    cards_card_types: list[str] = Field(
        default_factory=lambda: list(_DEFAULT_CARD_TYPES)
    )
    cards_bloom_filter: list[str] = Field(default_factory=list)

    # Cost
    cost_budget_limit: float = Field(default=1.00, ge=0.0)
    cost_warn_at: float = Field(default=0.80, ge=0.0, le=1.0)

    # Batch API
    batch_enabled: bool = False
    batch_poll_interval: float = Field(default=30.0, ge=0.0)
    batch_timeout: float = Field(default=3600.0, ge=0.0)

    # OCR
    ocr_enabled: bool = False
    ocr_lang: str = "jpn+eng"

    # Cache
    cache_enabled: bool = False
    cache_dir: str = ".cache/pdf2anki"

    # Vision
    vision_enabled: bool = False
    vision_coverage_threshold: float = Field(default=0.20, ge=0.0, le=1.0)
    vision_dpi: int = Field(default=150, gt=0)
    vision_max_images_per_page: int = Field(default=5, gt=0)


def _flatten_yaml(data: dict[str, Any], *, _prefix: str = "") -> dict[str, Any]:
    """Flatten nested YAML structure to flat config fields (recursive).

    Example: {"quality": {"confidence_threshold": 0.9}}
    becomes: {"quality_confidence_threshold": 0.9}

    Handles arbitrarily nested dicts by joining keys with '_'.
    """
    flat: dict[str, Any] = {}
    for key, value in data.items():
        full_key = f"{_prefix}{key}" if not _prefix else f"{_prefix}_{key}"
        if isinstance(value, dict):
            flat.update(_flatten_yaml(value, _prefix=full_key))
        else:
            flat[full_key] = value
    return flat


def _apply_env_overrides(config_dict: dict[str, Any]) -> dict[str, Any]:
    """Apply environment variable overrides to config dict."""
    result = dict(config_dict)
    for env_var, (field_name, converter) in _ENV_OVERRIDES.items():
        env_value = os.environ.get(env_var)
        if env_value is not None:
            result[field_name] = converter(env_value)
    return result


def load_config(config_path: str | None = None) -> AppConfig:
    """Load configuration from YAML file with env var overrides.

    Priority: env vars > YAML file > defaults.

    Args:
        config_path: Path to YAML config file, or None for defaults.

    Returns:
        Frozen AppConfig instance.

    Raises:
        FileNotFoundError: If config_path is specified but doesn't exist.
        ValueError: If YAML file is invalid.
    """
    config_dict: dict[str, Any] = {}

    if config_path is not None:
        path = Path(config_path)
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")

        raw = path.read_text(encoding="utf-8")
        try:
            parsed = yaml.safe_load(raw)
        except yaml.YAMLError as e:
            raise ValueError(f"Invalid YAML in {config_path}: {e}") from e

        if isinstance(parsed, dict):
            config_dict = _flatten_yaml(parsed)

    config_dict = _apply_env_overrides(config_dict)

    # Strip runtime-only fields that should not come from YAML/env
    config_dict.pop("model_overridden", None)

    return AppConfig(**config_dict)
