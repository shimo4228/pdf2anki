"""Tests for pdf2anki.config - TDD RED phase.

Tests cover:
- AppConfig model with nested settings
- Default values
- YAML loading
- Environment variable overrides
- Immutability
- Validation
"""

from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import patch

import pytest
from pydantic import ValidationError

from pdf2anki.config import AppConfig, load_config

FIXTURES_DIR = Path(__file__).parent / "fixtures"


# ============================================================
# AppConfig Model Tests
# ============================================================


class TestAppConfig:
    """Test AppConfig Pydantic model."""

    def test_default_values(self) -> None:
        config = AppConfig()
        assert config.model == "claude-sonnet-4-5-20250929"
        assert config.max_tokens == 8192

    def test_quality_defaults(self) -> None:
        config = AppConfig()
        assert config.quality_confidence_threshold == 0.90
        assert config.quality_enable_critique is True
        assert config.quality_max_critique_rounds == 2

    def test_output_defaults(self) -> None:
        config = AppConfig()
        assert config.output_format == "tsv"
        assert config.output_encoding == "utf-8"

    def test_cards_defaults(self) -> None:
        config = AppConfig()
        assert config.cards_max_cards == 50
        assert len(config.cards_card_types) == 6
        assert "qa" in config.cards_card_types
        assert "cloze" not in config.cards_card_types
        assert "image_occlusion" not in config.cards_card_types

    def test_cost_defaults(self) -> None:
        config = AppConfig()
        assert config.cost_budget_limit == 1.00
        assert config.cost_warn_at == 0.80

    def test_ocr_defaults(self) -> None:
        config = AppConfig()
        assert config.ocr_enabled is False
        assert config.ocr_lang == "jpn+eng"

    def test_custom_values(self) -> None:
        config = AppConfig(
            model="claude-haiku-4-5-20251001",
            max_tokens=4096,
            quality_confidence_threshold=0.80,
            cards_max_cards=100,
        )
        assert config.model == "claude-haiku-4-5-20251001"
        assert config.max_tokens == 4096
        assert config.quality_confidence_threshold == 0.80
        assert config.cards_max_cards == 100

    def test_frozen_immutability(self) -> None:
        config = AppConfig()
        with pytest.raises(ValidationError):
            config.model = "other-model"  # type: ignore[misc]

    def test_invalid_confidence_threshold(self) -> None:
        with pytest.raises(ValidationError):
            AppConfig(quality_confidence_threshold=1.5)

    def test_invalid_max_tokens_negative(self) -> None:
        with pytest.raises(ValidationError):
            AppConfig(max_tokens=-1)

    def test_invalid_budget_limit_negative(self) -> None:
        with pytest.raises(ValidationError):
            AppConfig(cost_budget_limit=-0.5)


# ============================================================
# load_config Tests
# ============================================================


class TestLoadConfig:
    """Test config loading from YAML and env vars."""

    def test_load_default_config(self, tmp_path: Path) -> None:
        """Loading without a file should return defaults."""
        config = load_config(config_path=None)
        assert config.model == "claude-sonnet-4-5-20250929"

    def test_load_from_yaml(self, tmp_path: Path) -> None:
        yaml_content = """
model: "claude-haiku-4-5-20251001"
max_tokens: 4096
quality:
  confidence_threshold: 0.85
cards:
  max_cards: 30
"""
        config_file = tmp_path / "config.yaml"
        config_file.write_text(yaml_content)

        config = load_config(config_path=str(config_file))
        assert config.model == "claude-haiku-4-5-20251001"
        assert config.max_tokens == 4096
        assert config.quality_confidence_threshold == 0.85
        assert config.cards_max_cards == 30

    def test_yaml_partial_override(self, tmp_path: Path) -> None:
        """YAML should only override specified fields, keeping defaults."""
        yaml_content = """
model: "claude-haiku-4-5-20251001"
"""
        config_file = tmp_path / "config.yaml"
        config_file.write_text(yaml_content)

        config = load_config(config_path=str(config_file))
        assert config.model == "claude-haiku-4-5-20251001"
        assert config.max_tokens == 8192  # default preserved

    def test_env_var_overrides(self) -> None:
        """Environment variables should override YAML and defaults."""
        with patch.dict(
            os.environ,
            {
                "PDF2ANKI_MODEL": "claude-opus-4-6",
                "PDF2ANKI_BUDGET_LIMIT": "5.00",
            },
        ):
            config = load_config(config_path=None)
            assert config.model == "claude-opus-4-6"
            assert config.cost_budget_limit == 5.00

    def test_env_var_overrides_yaml(self, tmp_path: Path) -> None:
        """Env vars take precedence over YAML values."""
        yaml_content = """
model: "claude-haiku-4-5-20251001"
"""
        config_file = tmp_path / "config.yaml"
        config_file.write_text(yaml_content)

        with patch.dict(os.environ, {"PDF2ANKI_MODEL": "claude-opus-4-6"}):
            config = load_config(config_path=str(config_file))
            assert config.model == "claude-opus-4-6"

    def test_nonexistent_config_file(self) -> None:
        """Non-existent file should raise FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            load_config(config_path="/nonexistent/config.yaml")

    def test_invalid_yaml(self, tmp_path: Path) -> None:
        """Invalid YAML should raise ValueError."""
        config_file = tmp_path / "bad.yaml"
        config_file.write_text("{unclosed: [")

        with pytest.raises(ValueError, match="Invalid YAML"):
            load_config(config_path=str(config_file))


# ============================================================
# Batch Config Tests (Phase 3)
# ============================================================


class TestBatchConfig:
    """Test batch-related configuration fields."""

    def test_batch_defaults(self) -> None:
        config = AppConfig()
        assert config.batch_enabled is False
        assert config.batch_poll_interval == 30.0
        assert config.batch_timeout == 3600.0

    def test_batch_enabled_override(self) -> None:
        config = AppConfig(batch_enabled=True)
        assert config.batch_enabled is True

    def test_batch_custom_poll_interval(self) -> None:
        config = AppConfig(batch_poll_interval=10.0)
        assert config.batch_poll_interval == 10.0

    def test_batch_custom_timeout(self) -> None:
        config = AppConfig(batch_timeout=7200.0)
        assert config.batch_timeout == 7200.0

    def test_batch_from_yaml(self, tmp_path: Path) -> None:
        yaml_content = """
batch:
  enabled: true
  poll_interval: 15.0
  timeout: 1800.0
"""
        config_file = tmp_path / "config.yaml"
        config_file.write_text(yaml_content)

        config = load_config(config_path=str(config_file))
        assert config.batch_enabled is True
        assert config.batch_poll_interval == 15.0
        assert config.batch_timeout == 1800.0

    def test_batch_invalid_poll_interval(self) -> None:
        with pytest.raises(ValidationError):
            AppConfig(batch_poll_interval=-1.0)

    def test_batch_invalid_timeout(self) -> None:
        with pytest.raises(ValidationError):
            AppConfig(batch_timeout=-1.0)


# ============================================================
# Cache Config Tests
# ============================================================


class TestCacheConfig:
    """Test cache-related configuration fields."""

    def test_cache_defaults(self) -> None:
        config = AppConfig()
        assert config.cache_enabled is False
        assert config.cache_dir == ".cache/pdf2anki"

    def test_cache_enabled_override(self) -> None:
        config = AppConfig(cache_enabled=True)
        assert config.cache_enabled is True

    def test_cache_custom_dir(self) -> None:
        config = AppConfig(cache_dir="/tmp/my_cache")
        assert config.cache_dir == "/tmp/my_cache"

    def test_cache_from_yaml(self, tmp_path: Path) -> None:
        yaml_content = """
cache:
  enabled: true
  dir: ".my_cache"
"""
        config_file = tmp_path / "config.yaml"
        config_file.write_text(yaml_content)

        config = load_config(config_path=str(config_file))
        assert config.cache_enabled is True
        assert config.cache_dir == ".my_cache"
