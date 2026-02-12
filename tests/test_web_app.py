"""Tests for pdf2anki.web.app module."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

from pdf2anki.anki_connect import PushResult
from pdf2anki.config import AppConfig  # noqa: F401
from pdf2anki.cost import CostTracker
from pdf2anki.schemas import AnkiCard, BloomLevel, CardType, ExtractionResult
from pdf2anki.web.app import (
    _build_config_from_ui,
    _write_temp,
    generate_cards,
    push_to_anki,
)

# ------------------------------------------------------------------
# _build_config_from_ui
# ------------------------------------------------------------------


class TestBuildConfigFromUI:
    """Tests for _build_config_from_ui helper."""

    def test_returns_app_config(self) -> None:
        cfg = _build_config_from_ui(
            model="claude-sonnet-4-5-20250929",
            quality="basic",
            max_cards=30,
            budget=2.0,
            vision=False,
        )
        assert isinstance(cfg, AppConfig)
        assert cfg.model == "claude-sonnet-4-5-20250929"
        assert cfg.cards_max_cards == 30
        assert cfg.cost_budget_limit == 2.0
        assert cfg.vision_enabled is False

    def test_quality_off_disables_critique(self) -> None:
        cfg = _build_config_from_ui(
            model="claude-haiku-4-5-20251001",
            quality="off",
            max_cards=10,
            budget=0.5,
            vision=False,
        )
        assert cfg.quality_enable_critique is False
        assert cfg.quality_confidence_threshold == 0.0

    def test_quality_full_enables_critique(self) -> None:
        cfg = _build_config_from_ui(
            model="claude-sonnet-4-5-20250929",
            quality="full",
            max_cards=50,
            budget=1.0,
            vision=False,
        )
        assert cfg.quality_enable_critique is True

    def test_vision_enabled(self) -> None:
        cfg = _build_config_from_ui(
            model="claude-sonnet-4-5-20250929",
            quality="basic",
            max_cards=50,
            budget=1.0,
            vision=True,
        )
        assert cfg.vision_enabled is True


# ------------------------------------------------------------------
# _write_temp
# ------------------------------------------------------------------


class TestWriteTemp:
    """Tests for _write_temp helper."""

    def test_writes_file_and_returns_path(self) -> None:
        path = _write_temp("hello", suffix=".txt", stem="test")
        result = Path(path)
        assert result.exists()
        assert result.read_text(encoding="utf-8") == "hello"
        assert result.suffix == ".txt"
        assert "test_" in result.name

    def test_utf8_content(self) -> None:
        path = _write_temp("日本語テスト", suffix=".tsv", stem="jp")
        assert Path(path).read_text(encoding="utf-8") == "日本語テスト"


# ------------------------------------------------------------------
# generate_cards
# ------------------------------------------------------------------

_SAMPLE_CARD = AnkiCard(
    front="What is X?",
    back="X is Y.",
    card_type=CardType.QA,
    bloom_level=BloomLevel.UNDERSTAND,
    tags=["test"],
)

_SAMPLE_RESULT = ExtractionResult(
    source_file="test.pdf",
    cards=[_SAMPLE_CARD],
    model_used="claude-sonnet-4-5-20250929",
)


class TestGenerateCards:
    """Tests for generate_cards handler."""

    def test_none_file_returns_error(self) -> None:
        status, cost, table, state, tsv, json_ = generate_cards(
            None, "claude-sonnet-4-5-20250929", "basic", 50, 1.0, False
        )
        assert "upload" in status.lower()
        assert state is None

    def test_unsupported_file_type(self, tmp_path: Path) -> None:
        bad_file = tmp_path / "test.xlsx"
        bad_file.write_text("data")
        status, _, _, state, _, _ = generate_cards(
            str(bad_file), "claude-sonnet-4-5-20250929", "basic", 50, 1.0, False
        )
        assert "unsupported" in status.lower()
        assert state is None

    @patch("pdf2anki.web.app.process_file")
    def test_successful_generation(self, mock_pf: MagicMock, tmp_path: Path) -> None:
        txt_file = tmp_path / "test.txt"
        txt_file.write_text("Some content")

        tracker = CostTracker(budget_limit=1.0)
        mock_pf.return_value = (_SAMPLE_RESULT, None, tracker)

        status, cost, table, state, tsv_path, json_path = generate_cards(
            str(txt_file), "claude-sonnet-4-5-20250929", "basic", 50, 1.0, False
        )

        assert "1 cards" in status
        assert state is not None
        assert len(table) == 1
        assert tsv_path is not None
        assert json_path is not None
        assert Path(tsv_path).exists()
        assert Path(json_path).exists()

    @patch("pdf2anki.web.app.process_file", side_effect=RuntimeError("API error"))
    def test_generation_failure(self, mock_pf: MagicMock, tmp_path: Path) -> None:
        txt_file = tmp_path / "test.txt"
        txt_file.write_text("Some content")

        status, _, _, state, _, _ = generate_cards(
            str(txt_file), "claude-sonnet-4-5-20250929", "basic", 50, 1.0, False
        )
        assert "failed" in status.lower()
        assert state is None


# ------------------------------------------------------------------
# push_to_anki
# ------------------------------------------------------------------


class TestPushToAnki:
    """Tests for push_to_anki handler."""

    def test_no_result_returns_error(self) -> None:
        msg = push_to_anki(None, "deck")
        assert "no cards" in msg.lower()

    def test_empty_cards_returns_error(self) -> None:
        empty = ExtractionResult(source_file="t.pdf", cards=[], model_used="m")
        msg = push_to_anki(empty, "deck")
        assert "no cards" in msg.lower()

    @patch("pdf2anki.anki_connect.push_cards")
    def test_successful_push(self, mock_push: MagicMock) -> None:
        mock_push.return_value = PushResult(total=1, added=1, failed=0, errors=())
        msg = push_to_anki(_SAMPLE_RESULT, "test-deck")
        assert "1 cards" in msg
        assert "test-deck" in msg

    @patch("pdf2anki.anki_connect.push_cards")
    def test_partial_failure(self, mock_push: MagicMock) -> None:
        mock_push.return_value = PushResult(
            total=2, added=1, failed=1, errors=("Card 2 failed",)
        )
        msg = push_to_anki(_SAMPLE_RESULT, "deck")
        assert "1 failed" in msg

    @patch("pdf2anki.anki_connect.push_cards", side_effect=ConnectionError("no anki"))
    def test_push_exception(self, mock_push: MagicMock) -> None:
        msg = push_to_anki(_SAMPLE_RESULT, "deck")
        assert "failed" in msg.lower()


# ------------------------------------------------------------------
# generate_cards: config build error (lines 93-95)
# ------------------------------------------------------------------


class TestGenerateCardsConfigError:
    """Test config build exception path in generate_cards."""

    @patch("pdf2anki.web.app.load_config", side_effect=ValueError("bad config"))
    def test_config_error_returns_message(
        self, mock_lc: MagicMock, tmp_path: Path
    ) -> None:
        txt_file = tmp_path / "test.txt"
        txt_file.write_text("content")
        status, _, _, state, _, _ = generate_cards(
            str(txt_file), "claude-sonnet-4-5-20250929", "basic", 50, 1.0, False
        )
        assert "config error" in status.lower()
        assert state is None


# ------------------------------------------------------------------
# create_interface (lines 165-237)
# ------------------------------------------------------------------


class TestCreateInterface:
    """Test create_interface returns a valid Gradio Blocks."""

    def test_returns_blocks(self) -> None:
        import gradio as gr

        from pdf2anki.web.app import create_interface

        demo = create_interface()
        assert isinstance(demo, gr.Blocks)


# ------------------------------------------------------------------
# launch_web (lines 246-248)
# ------------------------------------------------------------------


class TestLaunchWeb:
    """Test launch_web calls Gradio correctly."""

    @patch("pdf2anki.web.app.create_interface")
    def test_launch_calls_demo(self, mock_ci: MagicMock) -> None:
        from pdf2anki.web.app import launch_web

        mock_demo = MagicMock()
        mock_ci.return_value = mock_demo

        launch_web(host="127.0.0.1", port=7860)

        mock_demo.queue.assert_called_once_with(max_size=5)
        mock_demo.launch.assert_called_once_with(
            server_name="127.0.0.1", server_port=7860, share=False
        )
