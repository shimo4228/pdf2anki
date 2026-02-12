"""Gradio Web UI for pdf2anki.

Reuses existing service layer for all business logic.
Local-only by default (127.0.0.1). No authentication.
"""

from __future__ import annotations

import atexit
import glob as glob_mod
import logging
import shutil
import tempfile
from pathlib import Path
from typing import Any

import gradio as gr

from pdf2anki.config import AppConfig, load_config
from pdf2anki.convert import cards_to_json, cards_to_tsv
from pdf2anki.cost import CostTracker
from pdf2anki.schemas import ExtractionResult
from pdf2anki.service import process_file

logger = logging.getLogger(__name__)

_MAX_FILE_SIZE = 100 * 1024 * 1024  # 100 MB

# Clean stale temp dirs from previous crashed sessions.
for _old in glob_mod.glob(f"{tempfile.gettempdir()}/pdf2anki_web_*"):
    shutil.rmtree(_old, ignore_errors=True)

# Session-scoped temp dir; cleaned up on process exit.
_TEMP_DIR = tempfile.mkdtemp(prefix="pdf2anki_web_")
atexit.register(shutil.rmtree, _TEMP_DIR, True)

_MODELS = [
    ("Sonnet 4.5 (default)", "claude-sonnet-4-5-20250929"),
    ("Haiku 4.5 (fast/cheap)", "claude-haiku-4-5-20251001"),
    ("Opus 4.6 (best)", "claude-opus-4-6"),
]

_QUALITY_CHOICES = ["off", "basic", "full"]


def _build_config_from_ui(
    model: str,
    quality: str,
    max_cards: int,
    budget: float,
    vision: bool,
) -> AppConfig:
    """Build AppConfig from UI widget values."""
    base = load_config(None)
    overrides: dict[str, Any] = {
        "model": model,
        "cards_max_cards": int(max_cards),
        "cost_budget_limit": budget,
        "vision_enabled": vision,
    }
    if quality == "off":
        overrides["quality_enable_critique"] = False
        overrides["quality_confidence_threshold"] = 0.0
    elif quality == "full":
        overrides["quality_enable_critique"] = True
    return base.model_copy(update=overrides)


def _write_temp(content: str, *, suffix: str, stem: str) -> str:
    """Write content to a named temp file, return its path."""
    fd, path = tempfile.mkstemp(suffix=suffix, prefix=f"{stem}_", dir=_TEMP_DIR)
    with open(fd, "w", encoding="utf-8") as f:
        f.write(content)
    return path


# ------------------------------------------------------------------
# Event handlers
# ------------------------------------------------------------------


def generate_cards(
    file_path: str | None,
    model: str,
    quality: str,
    max_cards: float,
    budget: float,
    vision: bool,
) -> tuple[str, str, list[list[str]], ExtractionResult | None, str | None, str | None]:
    """Upload -> process_file -> (status, cost, table, state, tsv, json)."""
    import anthropic

    if file_path is None:
        return "Please upload a file first.", "", [], None, None, None

    path = Path(file_path)
    if path.suffix.lower() not in {".pdf", ".txt", ".md"}:
        return f"Unsupported file type: {path.suffix}", "", [], None, None, None

    if path.stat().st_size > _MAX_FILE_SIZE:
        return "File too large (max 100 MB).", "", [], None, None, None

    try:
        config = _build_config_from_ui(model, quality, int(max_cards), budget, vision)
    except (ValueError, TypeError) as e:
        logger.error("Config build failed: %s", e)
        return "Config error. Check server logs for details.", "", [], None, None, None

    cost_tracker = CostTracker(budget_limit=config.cost_budget_limit)

    try:
        result, _report, cost_tracker = process_file(
            file_path=path,
            config=config,
            cost_tracker=cost_tracker,
            quality=quality,
            focus_topics=None,
            additional_tags=None,
        )
    except (RuntimeError, ValueError, anthropic.APIError) as e:
        logger.error("Card generation failed: %s", type(e).__name__)
        return (
            "Generation failed. Check server logs for details.",
            "",
            [],
            None,
            None,
            None,
        )

    table = [
        [c.front[:100], c.back[:100], c.card_type.value, c.bloom_level.value]
        for c in result.cards
    ]

    status = f"Generated {result.card_count} cards from {path.name}"
    cost = f"${cost_tracker.total_cost:.4f} ({cost_tracker.request_count} API calls)"

    stem = path.stem
    tsv_path = _write_temp(cards_to_tsv(list(result.cards)), suffix=".tsv", stem=stem)
    json_path = _write_temp(cards_to_json(result), suffix=".json", stem=stem)

    return status, cost, table, result, tsv_path, json_path


def push_to_anki(
    result_state: ExtractionResult | None,
    deck_name: str,
) -> str:
    """Push generated cards to Anki via AnkiConnect."""
    if result_state is None or not result_state.cards:
        return "No cards to push. Generate cards first."

    from pdf2anki.anki_connect import AnkiConnectError, push_cards

    try:
        push_result = push_cards(list(result_state.cards), deck_name=deck_name)
    except (AnkiConnectError, OSError) as e:
        logger.error("Anki push failed: %s", e)
        return "Push failed. Check server logs for details."

    if push_result.failed > 0:
        return (
            f"Pushed {push_result.added}/{push_result.total} to '{deck_name}'. "
            f"{push_result.failed} failed."
        )
    return f"Pushed {push_result.added} cards to '{deck_name}'."


# ------------------------------------------------------------------
# Interface
# ------------------------------------------------------------------


def create_interface() -> gr.Blocks:
    """Build the Gradio Blocks interface."""
    with gr.Blocks(title="pdf2anki") as demo:
        gr.Markdown("# pdf2anki: Generate Anki Flashcards from Documents")

        result_state: gr.State = gr.State(value=None)

        with gr.Row():
            with gr.Column(scale=1):
                file_input = gr.File(
                    label="Upload File (PDF / TXT / MD)",
                    file_types=[".pdf", ".txt", ".md"],
                    file_count="single",
                )
                model_dd = gr.Dropdown(
                    label="Model",
                    choices=[(name, val) for name, val in _MODELS],
                    value=_MODELS[0][1],
                )
                quality_dd = gr.Dropdown(
                    label="Quality",
                    choices=_QUALITY_CHOICES,
                    value="basic",
                )
                max_cards_sl = gr.Slider(
                    label="Max Cards",
                    minimum=5,
                    maximum=100,
                    step=5,
                    value=50,
                )
                budget_num = gr.Number(
                    label="Budget Limit ($)",
                    value=1.00,
                    minimum=0.01,
                )
                vision_cb = gr.Checkbox(label="Vision API (image PDFs)", value=False)
                gen_btn = gr.Button("Generate Cards", variant="primary")

            with gr.Column(scale=2):
                status_tb = gr.Textbox(label="Status", interactive=False)
                cost_tb = gr.Textbox(label="Cost", interactive=False)
                cards_df = gr.Dataframe(
                    headers=["Front", "Back", "Type", "Bloom"],
                    label="Generated Cards",
                    interactive=False,
                    wrap=True,
                )
                with gr.Row():
                    tsv_dl = gr.File(label="Download TSV", interactive=False)
                    json_dl = gr.File(label="Download JSON", interactive=False)
                with gr.Row():
                    deck_tb = gr.Textbox(label="Deck Name", value="pdf2anki")
                    push_btn = gr.Button("Push to Anki")
                    push_st = gr.Textbox(label="Push Status", interactive=False)

        gen_btn.click(
            fn=generate_cards,
            inputs=[
                file_input,
                model_dd,
                quality_dd,
                max_cards_sl,
                budget_num,
                vision_cb,
            ],
            outputs=[status_tb, cost_tb, cards_df, result_state, tsv_dl, json_dl],
        )
        push_btn.click(
            fn=push_to_anki,
            inputs=[result_state, deck_tb],
            outputs=[push_st],
        )

    return demo  # type: ignore[no-any-return]


def launch_web(
    *,
    host: str = "127.0.0.1",
    port: int = 7860,
) -> None:
    """Launch the Gradio web interface."""
    demo = create_interface()
    demo.queue(max_size=5)
    demo.launch(server_name=host, server_port=port, share=False)
