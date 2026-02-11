"""Web UI for pdf2anki (optional dependency: gradio)."""

try:
    import gradio  # noqa: F401
except ImportError as e:
    raise ImportError(
        "Web UI requires Gradio. Install with: pip install pdf2anki[web]"
    ) from e

from pdf2anki.web.app import launch_web

__all__ = ["launch_web"]
