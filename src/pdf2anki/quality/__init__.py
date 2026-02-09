"""Quality assurance pipeline for pdf2anki.

Provides confidence scoring, LLM critique, and a full quality pipeline
for Anki card generation.
"""

from pdf2anki.quality.critique import (
    _parse_critique_response,
    critique_cards,
)
from pdf2anki.quality.duplicate import (
    _char_bigrams,
    _jaccard,
    _tokenize,
)
from pdf2anki.quality.heuristic import (
    score_card,
    score_cards,
)
from pdf2anki.quality.pipeline import (
    QualityReport,
    run_quality_pipeline,
)

__all__ = [
    "QualityReport",
    "_char_bigrams",
    "_jaccard",
    "_parse_critique_response",
    "_tokenize",
    "critique_cards",
    "run_quality_pipeline",
    "score_card",
    "score_cards",
]
