"""TSV and JSON output conversion for pdf2anki.

Converts AnkiCard lists to Anki-importable TSV format and
ExtractionResult to JSON with metadata. All functions are pure
(no side effects except file I/O in write_* functions).

TSV format:
  - Header: #separator:tab, #html:true, #tags column:3
  - Rows: front<TAB>back<TAB>tags
  - Tabs -> spaces, newlines -> <br>
  - Cloze: front only (back empty)
  - Reversible: expanded to 2 rows (forward + reverse)
  - Tags include bloom::<level>
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

from pdf2anki.schemas import AnkiCard, CardType, ExtractionResult

_SCHEMA_VERSION = "1.0"

_TSV_HEADER = "#separator:tab\n#html:true\n#tags column:3\n"


# ============================================================
# Internal helpers
# ============================================================


def _escape_tsv_field(text: str) -> str:
    """Escape tabs and newlines for TSV field safety.

    Tabs become spaces, newlines become <br> (Anki HTML mode).
    """
    return text.replace("\t", " ").replace("\n", "<br>")


def _build_tags(
    card: AnkiCard,
    additional_tags: list[str] | None = None,
) -> str:
    """Build space-separated tag string with bloom level.

    Adds bloom::<level> tag. Includes additional_tags if provided.
    """
    tags = list(card.tags)
    tags.append(f"bloom::{card.bloom_level.value}")
    if additional_tags:
        tags.extend(additional_tags)
    return " ".join(tags)


def _card_to_rows(
    card: AnkiCard,
    additional_tags: list[str] | None = None,
) -> list[str]:
    """Convert a single card to one or more TSV rows.

    Reversible cards produce 2 rows (forward + reverse).
    Cloze cards have an empty back field.
    """
    tags_str = _build_tags(card, additional_tags)
    front = _escape_tsv_field(card.front)
    back = _escape_tsv_field(card.back)

    if card.card_type == CardType.REVERSIBLE:
        return [
            f"{front}\t{back}\t{tags_str}",
            f"{back}\t{front}\t{tags_str}",
        ]

    return [f"{front}\t{back}\t{tags_str}"]


# ============================================================
# Public API
# ============================================================


def cards_to_tsv(
    cards: list[AnkiCard],
    additional_tags: list[str] | None = None,
) -> str:
    """Convert cards to Anki-importable TSV string.

    Args:
        cards: List of AnkiCards to convert.
        additional_tags: Extra tags to add to every card.

    Returns:
        TSV string with header directives and data rows.
    """
    rows: list[str] = []
    for card in cards:
        rows.extend(_card_to_rows(card, additional_tags))

    return _TSV_HEADER + "\n".join(rows)


def cards_to_json(
    result: ExtractionResult,
) -> str:
    """Convert ExtractionResult to JSON with metadata.

    Args:
        result: ExtractionResult containing cards and source info.

    Returns:
        Pretty-printed JSON string with _meta block.
    """
    data = result.model_dump(mode="json")
    data["_meta"] = {
        "schema_version": _SCHEMA_VERSION,
        "generated_at": datetime.now(UTC).isoformat(),
    }
    return json.dumps(data, ensure_ascii=False, indent=2)


def write_tsv(
    cards: list[AnkiCard],
    path: Path,
    additional_tags: list[str] | None = None,
) -> None:
    """Write cards to a TSV file (UTF-8, no BOM).

    Creates parent directories if they don't exist.

    Args:
        cards: List of AnkiCards to write.
        path: Output file path.
        additional_tags: Extra tags to add to every card.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    content = cards_to_tsv(cards, additional_tags)
    path.write_text(content, encoding="utf-8")


def write_json(
    result: ExtractionResult,
    path: Path,
) -> None:
    """Write ExtractionResult to a JSON file (UTF-8).

    Creates parent directories if they don't exist.

    Args:
        result: ExtractionResult to write.
        path: Output file path.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    content = cards_to_json(result)
    path.write_text(content, encoding="utf-8")
