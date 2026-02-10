"""Evaluation dataset schema and loader.

Defines frozen dataclasses for evaluation cases and loads
YAML dataset files for prompt quality measurement.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from pdf2anki.schemas import CardType


@dataclass(frozen=True, slots=True)
class ExpectedCard:
    """Expected card definition for evaluation (keyword-based)."""

    front_keywords: list[str]
    back_keywords: list[str]
    card_type: CardType | None = None
    tags: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class EvalCase:
    """Single evaluation case with input text and expected cards."""

    id: str
    text: str
    expected_cards: tuple[ExpectedCard, ...]
    description: str = ""


@dataclass(frozen=True, slots=True)
class EvalDataset:
    """Collection of evaluation cases."""

    name: str
    version: str
    cases: tuple[EvalCase, ...]


def _parse_expected_card(data: dict[str, Any]) -> ExpectedCard:
    """Parse a single expected card from YAML data."""
    card_type: CardType | None = None
    raw_type = data.get("card_type")
    if raw_type is not None:
        card_type = CardType(raw_type)

    tags = tuple(data.get("tags", ()))

    return ExpectedCard(
        front_keywords=data["front_keywords"],
        back_keywords=data["back_keywords"],
        card_type=card_type,
        tags=tags,
    )


def _parse_case(data: dict[str, Any]) -> EvalCase:
    """Parse a single evaluation case from YAML data."""
    expected = tuple(
        _parse_expected_card(ec) for ec in data["expected_cards"]
    )
    return EvalCase(
        id=data["id"],
        text=data["text"],
        expected_cards=expected,
        description=data.get("description", ""),
    )


def load_dataset(path: Path) -> EvalDataset:
    """Load an evaluation dataset from a YAML file.

    Raises:
        FileNotFoundError: If the YAML file does not exist.
    """
    if not path.exists():
        msg = f"Dataset file not found: {path}"
        raise FileNotFoundError(msg)

    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    cases = tuple(_parse_case(c) for c in raw["cases"])
    return EvalDataset(
        name=raw["name"],
        version=raw["version"],
        cases=cases,
    )
