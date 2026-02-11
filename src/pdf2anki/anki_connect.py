"""AnkiConnect API client for pushing cards to Anki.

Uses only urllib.request (no new dependencies).
AnkiConnect add-on (2055492159) must be installed in Anki.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any

from pdf2anki.schemas import AnkiCard, CardType

ANKICONNECT_URL = "http://127.0.0.1:8765"
ANKICONNECT_VERSION = 6


class AnkiConnectError(Exception):
    """AnkiConnect API error."""


@dataclass(frozen=True)
class PushResult:
    """Result of pushing cards to Anki."""

    total: int
    added: int
    failed: int
    errors: tuple[str, ...]


def _invoke(action: str, *, url: str = ANKICONNECT_URL, **params: Any) -> Any:
    """Call AnkiConnect API. Raises AnkiConnectError on failure."""
    payload = json.dumps(
        {
            "action": action,
            "version": ANKICONNECT_VERSION,
            "params": params,
        }
    ).encode()
    req = urllib.request.Request(
        url, data=payload, headers={"Content-Type": "application/json"}
    )
    try:
        with urllib.request.urlopen(req) as resp:
            body = json.loads(resp.read())
    except urllib.error.URLError as e:
        raise AnkiConnectError(
            "Anki is not running or AnkiConnect is not installed.\n"
            "Start Anki and install AnkiConnect add-on (code: 2055492159)."
        ) from e
    if body.get("error"):
        raise AnkiConnectError(body["error"])
    return body["result"]


def is_anki_running() -> bool:
    """Check if AnkiConnect is responsive."""
    try:
        _invoke("version")
    except AnkiConnectError:
        return False
    return True


def ensure_deck(deck_name: str) -> None:
    """Create deck if it doesn't exist."""
    _invoke("createDeck", deck=deck_name)


def card_to_note(card: AnkiCard, *, deck_name: str) -> dict[str, Any]:
    """Convert AnkiCard to AnkiConnect note dict."""
    tags = list(card.tags) + [f"bloom::{card.bloom_level.value}"]

    if card.card_type == CardType.CLOZE:
        return {
            "deckName": deck_name,
            "modelName": "Cloze",
            "fields": {"Text": card.front, "Extra": card.back},
            "tags": tags,
            "options": {"allowDuplicate": False},
        }

    if card.card_type == CardType.REVERSIBLE:
        return {
            "deckName": deck_name,
            "modelName": "Basic (and target: reversed card)",
            "fields": {"Front": card.front, "Back": card.back},
            "tags": tags,
            "options": {"allowDuplicate": False},
        }

    return {
        "deckName": deck_name,
        "modelName": "Basic",
        "fields": {"Front": card.front, "Back": card.back},
        "tags": tags,
        "options": {"allowDuplicate": False},
    }


def push_cards(
    cards: list[AnkiCard],
    *,
    deck_name: str = "pdf2anki",
) -> PushResult:
    """Push cards to Anki via AnkiConnect. Returns PushResult."""
    if not cards:
        return PushResult(total=0, added=0, failed=0, errors=())

    ensure_deck(deck_name)

    notes = [card_to_note(c, deck_name=deck_name) for c in cards]
    results: list[int | None] = _invoke("addNotes", notes=notes)

    added = sum(1 for r in results if r is not None)
    failed = sum(1 for r in results if r is None)
    errors = tuple(
        f"Card {i + 1} failed to add" for i, r in enumerate(results) if r is None
    )

    return PushResult(total=len(cards), added=added, failed=failed, errors=errors)
