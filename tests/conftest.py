"""Shared test fixtures for pdf2anki tests."""

from __future__ import annotations

from pathlib import Path

import pymupdf
import pytest

from pdf2anki.schemas import AnkiCard, BloomLevel, CardType


@pytest.fixture
def sample_qa_card() -> AnkiCard:
    """A basic QA-type card for testing."""
    return AnkiCard(
        front="ニューラルネットワークの活性化関数の役割は何ですか？",
        back="非線形変換を導入し、複雑なパターンの学習を可能にする。",
        card_type=CardType.QA,
        bloom_level=BloomLevel.UNDERSTAND,
        tags=["AI::基礎", "neural_network"],
    )


@pytest.fixture
def sample_cloze_card() -> AnkiCard:
    """A cloze-type card for testing."""
    return AnkiCard(
        front="{{c1::勾配降下法}}はパラメータを更新して損失関数を最小化するアルゴリズムである。",
        back="",
        card_type=CardType.CLOZE,
        bloom_level=BloomLevel.REMEMBER,
        tags=["AI::最適化"],
    )


@pytest.fixture
def sample_reversible_card() -> AnkiCard:
    """A reversible card for testing."""
    return AnkiCard(
        front="ReLU",
        back="Rectified Linear Unit: f(x) = max(0, x)",
        card_type=CardType.REVERSIBLE,
        bloom_level=BloomLevel.REMEMBER,
        tags=["AI::活性化関数"],
        related_concepts=["sigmoid", "tanh", "leaky_relu"],
        mnemonic_hint="REctified Linear = 負の値をREject(拒否)する",
    )


@pytest.fixture
def sample_pdf(tmp_path: Path) -> Path:
    """Create a minimal PDF for testing text extraction."""
    pdf_path = tmp_path / "sample.pdf"
    doc = pymupdf.Document()
    page = doc.new_page()
    page.insert_text(
        (72, 72),
        "ニューラルネットワークの基礎\n\n活性化関数はReLU、Sigmoid、Tanhがあります。",
        fontname="japan",
        fontsize=12,
    )
    doc.save(str(pdf_path))
    doc.close()
    return pdf_path
