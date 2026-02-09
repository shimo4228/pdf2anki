"""Tests for pdf2anki.quality - Quality Assurance Pipeline.

TDD RED phase: Tests written before implementation.

Tests cover:
- score_card(): Field-level confidence scoring with weighted totals
- Flag detection (6 quality flags)
- critique_cards(): LLM critique for low-confidence cards
- run_quality_pipeline(): Full pipeline orchestration
- QualityReport generation
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest
from pydantic import ValidationError

from pdf2anki.config import AppConfig
from pdf2anki.cost import CostTracker
from pdf2anki.quality import (
    QualityReport,
    _char_bigrams,
    _jaccard,
    _parse_critique_response,
    _tokenize,
    critique_cards,
    run_quality_pipeline,
    score_card,
    score_cards,
)
from pdf2anki.schemas import (
    AnkiCard,
    BloomLevel,
    CardConfidenceScore,
    CardType,
    QualityFlag,
)

# ============================================================
# Fixtures
# ============================================================


@pytest.fixture
def high_quality_qa_card() -> AnkiCard:
    """A well-formed QA card that should score high."""
    return AnkiCard(
        front="ニューラルネットワークの活性化関数の役割は何ですか？",
        back="非線形変換を導入し、複雑なパターンの学習を可能にする。",
        card_type=CardType.QA,
        bloom_level=BloomLevel.UNDERSTAND,
        tags=["AI::基礎", "neural_network"],
        related_concepts=["ReLU", "sigmoid"],
    )


@pytest.fixture
def high_quality_cloze_card() -> AnkiCard:
    """A well-formed cloze card that should score high."""
    return AnkiCard(
        front="{{c1::勾配降下法}}はパラメータを更新して損失関数を最小化するアルゴリズムである。",
        back="",
        card_type=CardType.CLOZE,
        bloom_level=BloomLevel.REMEMBER,
        tags=["AI::最適化"],
    )


@pytest.fixture
def vague_question_card() -> AnkiCard:
    """A card with a vague, non-question front."""
    return AnkiCard(
        front="概要",
        back="機械学習はデータからパターンを学習する手法である。",
        card_type=CardType.QA,
        bloom_level=BloomLevel.REMEMBER,
        tags=["AI"],
    )


@pytest.fixture
def too_long_answer_card() -> AnkiCard:
    """A card with a back exceeding 200 characters."""
    return AnkiCard(
        front="ディープラーニングの利点は何ですか？",
        back="あ" * 250,
        card_type=CardType.QA,
        bloom_level=BloomLevel.UNDERSTAND,
        tags=["AI::深層学習"],
    )


@pytest.fixture
def list_as_qa_card() -> AnkiCard:
    """A card that has a list in the back but uses QA type instead of cloze."""
    return AnkiCard(
        front="機械学習の3つの種類を挙げてください。",
        back="1. 教師あり学習 2. 教師なし学習 3. 強化学習",
        card_type=CardType.QA,
        bloom_level=BloomLevel.REMEMBER,
        tags=["AI::基礎"],
    )


@pytest.fixture
def multi_concept_card() -> AnkiCard:
    """A card that violates atomicity by combining multiple concepts."""
    return AnkiCard(
        front="CNNとRNNの違いは何ですか？また、それぞれの用途を説明してください。",
        back="CNNは画像認識に使われ、畳み込み層を使う。RNNは時系列データに使われ、再帰的構造を持つ。またLSTMは勾配消失を解決する。",
        card_type=CardType.QA,
        bloom_level=BloomLevel.ANALYZE,
        tags=["AI::深層学習"],
    )


@pytest.fixture
def type_mismatch_cloze_card() -> AnkiCard:
    """A card marked as cloze but without cloze syntax."""
    return AnkiCard(
        front="勾配降下法はパラメータを更新する手法です。",
        back="",
        card_type=CardType.CLOZE,
        bloom_level=BloomLevel.REMEMBER,
        tags=["AI::最適化"],
    )


@pytest.fixture
def well_tagged_card() -> AnkiCard:
    """A card with excellent hierarchical tags."""
    return AnkiCard(
        front="バッチ正規化の目的は何ですか？",
        back="内部共変量シフトを軽減し、学習を安定化させる。",
        card_type=CardType.QA,
        bloom_level=BloomLevel.UNDERSTAND,
        tags=["AI::深層学習::正規化", "training::technique"],
        related_concepts=["dropout", "layer_normalization"],
    )


@pytest.fixture
def reversible_card() -> AnkiCard:
    """A well-formed reversible card."""
    return AnkiCard(
        front="ReLU",
        back="Rectified Linear Unit: f(x) = max(0, x)",
        card_type=CardType.REVERSIBLE,
        bloom_level=BloomLevel.REMEMBER,
        tags=["AI::活性化関数"],
    )


# ============================================================
# score_card() Tests
# ============================================================


class TestScoreCard:
    """Test individual card confidence scoring."""

    def test_high_quality_qa_scores_above_threshold(
        self, high_quality_qa_card: AnkiCard
    ) -> None:
        """A well-formed QA card should pass the 0.90 threshold."""
        score = score_card(high_quality_qa_card)
        assert isinstance(score, CardConfidenceScore)
        assert score.weighted_total >= 0.90
        assert score.flags == []

    def test_high_quality_cloze_scores_above_threshold(
        self, high_quality_cloze_card: AnkiCard
    ) -> None:
        """A well-formed cloze card should pass the 0.90 threshold."""
        score = score_card(high_quality_cloze_card)
        assert score.weighted_total >= 0.90
        assert score.flags == []

    def test_vague_question_flagged(self, vague_question_card: AnkiCard) -> None:
        """A card with a vague, short, non-question front should be flagged."""
        score = score_card(vague_question_card)
        assert QualityFlag.VAGUE_QUESTION in score.flags
        assert score.front_quality < 0.7

    def test_too_long_answer_flagged(self, too_long_answer_card: AnkiCard) -> None:
        """A card with a back >200 chars should be flagged."""
        score = score_card(too_long_answer_card)
        assert QualityFlag.TOO_LONG_ANSWER in score.flags
        assert score.back_quality < 0.7

    def test_list_not_cloze_flagged(self, list_as_qa_card: AnkiCard) -> None:
        """A QA card with enumeration in back should flag list_not_cloze."""
        score = score_card(list_as_qa_card)
        assert QualityFlag.LIST_NOT_CLOZE in score.flags

    def test_atomicity_violation_flagged(self, multi_concept_card: AnkiCard) -> None:
        """A multi-concept card should score low on atomicity."""
        score = score_card(multi_concept_card)
        assert score.atomicity < 0.7

    def test_cloze_type_mismatch_flagged(
        self, type_mismatch_cloze_card: AnkiCard
    ) -> None:
        """A cloze card without {{c1::...}} should score low on card_type_fit."""
        score = score_card(type_mismatch_cloze_card)
        assert score.card_type_fit < 0.7

    def test_well_tagged_card_high_tags_quality(
        self, well_tagged_card: AnkiCard
    ) -> None:
        """Card with hierarchical multi-tags should score high on tags_quality."""
        score = score_card(well_tagged_card)
        assert score.tags_quality >= 0.9

    def test_reversible_card_scores_high(self, reversible_card: AnkiCard) -> None:
        """A well-formed reversible card should score above threshold."""
        score = score_card(reversible_card)
        assert score.weighted_total >= 0.85

    def test_all_score_fields_in_range(self, high_quality_qa_card: AnkiCard) -> None:
        """All individual scores should be between 0.0 and 1.0."""
        score = score_card(high_quality_qa_card)
        assert 0.0 <= score.front_quality <= 1.0
        assert 0.0 <= score.back_quality <= 1.0
        assert 0.0 <= score.card_type_fit <= 1.0
        assert 0.0 <= score.bloom_level_fit <= 1.0
        assert 0.0 <= score.tags_quality <= 1.0
        assert 0.0 <= score.atomicity <= 1.0

    def test_weighted_total_matches_manual_calculation(
        self, high_quality_qa_card: AnkiCard
    ) -> None:
        """weighted_total should match manual calculation from field weights."""
        score = score_card(high_quality_qa_card)
        expected = (
            score.front_quality * 0.25
            + score.back_quality * 0.25
            + score.card_type_fit * 0.15
            + score.bloom_level_fit * 0.10
            + score.tags_quality * 0.10
            + score.atomicity * 0.15
        )
        assert score.weighted_total == pytest.approx(expected)

    def test_score_card_returns_immutable(
        self, high_quality_qa_card: AnkiCard
    ) -> None:
        """CardConfidenceScore should be immutable (frozen=True)."""
        score = score_card(high_quality_qa_card)
        with pytest.raises(ValidationError):
            score.front_quality = 0.0  # type: ignore[misc]


# ============================================================
# score_cards() Tests
# ============================================================


class TestScoreCards:
    """Test batch scoring."""

    def test_score_empty_list(self) -> None:
        """Empty list should return empty results."""
        results = score_cards([])
        assert results == []

    def test_score_multiple_cards(
        self,
        high_quality_qa_card: AnkiCard,
        high_quality_cloze_card: AnkiCard,
        vague_question_card: AnkiCard,
    ) -> None:
        """Should return one score per card."""
        cards = [high_quality_qa_card, high_quality_cloze_card, vague_question_card]
        results = score_cards(cards)
        assert len(results) == 3
        assert all(isinstance(r, CardConfidenceScore) for r in results)

    def test_score_cards_order_preserved(
        self,
        high_quality_qa_card: AnkiCard,
        vague_question_card: AnkiCard,
    ) -> None:
        """Scores should correspond to input order."""
        cards = [high_quality_qa_card, vague_question_card]
        results = score_cards(cards)
        assert results[0].weighted_total > results[1].weighted_total


# ============================================================
# Front Quality Scoring Details
# ============================================================


class TestFrontQualityScoring:
    """Test front_quality heuristics in detail."""

    def test_question_mark_boosts_score(self) -> None:
        """QA card with question mark in front should score higher."""
        card_with_q = AnkiCard(
            front="活性化関数の役割は何ですか？",
            back="非線形変換を導入する。",
            card_type=CardType.QA,
            bloom_level=BloomLevel.UNDERSTAND,
            tags=["AI"],
        )
        card_no_q = AnkiCard(
            front="活性化関数の役割について",
            back="非線形変換を導入する。",
            card_type=CardType.QA,
            bloom_level=BloomLevel.UNDERSTAND,
            tags=["AI"],
        )
        score_q = score_card(card_with_q)
        score_no_q = score_card(card_no_q)
        assert score_q.front_quality >= score_no_q.front_quality

    def test_very_short_front_low_score(self) -> None:
        """Front under 10 chars should get lower score."""
        card = AnkiCard(
            front="概要は？",
            back="テストの回答です。",
            card_type=CardType.QA,
            bloom_level=BloomLevel.REMEMBER,
            tags=["test"],
        )
        score = score_card(card)
        assert score.front_quality < 0.8

    def test_cloze_syntax_accepted_as_valid_front(self) -> None:
        """Cloze card with proper {{c1::...}} should score high on front."""
        card = AnkiCard(
            front="{{c1::勾配降下法}}は最適化アルゴリズムである。",
            back="",
            card_type=CardType.CLOZE,
            bloom_level=BloomLevel.REMEMBER,
            tags=["AI"],
        )
        score = score_card(card)
        assert score.front_quality >= 0.85


# ============================================================
# Back Quality Scoring Details
# ============================================================


class TestBackQualityScoring:
    """Test back_quality heuristics in detail."""

    def test_empty_back_ok_for_cloze(self) -> None:
        """Cloze cards should get high back_quality even with empty back."""
        card = AnkiCard(
            front="{{c1::勾配降下法}}は最適化手法。",
            back="",
            card_type=CardType.CLOZE,
            bloom_level=BloomLevel.REMEMBER,
            tags=["AI"],
        )
        score = score_card(card)
        assert score.back_quality >= 0.9

    def test_empty_back_bad_for_qa(self) -> None:
        """QA cards with empty back should get very low back_quality."""
        card = AnkiCard(
            front="活性化関数とは？",
            back="a",
            card_type=CardType.QA,
            bloom_level=BloomLevel.REMEMBER,
            tags=["AI"],
        )
        score = score_card(card)
        assert score.back_quality < 0.8

    def test_reasonable_length_back_high_score(self) -> None:
        """Back of 10-200 chars should score well."""
        card = AnkiCard(
            front="ReLUとは何ですか？",
            back="Rectified Linear Unit: f(x) = max(0, x)。負の入力を0にする。",
            card_type=CardType.QA,
            bloom_level=BloomLevel.REMEMBER,
            tags=["AI"],
        )
        score = score_card(card)
        assert score.back_quality >= 0.9


# ============================================================
# Atomicity Scoring Details
# ============================================================


class TestAtomicityScoring:
    """Test atomicity heuristics in detail."""

    def test_short_back_is_atomic(self) -> None:
        """A card with a short, focused answer is likely atomic."""
        card = AnkiCard(
            front="ReLUの出力は？",
            back="max(0, x)",
            card_type=CardType.QA,
            bloom_level=BloomLevel.REMEMBER,
            tags=["AI"],
        )
        score = score_card(card)
        assert score.atomicity >= 0.9

    def test_long_multi_sentence_back_low_atomicity(self) -> None:
        """A long back with multiple sentences suggests non-atomic content."""
        card = AnkiCard(
            front="ニューラルネットワークの主要要素は？",
            back="入力層がデータを受け取る。隠れ層が特徴を抽出する。出力層が結果を生成する。活性化関数が非線形性を加える。",
            card_type=CardType.QA,
            bloom_level=BloomLevel.UNDERSTAND,
            tags=["AI"],
        )
        score = score_card(card)
        assert score.atomicity < 0.7


# ============================================================
# Duplicate Detection
# ============================================================


class TestDuplicateDetection:
    """Test duplicate concept detection across cards."""

    def test_duplicate_flagged_in_batch(self) -> None:
        """Nearly identical cards should be flagged as duplicates."""
        card_a = AnkiCard(
            front="ReLUとは何ですか？",
            back="max(0, x)を出力する活性化関数。",
            card_type=CardType.QA,
            bloom_level=BloomLevel.REMEMBER,
            tags=["AI::活性化関数"],
        )
        card_b = AnkiCard(
            front="ReLU関数の定義は？",
            back="f(x) = max(0, x)の活性化関数。",
            card_type=CardType.QA,
            bloom_level=BloomLevel.REMEMBER,
            tags=["AI::活性化関数"],
        )
        scores = score_cards([card_a, card_b])
        has_duplicate = any(
            QualityFlag.DUPLICATE_CONCEPT in s.flags for s in scores
        )
        assert has_duplicate

    def test_cross_section_duplicates_detected(self) -> None:
        """Cards from different sections with same concept should be flagged.

        This tests the Phase 4 requirement: when cards from separate sections
        (identifiable by _section:: tags) cover the same concept, the quality
        pipeline should still detect them as duplicates.
        """
        # Simulates cards from section-0 and section-1 with same content
        card_from_section_0 = AnkiCard(
            front="ReLUとは何ですか？",
            back="max(0, x)を出力する活性化関数。",
            card_type=CardType.QA,
            bloom_level=BloomLevel.REMEMBER,
            tags=["AI::活性化関数", "_section::section-0"],
        )
        card_from_section_1 = AnkiCard(
            front="ReLU関数の定義は？",
            back="f(x) = max(0, x)の活性化関数。",
            card_type=CardType.QA,
            bloom_level=BloomLevel.REMEMBER,
            tags=["AI::活性化関数", "_section::section-1"],
        )
        scores = score_cards([card_from_section_0, card_from_section_1])
        has_duplicate = any(
            QualityFlag.DUPLICATE_CONCEPT in s.flags for s in scores
        )
        assert has_duplicate


# ============================================================
# critique_cards() Tests
# ============================================================


class TestCritiqueCards:
    """Test LLM critique for low-confidence cards."""

    def test_critique_returns_improved_cards(self) -> None:
        """critique_cards should return improved cards for low-confidence input."""
        low_quality_card = AnkiCard(
            front="概要",
            back="あ" * 250,
            card_type=CardType.QA,
            bloom_level=BloomLevel.REMEMBER,
            tags=["test"],
        )

        mock_response_data = [
            {
                "card_index": 0,
                "action": "improve",
                "reason": "Vague front, too long back",
                "flags": ["vague_question", "too_long_answer"],
                "improved_cards": [
                    {
                        "front": "機械学習の概要を説明してください。",
                        "back": "データからパターンを学習する手法。",
                        "card_type": "qa",
                        "bloom_level": "understand",
                        "tags": ["AI::基礎"],
                    }
                ],
            }
        ]

        mock_message = MagicMock()
        mock_message.content = [MagicMock(text=json.dumps(mock_response_data))]
        mock_message.model = "claude-sonnet-4-5-20250929"
        mock_message.usage = MagicMock(input_tokens=500, output_tokens=200)

        with patch("pdf2anki.quality._call_critique_api", return_value=mock_message):
            result_cards, tracker = critique_cards(
                cards=[low_quality_card],
                source_text="機械学習の概要テスト",
                cost_tracker=CostTracker(budget_limit=1.0),
            )
            assert len(result_cards) >= 1
            assert result_cards[0].front != "概要"

    def test_critique_remove_action(self) -> None:
        """Cards with 'remove' action should be excluded from results."""
        card = AnkiCard(
            front="テスト",
            back="テスト",
            card_type=CardType.QA,
            bloom_level=BloomLevel.REMEMBER,
            tags=["test"],
        )

        mock_response_data = [
            {
                "card_index": 0,
                "action": "remove",
                "reason": "Too simple",
                "flags": ["too_simple"],
                "improved_cards": None,
            }
        ]

        mock_message = MagicMock()
        mock_message.content = [MagicMock(text=json.dumps(mock_response_data))]
        mock_message.model = "claude-sonnet-4-5-20250929"
        mock_message.usage = MagicMock(input_tokens=300, output_tokens=100)

        with patch("pdf2anki.quality._call_critique_api", return_value=mock_message):
            result_cards, _ = critique_cards(
                cards=[card],
                source_text="テスト",
                cost_tracker=CostTracker(budget_limit=1.0),
            )
            assert len(result_cards) == 0

    def test_critique_split_action(self) -> None:
        """Cards with 'split' action should produce multiple replacement cards."""
        card = AnkiCard(
            front="CNNとRNNの特徴は？",
            back="CNNは画像認識、RNNは時系列処理に使われる。",
            card_type=CardType.QA,
            bloom_level=BloomLevel.ANALYZE,
            tags=["AI::深層学習"],
        )

        mock_response_data = [
            {
                "card_index": 0,
                "action": "split",
                "reason": "Multiple concepts",
                "flags": [],
                "improved_cards": [
                    {
                        "front": "CNNは主にどの分野で使われますか？",
                        "back": "画像認識",
                        "card_type": "qa",
                        "bloom_level": "remember",
                        "tags": ["AI::深層学習::CNN"],
                    },
                    {
                        "front": "RNNは主にどの分野で使われますか？",
                        "back": "時系列データの処理",
                        "card_type": "qa",
                        "bloom_level": "remember",
                        "tags": ["AI::深層学習::RNN"],
                    },
                ],
            }
        ]

        mock_message = MagicMock()
        mock_message.content = [MagicMock(text=json.dumps(mock_response_data))]
        mock_message.model = "claude-sonnet-4-5-20250929"
        mock_message.usage = MagicMock(input_tokens=400, output_tokens=300)

        with patch("pdf2anki.quality._call_critique_api", return_value=mock_message):
            result_cards, _ = critique_cards(
                cards=[card],
                source_text="CNN RNN テスト",
                cost_tracker=CostTracker(budget_limit=1.0),
            )
            assert len(result_cards) == 2

    def test_critique_empty_input(self) -> None:
        """Empty card list should return empty results without API call."""
        tracker = CostTracker(budget_limit=1.0)
        result_cards, result_tracker = critique_cards(
            cards=[],
            source_text="テスト",
            cost_tracker=tracker,
        )
        assert result_cards == []
        assert result_tracker is tracker


# ============================================================
# QualityReport Tests
# ============================================================


class TestQualityReport:
    """Test quality report generation."""

    def test_report_creation(self) -> None:
        report = QualityReport(
            total_cards=10,
            passed_cards=8,
            critiqued_cards=2,
            removed_cards=1,
            improved_cards=1,
            split_cards=0,
            final_card_count=9,
        )
        assert report.total_cards == 10
        assert report.passed_cards == 8
        assert report.pass_rate == pytest.approx(0.8)

    def test_report_zero_cards(self) -> None:
        report = QualityReport(
            total_cards=0,
            passed_cards=0,
            critiqued_cards=0,
            removed_cards=0,
            improved_cards=0,
            split_cards=0,
            final_card_count=0,
        )
        assert report.pass_rate == 0.0

    def test_report_immutable(self) -> None:
        report = QualityReport(
            total_cards=10,
            passed_cards=8,
            critiqued_cards=2,
            removed_cards=0,
            improved_cards=2,
            split_cards=0,
            final_card_count=10,
        )
        with pytest.raises(ValidationError):
            report.total_cards = 5  # type: ignore[misc]


# ============================================================
# run_quality_pipeline() Tests
# ============================================================


class TestRunQualityPipeline:
    """Test the full quality pipeline orchestrator."""

    def test_pipeline_passes_high_quality_cards(
        self, high_quality_qa_card: AnkiCard, high_quality_cloze_card: AnkiCard
    ) -> None:
        """High quality cards should pass through without critique."""
        config = AppConfig(
            quality_confidence_threshold=0.90,
            quality_enable_critique=False,
        )
        tracker = CostTracker(budget_limit=1.0)

        result_cards, report, result_tracker = run_quality_pipeline(
            cards=[high_quality_qa_card, high_quality_cloze_card],
            source_text="ニューラルネットワーク 勾配降下法",
            config=config,
            cost_tracker=tracker,
        )
        assert len(result_cards) == 2
        assert report.passed_cards == 2
        assert report.critiqued_cards == 0

    def test_pipeline_with_critique_disabled(
        self, vague_question_card: AnkiCard
    ) -> None:
        """With critique disabled, all cards pass through (even low quality)."""
        config = AppConfig(quality_enable_critique=False)
        tracker = CostTracker(budget_limit=1.0)

        result_cards, report, _ = run_quality_pipeline(
            cards=[vague_question_card],
            source_text="テスト",
            config=config,
            cost_tracker=tracker,
        )
        assert len(result_cards) == 1
        assert report.critiqued_cards == 0

    def test_pipeline_filters_low_quality_for_critique(
        self,
        high_quality_qa_card: AnkiCard,
        vague_question_card: AnkiCard,
    ) -> None:
        """Low quality cards should be sent for critique when enabled."""
        config = AppConfig(
            quality_confidence_threshold=0.90,
            quality_enable_critique=True,
        )
        tracker = CostTracker(budget_limit=1.0)

        improved_card = AnkiCard(
            front="機械学習の概要を説明してください。",
            back="データからパターンを学習する手法。",
            card_type=CardType.QA,
            bloom_level=BloomLevel.UNDERSTAND,
            tags=["AI::基礎"],
        )

        with patch(
            "pdf2anki.quality.critique_cards",
            return_value=([improved_card], tracker),
        ):
            result_cards, report, _ = run_quality_pipeline(
                cards=[high_quality_qa_card, vague_question_card],
                source_text="テスト",
                config=config,
                cost_tracker=tracker,
            )
            assert report.passed_cards >= 1
            assert report.critiqued_cards >= 1

    def test_pipeline_respects_max_rounds(
        self,
        vague_question_card: AnkiCard,
    ) -> None:
        """Pipeline should respect quality_max_critique_rounds."""
        config = AppConfig(
            quality_confidence_threshold=0.90,
            quality_enable_critique=True,
            quality_max_critique_rounds=0,
        )
        tracker = CostTracker(budget_limit=1.0)

        result_cards, report, _ = run_quality_pipeline(
            cards=[vague_question_card],
            source_text="テスト",
            config=config,
            cost_tracker=tracker,
        )
        assert report.critiqued_cards == 0

    def test_pipeline_empty_input(self) -> None:
        """Empty card list should produce empty results."""
        config = AppConfig()
        tracker = CostTracker(budget_limit=1.0)

        result_cards, report, _ = run_quality_pipeline(
            cards=[],
            source_text="",
            config=config,
            cost_tracker=tracker,
        )
        assert result_cards == []
        assert report.total_cards == 0
        assert report.final_card_count == 0


# ============================================================
# Scoring Edge Cases (coverage gaps)
# ============================================================


class TestFrontQualityScoringEdgeCases:
    """Cover uncovered branches in _score_front_quality."""

    def test_very_long_front_penalized(self) -> None:
        """Front exceeding 200 chars should get a penalty."""
        card = AnkiCard(
            front="あ" * 250 + "？",
            back="回答です。",
            card_type=CardType.QA,
            bloom_level=BloomLevel.REMEMBER,
            tags=["test"],
        )
        score = score_card(card)
        assert score.front_quality < 0.9

    def test_short_cloze_front_gets_0_7(self) -> None:
        """Cloze with valid syntax but short front gets 0.7."""
        card = AnkiCard(
            front="{{c1::A}}",
            back="",
            card_type=CardType.CLOZE,
            bloom_level=BloomLevel.REMEMBER,
            tags=["test"],
        )
        score = score_card(card)
        assert score.front_quality == pytest.approx(0.7)

    def test_reversible_long_front(self) -> None:
        """Reversible card with front > 80 chars gets 0.8."""
        card = AnkiCard(
            front="あ" * 90,
            back="説明テキスト",
            card_type=CardType.REVERSIBLE,
            bloom_level=BloomLevel.REMEMBER,
            tags=["test"],
        )
        score = score_card(card)
        assert score.front_quality == pytest.approx(0.8)


class TestBackQualityScoringEdgeCases:
    """Cover uncovered branches in _score_back_quality."""

    def test_empty_back_for_qa_returns_zero(self) -> None:
        """QA card with completely empty back gets 0.0."""
        card = AnkiCard(
            front="テストの質問ですか？",
            back="",
            card_type=CardType.QA,
            bloom_level=BloomLevel.REMEMBER,
            tags=["test"],
        )
        score = score_card(card)
        assert score.back_quality == pytest.approx(0.0)

    def test_cloze_with_non_empty_back(self) -> None:
        """Cloze card with non-empty back gets 0.8."""
        card = AnkiCard(
            front="{{c1::勾配降下法}}は最適化手法。",
            back="補足情報あり",
            card_type=CardType.CLOZE,
            bloom_level=BloomLevel.REMEMBER,
            tags=["test"],
        )
        score = score_card(card)
        assert score.back_quality == pytest.approx(0.8)


class TestCardTypeFitEdgeCases:
    """Cover uncovered branches in _score_card_type_fit."""

    def test_qa_with_cloze_syntax_penalty(self) -> None:
        """QA card containing cloze syntax should get penalized."""
        card = AnkiCard(
            front="{{c1::テスト}}とは何ですか？",
            back="テストの回答です。",
            card_type=CardType.QA,
            bloom_level=BloomLevel.REMEMBER,
            tags=["test"],
        )
        score = score_card(card)
        assert score.card_type_fit == pytest.approx(0.5)

    def test_term_definition_long_front(self) -> None:
        """Term definition with long front (>50) gets 0.7."""
        card = AnkiCard(
            front="非常に長い用語名を持つ定義カードのテストです" + "あ" * 40,
            back="定義の内容",
            card_type=CardType.TERM_DEFINITION,
            bloom_level=BloomLevel.REMEMBER,
            tags=["test"],
        )
        score = score_card(card)
        assert score.card_type_fit == pytest.approx(0.7)

    def test_reversible_long_front_fit(self) -> None:
        """Reversible card with long front (>80) gets 0.7."""
        card = AnkiCard(
            front="あ" * 90,
            back="対応する回答",
            card_type=CardType.REVERSIBLE,
            bloom_level=BloomLevel.REMEMBER,
            tags=["test"],
        )
        score = score_card(card)
        assert score.card_type_fit == pytest.approx(0.7)

    def test_compare_contrast_with_keyword(self) -> None:
        """Compare/contrast card with matching keyword gets 1.0."""
        card = AnkiCard(
            front="CNNとRNNの違いは何ですか？",
            back="CNNは画像、RNNは時系列。",
            card_type=CardType.COMPARE_CONTRAST,
            bloom_level=BloomLevel.ANALYZE,
            tags=["AI"],
        )
        score = score_card(card)
        assert score.card_type_fit == pytest.approx(1.0)

    def test_compare_contrast_without_keyword(self) -> None:
        """Compare/contrast card without keyword gets 0.7."""
        card = AnkiCard(
            front="CNNとRNNについて",
            back="それぞれの特徴を述べよ。",
            card_type=CardType.COMPARE_CONTRAST,
            bloom_level=BloomLevel.ANALYZE,
            tags=["AI"],
        )
        score = score_card(card)
        assert score.card_type_fit == pytest.approx(0.7)

    def test_sequence_with_keyword(self) -> None:
        """Sequence card with matching keyword gets 1.0."""
        card = AnkiCard(
            front="データ前処理の次のステップは？",
            back="特徴量エンジニアリング。",
            card_type=CardType.SEQUENCE,
            bloom_level=BloomLevel.APPLY,
            tags=["ML"],
        )
        score = score_card(card)
        assert score.card_type_fit == pytest.approx(1.0)

    def test_sequence_without_keyword(self) -> None:
        """Sequence card without keyword gets 0.7."""
        card = AnkiCard(
            front="データ前処理について",
            back="重要な概念。",
            card_type=CardType.SEQUENCE,
            bloom_level=BloomLevel.APPLY,
            tags=["ML"],
        )
        score = score_card(card)
        assert score.card_type_fit == pytest.approx(0.7)

    def test_image_occlusion_default(self) -> None:
        """Image occlusion (fallback branch) gets 0.8."""
        card = AnkiCard(
            front="画像の一部を隠した問題",
            back="隠された部分の答え",
            card_type=CardType.IMAGE_OCCLUSION,
            bloom_level=BloomLevel.REMEMBER,
            tags=["test"],
        )
        score = score_card(card)
        assert score.card_type_fit == pytest.approx(0.8)


class TestAtomicityEdgeCases:
    """Cover uncovered branches in _score_atomicity."""

    def test_many_cloze_deletions_penalized(self) -> None:
        """Cloze card with >3 deletions should get penalized."""
        card = AnkiCard(
            front="{{c1::A}}と{{c2::B}}と{{c3::C}}と{{c4::D}}と{{c5::E}}は重要。",
            back="",
            card_type=CardType.CLOZE,
            bloom_level=BloomLevel.REMEMBER,
            tags=["test"],
        )
        score = score_card(card)
        assert score.atomicity < 1.0
        assert score.atomicity >= 0.5

    def test_empty_back_non_cloze_atomicity(self) -> None:
        """Non-cloze card with empty back gets 0.9 atomicity."""
        card = AnkiCard(
            front="テスト質問ですか？",
            back="",
            card_type=CardType.QA,
            bloom_level=BloomLevel.REMEMBER,
            tags=["test"],
        )
        score = score_card(card)
        assert score.atomicity == pytest.approx(0.9)


# ============================================================
# Similarity helper edge cases
# ============================================================


class TestSimilarityHelpers:
    """Test _char_bigrams, _tokenize, _jaccard edge cases."""

    def test_char_bigrams_single_char(self) -> None:
        """Single character text returns a set with that character."""
        assert _char_bigrams("A") == {"A"}

    def test_char_bigrams_empty(self) -> None:
        """Empty text returns empty set."""
        assert _char_bigrams("") == set()

    def test_jaccard_empty_sets(self) -> None:
        """Jaccard with empty set returns 0.0."""
        assert _jaccard(set(), {"a"}) == pytest.approx(0.0)
        assert _jaccard({"a"}, set()) == pytest.approx(0.0)

    def test_jaccard_identical_sets(self) -> None:
        """Jaccard with identical sets returns 1.0."""
        assert _jaccard({"a", "b"}, {"a", "b"}) == pytest.approx(1.0)

    def test_tokenize_punctuation_split(self) -> None:
        """Tokenize splits on Japanese and English punctuation."""
        tokens = _tokenize("AI、機械学習。深層学習")
        assert "AI" in tokens
        assert "機械学習" in tokens
        assert "深層学習" in tokens


class TestTokenizeCJK:
    """Test CJK-aware tokenization."""

    def test_japanese_text_produces_multiple_tokens(self) -> None:
        """Japanese text without spaces should produce character-level tokens."""
        tokens = _tokenize("活性化関数の役割")
        assert len(tokens) > 1

    def test_mixed_japanese_english(self) -> None:
        """Mixed JP/EN text should tokenize both parts."""
        tokens = _tokenize("ReLU活性化関数")
        assert any("ReLU" in t for t in tokens)
        assert len(tokens) >= 2

    def test_chinese_text_produces_tokens(self) -> None:
        """Chinese text should also be properly tokenized."""
        tokens = _tokenize("神经网络的激活函数")
        assert len(tokens) > 1

    def test_english_only_unchanged(self) -> None:
        """Pure English tokenization should still work as before."""
        tokens = _tokenize("neural network activation function")
        assert "neural" in tokens
        assert "network" in tokens


class TestCardsSimilarAdditionalBranches:
    """Cover remaining similarity detection branches."""

    def test_high_front_similarity_same_type(self) -> None:
        """Cards with >0.7 front bigram Jaccard and same type are duplicates."""
        card_a = AnkiCard(
            front="ニューラルネットワークの活性化関数の役割",
            back="非線形変換",
            card_type=CardType.QA,
            bloom_level=BloomLevel.UNDERSTAND,
            tags=["AI"],
        )
        card_b = AnkiCard(
            front="ニューラルネットワークの活性化関数の機能",
            back="入力の変換",
            card_type=CardType.QA,
            bloom_level=BloomLevel.UNDERSTAND,
            tags=["ML"],
        )
        scores = score_cards([card_a, card_b])
        has_dup = any(QualityFlag.DUPLICATE_CONCEPT in s.flags for s in scores)
        assert has_dup

    def test_shared_tags_moderate_front_similarity(self) -> None:
        """Cards with same tags and moderate front similarity are duplicates."""
        card_a = AnkiCard(
            front="勾配降下法のアルゴリズムの説明",
            back="パラメータを更新する手法。",
            card_type=CardType.QA,
            bloom_level=BloomLevel.UNDERSTAND,
            tags=["AI::最適化"],
        )
        card_b = AnkiCard(
            front="勾配降下法のアルゴリズムの概要",
            back="損失関数を最小化する方法。",
            card_type=CardType.TERM_DEFINITION,
            bloom_level=BloomLevel.REMEMBER,
            tags=["AI::最適化"],
        )
        scores = score_cards([card_a, card_b])
        has_dup = any(QualityFlag.DUPLICATE_CONCEPT in s.flags for s in scores)
        assert has_dup

    def test_shared_tags_moderate_back_similarity(self) -> None:
        """Cards with same tags, same type, moderate back similarity are duplicates."""
        card_a = AnkiCard(
            front="ReLU？",
            back="Rectified Linear Unit 活性化関数 f(x) = max(0, x)",
            card_type=CardType.QA,
            bloom_level=BloomLevel.REMEMBER,
            tags=["AI::活性化関数"],
        )
        card_b = AnkiCard(
            front="活性化関数ReLUの定義",
            back="Rectified Linear Unit 関数 f(x) = max(0, x) の定義",
            card_type=CardType.QA,
            bloom_level=BloomLevel.REMEMBER,
            tags=["AI::活性化関数"],
        )
        scores = score_cards([card_a, card_b])
        has_dup = any(QualityFlag.DUPLICATE_CONCEPT in s.flags for s in scores)
        assert has_dup

    def test_not_similar_cards(self) -> None:
        """Completely different cards should not be flagged as duplicates."""
        card_a = AnkiCard(
            front="ReLUとは？",
            back="活性化関数の一種。",
            card_type=CardType.QA,
            bloom_level=BloomLevel.REMEMBER,
            tags=["AI::活性化関数"],
        )
        card_b = AnkiCard(
            front="SQLインジェクションとは？",
            back="悪意あるSQL文を注入する攻撃。",
            card_type=CardType.QA,
            bloom_level=BloomLevel.REMEMBER,
            tags=["security::攻撃"],
        )
        scores = score_cards([card_a, card_b])
        has_dup = any(QualityFlag.DUPLICATE_CONCEPT in s.flags for s in scores)
        assert not has_dup


# ============================================================
# _parse_critique_response edge cases
# ============================================================


class TestParseCritiqueResponse:
    """Cover uncovered paths in _parse_critique_response."""

    def test_json_in_code_block(self) -> None:
        """Should extract JSON from ```json code block."""
        response = '```json\n[{"card_index": 0, "action": "keep"}]\n```'
        result = _parse_critique_response(response)
        assert len(result) == 1
        assert result[0]["action"] == "keep"

    def test_invalid_json(self) -> None:
        """Should return empty list for invalid JSON."""
        result = _parse_critique_response("{not valid json")
        assert result == []

    def test_non_list_json(self) -> None:
        """Should return empty list for non-array JSON."""
        result = _parse_critique_response('{"key": "value"}')
        assert result == []

    def test_non_dict_items_skipped(self) -> None:
        """Non-dict items in the array should be skipped."""
        response = json.dumps([
            "string_item",
            {"card_index": 0, "action": "keep"},
        ])
        result = _parse_critique_response(response)
        assert len(result) == 1

    def test_missing_required_fields_skipped(self) -> None:
        """Items missing card_index or action should be skipped."""
        response = json.dumps([
            {"card_index": 0},
            {"action": "keep"},
            {"card_index": 1, "action": "improve"},
        ])
        result = _parse_critique_response(response)
        assert len(result) == 1
        assert result[0]["card_index"] == 1


# ============================================================
# critique_cards edge cases
# ============================================================


class TestCritiqueCardsEdgeCases:
    """Cover uncovered paths in critique_cards."""

    def test_api_error_returns_original_cards(self) -> None:
        """API error should return original cards unchanged."""
        import anthropic

        card = AnkiCard(
            front="テスト質問ですか？",
            back="テスト回答。",
            card_type=CardType.QA,
            bloom_level=BloomLevel.REMEMBER,
            tags=["test"],
        )
        tracker = CostTracker(budget_limit=1.0)

        with patch(
            "pdf2anki.quality._call_critique_api",
            side_effect=anthropic.APIConnectionError(request=MagicMock()),
        ):
            result_cards, result_tracker = critique_cards(
                cards=[card],
                source_text="テスト",
                cost_tracker=tracker,
            )
            assert len(result_cards) == 1
            assert result_cards[0] is card
            assert result_tracker is tracker

    def test_empty_response_content(self) -> None:
        """Empty response content returns original cards."""
        card = AnkiCard(
            front="テスト質問ですか？",
            back="テスト回答。",
            card_type=CardType.QA,
            bloom_level=BloomLevel.REMEMBER,
            tags=["test"],
        )
        mock_message = MagicMock()
        mock_message.content = []
        mock_message.model = "claude-sonnet-4-5-20250929"
        mock_message.usage = MagicMock(input_tokens=100, output_tokens=0)

        with patch("pdf2anki.quality._call_critique_api", return_value=mock_message):
            result_cards, _ = critique_cards(
                cards=[card],
                source_text="テスト",
                cost_tracker=CostTracker(budget_limit=1.0),
            )
            assert len(result_cards) == 1

    def test_keep_action_preserves_card(self) -> None:
        """Cards with 'keep' action are preserved in output."""
        card = AnkiCard(
            front="良い質問ですか？",
            back="良い回答です。",
            card_type=CardType.QA,
            bloom_level=BloomLevel.UNDERSTAND,
            tags=["test"],
        )
        mock_response = [
            {"card_index": 0, "action": "keep", "reason": "Good card"},
        ]
        mock_message = MagicMock()
        mock_message.content = [MagicMock(text=json.dumps(mock_response))]
        mock_message.model = "claude-sonnet-4-5-20250929"
        mock_message.usage = MagicMock(input_tokens=200, output_tokens=100)

        with patch("pdf2anki.quality._call_critique_api", return_value=mock_message):
            result_cards, _ = critique_cards(
                cards=[card],
                source_text="テスト",
                cost_tracker=CostTracker(budget_limit=1.0),
            )
            assert len(result_cards) == 1
            assert result_cards[0].front == "良い質問ですか？"

    def test_invalid_card_index_skipped(self) -> None:
        """Reviews with None or non-int card_index should be skipped."""
        card = AnkiCard(
            front="テスト質問ですか？",
            back="テスト回答。",
            card_type=CardType.QA,
            bloom_level=BloomLevel.REMEMBER,
            tags=["test"],
        )
        mock_response = [
            {"card_index": None, "action": "remove"},
            {"card_index": "bad", "action": "remove"},
        ]
        mock_message = MagicMock()
        mock_message.content = [MagicMock(text=json.dumps(mock_response))]
        mock_message.model = "claude-sonnet-4-5-20250929"
        mock_message.usage = MagicMock(input_tokens=200, output_tokens=100)

        with patch("pdf2anki.quality._call_critique_api", return_value=mock_message):
            result_cards, _ = critique_cards(
                cards=[card],
                source_text="テスト",
                cost_tracker=CostTracker(budget_limit=1.0),
            )
            # Card not reviewed -> passes through
            assert len(result_cards) == 1

    def test_unreviewed_cards_pass_through(self) -> None:
        """Cards not mentioned in critique response should pass through."""
        cards = [
            AnkiCard(
                front="質問1ですか？",
                back="回答1。",
                card_type=CardType.QA,
                bloom_level=BloomLevel.REMEMBER,
                tags=["test"],
            ),
            AnkiCard(
                front="質問2ですか？",
                back="回答2。",
                card_type=CardType.QA,
                bloom_level=BloomLevel.REMEMBER,
                tags=["test"],
            ),
        ]
        mock_response = [
            {"card_index": 0, "action": "remove"},
        ]
        mock_message = MagicMock()
        mock_message.content = [MagicMock(text=json.dumps(mock_response))]
        mock_message.model = "claude-sonnet-4-5-20250929"
        mock_message.usage = MagicMock(input_tokens=200, output_tokens=100)

        with patch("pdf2anki.quality._call_critique_api", return_value=mock_message):
            result_cards, _ = critique_cards(
                cards=cards,
                source_text="テスト",
                cost_tracker=CostTracker(budget_limit=1.0),
            )
            assert len(result_cards) == 1
            assert result_cards[0].front == "質問2ですか？"


# ============================================================
# Pipeline report counting edge cases
# ============================================================


class TestPipelineReportCounting:
    """Cover removed_count and split_count branches in run_quality_pipeline."""

    def test_pipeline_counts_removed_cards(
        self, vague_question_card: AnkiCard
    ) -> None:
        """Report should count removed cards when critique removes some."""
        config = AppConfig(
            quality_confidence_threshold=0.90,
            quality_enable_critique=True,
        )
        tracker = CostTracker(budget_limit=1.0)

        with patch(
            "pdf2anki.quality.critique_cards",
            return_value=([], tracker),
        ):
            _, report, _ = run_quality_pipeline(
                cards=[vague_question_card],
                source_text="テスト",
                config=config,
                cost_tracker=tracker,
            )
            assert report.removed_cards >= 1

    def test_pipeline_counts_split_cards(
        self, vague_question_card: AnkiCard
    ) -> None:
        """Report should count split cards when critique splits some."""
        config = AppConfig(
            quality_confidence_threshold=0.90,
            quality_enable_critique=True,
        )
        tracker = CostTracker(budget_limit=1.0)

        split_card_a = AnkiCard(
            front="質問Aですか？",
            back="回答A。",
            card_type=CardType.QA,
            bloom_level=BloomLevel.REMEMBER,
            tags=["test"],
        )
        split_card_b = AnkiCard(
            front="質問Bですか？",
            back="回答B。",
            card_type=CardType.QA,
            bloom_level=BloomLevel.REMEMBER,
            tags=["test"],
        )

        with patch(
            "pdf2anki.quality.critique_cards",
            return_value=([split_card_a, split_card_b], tracker),
        ):
            _, report, _ = run_quality_pipeline(
                cards=[vague_question_card],
                source_text="テスト",
                config=config,
                cost_tracker=tracker,
            )
            assert report.split_cards >= 1
