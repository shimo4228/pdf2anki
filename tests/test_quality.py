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

from pdf2anki.quality import (
    QualityReport,
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
def too_simple_card() -> AnkiCard:
    """A trivially simple card."""
    return AnkiCard(
        front="AIは何の略ですか？",
        back="人工知能",
        card_type=CardType.TERM_DEFINITION,
        bloom_level=BloomLevel.REMEMBER,
        tags=["AI"],
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
        with pytest.raises(Exception):
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
            from pdf2anki.cost import CostTracker

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
            from pdf2anki.cost import CostTracker

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
            from pdf2anki.cost import CostTracker

            result_cards, _ = critique_cards(
                cards=[card],
                source_text="CNN RNN テスト",
                cost_tracker=CostTracker(budget_limit=1.0),
            )
            assert len(result_cards) == 2

    def test_critique_empty_input(self) -> None:
        """Empty card list should return empty results without API call."""
        from pdf2anki.cost import CostTracker

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
        with pytest.raises(Exception):
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
        from pdf2anki.config import AppConfig
        from pdf2anki.cost import CostTracker

        config = AppConfig(quality_confidence_threshold=0.90, quality_enable_critique=False)
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
        from pdf2anki.config import AppConfig
        from pdf2anki.cost import CostTracker

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
        from pdf2anki.config import AppConfig
        from pdf2anki.cost import CostTracker

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
        from pdf2anki.config import AppConfig
        from pdf2anki.cost import CostTracker

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
        from pdf2anki.config import AppConfig
        from pdf2anki.cost import CostTracker

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
