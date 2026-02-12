"""Tests for pdf2anki.eval — Prompt Evaluation Framework."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from pdf2anki.schemas import AnkiCard, BloomLevel, CardType

# ---------------------------------------------------------------------------
# dataset.py tests
# ---------------------------------------------------------------------------


class TestExpectedCard:
    """Tests for ExpectedCard dataclass."""

    def test_create_with_required_fields(self) -> None:
        from pdf2anki.eval.dataset import ExpectedCard

        ec = ExpectedCard(
            front_keywords=["四諦", "何"],
            back_keywords=["苦諦", "集諦"],
        )
        assert ec.front_keywords == ["四諦", "何"]
        assert ec.back_keywords == ["苦諦", "集諦"]
        assert ec.card_type is None
        assert ec.tags == ()

    def test_create_with_all_fields(self) -> None:
        from pdf2anki.eval.dataset import ExpectedCard

        ec = ExpectedCard(
            front_keywords=["ML"],
            back_keywords=["supervised"],
            card_type=CardType.QA,
            tags=("ai", "ml"),
        )
        assert ec.card_type == CardType.QA
        assert ec.tags == ("ai", "ml")

    def test_frozen(self) -> None:
        from pdf2anki.eval.dataset import ExpectedCard

        ec = ExpectedCard(front_keywords=["a"], back_keywords=["b"])
        with pytest.raises(AttributeError):
            ec.front_keywords = ["x"]  # type: ignore[misc]


class TestEvalCase:
    """Tests for EvalCase dataclass."""

    def test_create(self) -> None:
        from pdf2anki.eval.dataset import EvalCase, ExpectedCard

        ec = ExpectedCard(front_keywords=["x"], back_keywords=["y"])
        case = EvalCase(
            id="test-01",
            text="Sample text",
            expected_cards=(ec,),
            description="A test case",
        )
        assert case.id == "test-01"
        assert len(case.expected_cards) == 1

    def test_default_description(self) -> None:
        from pdf2anki.eval.dataset import EvalCase, ExpectedCard

        ec = ExpectedCard(front_keywords=["x"], back_keywords=["y"])
        case = EvalCase(id="t", text="t", expected_cards=(ec,))
        assert case.description == ""


class TestEvalDataset:
    """Tests for EvalDataset dataclass."""

    def test_create(self) -> None:
        from pdf2anki.eval.dataset import EvalCase, EvalDataset, ExpectedCard

        ec = ExpectedCard(front_keywords=["a"], back_keywords=["b"])
        case = EvalCase(id="c1", text="text", expected_cards=(ec,))
        ds = EvalDataset(name="test", version="1.0", cases=(case,))
        assert ds.name == "test"
        assert len(ds.cases) == 1


class TestLoadDataset:
    """Tests for load_dataset function."""

    def test_load_valid_yaml(self, tmp_path: Path) -> None:
        from pdf2anki.eval.dataset import load_dataset

        data = {
            "name": "test-ds",
            "version": "1.0",
            "cases": [
                {
                    "id": "case-01",
                    "text": "四諦とは仏教の根本教義",
                    "expected_cards": [
                        {
                            "front_keywords": ["四諦"],
                            "back_keywords": ["苦諦", "集諦"],
                        },
                    ],
                },
            ],
        }
        yaml_path = tmp_path / "dataset.yaml"
        yaml_path.write_text(yaml.dump(data, allow_unicode=True))

        ds = load_dataset(yaml_path)
        assert ds.name == "test-ds"
        assert ds.version == "1.0"
        assert len(ds.cases) == 1
        assert ds.cases[0].id == "case-01"
        assert ds.cases[0].expected_cards[0].front_keywords == ["四諦"]

    def test_load_with_card_type(self, tmp_path: Path) -> None:
        from pdf2anki.eval.dataset import load_dataset

        data = {
            "name": "typed",
            "version": "1.0",
            "cases": [
                {
                    "id": "c1",
                    "text": "text",
                    "expected_cards": [
                        {
                            "front_keywords": ["x"],
                            "back_keywords": ["y"],
                            "card_type": "qa",
                        },
                    ],
                },
            ],
        }
        yaml_path = tmp_path / "ds.yaml"
        yaml_path.write_text(yaml.dump(data))

        ds = load_dataset(yaml_path)
        assert ds.cases[0].expected_cards[0].card_type == CardType.QA

    def test_load_with_tags(self, tmp_path: Path) -> None:
        from pdf2anki.eval.dataset import load_dataset

        data = {
            "name": "tagged",
            "version": "1.0",
            "cases": [
                {
                    "id": "c1",
                    "text": "text",
                    "expected_cards": [
                        {
                            "front_keywords": ["x"],
                            "back_keywords": ["y"],
                            "tags": ["ai", "ml"],
                        },
                    ],
                },
            ],
        }
        yaml_path = tmp_path / "ds.yaml"
        yaml_path.write_text(yaml.dump(data))

        ds = load_dataset(yaml_path)
        assert ds.cases[0].expected_cards[0].tags == ("ai", "ml")

    def test_load_missing_file(self, tmp_path: Path) -> None:
        from pdf2anki.eval.dataset import load_dataset

        with pytest.raises(FileNotFoundError):
            load_dataset(tmp_path / "missing.yaml")

    def test_load_multiple_cases(self, tmp_path: Path) -> None:
        from pdf2anki.eval.dataset import load_dataset

        data = {
            "name": "multi",
            "version": "2.0",
            "cases": [
                {
                    "id": f"c{i}",
                    "text": f"text {i}",
                    "expected_cards": [
                        {
                            "front_keywords": [f"k{i}"],
                            "back_keywords": [f"v{i}"],
                        },
                    ],
                }
                for i in range(5)
            ],
        }
        yaml_path = tmp_path / "ds.yaml"
        yaml_path.write_text(yaml.dump(data))

        ds = load_dataset(yaml_path)
        assert len(ds.cases) == 5


# ---------------------------------------------------------------------------
# matcher.py tests
# ---------------------------------------------------------------------------


def _make_card(
    front: str = "Q",
    back: str = "A",
    card_type: CardType = CardType.QA,
) -> AnkiCard:
    """Helper to create AnkiCard for tests."""
    return AnkiCard(
        front=front,
        back=back,
        card_type=card_type,
        bloom_level=BloomLevel.REMEMBER,
        tags=["test"],
    )


class TestMatchCards:
    """Tests for match_cards function."""

    def test_perfect_match(self) -> None:
        from pdf2anki.eval.dataset import ExpectedCard
        from pdf2anki.eval.matcher import match_cards

        expected = [
            ExpectedCard(
                front_keywords=["四諦", "何"],
                back_keywords=["苦諦", "集諦", "滅諦", "道諦"],
                card_type=CardType.QA,
            ),
        ]
        generated = [
            _make_card(
                front="四諦とは何ですか？",
                back="苦諦、集諦、滅諦、道諦の4つの真理",
            ),
        ]
        result = match_cards(expected, generated, case_id="t1")
        assert len(result.matches) == 1
        assert result.matches[0].matched_card is not None
        assert result.matches[0].similarity > 0.5

    def test_no_match(self) -> None:
        from pdf2anki.eval.dataset import ExpectedCard
        from pdf2anki.eval.matcher import match_cards

        expected = [
            ExpectedCard(
                front_keywords=["量子コンピュータ"],
                back_keywords=["量子ビット"],
            ),
        ]
        generated = [
            _make_card(front="機械学習とは？", back="データからパターン"),
        ]
        result = match_cards(expected, generated, case_id="t2")
        assert len(result.matches) == 1
        assert result.matches[0].matched_card is None
        assert result.matches[0].similarity < 0.5

    def test_unmatched_generated(self) -> None:
        from pdf2anki.eval.dataset import ExpectedCard
        from pdf2anki.eval.matcher import match_cards

        expected = [
            ExpectedCard(
                front_keywords=["四諦"],
                back_keywords=["苦諦"],
            ),
        ]
        generated = [
            _make_card(front="四諦とは何か", back="苦諦を含む"),
            _make_card(front="余分なカード", back="関係ないもの"),
        ]
        result = match_cards(expected, generated, case_id="t3")
        assert len(result.unmatched_generated) >= 1

    def test_multiple_expected_one_match(self) -> None:
        from pdf2anki.eval.dataset import ExpectedCard
        from pdf2anki.eval.matcher import match_cards

        expected = [
            ExpectedCard(
                front_keywords=["四諦"],
                back_keywords=["苦諦"],
            ),
            ExpectedCard(
                front_keywords=["八正道"],
                back_keywords=["正見"],
            ),
        ]
        generated = [
            _make_card(front="四諦について", back="苦諦は第一の真理"),
        ]
        result = match_cards(expected, generated, case_id="t4")
        matched_count = sum(1 for m in result.matches if m.matched_card is not None)
        assert matched_count == 1

    def test_card_type_bonus(self) -> None:
        from pdf2anki.eval.dataset import ExpectedCard
        from pdf2anki.eval.matcher import match_cards

        expected = [
            ExpectedCard(
                front_keywords=["ML"],
                back_keywords=["学習"],
                card_type=CardType.QA,
            ),
        ]
        card_qa = _make_card(front="MLとは", back="機械学習", card_type=CardType.QA)
        card_cloze = _make_card(
            front="MLとは", back="機械学習", card_type=CardType.CLOZE
        )

        result_qa = match_cards(expected, [card_qa], case_id="qa")
        result_cloze = match_cards(expected, [card_cloze], case_id="cloze")

        # QA match should have higher similarity due to type bonus
        if result_qa.matches[0].matched_card and result_cloze.matches[0].matched_card:
            assert result_qa.matches[0].similarity >= result_cloze.matches[0].similarity

    def test_empty_generated(self) -> None:
        from pdf2anki.eval.dataset import ExpectedCard
        from pdf2anki.eval.matcher import match_cards

        expected = [
            ExpectedCard(front_keywords=["x"], back_keywords=["y"]),
        ]
        result = match_cards(expected, [], case_id="empty")
        assert len(result.matches) == 1
        assert result.matches[0].matched_card is None
        assert result.matches[0].similarity == 0.0

    def test_empty_expected(self) -> None:
        from pdf2anki.eval.matcher import match_cards

        generated = [_make_card()]
        result = match_cards([], generated, case_id="no-exp")
        assert len(result.matches) == 0
        assert len(result.unmatched_generated) == 1


class TestKeywordSimilarity:
    """Tests for _keyword_similarity helper."""

    def test_full_overlap(self) -> None:
        from pdf2anki.eval.matcher import _keyword_similarity

        assert _keyword_similarity(["a", "b"], "a and b are here") == 1.0

    def test_no_overlap(self) -> None:
        from pdf2anki.eval.matcher import _keyword_similarity

        assert _keyword_similarity(["x", "y"], "nothing matches") == 0.0

    def test_partial_overlap(self) -> None:
        from pdf2anki.eval.matcher import _keyword_similarity

        sim = _keyword_similarity(["a", "b", "c"], "a and b")
        assert 0.3 < sim < 0.8

    def test_empty_keywords(self) -> None:
        from pdf2anki.eval.matcher import _keyword_similarity

        assert _keyword_similarity([], "some text") == 0.0


# ---------------------------------------------------------------------------
# metrics.py tests
# ---------------------------------------------------------------------------


class TestCalculateMetrics:
    """Tests for calculate_metrics function."""

    def test_perfect_metrics(self) -> None:
        from pdf2anki.eval.dataset import ExpectedCard
        from pdf2anki.eval.matcher import CaseResult, MatchResult
        from pdf2anki.eval.metrics import calculate_metrics

        card = _make_card(front="Q", back="A")
        ec = ExpectedCard(front_keywords=["Q"], back_keywords=["A"])
        case_results = [
            CaseResult(
                case_id="c1",
                generated_cards=(card,),
                matches=(MatchResult(expected=ec, matched_card=card, similarity=1.0),),
                unmatched_generated=(),
            ),
        ]
        m = calculate_metrics(case_results)
        assert m.recall == 1.0
        assert m.precision == 1.0
        assert m.f1 == 1.0
        assert m.total_expected == 1
        assert m.total_generated == 1
        assert m.total_matched == 1

    def test_zero_recall(self) -> None:
        from pdf2anki.eval.dataset import ExpectedCard
        from pdf2anki.eval.matcher import CaseResult, MatchResult
        from pdf2anki.eval.metrics import calculate_metrics

        ec = ExpectedCard(front_keywords=["Q"], back_keywords=["A"])
        case_results = [
            CaseResult(
                case_id="c1",
                generated_cards=(),
                matches=(MatchResult(expected=ec, matched_card=None, similarity=0.0),),
                unmatched_generated=(),
            ),
        ]
        m = calculate_metrics(case_results)
        assert m.recall == 0.0
        assert m.total_matched == 0

    def test_low_precision(self) -> None:
        from pdf2anki.eval.dataset import ExpectedCard
        from pdf2anki.eval.matcher import CaseResult, MatchResult
        from pdf2anki.eval.metrics import calculate_metrics

        card = _make_card()
        extra1 = _make_card(front="Extra 1", back="Noise")
        extra2 = _make_card(front="Extra 2", back="Noise")
        ec = ExpectedCard(front_keywords=["Q"], back_keywords=["A"])
        case_results = [
            CaseResult(
                case_id="c1",
                generated_cards=(card, extra1, extra2),
                matches=(MatchResult(expected=ec, matched_card=card, similarity=0.8),),
                unmatched_generated=(extra1, extra2),
            ),
        ]
        m = calculate_metrics(case_results)
        assert m.recall == 1.0
        assert m.precision == pytest.approx(1 / 3, abs=0.01)

    def test_with_cost(self) -> None:
        from pdf2anki.eval.metrics import calculate_metrics

        m = calculate_metrics([], cost_usd=0.42)
        assert m.total_cost_usd == 0.42

    def test_empty_results(self) -> None:
        from pdf2anki.eval.metrics import calculate_metrics

        m = calculate_metrics([])
        assert m.recall == 0.0
        assert m.precision == 0.0
        assert m.f1 == 0.0
        assert m.total_expected == 0
        assert m.total_generated == 0

    def test_multiple_cases(self) -> None:
        from pdf2anki.eval.dataset import ExpectedCard
        from pdf2anki.eval.matcher import CaseResult, MatchResult
        from pdf2anki.eval.metrics import calculate_metrics

        c1 = _make_card(front="c1-q", back="c1-a")
        c2 = _make_card(front="c2-q", back="c2-a")
        ec1 = ExpectedCard(front_keywords=["c1"], back_keywords=["a"])
        ec2 = ExpectedCard(front_keywords=["c2"], back_keywords=["a"])
        case_results = [
            CaseResult(
                case_id="c1",
                generated_cards=(c1,),
                matches=(MatchResult(expected=ec1, matched_card=c1, similarity=0.9),),
                unmatched_generated=(),
            ),
            CaseResult(
                case_id="c2",
                generated_cards=(c2,),
                matches=(MatchResult(expected=ec2, matched_card=None, similarity=0.1),),
                unmatched_generated=(c2,),
            ),
        ]
        m = calculate_metrics(case_results)
        assert m.total_expected == 2
        assert m.total_matched == 1
        assert m.recall == 0.5


# ---------------------------------------------------------------------------
# report.py tests
# ---------------------------------------------------------------------------


class TestEvalReport:
    """Tests for report generation functions."""

    def test_print_eval_report_no_error(
        self,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        from pdf2anki.eval.metrics import EvalMetrics
        from pdf2anki.eval.report import print_eval_report

        metrics = EvalMetrics(
            recall=0.8,
            precision=0.75,
            f1=0.77,
            avg_similarity=0.85,
            total_cost_usd=0.05,
            total_expected=10,
            total_generated=12,
            total_matched=8,
        )
        # Should not raise
        print_eval_report(metrics, [])

    def test_print_comparison_report_no_error(self) -> None:
        from pdf2anki.eval.metrics import EvalMetrics
        from pdf2anki.eval.report import print_comparison_report

        m1 = EvalMetrics(
            recall=0.6,
            precision=0.5,
            f1=0.55,
            avg_similarity=0.7,
            total_cost_usd=0.1,
            total_expected=10,
            total_generated=15,
            total_matched=6,
        )
        m2 = EvalMetrics(
            recall=0.8,
            precision=0.75,
            f1=0.77,
            avg_similarity=0.85,
            total_cost_usd=0.08,
            total_expected=10,
            total_generated=12,
            total_matched=8,
        )
        # Should not raise
        print_comparison_report(m1, m2, label_a="v1", label_b="v2")

    def test_print_eval_report_with_case_results(self) -> None:
        """Test per-case breakdown printing (lines 42-66)."""
        from pdf2anki.eval.dataset import ExpectedCard
        from pdf2anki.eval.matcher import CaseResult, MatchResult
        from pdf2anki.eval.metrics import EvalMetrics
        from pdf2anki.eval.report import print_eval_report

        card = _make_card(front="Q", back="A")
        ec = ExpectedCard(front_keywords=["Q"], back_keywords=["A"])
        case_results = [
            CaseResult(
                case_id="c1",
                generated_cards=(card,),
                matches=(MatchResult(expected=ec, matched_card=card, similarity=0.9),),
                unmatched_generated=(),
            ),
            CaseResult(
                case_id="c2",
                generated_cards=(card, card),
                matches=(MatchResult(expected=ec, matched_card=None, similarity=0.1),),
                unmatched_generated=(card,),
            ),
        ]
        metrics = EvalMetrics(
            recall=0.5,
            precision=0.33,
            f1=0.4,
            avg_similarity=0.9,
            total_cost_usd=0.05,
            total_expected=2,
            total_generated=3,
            total_matched=1,
        )
        # Should not raise; exercises the per-case detail table
        print_eval_report(metrics, case_results)

    def test_write_eval_json_with_cases(self, tmp_path: Path) -> None:
        """Test write_eval_json with non-empty case_results."""
        from pdf2anki.eval.dataset import ExpectedCard
        from pdf2anki.eval.matcher import CaseResult, MatchResult
        from pdf2anki.eval.metrics import EvalMetrics
        from pdf2anki.eval.report import write_eval_json

        card = _make_card()
        ec = ExpectedCard(front_keywords=["Q"], back_keywords=["A"])
        case_results = [
            CaseResult(
                case_id="test-case",
                generated_cards=(card,),
                matches=(MatchResult(expected=ec, matched_card=card, similarity=0.8),),
                unmatched_generated=(),
            ),
        ]
        metrics = EvalMetrics(
            recall=1.0,
            precision=1.0,
            f1=1.0,
            avg_similarity=0.8,
            total_cost_usd=0.01,
            total_expected=1,
            total_generated=1,
            total_matched=1,
        )
        out = tmp_path / "report.json"
        write_eval_json(metrics, case_results, out)
        data = json.loads(out.read_text())
        assert len(data["cases"]) == 1
        assert data["cases"][0]["case_id"] == "test-case"
        assert data["cases"][0]["matched_count"] == 1

    def test_write_eval_json(self, tmp_path: Path) -> None:
        from pdf2anki.eval.metrics import EvalMetrics
        from pdf2anki.eval.report import write_eval_json

        metrics = EvalMetrics(
            recall=0.9,
            precision=0.85,
            f1=0.87,
            avg_similarity=0.9,
            total_cost_usd=0.03,
            total_expected=5,
            total_generated=6,
            total_matched=5,
        )
        out = tmp_path / "report.json"
        write_eval_json(metrics, [], out)
        assert out.exists()
        data = json.loads(out.read_text())
        assert data["metrics"]["recall"] == 0.9
        assert data["metrics"]["precision"] == 0.85
