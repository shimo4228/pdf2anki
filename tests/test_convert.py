"""Tests for pdf2anki.convert - TSV/JSON output conversion.

TDD RED phase: Tests written before implementation.

Tests cover:
- _escape_tsv_field(): Tab/newline escaping for TSV
- _build_tags(): Tag formatting with bloom level and difficulty
- _expand_card_rows(): Reversible/cloze/normal card row expansion
- cards_to_tsv(): Full TSV string generation
- cards_to_json(): JSON output with metadata
- write_tsv() / write_json(): File writing
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import pytest

from pdf2anki.convert import (
    cards_to_json,
    cards_to_tsv,
    write_json,
    write_tsv,
)
from pdf2anki.schemas import (
    AnkiCard,
    BloomLevel,
    CardType,
    ExtractionResult,
)

# ============================================================
# Fixtures
# ============================================================


@pytest.fixture
def qa_card() -> AnkiCard:
    return AnkiCard(
        front="ニューラルネットワークの活性化関数の役割は何ですか？",
        back="非線形変換を導入し、複雑なパターンの学習を可能にする。",
        card_type=CardType.QA,
        bloom_level=BloomLevel.UNDERSTAND,
        tags=["AI::基礎", "neural_network"],
    )


@pytest.fixture
def cloze_card() -> AnkiCard:
    return AnkiCard(
        front="{{c1::勾配降下法}}はパラメータを更新して損失関数を最小化するアルゴリズムである。",
        back="",
        card_type=CardType.CLOZE,
        bloom_level=BloomLevel.REMEMBER,
        tags=["AI::最適化"],
    )


@pytest.fixture
def reversible_card() -> AnkiCard:
    return AnkiCard(
        front="ReLU",
        back="Rectified Linear Unit: f(x) = max(0, x)",
        card_type=CardType.REVERSIBLE,
        bloom_level=BloomLevel.REMEMBER,
        tags=["AI::活性化関数"],
    )


@pytest.fixture
def term_def_card() -> AnkiCard:
    return AnkiCard(
        front="過学習",
        back="訓練データに対して過度に適合し、未知のデータに対する汎化性能が低下する現象。",
        card_type=CardType.TERM_DEFINITION,
        bloom_level=BloomLevel.REMEMBER,
        tags=["AI::基礎"],
    )


@pytest.fixture
def card_with_special_chars() -> AnkiCard:
    """Card with tabs and newlines in content."""
    return AnkiCard(
        front="以下の手順を\n説明してください。",
        back="ステップ1:\tデータ収集\nステップ2:\t前処理",
        card_type=CardType.QA,
        bloom_level=BloomLevel.APPLY,
        tags=["AI::手順"],
    )


@pytest.fixture
def sample_extraction_result(
    qa_card: AnkiCard, cloze_card: AnkiCard
) -> ExtractionResult:
    return ExtractionResult(
        source_file="sample.pdf",
        cards=[qa_card, cloze_card],
        model_used="claude-sonnet-4-5-20250929",
    )


# ============================================================
# cards_to_tsv() - TSV Header
# ============================================================


class TestTsvHeader:
    """Test TSV output header format."""

    def test_header_contains_separator_directive(self, qa_card: AnkiCard) -> None:
        """TSV must start with #separator:tab directive."""
        tsv = cards_to_tsv([qa_card])
        first_line = tsv.split("\n")[0]
        assert first_line == "#separator:tab"

    def test_header_contains_html_directive(self, qa_card: AnkiCard) -> None:
        """TSV must contain #html:true directive."""
        tsv = cards_to_tsv([qa_card])
        lines = tsv.split("\n")
        assert "#html:true" in lines

    def test_header_contains_tags_column_directive(self, qa_card: AnkiCard) -> None:
        """TSV must contain #tags column:3 directive."""
        tsv = cards_to_tsv([qa_card])
        lines = tsv.split("\n")
        assert "#tags column:3" in lines


def _data_lines(tsv: str) -> list[str]:
    """Extract non-comment, non-blank lines from TSV."""
    return [
        ln for ln in tsv.split("\n")
        if not ln.startswith("#") and ln.strip()
    ]


# ============================================================
# cards_to_tsv() - Basic Row Format
# ============================================================


class TestTsvRowFormat:
    """Test TSV row format: front<TAB>back<TAB>tags."""

    def test_qa_card_has_three_tab_separated_fields(
        self, qa_card: AnkiCard
    ) -> None:
        """QA card row should have exactly 3 tab-separated fields."""
        tsv = cards_to_tsv([qa_card])
        data_lines = _data_lines(tsv)
        assert len(data_lines) == 1
        fields = data_lines[0].split("\t")
        assert len(fields) == 3

    def test_qa_card_front_in_first_field(self, qa_card: AnkiCard) -> None:
        """Front text should appear in the first field."""
        tsv = cards_to_tsv([qa_card])
        data_lines = _data_lines(tsv)
        fields = data_lines[0].split("\t")
        assert "ニューラルネットワークの活性化関数の役割は何ですか？" in fields[0]

    def test_qa_card_back_in_second_field(self, qa_card: AnkiCard) -> None:
        """Back text should appear in the second field."""
        tsv = cards_to_tsv([qa_card])
        data_lines = _data_lines(tsv)
        fields = data_lines[0].split("\t")
        assert "非線形変換を導入し" in fields[1]

    def test_qa_card_tags_in_third_field(self, qa_card: AnkiCard) -> None:
        """Tags should appear in the third field."""
        tsv = cards_to_tsv([qa_card])
        data_lines = _data_lines(tsv)
        fields = data_lines[0].split("\t")
        assert "AI::基礎" in fields[2]
        assert "neural_network" in fields[2]


# ============================================================
# cards_to_tsv() - Special Character Escaping
# ============================================================


class TestTsvEscaping:
    """Test tab/newline escaping in TSV fields."""

    def test_tabs_replaced_with_spaces(
        self, card_with_special_chars: AnkiCard
    ) -> None:
        """Tabs in content must be replaced with spaces to avoid TSV corruption."""
        tsv = cards_to_tsv([card_with_special_chars])
        data_lines = _data_lines(tsv)
        # Each data row should have exactly 2 tabs (3 fields)
        for line in data_lines:
            assert line.count("\t") == 2

    def test_newlines_replaced_with_br(
        self, card_with_special_chars: AnkiCard
    ) -> None:
        """Newlines in content must be replaced with <br> for Anki HTML mode."""
        tsv = cards_to_tsv([card_with_special_chars])
        data_lines = _data_lines(tsv)
        for line in data_lines:
            fields = line.split("\t")
            # front and back fields should not contain raw newlines
            # (they've already been split by \n, so check for <br>)
            assert "<br>" in fields[0] or "<br>" in fields[1]


# ============================================================
# cards_to_tsv() - Cloze Card Handling
# ============================================================


class TestTsvCloze:
    """Test cloze card TSV output."""

    def test_cloze_front_preserved(self, cloze_card: AnkiCard) -> None:
        """Cloze front with {{c1::...}} must be preserved."""
        tsv = cards_to_tsv([cloze_card])
        data_lines = _data_lines(tsv)
        fields = data_lines[0].split("\t")
        assert "{{c1::勾配降下法}}" in fields[0]

    def test_cloze_back_is_empty(self, cloze_card: AnkiCard) -> None:
        """Cloze card back field should be empty (Anki auto-generates)."""
        tsv = cards_to_tsv([cloze_card])
        data_lines = _data_lines(tsv)
        fields = data_lines[0].split("\t")
        assert fields[1] == ""


# ============================================================
# cards_to_tsv() - Reversible Card Expansion
# ============================================================


class TestTsvReversible:
    """Test reversible card expansion to 2 rows."""

    def test_reversible_produces_two_rows(
        self, reversible_card: AnkiCard
    ) -> None:
        """Reversible card should expand to 2 TSV rows (forward + reverse)."""
        tsv = cards_to_tsv([reversible_card])
        data_lines = _data_lines(tsv)
        assert len(data_lines) == 2

    def test_reversible_forward_row(self, reversible_card: AnkiCard) -> None:
        """First row: original front -> back."""
        tsv = cards_to_tsv([reversible_card])
        data_lines = _data_lines(tsv)
        fields = data_lines[0].split("\t")
        assert "ReLU" in fields[0]
        assert "Rectified Linear Unit" in fields[1]

    def test_reversible_reverse_row(self, reversible_card: AnkiCard) -> None:
        """Second row: original back -> front (reversed)."""
        tsv = cards_to_tsv([reversible_card])
        data_lines = _data_lines(tsv)
        fields = data_lines[1].split("\t")
        assert "Rectified Linear Unit" in fields[0]
        assert "ReLU" in fields[1]


# ============================================================
# cards_to_tsv() - Tag Formatting
# ============================================================


class TestTsvTags:
    """Test tag formatting in TSV output."""

    def test_bloom_level_tag_added(self, qa_card: AnkiCard) -> None:
        """Bloom level should be added as bloom::<level> tag."""
        tsv = cards_to_tsv([qa_card])
        data_lines = _data_lines(tsv)
        fields = data_lines[0].split("\t")
        assert "bloom::understand" in fields[2]

    def test_tags_space_separated(self, qa_card: AnkiCard) -> None:
        """Tags should be space-separated in TSV."""
        tsv = cards_to_tsv([qa_card])
        data_lines = _data_lines(tsv)
        fields = data_lines[0].split("\t")
        tags = fields[2]
        # Should have multiple space-separated tags
        assert " " in tags

    def test_additional_tags_included(self, qa_card: AnkiCard) -> None:
        """Additional tags passed via parameter should be included."""
        tsv = cards_to_tsv([qa_card], additional_tags=["source::textbook"])
        data_lines = _data_lines(tsv)
        fields = data_lines[0].split("\t")
        assert "source::textbook" in fields[2]


# ============================================================
# cards_to_tsv() - Multiple Cards
# ============================================================


class TestTsvMultipleCards:
    """Test TSV output with multiple cards."""

    def test_multiple_cards_produce_correct_row_count(
        self, qa_card: AnkiCard, cloze_card: AnkiCard, term_def_card: AnkiCard
    ) -> None:
        """N normal cards should produce N data rows."""
        tsv = cards_to_tsv([qa_card, cloze_card, term_def_card])
        data_lines = _data_lines(tsv)
        assert len(data_lines) == 3

    def test_mixed_cards_with_reversible(
        self,
        qa_card: AnkiCard,
        reversible_card: AnkiCard,
        cloze_card: AnkiCard,
    ) -> None:
        """1 QA + 1 reversible + 1 cloze = 4 data rows."""
        tsv = cards_to_tsv([qa_card, reversible_card, cloze_card])
        data_lines = _data_lines(tsv)
        assert len(data_lines) == 4  # 1 + 2 + 1

    def test_empty_card_list(self) -> None:
        """Empty card list should produce only header lines."""
        tsv = cards_to_tsv([])
        data_lines = _data_lines(tsv)
        assert len(data_lines) == 0


# ============================================================
# cards_to_tsv() - Encoding
# ============================================================


class TestTsvEncoding:
    """Test TSV output encoding."""

    def test_output_is_utf8_string(self, qa_card: AnkiCard) -> None:
        """Output should be a valid UTF-8 string."""
        tsv = cards_to_tsv([qa_card])
        assert isinstance(tsv, str)
        tsv.encode("utf-8")  # Should not raise

    def test_japanese_characters_preserved(self, qa_card: AnkiCard) -> None:
        """Japanese characters should be preserved without escaping."""
        tsv = cards_to_tsv([qa_card])
        assert "ニューラルネットワーク" in tsv


# ============================================================
# cards_to_json() Tests
# ============================================================


class TestCardsToJson:
    """Test JSON output with metadata."""

    def test_json_is_valid(
        self, sample_extraction_result: ExtractionResult
    ) -> None:
        """Output should be valid JSON."""
        output = cards_to_json(sample_extraction_result)
        parsed = json.loads(output)
        assert isinstance(parsed, dict)

    def test_json_contains_cards(
        self, sample_extraction_result: ExtractionResult
    ) -> None:
        """JSON should contain the cards array."""
        output = cards_to_json(sample_extraction_result)
        parsed = json.loads(output)
        assert "cards" in parsed
        assert len(parsed["cards"]) == 2

    def test_json_contains_source_file(
        self, sample_extraction_result: ExtractionResult
    ) -> None:
        """JSON should contain the source file path."""
        output = cards_to_json(sample_extraction_result)
        parsed = json.loads(output)
        assert parsed["source_file"] == "sample.pdf"

    def test_json_contains_model_used(
        self, sample_extraction_result: ExtractionResult
    ) -> None:
        """JSON should contain the model used."""
        output = cards_to_json(sample_extraction_result)
        parsed = json.loads(output)
        assert parsed["model_used"] == "claude-sonnet-4-5-20250929"

    def test_json_contains_meta(
        self, sample_extraction_result: ExtractionResult
    ) -> None:
        """JSON should contain _meta with schema version and generated_at."""
        output = cards_to_json(sample_extraction_result)
        parsed = json.loads(output)
        assert "_meta" in parsed
        meta = parsed["_meta"]
        assert "schema_version" in meta
        assert "generated_at" in meta

    def test_json_meta_schema_version(
        self, sample_extraction_result: ExtractionResult
    ) -> None:
        """_meta.schema_version should be a string like '1.0'."""
        output = cards_to_json(sample_extraction_result)
        parsed = json.loads(output)
        assert isinstance(parsed["_meta"]["schema_version"], str)

    def test_json_meta_generated_at_is_iso_format(
        self, sample_extraction_result: ExtractionResult
    ) -> None:
        """_meta.generated_at should be ISO 8601 format."""
        output = cards_to_json(sample_extraction_result)
        parsed = json.loads(output)
        generated_at = parsed["_meta"]["generated_at"]
        datetime.fromisoformat(generated_at)  # Should not raise

    def test_json_card_fields_present(
        self, sample_extraction_result: ExtractionResult
    ) -> None:
        """Each card in JSON should have front, back, card_type, bloom_level, tags."""
        output = cards_to_json(sample_extraction_result)
        parsed = json.loads(output)
        card = parsed["cards"][0]
        assert "front" in card
        assert "back" in card
        assert "card_type" in card
        assert "bloom_level" in card
        assert "tags" in card

    def test_json_preserves_japanese(
        self, sample_extraction_result: ExtractionResult
    ) -> None:
        """JSON should preserve Japanese characters (ensure_ascii=False)."""
        output = cards_to_json(sample_extraction_result)
        assert "ニューラルネットワーク" in output

    def test_json_is_indented(
        self, sample_extraction_result: ExtractionResult
    ) -> None:
        """JSON should be pretty-printed with indentation."""
        output = cards_to_json(sample_extraction_result)
        assert "\n  " in output  # indented


# ============================================================
# write_tsv() Tests
# ============================================================


class TestWriteTsv:
    """Test TSV file writing."""

    def test_write_tsv_creates_file(
        self, tmp_path: Path, qa_card: AnkiCard
    ) -> None:
        """write_tsv should create a .tsv file."""
        output_path = tmp_path / "output.tsv"
        write_tsv([qa_card], output_path)
        assert output_path.exists()

    def test_write_tsv_content_matches(
        self, tmp_path: Path, qa_card: AnkiCard
    ) -> None:
        """Written file content should match cards_to_tsv output."""
        output_path = tmp_path / "output.tsv"
        write_tsv([qa_card], output_path)
        content = output_path.read_text(encoding="utf-8")
        expected = cards_to_tsv([qa_card])
        assert content == expected

    def test_write_tsv_utf8_no_bom(
        self, tmp_path: Path, qa_card: AnkiCard
    ) -> None:
        """File should be UTF-8 without BOM."""
        output_path = tmp_path / "output.tsv"
        write_tsv([qa_card], output_path)
        raw = output_path.read_bytes()
        assert not raw.startswith(b"\xef\xbb\xbf")  # No UTF-8 BOM

    def test_write_tsv_creates_parent_dirs(
        self, tmp_path: Path, qa_card: AnkiCard
    ) -> None:
        """write_tsv should create parent directories if needed."""
        output_path = tmp_path / "sub" / "dir" / "output.tsv"
        write_tsv([qa_card], output_path)
        assert output_path.exists()

    def test_write_tsv_with_additional_tags(
        self, tmp_path: Path, qa_card: AnkiCard
    ) -> None:
        """Additional tags should be included in written file."""
        output_path = tmp_path / "output.tsv"
        write_tsv([qa_card], output_path, additional_tags=["source::test"])
        content = output_path.read_text(encoding="utf-8")
        assert "source::test" in content


# ============================================================
# write_json() Tests
# ============================================================


class TestWriteJson:
    """Test JSON file writing."""

    def test_write_json_creates_file(
        self, tmp_path: Path, sample_extraction_result: ExtractionResult
    ) -> None:
        """write_json should create a .json file."""
        output_path = tmp_path / "output.json"
        write_json(sample_extraction_result, output_path)
        assert output_path.exists()

    def test_write_json_content_is_valid(
        self, tmp_path: Path, sample_extraction_result: ExtractionResult
    ) -> None:
        """Written file should contain valid JSON."""
        output_path = tmp_path / "output.json"
        write_json(sample_extraction_result, output_path)
        content = output_path.read_text(encoding="utf-8")
        parsed = json.loads(content)
        assert "cards" in parsed

    def test_write_json_creates_parent_dirs(
        self, tmp_path: Path, sample_extraction_result: ExtractionResult
    ) -> None:
        """write_json should create parent directories if needed."""
        output_path = tmp_path / "sub" / "dir" / "output.json"
        write_json(sample_extraction_result, output_path)
        assert output_path.exists()

    def test_write_json_utf8(
        self, tmp_path: Path, sample_extraction_result: ExtractionResult
    ) -> None:
        """File should be UTF-8 encoded."""
        output_path = tmp_path / "output.json"
        write_json(sample_extraction_result, output_path)
        content = output_path.read_text(encoding="utf-8")
        assert "ニューラルネットワーク" in content
