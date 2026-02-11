"""End-to-end tests for pdf2anki.

Unlike test_main.py (which mocks extract_text, extract_cards, run_quality_pipeline),
these tests mock ONLY at the Anthropic API boundary (client.messages.create)
and let the full pipeline run for real:

  Input file → text extraction → prompt building → (mock) API call →
  response parsing → quality scoring → output writing

This ensures the entire pipeline integrates correctly.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pymupdf
import pytest
from typer.testing import CliRunner

# ---------------------------------------------------------------------------
# Helpers: build fake Anthropic API responses
# ---------------------------------------------------------------------------

_SAMPLE_CARDS_JSON = json.dumps(
    [
        {
            "front": "ニューラルネットワークにおける活性化関数の役割は何ですか？",
            "back": "非線形変換を導入し、複雑なパターンの学習を可能にする。",
            "card_type": "qa",
            "bloom_level": "understand",
            "tags": ["AI::neural_network", "AI::活性化関数"],
            "related_concepts": ["ReLU", "sigmoid", "tanh"],
            "mnemonic_hint": None,
        },
        {
            "front": "{{c1::ReLU}}の数式はf(x) = max(0, x)である。",
            "back": "",
            "card_type": "cloze",
            "bloom_level": "remember",
            "tags": ["AI::活性化関数"],
            "related_concepts": ["leaky_relu"],
            "mnemonic_hint": "REctified = 負をREject",
        },
        {
            "front": "Sigmoid",
            "back": "σ(x) = 1/(1+e^(-x)) 出力を0〜1に変換する活性化関数",
            "card_type": "term_definition",
            "bloom_level": "remember",
            "tags": ["AI::活性化関数"],
            "related_concepts": ["logistic"],
            "mnemonic_hint": None,
        },
    ],
    ensure_ascii=False,
)

_CRITIQUE_RESPONSE_JSON = json.dumps(
    [
        {
            "card_index": 0,
            "action": "improve",
            "reason": "Question could be more specific",
            "flags": ["vague_question"],
            "improved_cards": [
                {
                    "front": "ニューラルネットワークで"
                    "活性化関数が必要な理由は何ですか？",
                    "back": "線形変換のみでは複雑なパターンを"
                    "学習できないため、非線形性を導入する。",
                    "card_type": "qa",
                    "bloom_level": "understand",
                    "tags": ["AI::neural_network", "AI::活性化関数"],
                    "related_concepts": ["ReLU", "sigmoid"],
                    "mnemonic_hint": None,
                }
            ],
        }
    ],
    ensure_ascii=False,
)


def _make_api_response(
    text: str,
    model: str = "claude-haiku-4-5-20251001",
    input_tokens: int = 500,
    output_tokens: int = 300,
) -> SimpleNamespace:
    """Build a fake anthropic.types.Message response."""
    content_block = SimpleNamespace(type="text", text=text)
    usage = SimpleNamespace(input_tokens=input_tokens, output_tokens=output_tokens)
    return SimpleNamespace(
        content=[content_block],
        model=model,
        usage=usage,
        stop_reason="end_turn",
    )


def _build_mock_client(
    extraction_text: str = _SAMPLE_CARDS_JSON,
    critique_text: str = _CRITIQUE_RESPONSE_JSON,
) -> MagicMock:
    """Build a mock Anthropic client that returns cards then critique."""
    client = MagicMock()
    # First call = card extraction, subsequent calls = critique
    client.messages.create.side_effect = [
        _make_api_response(extraction_text),
        _make_api_response(critique_text),
        # Extra responses for batch processing
        _make_api_response(extraction_text),
        _make_api_response(critique_text),
        _make_api_response(extraction_text),
        _make_api_response(critique_text),
    ]
    return client


@pytest.fixture
def runner() -> CliRunner:
    return CliRunner()


@pytest.fixture
def sample_txt(tmp_path: Path) -> Path:
    p = tmp_path / "neural_network.txt"
    p.write_text(
        "ニューラルネットワークの基礎\n\n"
        "活性化関数はReLU、Sigmoid、Tanhがある。\n"
        "ReLUはf(x)=max(0,x)で計算される。\n"
        "Sigmoidはσ(x)=1/(1+e^(-x))である。\n"
        "これらは非線形変換を導入する役割がある。\n",
        encoding="utf-8",
    )
    return p


@pytest.fixture
def sample_md(tmp_path: Path) -> Path:
    p = tmp_path / "machine_learning.md"
    p.write_text(
        "# 機械学習の基礎\n\n"
        "## 教師あり学習\n\n"
        "教師あり学習では、入力と正解ラベルのペアからモデルを学習する。\n"
        "代表的なアルゴリズムには線形回帰、決定木、SVM がある。\n\n"
        "## 教師なし学習\n\n"
        "教師なし学習では、ラベルなしデータから構造を発見する。\n"
        "クラスタリングや次元削減が代表的な手法である。\n",
        encoding="utf-8",
    )
    return p


@pytest.fixture
def sample_pdf(tmp_path: Path) -> Path:
    pdf_path = tmp_path / "sample.pdf"
    doc = pymupdf.Document()
    page = doc.new_page()
    lines = [
        "ニューラルネットワークの基礎についての解説文書である。",
        "活性化関数にはReLU、Sigmoid、Tanhなどがある。",
        "ReLUは入力が負のとき0を出力し正のときはそのまま出力する。",
        "Sigmoidは出力を0から1の範囲に収める関数である。",
    ]
    y = 72
    for line in lines:
        page.insert_text((72, y), line, fontname="japan", fontsize=12)
        y += 20
    doc.save(str(pdf_path))
    doc.close()
    return pdf_path


@pytest.fixture
def sample_dir(tmp_path: Path) -> Path:
    d = tmp_path / "docs"
    d.mkdir()
    (d / "a.txt").write_text(
        "文書A: ニューラルネットワークのテスト用テキストである。"
        "活性化関数にはReLU、Sigmoid、Tanhがあり、それぞれ異なる特性を持つ。"
        "バックプロパゲーションは誤差逆伝播法とも呼ばれ、勾配を計算する手法である。\n",
        encoding="utf-8",
    )
    (d / "b.md").write_text(
        "# 文書B\n\n深層学習の基礎について解説する。"
        "CNNは画像認識に広く使われる畳み込みニューラルネットワークである。"
        "RNNは時系列データの処理に適した回帰結合を持つニューラルネットワークである。\n",
        encoding="utf-8",
    )
    (d / "ignore.csv").write_text("col1,col2\n1,2", encoding="utf-8")
    return d


def _get_app():
    """Lazily import the app."""
    from pdf2anki.main import app

    return app


# ============================================================
# E2E: TXT → TSV (full pipeline with quality=basic)
# ============================================================


class TestE2ETxtToTsv:
    """Full pipeline: real text extraction → mocked API → real quality scoring → TSV."""

    @patch("pdf2anki.structure.anthropic.Anthropic")
    def test_txt_to_tsv_quality_off(
        self,
        mock_anthropic_cls: MagicMock,
        runner: CliRunner,
        sample_txt: Path,
        tmp_path: Path,
    ) -> None:
        """TXT → TSV with quality=off: extraction + card generation only."""
        mock_anthropic_cls.return_value = _build_mock_client()

        output = tmp_path / "output.tsv"
        result = runner.invoke(
            _get_app(),
            ["convert", str(sample_txt), "-o", str(output), "--quality", "off"],
        )

        assert result.exit_code == 0, f"CLI failed: {result.output}"
        assert output.exists()

        content = output.read_text(encoding="utf-8")
        # TSV header
        assert "#separator:tab" in content
        assert "#html:true" in content
        # Card content present
        assert "活性化関数" in content
        assert "ReLU" in content
        # Bloom tags
        assert "bloom::understand" in content or "bloom::remember" in content
        # Verify it has actual tab-separated data rows
        data_lines = [
            ln for ln in content.strip().split("\n") if not ln.startswith("#")
        ]
        assert len(data_lines) >= 3  # 3 cards

    @patch("pdf2anki.quality.critique.anthropic.Anthropic")
    @patch("pdf2anki.structure.anthropic.Anthropic")
    def test_txt_to_tsv_quality_full(
        self,
        mock_struct_anthropic: MagicMock,
        mock_qual_anthropic: MagicMock,
        runner: CliRunner,
        sample_txt: Path,
        tmp_path: Path,
    ) -> None:
        """TXT → TSV with quality=full: includes LLM critique."""
        mock_struct_anthropic.return_value = _build_mock_client()
        mock_qual_anthropic.return_value = _build_mock_client(
            extraction_text=_CRITIQUE_RESPONSE_JSON,
        )

        output = tmp_path / "output.tsv"
        result = runner.invoke(
            _get_app(),
            ["convert", str(sample_txt), "-o", str(output), "--quality", "full"],
        )

        assert result.exit_code == 0, f"CLI failed: {result.output}"
        assert output.exists()

        content = output.read_text(encoding="utf-8")
        assert "#separator:tab" in content
        # Summary should show quality report info
        assert "QA" in result.output or "Cards" in result.output or "2" in result.output


# ============================================================
# E2E: MD → JSON (full pipeline)
# ============================================================


class TestE2EMdToJson:
    """Full pipeline: MD → JSON output format."""

    @patch("pdf2anki.structure.anthropic.Anthropic")
    def test_md_to_json(
        self,
        mock_anthropic_cls: MagicMock,
        runner: CliRunner,
        sample_md: Path,
        tmp_path: Path,
    ) -> None:
        """MD → JSON with quality=off."""
        mock_anthropic_cls.return_value = _build_mock_client()

        output = tmp_path / "output.json"
        result = runner.invoke(
            _get_app(),
            [
                "convert",
                str(sample_md),
                "-o",
                str(output),
                "--format",
                "json",
                "--quality",
                "off",
            ],
        )

        assert result.exit_code == 0, f"CLI failed: {result.output}"
        assert output.exists()

        data = json.loads(output.read_text(encoding="utf-8"))
        assert "source_file" in data
        assert "cards" in data
        assert "_meta" in data
        assert data["_meta"]["schema_version"] == "1.0"
        assert len(data["cards"]) >= 1
        # Verify card structure
        card = data["cards"][0]
        assert "front" in card
        assert "back" in card
        assert "card_type" in card
        assert "bloom_level" in card
        assert "tags" in card

    @patch("pdf2anki.structure.anthropic.Anthropic")
    def test_md_to_both(
        self,
        mock_anthropic_cls: MagicMock,
        runner: CliRunner,
        sample_md: Path,
        tmp_path: Path,
    ) -> None:
        """MD → both TSV and JSON output."""
        mock_anthropic_cls.return_value = _build_mock_client()

        output_dir = tmp_path / "output"
        result = runner.invoke(
            _get_app(),
            [
                "convert",
                str(sample_md),
                "-o",
                str(output_dir),
                "--format",
                "both",
                "--quality",
                "off",
            ],
        )

        assert result.exit_code == 0, f"CLI failed: {result.output}"
        tsv_files = list(output_dir.glob("*.tsv"))
        json_files = list(output_dir.glob("*.json"))
        assert len(tsv_files) == 1
        assert len(json_files) == 1

        # Verify both files have valid content
        tsv_content = tsv_files[0].read_text(encoding="utf-8")
        assert "#separator:tab" in tsv_content

        json_data = json.loads(json_files[0].read_text(encoding="utf-8"))
        assert "cards" in json_data


# ============================================================
# E2E: PDF → TSV (full pipeline)
# ============================================================


class TestE2EPdfToTsv:
    """Full pipeline with real PDF extraction."""

    @patch("pdf2anki.structure.anthropic.Anthropic")
    def test_pdf_to_tsv(
        self,
        mock_anthropic_cls: MagicMock,
        runner: CliRunner,
        sample_pdf: Path,
        tmp_path: Path,
    ) -> None:
        """PDF → TSV: real pymupdf extraction + mocked API."""
        mock_anthropic_cls.return_value = _build_mock_client()

        output = tmp_path / "output.tsv"
        result = runner.invoke(
            _get_app(),
            ["convert", str(sample_pdf), "-o", str(output), "--quality", "off"],
        )

        assert result.exit_code == 0, f"CLI failed: {result.output}"
        assert output.exists()

        content = output.read_text(encoding="utf-8")
        assert "#separator:tab" in content
        data_lines = [
            ln for ln in content.strip().split("\n") if not ln.startswith("#")
        ]
        assert len(data_lines) >= 1


# ============================================================
# E2E: Directory batch processing
# ============================================================


class TestE2EBatchDirectory:
    """Full pipeline for directory batch processing."""

    @patch("pdf2anki.structure.anthropic.Anthropic")
    def test_batch_directory(
        self,
        mock_anthropic_cls: MagicMock,
        runner: CliRunner,
        sample_dir: Path,
        tmp_path: Path,
    ) -> None:
        """Process all supported files in a directory."""
        mock_anthropic_cls.return_value = _build_mock_client()

        output_dir = tmp_path / "output"
        result = runner.invoke(
            _get_app(),
            ["convert", str(sample_dir), "-o", str(output_dir), "--quality", "off"],
        )

        assert result.exit_code == 0, f"CLI failed: {result.output}"
        # Should produce TSV for each supported file (a.txt, b.md), not ignore.csv
        tsv_files = list(output_dir.glob("*.tsv"))
        assert len(tsv_files) == 2

        # API should have been called for each file
        client = mock_anthropic_cls.return_value
        assert client.messages.create.call_count == 2


# ============================================================
# E2E: Preview command (no API calls)
# ============================================================


class TestE2EPreview:
    """Preview is truly end-to-end with no mocking needed (no API calls)."""

    def test_preview_txt(
        self,
        runner: CliRunner,
        sample_txt: Path,
    ) -> None:
        """Preview of a TXT file (fully real, no mocks)."""
        result = runner.invoke(_get_app(), ["preview", str(sample_txt)])

        assert result.exit_code == 0
        assert "txt" in result.output
        # Should show text length info
        assert "chars" in result.output
        # Should show chunk count
        assert "1" in result.output  # 1 chunk for short text

    def test_preview_md(
        self,
        runner: CliRunner,
        sample_md: Path,
    ) -> None:
        """Preview of a MD file (fully real, no mocks)."""
        result = runner.invoke(_get_app(), ["preview", str(sample_md)])

        assert result.exit_code == 0
        assert "md" in result.output

    def test_preview_pdf(
        self,
        runner: CliRunner,
        sample_pdf: Path,
    ) -> None:
        """Preview of a PDF file (fully real, no mocks)."""
        result = runner.invoke(_get_app(), ["preview", str(sample_pdf)])

        assert result.exit_code == 0
        assert "pdf" in result.output


# ============================================================
# E2E: CLI options integration
# ============================================================


class TestE2ECliOptions:
    """Test that CLI options propagate correctly through the full pipeline."""

    @patch("pdf2anki.structure.anthropic.Anthropic")
    def test_tags_appear_in_output(
        self,
        mock_anthropic_cls: MagicMock,
        runner: CliRunner,
        sample_txt: Path,
        tmp_path: Path,
    ) -> None:
        """--tags are included in the TSV output."""
        mock_anthropic_cls.return_value = _build_mock_client()

        output = tmp_path / "output.tsv"
        result = runner.invoke(
            _get_app(),
            [
                "convert",
                str(sample_txt),
                "-o",
                str(output),
                "--tags",
                "custom::tag,extra",
                "--quality",
                "off",
            ],
        )

        assert result.exit_code == 0, f"CLI failed: {result.output}"
        content = output.read_text(encoding="utf-8")
        assert "custom::tag" in content
        assert "extra" in content

    @patch("pdf2anki.structure.anthropic.Anthropic")
    def test_focus_topics_passed_to_api(
        self,
        mock_anthropic_cls: MagicMock,
        runner: CliRunner,
        sample_txt: Path,
        tmp_path: Path,
    ) -> None:
        """--focus topics are included in the API prompt."""
        mock_anthropic_cls.return_value = _build_mock_client()

        output = tmp_path / "output.tsv"
        result = runner.invoke(
            _get_app(),
            [
                "convert",
                str(sample_txt),
                "-o",
                str(output),
                "--focus",
                "ReLU,活性化関数",
                "--quality",
                "off",
            ],
        )

        assert result.exit_code == 0, f"CLI failed: {result.output}"
        # Verify focus topics were in the API call
        client = mock_anthropic_cls.return_value
        call_kwargs = client.messages.create.call_args
        messages = call_kwargs.kwargs.get("messages") or call_kwargs[1].get("messages")
        user_content = messages[0]["content"]
        assert "ReLU" in user_content
        assert "活性化関数" in user_content

    @patch("pdf2anki.structure.anthropic.Anthropic")
    def test_model_override(
        self,
        mock_anthropic_cls: MagicMock,
        runner: CliRunner,
        sample_txt: Path,
        tmp_path: Path,
    ) -> None:
        """--model overrides automatic model selection."""
        mock_anthropic_cls.return_value = _build_mock_client()

        output = tmp_path / "output.tsv"
        result = runner.invoke(
            _get_app(),
            [
                "convert",
                str(sample_txt),
                "-o",
                str(output),
                "--model",
                "claude-sonnet-4-5-20250929",
                "--quality",
                "off",
            ],
        )

        assert result.exit_code == 0, f"CLI failed: {result.output}"
        client = mock_anthropic_cls.return_value
        call_kwargs = client.messages.create.call_args
        model_used = call_kwargs.kwargs.get("model") or call_kwargs[1].get("model")
        assert model_used == "claude-sonnet-4-5-20250929"

    @patch("pdf2anki.structure.anthropic.Anthropic")
    def test_max_cards_in_prompt(
        self,
        mock_anthropic_cls: MagicMock,
        runner: CliRunner,
        sample_txt: Path,
        tmp_path: Path,
    ) -> None:
        """--max-cards is reflected in the user prompt."""
        mock_anthropic_cls.return_value = _build_mock_client()

        output = tmp_path / "output.tsv"
        result = runner.invoke(
            _get_app(),
            [
                "convert",
                str(sample_txt),
                "-o",
                str(output),
                "--max-cards",
                "5",
                "--quality",
                "off",
            ],
        )

        assert result.exit_code == 0, f"CLI failed: {result.output}"
        client = mock_anthropic_cls.return_value
        call_kwargs = client.messages.create.call_args
        messages = call_kwargs.kwargs.get("messages") or call_kwargs[1].get("messages")
        user_content = messages[0]["content"]
        assert "5" in user_content


# ============================================================
# E2E: Error scenarios
# ============================================================


class TestE2EErrors:
    """Test error handling through the full pipeline."""

    def test_missing_file(self, runner: CliRunner) -> None:
        """Error when input file doesn't exist."""
        result = runner.invoke(_get_app(), ["convert", "/nonexistent/file.txt"])
        assert result.exit_code != 0

    def test_unsupported_file(self, runner: CliRunner, tmp_path: Path) -> None:
        """Error for unsupported file type."""
        csv_file = tmp_path / "bad.csv"
        csv_file.write_text("a,b,c", encoding="utf-8")
        result = runner.invoke(_get_app(), ["convert", str(csv_file)])
        assert result.exit_code != 0

    @patch("pdf2anki.structure.anthropic.Anthropic")
    def test_empty_api_response(
        self,
        mock_anthropic_cls: MagicMock,
        runner: CliRunner,
        sample_txt: Path,
        tmp_path: Path,
    ) -> None:
        """Handle empty API response gracefully."""
        client = MagicMock()
        client.messages.create.return_value = SimpleNamespace(
            content=[],
            model="claude-haiku-4-5-20251001",
            usage=SimpleNamespace(input_tokens=100, output_tokens=0),
            stop_reason="end_turn",
        )
        mock_anthropic_cls.return_value = client

        output = tmp_path / "output.tsv"
        result = runner.invoke(
            _get_app(),
            ["convert", str(sample_txt), "-o", str(output), "--quality", "off"],
        )

        # Should not crash; produces empty output
        assert result.exit_code == 0

    @patch("pdf2anki.structure.anthropic.Anthropic")
    def test_invalid_json_response(
        self,
        mock_anthropic_cls: MagicMock,
        runner: CliRunner,
        sample_txt: Path,
        tmp_path: Path,
    ) -> None:
        """Handle malformed JSON from API gracefully."""
        mock_anthropic_cls.return_value = _build_mock_client(
            extraction_text="This is not JSON at all",
        )

        output = tmp_path / "output.tsv"
        result = runner.invoke(
            _get_app(),
            ["convert", str(sample_txt), "-o", str(output), "--quality", "off"],
        )

        # Should report error but not crash (main.py catches RuntimeError/ValueError)
        assert result.exit_code == 0

    @patch("pdf2anki.structure.anthropic.Anthropic")
    def test_budget_limit_enforcement(
        self,
        mock_anthropic_cls: MagicMock,
        runner: CliRunner,
        sample_txt: Path,
        tmp_path: Path,
    ) -> None:
        """--budget-limit is enforced through the pipeline."""
        mock_anthropic_cls.return_value = _build_mock_client()

        output = tmp_path / "output.tsv"
        result = runner.invoke(
            _get_app(),
            [
                "convert",
                str(sample_txt),
                "-o",
                str(output),
                "--budget-limit",
                "0.50",
                "--quality",
                "off",
            ],
        )

        # Should succeed since mock costs are tiny
        assert result.exit_code == 0


# ============================================================
# E2E: TSV output format verification
# ============================================================


class TestE2ETsvFormat:
    """Verify the TSV output is correctly formatted for Anki import."""

    @patch("pdf2anki.structure.anthropic.Anthropic")
    def test_tsv_structure(
        self,
        mock_anthropic_cls: MagicMock,
        runner: CliRunner,
        sample_txt: Path,
        tmp_path: Path,
    ) -> None:
        """Verify TSV has header + tab-separated rows."""
        mock_anthropic_cls.return_value = _build_mock_client()

        output = tmp_path / "output.tsv"
        runner.invoke(
            _get_app(),
            ["convert", str(sample_txt), "-o", str(output), "--quality", "off"],
        )

        content = output.read_text(encoding="utf-8")
        lines = content.strip().split("\n")

        # Header lines
        assert lines[0] == "#separator:tab"
        assert lines[1] == "#html:true"
        assert lines[2] == "#tags column:3"

        # Data rows: each should have exactly 2 tabs (3 columns)
        data_lines = [ln for ln in lines[3:] if ln.strip()]
        for line in data_lines:
            parts = line.split("\t")
            assert len(parts) == 3, f"Expected 3 columns, got {len(parts)}: {line}"

    @patch("pdf2anki.structure.anthropic.Anthropic")
    def test_cloze_card_has_empty_back(
        self,
        mock_anthropic_cls: MagicMock,
        runner: CliRunner,
        sample_txt: Path,
        tmp_path: Path,
    ) -> None:
        """Cloze cards should have empty back field in TSV."""
        mock_anthropic_cls.return_value = _build_mock_client()

        output = tmp_path / "output.tsv"
        runner.invoke(
            _get_app(),
            ["convert", str(sample_txt), "-o", str(output), "--quality", "off"],
        )

        content = output.read_text(encoding="utf-8")
        # Find the cloze card line (contains {{c1::)
        cloze_lines = [ln for ln in content.strip().split("\n") if "{{c1::" in ln]
        assert len(cloze_lines) >= 1
        # Cloze card: front\t\ttags (empty back)
        parts = cloze_lines[0].split("\t")
        assert parts[1] == "", f"Cloze back should be empty, got: {parts[1]}"


# ============================================================
# E2E: JSON output format verification
# ============================================================


class TestE2EJsonFormat:
    """Verify the JSON output structure."""

    @patch("pdf2anki.structure.anthropic.Anthropic")
    def test_json_schema(
        self,
        mock_anthropic_cls: MagicMock,
        runner: CliRunner,
        sample_txt: Path,
        tmp_path: Path,
    ) -> None:
        """Verify JSON has correct schema with metadata."""
        mock_anthropic_cls.return_value = _build_mock_client()

        output = tmp_path / "output.json"
        runner.invoke(
            _get_app(),
            [
                "convert",
                str(sample_txt),
                "-o",
                str(output),
                "--format",
                "json",
                "--quality",
                "off",
            ],
        )

        data = json.loads(output.read_text(encoding="utf-8"))

        # Top-level structure
        assert "source_file" in data
        assert "cards" in data
        assert "model_used" in data
        assert "_meta" in data
        assert "schema_version" in data["_meta"]
        assert "generated_at" in data["_meta"]

        # Card structure
        for card in data["cards"]:
            assert "front" in card
            assert "back" in card
            assert "card_type" in card
            assert "bloom_level" in card
            assert "tags" in card
            assert isinstance(card["tags"], list)
            assert len(card["tags"]) >= 1


# ============================================================
# E2E: Summary output verification
# ============================================================


class TestE2ESummary:
    """Verify the CLI summary output."""

    @patch("pdf2anki.structure.anthropic.Anthropic")
    def test_summary_shows_card_count(
        self,
        mock_anthropic_cls: MagicMock,
        runner: CliRunner,
        sample_txt: Path,
        tmp_path: Path,
    ) -> None:
        """Summary includes card count and cost info."""
        mock_anthropic_cls.return_value = _build_mock_client()

        output = tmp_path / "output.tsv"
        result = runner.invoke(
            _get_app(),
            ["convert", str(sample_txt), "-o", str(output), "--quality", "off"],
        )

        assert result.exit_code == 0
        # Should mention card count (3 cards in our mock)
        assert "3" in result.output
        # Should show cost info
        assert "$" in result.output or "cost" in result.output.lower()
