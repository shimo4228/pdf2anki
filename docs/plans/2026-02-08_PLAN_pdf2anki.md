# 実装計画: pdf2anki -- 最高水準のAnki Q&Aカード自動生成ツール

**作成日:** 2026-02-08
**ステータス:** Draft
**著者:** Claude Code + shimomoto_tatsuya

---

## Context

### 背景
PDF/テキスト/Markdownからの学習カード自動生成ツールを構築する。既存の`pdf2anki-spec.md`と`text-to-anki-research-report.md`の設計をベースに、以下3つの知見を統合して「最も洗練された」ツールを目指す:

1. **g-kentei-ios の実証済みパターン**: FSRS v5.0、Hybrid Parser + Confidence Scorer + LLM Validator
2. **Wozniakの知識定式化20原則**: 認知科学に基づくカード品質基準
3. **2025-2026年の最新ツール調査**: AnkiAIUtils, AnkiAny, Anki MCP Server 等の分析

### 既存ツールとの差別化

| 既存ツール | 強み | 弱み | pdf2ankiで解決 |
|-----------|------|------|---------------|
| [AnkiAIUtils](https://github.com/thiswillbeyourgithub/AnkiAIUtils) | セマンティックフィルタリング、ニーモニック生成 | 既存カードの強化のみ、新規生成不可 | PDF/テキストから新規生成 |
| [AnkiAny](https://github.com/alingse/ankiany) (Claude Agent SDK) | QA/Cloze/MCQ、genanki出力 | 品質保証なし | Confidence Scorer + LLM批評 |
| [Anki_FlashCard_Generator](https://github.com/PromtEngineer/Anki_FlashCard_Generator) | PDF→GPT→Anki | チャンク分割でコンテキスト喪失、日本語弱い | pymupdf4llm + 日本語最適化 |
| [Anki MCP Servers](https://github.com/dhkim0124/anki-mcp-server) | Claude Desktopから自然言語でカード作成 | バッチ処理不可、構造化なし | CLI + Structured Outputs |

### 3つの主要イノベーション

**1. 品質保証パイプライン** (g-kentei-iosパターン適用)
```
[LLM生成] -> [信頼度スコア] -> 高(>=0.90) -> そのまま出力
                              -> 低(<0.90) -> [LLM批評] -> 改善/分割/除去
```

**2. Wozniak原則ベースのプロンプト**
- 最小情報原則（1カード1概念）
- リスト→cloze自動変換
- 冗長性の活用（同概念を複数角度から）
- 記憶術ヒント

**3. 8種カード + Bloomの分類法**
- 既存4種 + reversible, sequence, compare_contrast, image_occlusion
- 全カードにBloomレベル (remember〜create) を付与

---

## アーキテクチャ

```
[入力ファイル] ─ PDF / TXT / MD
    │
    ▼
[Step 1: テキスト抽出] ── extract.py
    │  pymupdf4llm (PDF→Markdown)
    │  OCRフォールバック (Tesseract)
    ▼
[Step 2: LLM構造化抽出] ── structure.py + prompts.py
    │  Claude API Structured Outputs
    │  Wozniak原則プロンプト
    │  8種カード + Bloomレベル
    ▼
[Step 3: 品質保証] ── quality.py
    │  信頼度スコアリング (g-kentei-iosパターン)
    │  低信頼度 → LLM批評 + 改善
    ▼
[Step 4: 出力変換] ── convert.py
    │  TSV (Ankiインポート形式) ── 主力出力
    │  JSON (メタデータ付き) ── 連携用
    ▼
[出力] + QualityReport
```

---

## プロジェクト構成

```
Anki-QA/
├── pyproject.toml
├── config.yaml
├── .env.example
├── src/
│   └── pdf2anki/
│       ├── __init__.py
│       ├── main.py          # CLI (typer + rich)
│       ├── extract.py       # Step 1: テキスト抽出
│       ├── structure.py     # Step 2: LLM構造化
│       ├── quality.py       # Step 3: 品質保証パイプライン
│       ├── convert.py       # Step 4: TSV/JSON出力
│       ├── schemas.py       # 拡張Pydanticスキーマ
│       ├── prompts.py       # Wozniak原則プロンプト
│       ├── config.py        # 設定読み込み
│       └── cost.py          # コスト追跡
├── tests/
│   ├── conftest.py
│   ├── test_schemas.py
│   ├── test_extract.py
│   ├── test_structure.py
│   ├── test_quality.py
│   ├── test_convert.py
│   ├── test_cost.py
│   └── fixtures/
│       ├── sample.pdf
│       ├── sample.txt
│       └── sample.md
└── output/                  # デフォルト出力先 (.gitignore)
```

---

## 実装フェーズ（全6フェーズ）

### Phase 1: 基盤構築（スキーマ + 設定）

**ファイル:**
- `pyproject.toml` -- uv, 依存パッケージ (anthropic, pymupdf4llm, pydantic, typer, rich, pyyaml)
- `src/pdf2anki/schemas.py` -- 8種CardType, BloomLevel, AnkiCard (frozen=True), ExtractionResult, CardConfidenceScore
- `src/pdf2anki/config.py` -- YAML + 環境変数の統合管理
- `config.yaml`, `.env.example`
- `tests/test_schemas.py`, `tests/conftest.py`

**スキーマ設計のポイント:**
- `CardType` enum: qa, term_definition, summary_point, cloze, reversible, sequence, compare_contrast, image_occlusion
- `BloomLevel` enum: remember, understand, apply, analyze, evaluate, create
- 全モデル `frozen=True` (g-kentei-iosのイミュータブルパターン踏襲)
- `related_concepts`, `mnemonic_hint` フィールド追加

**g-kentei-ios参照:** `g-kentei-ios/Scripts/models.py` (frozen dataclass パターン)

---

### Phase 2: テキスト抽出

**ファイル:**
- `src/pdf2anki/extract.py` -- PDF (pymupdf4llm), TXT, MD 抽出 + OCRフォールバック
- `tests/test_extract.py`

**処理:** PDF→pymupdf4llm→Markdown、空/少文字→OCR、前処理（空行正規化、制御文字除去）、150Kトークン上限分割

---

### Phase 3: LLM構造化抽出 + Wozniakプロンプト

**ファイル:**
- `src/pdf2anki/prompts.py` -- SYSTEM_PROMPT (Wozniak原則統合), CRITIQUE_PROMPT, build_user_prompt()
- `src/pdf2anki/structure.py` -- Claude API Structured Outputs呼び出し, リトライ, Prompt Caching
- `src/pdf2anki/cost.py` -- CostTracker, 予算制限, モデルルーティング (Haiku/Sonnet)
- `tests/test_prompts.py`, `tests/test_structure.py`, `tests/test_cost.py`

**Wozniak原則のプロンプト統合:**
- 最小情報原則の例示（良い例/悪い例）
- リスト→cloze変換ルール
- 冗長性活用ルール（同概念を正方向/逆方向/穴埋めで）
- Bloomレベル判定基準

**Structured Outputs注意:** Haiku 4.5未対応の場合 → tool_use strict mode or Sonnet 4.5（月$0.78で許容範囲）

**g-kentei-ios参照:** `g-kentei-ios/Scripts/llm_validator.py` (CostTracker, 予算制限バッチ処理)

---

### Phase 4: 品質保証パイプライン ★最重要

**ファイル:**
- `src/pdf2anki/quality.py` -- CardConfidenceScorer, critique_cards(), run_quality_pipeline()
- `tests/test_quality.py`

**信頼度スコアリング（重み付け）:**
| フィールド | 重み | チェック内容 |
|-----------|------|------------|
| front品質 | 0.25 | 疑問文形式、適切な長さ、明確さ |
| back品質 | 0.25 | 簡潔さ、正確さ、200文字以内 |
| card_type適合 | 0.15 | 種別と内容の一致 |
| bloom_level | 0.10 | 認知レベルの妥当性 |
| tags品質 | 0.10 | 階層構造、存在 |
| 原子性 | 0.15 | 1カード1概念 |

**フラグ検出:** vague_question, too_long_answer, list_not_cloze, duplicate_concept, too_simple, hallucination_risk

**閾値:** ≥0.90 = パス、<0.90 = LLM批評 → 改善/分割/除去

**g-kentei-ios参照:**
- `g-kentei-ios/Scripts/confidence_scorer.py` (フィールド別スコアリング、重み付け、フラグ検出)
- `g-kentei-ios/Scripts/hybrid_parser.py` (パイプライン統合)

---

### Phase 5: TSV/JSON出力

**ファイル:**
- `src/pdf2anki/convert.py` -- TSV (Ankiインポート形式), JSON (メタデータ付き)
- `tests/test_convert.py`

**TSV出力仕様（主力フォーマット）:**
- ヘッダー: `#separator:tab`, `#html:true`, `#tags column:3`
- フォーマット: `front<TAB>back<TAB>tags`
- タブ→スペース置換、改行→`<br>`置換
- cloze型: frontのみ出力（backは空、Ankiが自動処理）
- reversible型: 正方向+逆方向の2行に展開
- タグにBloomレベル追加（`bloom::remember`等）
- タグに難易度追加（`difficulty::easy`等）
- エンコーディング: UTF-8 (BOMなし)

**JSON出力（連携用）:**
- `ExtractionResult.model_dump_json(indent=2)` + `_meta`情報
- スキーマバージョン、ソースファイル、生成日時、使用モデル

---

### Phase 6: CLI統合 + E2Eテスト

**ファイル:**
- `src/pdf2anki/main.py` -- typer CLI (convert, preview サブコマンド)

**CLIコマンド例:**
```bash
pdf2anki convert input.pdf                          # 基本変換 → TSV出力
pdf2anki convert input.pdf -o output.tsv            # 出力先指定
pdf2anki convert input.pdf --format both            # TSV + JSON 同時出力
pdf2anki convert ./docs/ -o ./output/ --format both # ディレクトリ一括処理
pdf2anki convert input.pdf --quality full           # 品質保証パイプライン有効
pdf2anki convert input.pdf --max-cards 30           # カード数上限
pdf2anki convert input.pdf --tags "AI::基礎"        # 追加タグ
pdf2anki convert input.pdf --focus "CNN,RNN"        # 重点トピック
pdf2anki preview input.pdf                          # dry-run（テキスト抽出のみ）
```

**主要オプション:**
- `-o, --output` -- 出力先
- `--format` -- tsv / json / both (default: tsv)
- `--quality` -- off / basic / full (default: basic)
- `--model` -- Claude モデル名
- `--max-cards` -- 最大カード数
- `--tags` -- 追加タグ（カンマ区切り）
- `--focus` -- 重点トピック（カンマ区切り）
- `--card-types` -- 生成するカード種別（カンマ区切り）
- `--bloom-filter` -- 特定Bloomレベルのみ
- `--budget-limit` -- 予算上限 (USD)
- `--ocr` -- OCR有効化
- `--lang` -- OCR言語 (default: jpn+eng)
- `--config` -- 設定ファイルパス
- `--verbose` -- 詳細ログ

---

## 将来の拡張（コア完成後に追加）

### 拡張A: 知識グラフ抽出（`--knowledge-graph`オプション）

**ファイル:** `src/pdf2anki/knowledge_graph.py`, `tests/test_knowledge_graph.py`

**概要:** テキストから概念間の関係 (is_a, causes, requires, contrasts_with) をLLMで抽出し、以下を実現:
- **関係ベースカード自動生成**: 「AとBの違いは？」「Aの前提条件は？」
- **重複カード検出**: 同概念を同角度からテストするカードを統合
- **学習順序の推定**: 前提条件のトポロジカルソートで最適な学習順を提案
- **カバレッジの穴検出**: テキストに登場するがカード化されていない概念を警告

**実装タイミング:** Step 2のLLM呼び出しで同時抽出（追加コストゼロ）か、独立した後処理ステップか選択可能。`ExtractionResult`スキーマの`concept_relations`フィールドは初回から定義しておく（空リストがデフォルト）。

### 拡張B: AnkiConnect直接push（`pdf2anki push`コマンド）

**ファイル:** `src/pdf2anki/anki_connect.py`, `tests/test_anki_connect.py`

**概要:** Anki Desktop起動中にlocalhost:8765経由でカードを直接追加。ファイルインポート不要。

### 拡張C: .apkg出力（`--format apkg`オプション）

**ファイル:** `src/pdf2anki/apkg_builder.py`, `tests/test_apkg_builder.py`

**概要:** genankiで.apkgファイルを生成。CSS/フォントは利用者の好みに委ねるため、Ankiデフォルトのノートタイプを使用。安定GUIDで再インポート時の更新に対応。

---

## 検証方法

```bash
# プロジェクト初期化
cd /Users/shimomoto_tatsuya/MyAI_Lab/Anki-QA
uv sync --all-extras

# テスト実行
uv run pytest tests/ -v --cov=src/pdf2anki --cov-report=term-missing

# 基本動作確認
uv run pdf2anki convert sample.txt -o output.tsv

# 品質保証付き変換
uv run pdf2anki convert sample.pdf -o output/ --format both --quality full

# dry-run
uv run pdf2anki preview sample.pdf
```

---

## リスクと軽減策

| リスク | 軽減策 |
|--------|--------|
| Haiku 4.5がStructured Outputs未対応 | tool_use strict mode。Sonnet 4.5使用（月$0.78） |
| 品質パイプラインのコスト増加 | budget_limit設定。高信頼度カードはLLM批評スキップ |

---

## 参考資料

- [Wozniakの20原則](https://supermemo.guru/wiki/20_rules_of_knowledge_formulation)
- [AnkiAIUtils](https://github.com/thiswillbeyourgithub/AnkiAIUtils) -- セマンティックフィルタリング、ニーモニック生成
- [AnkiAny](https://github.com/alingse/ankiany) -- Claude Agent SDK ベースのカード生成
- [Anki MCP Server](https://github.com/dhkim0124/anki-mcp-server) -- Claude Desktop連携
- [FSRS Algorithm](https://github.com/open-spaced-repetition/fsrs4anki) -- 最新間隔反復アルゴリズム
- g-kentei-ios: `Scripts/confidence_scorer.py`, `Scripts/models.py`, `Packages/GKenteiCore/Sources/Services/FSRSAlgorithm.swift`
