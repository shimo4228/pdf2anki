# Long-Document Processing Implementation Plan

> 作成日: 2026-02-09
> 対象プロジェクト: pdf2anki (Anki-QA)
> 基礎資料: [long-document-processing.md](../research/long-document-processing.md)

---

## Context

572K文字の日本語PDF「正理の海全編」の処理で、現行パイプラインの限界が顕在化:

| 問題 | 現状 | 改善後 |
|------|------|--------|
| **コスト** | ~$0.90/回（全文1チャンク送信） | ~$0.35/回（**60%削減**） |
| **品質** | "Lost in the Middle" で中間部30%+劣化 | セクション単位で注意力維持 |
| **構造** | `\n\n` 段落分割のみ | 章・節の論理構造を活用 |
| **タグ** | 階層タグなし | `正理の海::本論::第1章` 自動生成 |
| **並列処理** | 逐次処理のみ | Batch API対応（50%割引） |

---

## Phase 1: 構造的チャンキング（最優先）

### 新規: `src/pdf2anki/section.py` (~250 LOC)

**データモデル:**

```python
@dataclass(frozen=True, slots=True)
class Section:
    id: str               # "section-0", "section-1-2"
    heading: str           # "序論", "第1章 論書名の意味"
    level: int             # 1=H1, 2=H2, 3=H3, 0=preamble
    breadcrumb: str        # "正理の海 > 本論 > 第1章"
    text: str              # セクション本文（見出し行含む）
    page_range: str        # "pp.3-18" or ""
    char_count: int        # len(text), 事前計算
```

**関数:**

| 関数 | 説明 |
|------|------|
| `split_by_headings(markdown_text, *, max_chars=30_000, document_title="") -> list[Section]` | `#`/`##`/`###` 境界で分割。パンくずスタック構築。超過セクションは `\n\n` でサブ分割。フォールバック: 日本語見出し検出 → 単一Section |
| `_detect_japanese_headings(text) -> list[tuple[int, str, int]]` | `第\d+章`, `第[一二三...]章`, `序論\|本論\|結論`, `\d+\.\s+`, `（\d+）`, `[一二三...]、` |
| `_subsplit_section(section, max_chars) -> list[Section]` | 超過セクションの段落境界サブ分割 |
| `_build_breadcrumb(heading_stack: list[str]) -> str` | 見出しスタックを ` > ` で結合 |
| `extract_page_ranges(page_chunks: list[dict]) -> dict[str, str]` | `toc_items` からページ範囲をマッピング |

### 変更: `src/pdf2anki/extract.py`

1. `ExtractedDocument` に `sections: tuple[Section, ...] = ()` を追加（後方互換のデフォルト値）
2. 新関数 `_extract_pdf_with_sections(path)` — `pymupdf4llm.to_markdown(str(path), page_chunks=True)` を呼び出し、ページテキストを結合、`split_by_headings()` に委譲
3. `extract_text()` — PDFの場合、`chunks`（セクションテキストから）と `sections`（完全なSectionオブジェクト）の両方を設定。MDファイルも見出しベース分割を試行。TXTファイルは段落境界チャンキングのみ維持

**後方互換性:** 既存の `chunks` フィールドはセクションテキストから設定される。`doc.chunks` を使うコンシューマーには変更なし。

### テスト: `tests/test_section.py` (~300 LOC)

- H1/H2/H3分割、パンくず生成、超過セクションのサブ分割
- 日本語見出し検出（各パターン）
- 空テキスト、見出しなしフォールバック、混合見出しレベル
- `toc_items` からのページ範囲抽出

### テスト: `tests/test_extract.py` への追加 (~50 LOC)

- PDF抽出が `sections` フィールドを設定
- MD抽出が `sections` フィールドを設定
- `chunks` がセクションテキストと一致（後方互換確認）

---

## Phase 2: パンくずコンテキスト付きプロンプト

### 変更: `src/pdf2anki/prompts.py`

既存の `build_user_prompt()` は変更なし。**新関数** `build_section_prompt()` を追加:

```python
def build_section_prompt(
    section: Section,
    *,
    document_title: str = "",
    max_cards: int = 20,
    card_types: list[str] | None = None,
    focus_topics: list[str] | None = None,
    bloom_filter: list[str] | None = None,
    additional_tags: list[str] | None = None,
) -> str:
```

- ドキュメントタイトル、パンくず、ページ範囲をヘッダーに含める
- セクションパスからの階層タグ生成をLLMに指示（例: `正理の海::本論::第1章`）
- セクション単位のため `max_cards=20`（ドキュメント単位の50から削減）

### 変更: `src/pdf2anki/structure.py`

- `extract_cards()` に `sections: list[Section] | None = None` パラメータを追加
- `sections` が提供された場合: セクションをイテレート、`build_section_prompt()` を使用、セクション単位モデルルーティング（`select_model(text_length=section.char_count, ...)`）
- 既存の `chunks` パスは変更なし

### 変更: `src/pdf2anki/main.py`

- `_process_file()`: `doc.sections` が非空の場合、`chunks` の代わりに `sections=list(doc.sections)` を `extract_cards()` に渡す

### テスト (~140 LOC追加)

- `test_prompts.py`: パンくず含有、タグ階層、ページ範囲、ドキュメントタイトル
- `test_structure.py`: セクション対応抽出、セクション単位モデルルーティング
- `test_main.py`: `_process_file()` がセクション利用可能時にセクションを渡す

---

## Phase 3: Batch API対応

### 新規: `src/pdf2anki/batch.py` (~250 LOC)

**データモデル:**

```python
@dataclass(frozen=True, slots=True)
class BatchRequest:
    custom_id: str          # "section-0", "section-1"
    section: Section
    user_prompt: str

@dataclass(frozen=True, slots=True)
class BatchResult:
    custom_id: str
    cards: list[AnkiCard]
    input_tokens: int
    output_tokens: int
    model: str
```

**関数:**

| 関数 | 説明 |
|------|------|
| `create_batch_requests(sections, *, document_title, config) -> list[BatchRequest]` | セクションからバッチリクエストを生成 |
| `submit_batch(requests, *, model, max_tokens) -> str` | バッチ送信、batch_idを返す |
| `poll_batch(batch_id, *, poll_interval=30.0, timeout=3600.0) -> list[BatchResult]` | ポーリングで完了を待つ |

- `anthropic.Anthropic().messages.batches.create()` + プロンプトキャッシュ
- `rich.progress` によるプログレスバー

### 変更: `src/pdf2anki/cost.py`

- `BATCH_PRICING` 辞書を追加（標準価格の50%）
- `estimate_cost()` に `batch: bool = False` パラメータを追加

### 変更: `src/pdf2anki/config.py`

- フィールド追加: `batch_enabled: bool = False`, `batch_poll_interval: float = 30.0`, `batch_timeout: float = 3600.0`

### 変更: `src/pdf2anki/structure.py`

- `_parse_cards_response` → `parse_cards_response` にリネーム（batch.pyと共有のため）

### 変更: `src/pdf2anki/main.py`

- `--batch` CLIフラグを追加
- `--batch` + sections の場合: バッチ送信 → ポーリング → 結果パース → 品質パイプライン
- `--batch-poll-interval`, `--batch-timeout` オプションも追加

### テスト: `tests/test_batch.py` (~200 LOC)

- `messages.batches` をモックして submit/poll/results をテスト
- バッチ価格設定、タイムアウト処理、部分的失敗

---

## Phase 4: セクション間重複排除

### 既存パイプラインへの最小限の変更

現在の `quality.py` は既にマージされた全カードリストに対して `_detect_duplicates()` を実行する。セクション単位で独立生成されたカードが品質パイプライン前にマージされるため、セクション間重複は自動的に検出される。

**強化策:** `structure.py` で、各セクションからカード生成後にセクション起源タグ（例: `_section::section-0`）を各カードに注入。`AnkiCard` スキーマ変更なしで重複追跡を支援。

### テスト (~30 LOC追加)

- `test_quality.py` に統合テスト: 異なる「セクション」からの重複カードがフラグ付けされることを確認

---

## Phase 5: 非同期並列処理（DEFERRED）

**延期の根拠:**

- Batch API（Phase 3）が50%割引でコスト問題を既に解決
- Asyncは `pytest-asyncio`, `asyncio.Lock` (CostTracker), async CLI統合が必要
- バッチ処理と比較して複雑さに対するリターンが低い
- 主要ユースケース（長い日本語PDF）はバッチ向き、レイテンシー非依存

将来必要な場合: `async_structure.py` に `AsyncAnthropic` + `Semaphore(N)` + `--parallel N` フラグ。

---

## 実装順序

```
Phase 1 (section.py + extract.py)      ← 基盤、最初に着手
    ↓
Phase 2 (prompts.py + structure.py)    ← Section データクラスに依存
    ↓
Phase 3 (batch.py + cost.py + CLI)     ← build_section_prompt に依存
  + Phase 4 (quality.py, 軽量)          ← Phase 3 と並行可能
    ↓
Phase 5 (延期)
```

---

## ファイル変更サマリー

### ソースコード

| ファイル | アクション | 差分 |
|----------|-----------|------|
| `src/pdf2anki/section.py` | **新規作成** | +250 LOC |
| `src/pdf2anki/batch.py` | **新規作成** | +250 LOC |
| `src/pdf2anki/extract.py` | 変更 | +62 LOC |
| `src/pdf2anki/prompts.py` | 変更 | +67 LOC |
| `src/pdf2anki/structure.py` | 変更 | +61 LOC |
| `src/pdf2anki/config.py` | 変更 | +22 LOC |
| `src/pdf2anki/cost.py` | 変更 | +23 LOC |
| `src/pdf2anki/main.py` | 変更 | +59 LOC |
| `src/pdf2anki/quality.py` | 変更 | +14 LOC |

### テストコード

| ファイル | アクション | 差分 |
|----------|-----------|------|
| `tests/test_section.py` | **新規作成** | +300 LOC |
| `tests/test_batch.py` | **新規作成** | +200 LOC |
| `tests/test_extract.py` | 追加 | +50 LOC |
| `tests/test_prompts.py` | 追加 | +60 LOC |
| `tests/test_structure.py` | 追加 | +80 LOC |
| `tests/test_main.py` | 追加 | +40 LOC |
| `tests/test_quality.py` | 追加 | +30 LOC |

全ファイル800 LOC上限以内。新規ファイルは400 LOC以内。

---

## リスクと緩和策

| リスク | 深刻度 | 緩和策 |
|--------|--------|--------|
| `page_chunks=True` が想定外の構造を返す | **HIGH** | 実PDFで統合テスト。pymupdf4llmバージョンを固定。`page_chunks=False` へのtry/exceptフォールバック |
| 本文中の日本語見出しパターン誤検出 | **MEDIUM** | `^` 行頭一致 + 前後空行を要求。最小セクションサイズ閾値 |
| `\n\n` 境界のない超過セクション | **LOW** | 文字境界フォールバック（既存の `split_into_chunks()` を活用） |
| Batch APIタイムアウト | **LOW** | タイムアウトを設定可能に（デフォルト1時間）。batch IDでの再確認を許可 |
| セクション間の予算トラッキング | **MEDIUM** | 逐次処理を維持。各セクション前に予算チェック |

---

## 検証手順

1. **ユニットテスト**: `pytest --cov=src --cov-report=term-missing` — 80%+カバレッジ維持
2. **統合テスト**: `pdf2anki preview` をマルチセクションPDFで実行 → セクション検出を確認
3. **E2Eテスト**: `pdf2anki convert sample.pdf --verbose` を小さな構造化PDFで実行 → 出力カードにパンくずタグが含まれることを確認
4. **バッチテスト**: `pdf2anki convert large.pdf --batch --verbose` → バッチ送信、ポーリング、カード出力を確認
5. **リント/型チェック**: `ruff check src/ tests/` + `mypy src/`

---

## アーキテクチャ図

### 改善後のパイプライン

```
Input (PDF/TXT/MD)
    ↓
[Step 1] Text Extraction (extract.py)
    ├── PDF: pymupdf4llm.to_markdown(page_chunks=True)
    │       ↓
    │   section.py: split_by_headings()
    │       ├── Markdown headings (#, ##, ###)
    │       ├── Japanese heading fallback
    │       └── Oversized section sub-split
    │       ↓
    │   ExtractedDocument(sections=(...), chunks=(...))
    │
    ├── MD: split_by_headings() (直接)
    └── TXT: split_into_chunks() (既存)
    ↓
[Step 2] Card Generation (structure.py / batch.py)
    ├── Standard: セクション単位で逐次API呼び出し
    │   └── build_section_prompt() with breadcrumb context
    │
    └── --batch: Batch API で全セクション一括送信 (50%割引)
        ├── submit_batch()
        ├── poll_batch() with progress bar
        └── parse results
    ↓
[Step 3] Quality Pipeline (quality.py)
    ├── Heuristic scoring (6 dimensions)
    ├── Cross-section duplicate detection
    └── LLM critique (if enabled)
    ↓
[Step 4] Output (convert.py)
    ├── TSV (Anki importable)
    └── JSON (with metadata)
```

### コスト比較

```
現状:   572K文字 → 1チャンク → Sonnet     = ~$0.90
改善後: 572K文字 → ~20セクション → Batch API = ~$0.35 (約60%削減)
```
