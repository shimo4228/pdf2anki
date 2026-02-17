# pdf2anki

PDF/テキストから Anki フラッシュカードを自動生成する Python CLI ツール。Claude API で構造化抽出 → TSV/JSON 出力。

## Tech Stack

- Python >=3.12 / uv + pyproject.toml
- CLI: Typer + Rich
- Web UI: Gradio
- AI: Anthropic Claude API (テキスト + Vision)
- PDF: PyMuPDF
- テスト: pytest + pytest-cov (カバレッジ 92%)
- セキュリティ: pre-commit + detect-secrets

## Directory Structure

```
src/pdf2anki/
├── main.py              # Typer CLI エントリポイント
├── service.py           # ビジネスロジック (process_file, extract_with_cache)
├── structure.py         # Claude API カード抽出 (extract_cards)
├── vision.py            # Vision API カード抽出
├── batch.py             # Batch API (50% コスト削減)
├── extract.py           # PDF/テキスト抽出
├── section.py           # セクション分割
├── quality.py           # 品質パイプライン (critique → improve → split)
├── convert.py           # TSV/JSON 出力
├── cost.py              # コスト追跡 + 見積もり
├── cache.py             # SHA-256 ベースキャッシュ
├── image.py             # PDF 画像抽出
├── config.py            # AppConfig (frozen dataclass)
├── schemas.py           # AnkiCard, ExtractionResult
├── web.py               # Gradio Web UI
├── anki_connect.py      # AnkiConnect API 連携
├── eval.py              # 評価フレームワーク
└── tui.py               # インタラクティブカードレビュー TUI
tests/                   # pytest テスト群
docs/                    # Phase 計画、コードレビュー結果
```

## Build / Test / Run

```bash
# セットアップ
uv venv && uv sync
cp .env.example .env     # ANTHROPIC_API_KEY を設定

# CLI 実行
.venv/bin/python -m pdf2anki input.pdf
.venv/bin/python -m pdf2anki input.pdf --format both --quality full
.venv/bin/python -m pdf2anki input.pdf --vision    # Vision API モード
.venv/bin/python -m pdf2anki input.pdf --batch     # Batch API モード
.venv/bin/python -m pdf2anki input.pdf --review    # TUI レビュー

# Web UI
.venv/bin/python -m pdf2anki --web

# テスト
.venv/bin/python -m pytest tests/ -v
.venv/bin/python -m pytest tests/ --cov=pdf2anki --cov-report=term-missing
```

## Environment Variables

| 変数 | 必須 | 用途 |
|------|------|------|
| `ANTHROPIC_API_KEY` | Yes | Claude API キー |
| `PDF2ANKI_MODEL` | No | モデルオーバーライド |
| `PDF2ANKI_BUDGET_LIMIT` | No | USD 予算上限 |

## Conventions

- Immutable: `AppConfig` は `frozen=True` dataclass。`CostTracker` も immutable (`.add()` で新インスタンス返却)
- エラーハンドリング: リトライは `anthropic.RateLimitError` 等の特定例外のみ
- コスト管理: 全 API コールを `CostTracker` で追跡、予算超過時は停止
- pre-commit: `detect-secrets` でシークレットスキャン
- CJK 対応: トークン見積もりは日本語文字を考慮

## Status

Phase 3 完了 (v0.3.0+)。Phase 4 バックログあり。
