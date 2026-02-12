# Phase 3 Task 3: Gradio Web UI PoC

**作成日:** 2026-02-11
**推定工数:** 4 時間
**ステータス:** Draft

---

## Context

Phase 3 Task 1 (AnkiConnect) と Task 2 (CJK Token Estimation) 完了後、非開発者向け GUI の PoC として Gradio ベースの Web UI を構築する。CLI の service 層を再利用し、ブラウザから PDF/TXT/MD のアップロード → カード生成 → エクスポート/Anki Push を実現する。

**PoC なので自動テストは不要。手動テストのみ。**

---

## ファイル構成

| ファイル | 操作 | LOC目安 | 内容 |
|---------|------|---------|------|
| `src/pdf2anki/web/__init__.py` | **新規** | ~15 | Import guard + re-export |
| `src/pdf2anki/web/app.py` | **新規** | ~150 | Gradio interface + handlers |
| `src/pdf2anki/main.py` | 修正 | +15 | `web` サブコマンド追加 (L486以降) |
| `pyproject.toml` | 修正 | +3 | `[web]` optional dependency 追加 |

**合計:** ~183 LOC (200 LOC 以内)

---

## 設計

### UI レイアウト

```
┌────────────────────────────────────────────────────────────────┐
│  pdf2anki: Generate Anki Flashcards from Documents             │
├────────────────────────┬───────────────────────────────────────┤
│  Upload & Configure    │  Results                              │
│  ┌──────────────────┐  │  ┌─────────────────────────────────┐ │
│  │ [Upload File]    │  │  │ Status (Textbox)                │ │
│  │ PDF/TXT/MD       │  │  │ ✓ Generated 23 cards            │ │
│  └──────────────────┘  │  └─────────────────────────────────┘ │
│  Model: [Sonnet ▾]    │  ┌─────────────────────────────────┐ │
│  Quality: [Basic ▾]   │  │ Cost: $0.0245 (Textbox)         │ │
│  Max Cards: [━━●━] 50 │  └─────────────────────────────────┘ │
│  Budget: [$1.00]      │  ┌─────────────────────────────────┐ │
│  ☑ Vision API         │  │ Cards Table (Dataframe)         │ │
│                        │  │ Front | Back | Type | Bloom     │ │
│  [Generate Cards]      │  └─────────────────────────────────┘ │
│                        │  [Download TSV] [Download JSON]      │
│                        │  [Push to Anki] + Status             │
└────────────────────────┴───────────────────────────────────────┘
```

### service 層の再利用

既存の公開 API をそのまま呼び出し、ロジック重複なし:

```python
from pdf2anki.config import load_config, AppConfig      # 設定ロード
from pdf2anki.cost import CostTracker                    # コスト追跡
from pdf2anki.service import process_file                # メイン処理
from pdf2anki.convert import cards_to_tsv, cards_to_json # エクスポート
from pdf2anki.anki_connect import push_cards, is_anki_running  # Anki push
```

### web/__init__.py

```python
"""Web UI (optional dependency: gradio)."""
try:
    import gradio  # noqa: F401
except ImportError as e:
    raise ImportError(
        "Web UI requires Gradio. Install: pip install pdf2anki[web]"
    ) from e

from pdf2anki.web.app import launch_web
__all__ = ["launch_web"]
```

### web/app.py 主要関数

1. **`_build_config_from_ui()`** — UI 入力 → AppConfig 構築 (main.py の `_build_config()` パターン踏襲)
2. **`generate_cards()`** — アップロードファイル → `process_file()` → (status, cost, cards_table, result_state)
3. **`export_tsv()` / `export_json()`** — State から一時ファイル生成 → ダウンロード
4. **`push_to_anki()`** — `push_cards()` 呼び出し → ステータス表示
5. **`create_interface()`** — `gr.Blocks` で UI 構築
6. **`launch_web(host, port, share)`** — エントリーポイント

### main.py 追加 (L486 以降)

```python
@app.command()
def web(
    host: str = typer.Option("127.0.0.1", "--host"),
    port: int = typer.Option(7860, "--port"),
    share: bool = typer.Option(False, "--share"),
) -> None:
    """Launch Gradio web interface."""
    try:
        from pdf2anki.web import launch_web
    except ImportError:
        console.print("[red]Error:[/red] pip install pdf2anki[web]")
        raise typer.Exit(code=1)
    launch_web(host=host, port=port, share=share)
```

### pyproject.toml 追加

```toml
web = [
    "gradio>=5.0",
]
```

`[project.optional-dependencies]` の `ocr` と `dev` の間に挿入。

---

## 実装手順

| # | タスク | 工数 |
|---|--------|------|
| 1 | `pyproject.toml` に `[web]` extra 追加 + `uv sync --extra web` | 10min |
| 2 | `web/__init__.py` 作成 (import guard) | 5min |
| 3 | `web/app.py` 作成 (UI + handlers) | 2h |
| 4 | `main.py` に `web` サブコマンド追加 | 15min |
| 5 | ruff + mypy チェック | 10min |
| 6 | 手動テスト (ファイルアップロード、生成、エクスポート) | 30min |
| 7 | Phase 3 プラン更新 + MEMORY 更新 | 10min |

---

## 検証方法 (手動)

```bash
# 1. 依存インストール
uv sync --extra web

# 2. Web UI 起動
uv run pdf2anki web

# 3. ブラウザで http://127.0.0.1:7860 にアクセス

# 4. テストチェックリスト:
#    - [ ] TXT ファイルアップロード → カード生成
#    - [ ] PDF ファイルアップロード → カード生成
#    - [ ] Model を Haiku に変更 → 生成
#    - [ ] Quality を off/full に切り替え
#    - [ ] TSV ダウンロード
#    - [ ] JSON ダウンロード
#    - [ ] Push to Anki (Anki 起動時)
#    - [ ] エラーケース: ファイル未選択で Generate
```

---

## 注意事項

- Gradio は optional dependency → 未インストール時は明確なエラーメッセージ
- `ANTHROPIC_API_KEY` 環境変数が必要 (CLI と同様)
- ローカル専用想定 (認証なし)。`--share` はオプトイン
- テンポラリファイルは処理後に削除
