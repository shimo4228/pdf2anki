# AnkiConnect API 統合 — 実装計画

**作成日:** 2026-02-11
**親プラン:** `2026-02-11_PLAN_phase3.md` Task 1
**ステータス:** Ready to implement
**推定工数:** 6h

---

## 目的

`pdf2anki convert input.pdf --push` で Anki にカードを直接追加する。
TSV 手動インポートの摩擦を排除し、ワンコマンド完結を実現。

---

## 調査結果

### AnkiConnect API 仕様 (v6)

**エンドポイント:** `POST http://127.0.0.1:8765`

**リクエスト形式:**
```json
{
    "action": "actionName",
    "version": 6,
    "params": {}
}
```

**レスポンス形式:**
```json
{
    "result": <any>,
    "error": <string | null>
}
```

**使用するアクション:**

| アクション | 用途 | params |
|-----------|------|--------|
| `version` | 接続確認 | なし |
| `createDeck` | デッキ作成/確認 | `deck: str` |
| `addNotes` | バッチノート追加 | `notes: list[Note]` |

**Note オブジェクト形式:**
```json
{
    "deckName": "pdf2anki",
    "modelName": "Basic",
    "fields": {"Front": "...", "Back": "..."},
    "tags": ["tag1", "tag2"],
    "options": {"allowDuplicate": false}
}
```

**ノートタイプ別フィールド:**
- **Basic**: `Front`, `Back`
- **Basic (and target: reversed card)**: `Front`, `Back` (自動で逆方向カードも作成)
- **Cloze**: `Text`, `Extra`

**addNotes レスポンス:** `[noteId, noteId, null, ...]` — null は追加失敗を意味する

### 既存コードベースのパターン

**service.py (380 LOC):**
- `process_file()` → `ExtractionResult` を返す
- `write_output()` → ファイル書き出し
- main.py の convert コマンドフロー:
  1. `process_file()` → result
  2. `launch_review()` (--review 時) → filtered result
  3. `write_output()` → files
  4. `_print_summary()` → 表示

**→ push は `write_output()` の後に配置するのが自然**

**main.py convert コマンドのオプション追加パターン:**
```python
# 既存パターン: bool フラグ
review: bool = typer.Option(False, "--review", help="...")
batch: bool = typer.Option(False, "--batch", help="...")
# 既存パターン: Optional str
model: str | None = typer.Option(None, "--model", help="...")
```

**schemas.py AnkiCard:**
```python
class AnkiCard(BaseModel, frozen=True):
    front: str
    back: str = ""
    card_type: CardType   # QA, CLOZE, REVERSIBLE, TERM_DEFINITION, etc.
    bloom_level: BloomLevel
    tags: list[str]
    media: list[str] = []
```

**CardType → AnkiConnect モデル名マッピング:**
- `CLOZE` → `"Cloze"` (fields: Text, Extra)
- `REVERSIBLE` → `"Basic (and target: reversed card)"` (fields: Front, Back)
- その他すべて → `"Basic"` (fields: Front, Back)

**テストパターン (conftest.py):**
- `sample_qa_card`, `sample_cloze_card`, `sample_reversible_card` fixtures
- `unittest.mock.patch` でモック
- 外部 API テストはすべてモックベース

---

## ファイル構成

| ファイル | 操作 | LOC目安 | 内容 |
|---------|------|---------|------|
| `src/pdf2anki/anki_connect.py` | **新規** | ~160 | AnkiConnect HTTP クライアント |
| `src/pdf2anki/main.py` | 修正 | +20 | `--push`, `--deck` オプション追加 |
| `tests/test_anki_connect.py` | **新規** | ~200 | モックベーステスト |

**修正しないファイル:** service.py, convert.py, schemas.py, config.py
（push は CLI 層の関心事。service.py に入れると AnkiConnect が起動必須になる）

---

## anki_connect.py 設計

```python
"""AnkiConnect API client for pushing cards to Anki.

Uses only urllib.request (no new dependencies).
AnkiConnect add-on (2055492159) must be installed in Anki.
"""
from __future__ import annotations

import json
import urllib.request
import urllib.error
from dataclasses import dataclass

from pdf2anki.schemas import AnkiCard, CardType

ANKICONNECT_URL = "http://127.0.0.1:8765"
ANKICONNECT_VERSION = 6


class AnkiConnectError(Exception):
    """AnkiConnect API error."""


@dataclass(frozen=True)
class PushResult:
    total: int
    added: int
    failed: int
    errors: tuple[str, ...]


def _invoke(action: str, *, url: str = ANKICONNECT_URL, **params: Any) -> Any:
    """Call AnkiConnect API. Raises AnkiConnectError on failure."""
    payload = json.dumps({
        "action": action,
        "version": ANKICONNECT_VERSION,
        "params": params,
    }).encode()
    req = urllib.request.Request(url, data=payload, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req) as resp:
            body = json.loads(resp.read())
    except urllib.error.URLError as e:
        raise AnkiConnectError(
            "Anki is not running or AnkiConnect is not installed.\n"
            "Start Anki and install AnkiConnect add-on (code: 2055492159)."
        ) from e
    if body.get("error"):
        raise AnkiConnectError(body["error"])
    return body["result"]


def is_anki_running(*, url: str = ANKICONNECT_URL) -> bool:
    """Check if AnkiConnect is responsive."""
    try:
        _invoke("version", url=url)
        return True
    except AnkiConnectError:
        return False


def ensure_deck(deck_name: str, *, url: str = ANKICONNECT_URL) -> None:
    """Create deck if it doesn't exist."""
    _invoke("createDeck", url=url, deck=deck_name)


def card_to_note(card: AnkiCard, *, deck_name: str) -> dict:
    """Convert AnkiCard to AnkiConnect note dict."""
    tags = list(card.tags) + [f"bloom::{card.bloom_level.value}"]

    if card.card_type == CardType.CLOZE:
        return {
            "deckName": deck_name,
            "modelName": "Cloze",
            "fields": {"Text": card.front, "Extra": card.back},
            "tags": tags,
            "options": {"allowDuplicate": False},
        }

    if card.card_type == CardType.REVERSIBLE:
        return {
            "deckName": deck_name,
            "modelName": "Basic (and target: reversed card)",
            "fields": {"Front": card.front, "Back": card.back},
            "tags": tags,
            "options": {"allowDuplicate": False},
        }

    # QA, TERM_DEFINITION, SUMMARY_POINT, SEQUENCE, COMPARE_CONTRAST, IMAGE_OCCLUSION
    return {
        "deckName": deck_name,
        "modelName": "Basic",
        "fields": {"Front": card.front, "Back": card.back},
        "tags": tags,
        "options": {"allowDuplicate": False},
    }


def push_cards(
    cards: list[AnkiCard],
    *,
    deck_name: str = "pdf2anki",
    url: str = ANKICONNECT_URL,
) -> PushResult:
    """Push cards to Anki via AnkiConnect. Returns PushResult."""
    if not cards:
        return PushResult(total=0, added=0, failed=0, errors=())

    ensure_deck(deck_name, url=url)

    notes = [card_to_note(c, deck_name=deck_name) for c in cards]
    results = _invoke("addNotes", url=url, notes=notes)

    added = sum(1 for r in results if r is not None)
    failed = sum(1 for r in results if r is None)
    errors = tuple(
        f"Card {i+1} failed to add"
        for i, r in enumerate(results) if r is None
    )

    return PushResult(total=len(cards), added=added, failed=failed, errors=errors)
```

---

## main.py 修正箇所

**追加オプション (convert コマンドに):**
```python
push: bool = typer.Option(
    False, "--push", help="Push cards to Anki via AnkiConnect"
),
deck: str = typer.Option(
    "pdf2anki", "--deck", help="Anki deck name for --push"
),
```

**push ロジック挿入箇所 (L285 付近、write_output の後):**
```python
# 既存: write_output → all_written.extend(written)

if push and result.cards:
    from pdf2anki.anki_connect import push_cards

    push_result = push_cards(list(result.cards), deck_name=deck)
    total_pushed += push_result.added
    if push_result.failed > 0:
        console.print(
            f"[yellow]Warning:[/yellow] {push_result.failed} card(s) "
            f"failed to push"
        )
```

**_print_summary 拡張:**
```python
if total_pushed > 0:
    table.add_row("Pushed to Anki", str(total_pushed))
```

---

## テスト計画

### test_anki_connect.py (~200 LOC)

**TestInvoke:**
- `test_invoke_success` — 正常レスポンス
- `test_invoke_sends_correct_json` — リクエスト形式検証
- `test_invoke_raises_on_api_error` — error フィールド非 null
- `test_invoke_raises_on_connection_error` — URLError

**TestIsAnkiRunning:**
- `test_returns_true_when_connected`
- `test_returns_false_on_connection_error`

**TestEnsureDeck:**
- `test_creates_deck`
- `test_propagates_error`

**TestCardToNote:**
- `test_qa_card_uses_basic_model`
- `test_cloze_card_uses_cloze_model`
- `test_term_card_uses_basic_model`
- `test_reversible_card_uses_basic_reversed_model`
- `test_tags_are_included`
- `test_bloom_tag_added`
- `test_options_no_duplicate`

**TestPushCards:**
- `test_push_empty_list`
- `test_push_successful`
- `test_push_partial_failure` — addNotes が null を含む
- `test_push_checks_anki_running` — 接続失敗時エラー
- `test_push_calls_ensure_deck`

**モック戦略:** `unittest.mock.patch("pdf2anki.anki_connect.urllib.request.urlopen")`

---

## 実装ステップ (TDD)

| # | タスク | 内容 |
|---|--------|------|
| 1 | テスト作成 (RED) | `tests/test_anki_connect.py` 全テスト作成 → import error で全 FAIL |
| 2 | 実装 (GREEN) | `src/pdf2anki/anki_connect.py` 作成 → 全テスト PASS |
| 3 | CLI 統合 | `main.py` に `--push`, `--deck` 追加 + test_main.py にテスト追加 |
| 4 | 検証 | `uv run pytest --cov` 全テスト PASS + カバレッジ 80%+ + ruff + mypy |

---

## 検証コマンド

```bash
# ユニットテスト
uv run pytest tests/test_anki_connect.py -v

# 全テスト + カバレッジ
uv run pytest --cov=src/pdf2anki --cov-report=term-missing

# Lint + Type check
uv run ruff check src/pdf2anki/anki_connect.py
uv run mypy src/pdf2anki/anki_connect.py --strict

# E2E テスト (Anki 起動状態で手動)
uv run pdf2anki convert tests/fixtures/sample.txt --push --deck "pdf2anki-test"
```

---

## 注意事項

1. **依存追加なし**: `urllib.request` のみ使用。依存は 8 個のまま
2. **画像 push は Phase 3 スコープ外**: テキストカードのみ対応
3. **Cloze 形式**: AnkiCard.front に既に `{{c1::...}}` が入っている前提（schemas.py の cloze fixture で確認済み）
4. **push は CLI 層のみ**: service.py に push ロジックを入れない。理由: service.py は Anki 非依存であるべき
5. **エラーメッセージ**: Anki 未起動時は AnkiConnect のインストール手順を含むメッセージを表示
