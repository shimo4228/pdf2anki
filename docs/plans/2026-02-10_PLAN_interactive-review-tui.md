# Task 4: Interactive Review TUI 実装ドキュメント

**作成日:** 2026-02-10
**ステータス:** ✅ 実装完了（624テスト全パス、カバレッジ92.26%）
**推定工数:** 6 時間
**著者:** Claude Code + shimomoto_tatsuya

---

## 背景と目的

Phase 2 の Task 1-3 が完了（Cache, Vision, Eval）。現在のパイプラインでは `process_file()` → `write_output()` と直結しており、生成カードの品質確認・取捨選択手段がない。

**解決:** Textual ベースの対話的カードレビュー TUI を挿入し、ユーザーが Anki インポート前にカードを Accept/Reject/Edit できるようにする。

---

## 技術選定

### Textual (7番目の依存)

| 観点 | 判断 |
|------|------|
| 選定理由 | 既存 `rich` と同じ Textualize 社、TUI 特化ライブラリ |
| 代替案 | curses（5x コード量増）、rich のみ（UX 劣化）、prompt_toolkit（依存追加は同じ） |
| Micro-Dependencies 原則 | Textual は「車輪」であり「倉庫」ではない。~150KB、サブ依存軽量 |
| テスト | Textual Pilot API で非同期 UI テスト可能 |

### 設計原則

- **不変状態管理**: frozen dataclass + Pydantic `model_copy(update={...})`
- **純粋関数**: 状態変更はすべて `state → new_state` の純粋関数
- **SRP**: TUI パッケージは自己完結、コア処理パイプラインに変更なし
- **TDD**: テスト先行、Textual pilot でウィジェットテスト

---

## アーキテクチャ

### パイプライン挿入ポイント

```
[既存] process_file() → ExtractionResult
            ↓
[NEW]  launch_review(result, scores) → ExtractionResult (承認カードのみ)
            ↓
[既存] write_output(result, ...)
```

### ファイル構成 (4ファイル、~570 LOC)

```
src/pdf2anki/tui/
├── __init__.py   (~30 LOC)  # launch_review() 公開 API
├── state.py      (~150 LOC) # ReviewState, CardStatus, 状態変更ヘルパー
├── widgets.py    (~140 LOC) # StatsBar, CardDisplay, ActionBar
└── app.py        (~250 LOC) # ReviewApp + EditCardScreen (モーダル)
```

---

## データモデル

### tui/state.py

```python
from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from pdf2anki.schemas import AnkiCard, CardConfidenceScore


class CardStatus(StrEnum):
    """カードのレビュー状態。"""
    PENDING = "pending"
    ACCEPTED = "accepted"
    REJECTED = "rejected"


@dataclass(frozen=True)
class ReviewCard:
    """レビュー対象カード。不変。"""
    card: AnkiCard
    original_index: int
    status: CardStatus = CardStatus.PENDING
    quality_score: CardConfidenceScore | None = None


@dataclass(frozen=True)
class ReviewState:
    """レビューセッション全体の不変状態。"""
    items: tuple[ReviewCard, ...]
    current_index: int = 0
    filter_status: CardStatus | None = None  # None = 全件表示

    @property
    def stats(self) -> dict[str, int]:
        return {
            "total": len(self.items),
            "accepted": sum(1 for i in self.items if i.status == CardStatus.ACCEPTED),
            "rejected": sum(1 for i in self.items if i.status == CardStatus.REJECTED),
            "pending": sum(1 for i in self.items if i.status == CardStatus.PENDING),
        }

    def filtered_items(self) -> list[ReviewCard]:
        if self.filter_status is None:
            return list(self.items)
        return [i for i in self.items if i.status == self.filter_status]
```

### 状態変更ヘルパー（純粋関数）

```python
def create_initial_state(
    cards: list[AnkiCard],
    scores: list[CardConfidenceScore] | None = None,
) -> ReviewState:
    """カードリストから初期 ReviewState を生成。"""

def set_card_status(state: ReviewState, item_index: int, status: CardStatus) -> ReviewState:
    """指定カードのステータスを変更した新 ReviewState を返す。"""

def edit_card(state: ReviewState, item_index: int, front: str, back: str) -> ReviewState:
    """指定カードの front/back を編集した新 ReviewState を返す。"""

def navigate(state: ReviewState, delta: int) -> ReviewState:
    """current_index を delta 分移動した新 ReviewState を返す (循環)。"""

def cycle_filter(state: ReviewState) -> ReviewState:
    """フィルタを None→pending→accepted→rejected→None とサイクル。"""
```

**不変性のポイント:**
- `ReviewCard` は frozen dataclass → 変更時は新インスタンス生成
- `AnkiCard` は Pydantic frozen → `card.model_copy(update={"front": new_front})`
- `ReviewState.items` は tuple → 変更時は list 化して修正後に再 tuple 化

---

## TUI 画面設計

```
┌─ pdf2anki Review ───────────────────────────────┐
│ Total: 47 | Accepted: 12 | Rejected: 3 | Pend: 32│ ← StatsBar
├─────────────────────────────────────────────────┤
│ Card 3/47              [QA]  Quality: 0.85      │
│                                                 │
│ Front: ニューラルネットワークの活性化関数の       │ ← CardDisplay
│        役割は何ですか？                          │
│ ─────────────────────────────────────────────── │
│ Back:  非線形変換を導入し、複雑なパターンの      │
│        学習を可能にする。                        │
│ ─────────────────────────────────────────────── │
│ Tags: AI::基礎, neural_network                  │
│ Bloom: understand                               │
├─────────────────────────────────────────────────┤
│ [A]ccept [R]eject [E]dit [N]ext [P]rev [F]ilter │ ← ActionBar
│ [S]ave & Quit  [Q]uit                           │
└─────────────────────────────────────────────────┘
```

### キーバインド

| キー | アクション | 説明 |
|------|----------|------|
| `a` | accept | カード承認 → 自動で次へ |
| `r` | reject | カード棄却 → 自動で次へ |
| `e` | edit | 編集モーダル表示 |
| `n` | next | 次のカード |
| `p` | prev | 前のカード |
| `f` | filter | フィルタ切替: 全件→pending→accepted→rejected→全件 |
| `s` | save | 承認カードのみ保存して終了 |
| `q` | quit | 保存せず終了 |

### EditCardScreen (モーダル)

`e` キーで表示。front/back を TextArea で編集し、Save/Cancel。

```python
class EditCardScreen(Screen[tuple[str, str] | None]):
    """カード編集モーダル。Save → (new_front, new_back)、Cancel → None。"""
```

---

## ウィジェット設計

### tui/widgets.py

| ウィジェット | 親クラス | 役割 |
|-------------|---------|------|
| `StatsBar` | `Static` | 上部: Total/Accepted/Rejected/Pending カウント |
| `CardDisplay` | `Container` | 中央: カード詳細 (front/back/tags/bloom/quality) |
| `ActionBar` | `Static` | 下部: キーバインドヘルプ + カード位置 |

CSS はインラインスタイル (`STYLES = """..."""`) を使用。`.tcss` ファイル不要。

---

## アプリケーション設計

### tui/app.py

```python
class ReviewApp(App[None]):
    """カードレビュー TUI メインアプリケーション。"""
    BINDINGS = [
        Binding("a", "accept_card", "Accept"),
        Binding("r", "reject_card", "Reject"),
        Binding("e", "edit_card", "Edit"),
        Binding("n", "next_card", "Next"),
        Binding("p", "prev_card", "Previous"),
        Binding("f", "cycle_filter", "Filter"),
        Binding("s", "save_and_quit", "Save"),
        Binding("q", "quit_app", "Quit"),
    ]

    def __init__(self, initial_state: ReviewState) -> None:
        self.state = initial_state
        self.save_requested = False
        super().__init__()
```

**状態更新パターン:**
```python
def action_accept_card(self) -> None:
    self.state = set_card_status(self.state, current_idx, CardStatus.ACCEPTED)
    self.state = navigate(self.state, +1)
    self._refresh_ui()
```

---

## 公開 API

### tui/__init__.py

```python
def launch_review(
    result: ExtractionResult,
    scores: list[CardConfidenceScore] | None = None,
) -> ExtractionResult:
    """TUI を起動しレビュー済み ExtractionResult を返す。

    save_requested=True → 承認カードのみ
    save_requested=False → 元の result をそのまま返す
    """
```

---

## CLI 統合

### main.py 修正

`convert` コマンドに `--review` フラグ追加:

```python
review: bool = typer.Option(False, "--review", help="Launch interactive review TUI")
```

挿入位置: `main.py:267-275` — `process_file()` 後、`write_output()` 前

```python
result, report, cost_tracker = process_file(...)

# NEW: Interactive review
if review and result.cards:
    from pdf2anki.quality.heuristic import score_cards
    from pdf2anki.tui import launch_review
    scores = score_cards(list(result.cards))
    result = launch_review(result, scores)

written = write_output(result=result, ...)
```

---

## テスト戦略

### test_tui_state.py (~120 LOC)

| テスト対象 | テスト内容 |
|-----------|-----------|
| `CardStatus` | 3状態の存在確認 |
| `ReviewCard` | frozen 不変性、デフォルト値 |
| `ReviewState.stats` | カウント正確性 |
| `ReviewState.filtered_items()` | フィルタ正確性 |
| `create_initial_state()` | カードリスト → 初期状態 |
| `set_card_status()` | ステータス変更 + 元状態不変確認 |
| `edit_card()` | front/back 変更 + 元カード不変確認 |
| `navigate()` | 境界値（先頭→末尾の循環）|
| `cycle_filter()` | 4状態サイクル |

### test_tui_app.py (~150 LOC)

Textual Pilot API で非同期テスト:

```python
async def test_accept_card():
    app = ReviewApp(sample_state)
    async with app.run_test() as pilot:
        await pilot.press("a")
        assert app.state.items[0].status == CardStatus.ACCEPTED

async def test_edit_card_modal():
    app = ReviewApp(sample_state)
    async with app.run_test() as pilot:
        await pilot.press("e")
        assert isinstance(app.screen, EditCardScreen)
```

---

## 実装順序

| # | タスク | 工数 | 状態 |
|---|--------|------|------|
| 1 | `uv add textual` | 5min | ✅ 完了 |
| 2 | `tui/state.py` + `test_tui_state.py` | 1.5h | ✅ 完了 |
| 3 | `tui/widgets.py` + `tui/app.py` + `test_tui_app.py` | 2.5h | ✅ 完了 |
| 4 | `tui/__init__.py` (launch_review) | 30min | ✅ 完了 |
| 5 | `main.py` 統合 (`--review` フラグ) | 30min | ✅ 完了 |
| 6 | 全テスト + ruff + mypy 検証 | 30min | ✅ 完了 |

---

## 検証方法

```bash
# ユニットテスト
uv run pytest tests/test_tui_state.py tests/test_tui_app.py -v

# TUI モジュールカバレッジ (85%+)
uv run pytest tests/test_tui_state.py tests/test_tui_app.py \
  --cov=pdf2anki.tui --cov-report=term-missing

# 全テスト + カバレッジ (80%+)
uv run pytest --cov=src/pdf2anki --cov-report=term-missing

# 型チェック + リンター
uv run mypy src/pdf2anki/tui --strict
uv run ruff check src/pdf2anki/tui

# 手動 E2E
uv run pdf2anki convert tests/fixtures/sample.txt --review --format json -o /tmp/out
```

---

## 重要ファイル参照

| ファイル | 参照理由 |
|---------|---------|
| `src/pdf2anki/schemas.py` | AnkiCard, CardConfidenceScore, ExtractionResult (全て Pydantic frozen) |
| `src/pdf2anki/main.py:258-280` | TUI 挿入ポイント |
| `src/pdf2anki/quality/heuristic.py` | `score_cards()` で品質スコア取得 |
| `src/pdf2anki/service.py` | `process_file()` の戻り値型確認 |
| `src/pdf2anki/convert.py` | `write_output()` の引数確認 |
| `tests/conftest.py` | 既存fixture: sample_qa_card, sample_cloze_card, sample_reversible_card |

---

## 設計判断メモ

### なぜ 4 ファイル（5 ではなく）？
- 既存プランでは `screens.py` を別ファイルとしていたが、EditCardScreen 1つだけなので `app.py` に統合
- `eval/` パッケージと同規模（4ファイル、~270 LOC）

### なぜインライン CSS？
- `.tcss` ファイル管理不要
- Textual 推奨パターン（小規模アプリ向け）
- Python コード内で完結 → メンテナンス容易

### なぜ review サブコマンドを（今は）追加しない？
- `--review` フラグで必要十分
- JSON 再読み込み + レビューは将来のスコープ
- 最小実装で価値を出す
