# Code Review Fix Plan

**Date**: 2026-02-08
**Reviewer**: Claude Code
**Scope**: pdf2anki codebase full review
**Test Status**: 266 passed, 92% coverage

---

## HIGH Issues (3 件)

### H-1: 広すぎる例外キャッチ (`structure.py`)

**File**: `src/pdf2anki/structure.py:239`
**Severity**: HIGH
**Category**: Error Handling

#### 現状コード

```python
def _call_with_retry(
    *,
    client: anthropic.Anthropic,
    model: str,
    max_tokens: int,
    messages: list[dict[str, str]],
) -> anthropic.types.Message:
    last_error: Exception | None = None

    for attempt in range(_MAX_RETRIES):
        try:
            return _call_claude_api(
                client=client,
                model=model,
                max_tokens=max_tokens,
                messages=messages,
            )
        except Exception as e:  # <- 問題: 全例外をキャッチ
            last_error = e
            logger.warning(
                "API call attempt %d/%d failed: %s",
                attempt + 1,
                _MAX_RETRIES,
                e,
            )
            if attempt < _MAX_RETRIES - 1:
                time.sleep(_RETRY_DELAY_SECONDS * (attempt + 1))

    raise RuntimeError(
        f"API call failed after {_MAX_RETRIES} retries: {last_error}"
    )
```

#### 問題点

- `Exception` を全捕捉するため、プログラミングバグ（`TypeError`, `AttributeError` など）もリトライ対象になる
- バグの原因が隠蔽され、デバッグが困難になる
- リトライすべきはネットワーク・API 関連エラーのみ

#### 修正後コード

```python
# リトライ対象の例外を定義
_RETRYABLE_ERRORS = (
    anthropic.APIConnectionError,
    anthropic.RateLimitError,
    anthropic.InternalServerError,
)


def _call_with_retry(
    *,
    client: anthropic.Anthropic,
    model: str,
    max_tokens: int,
    messages: list[dict[str, str]],
) -> anthropic.types.Message:
    last_error: Exception | None = None

    for attempt in range(_MAX_RETRIES):
        try:
            return _call_claude_api(
                client=client,
                model=model,
                max_tokens=max_tokens,
                messages=messages,
            )
        except _RETRYABLE_ERRORS as e:  # <- リトライ可能なエラーのみ
            last_error = e
            logger.warning(
                "API call attempt %d/%d failed: %s",
                attempt + 1,
                _MAX_RETRIES,
                e,
            )
            if attempt < _MAX_RETRIES - 1:
                time.sleep(_RETRY_DELAY_SECONDS * (attempt + 1))

    raise RuntimeError(
        f"API call failed after {_MAX_RETRIES} retries: {last_error}"
    )
```

#### テスト修正

`tests/test_structure.py` の既存テストも例外の型を合わせる:

```python
# test_api_error_retries (L238-252)
@patch("pdf2anki.structure._call_claude_api")
def test_api_error_retries(self, mock_api: MagicMock) -> None:
    mock_api.side_effect = [
        anthropic.APIConnectionError(request=MagicMock()),  # <- 変更
        _make_mock_response(SAMPLE_CARDS_JSON),
    ]
    # ... 以下同じ

# test_api_error_exhausts_retries (L254-265)
@patch("pdf2anki.structure._call_claude_api")
def test_api_error_exhausts_retries(self, mock_api: MagicMock) -> None:
    mock_api.side_effect = anthropic.APIConnectionError(
        request=MagicMock()
    )  # <- 変更
    # ... 以下同じ
```

#### 影響範囲

- `src/pdf2anki/structure.py`: `_call_with_retry` 関数
- `tests/test_structure.py`: `test_api_error_retries`, `test_api_error_exhausts_retries`

---

### H-2: 広すぎる例外キャッチ (`quality.py`)

**File**: `src/pdf2anki/quality.py:530`
**Severity**: HIGH
**Category**: Error Handling

#### 現状コード

```python
    try:
        client = anthropic.Anthropic()
        response = _call_critique_api(
            client=client,
            model=model,
            cards_json=cards_json,
            source_text=source_text,
        )
    except (anthropic.APIError, Exception) as e:  # <- 問題: Exception が全包含
        logger.error("API error during critique: %s", e)
        return list(cards), cost_tracker
```

#### 問題点

- `Exception` は `anthropic.APIError` のスーパークラスなので、`anthropic.APIError` の指定が冗長
- 実質 bare `except Exception` であり、全エラーをサイレントにフォールバックする
- プログラミングバグが隠蔽される

#### 修正後コード

```python
    try:
        client = anthropic.Anthropic()
        response = _call_critique_api(
            client=client,
            model=model,
            cards_json=cards_json,
            source_text=source_text,
        )
    except anthropic.APIError as e:  # <- API エラーのみキャッチ
        logger.error("API error during critique: %s", e)
        return list(cards), cost_tracker
```

#### 影響範囲

- `src/pdf2anki/quality.py`: `critique_cards` 関数 L530

---

### H-3: `response.content[0].text` の未検証アクセス (`structure.py`)

**File**: `src/pdf2anki/structure.py:182`
**Severity**: HIGH
**Category**: Robustness

#### 現状コード

```python
        response = _call_with_retry(
            client=client,
            model=model,
            max_tokens=config.max_tokens,
            messages=messages,
        )

        response_text = response.content[0].text  # <- ガードなし
        cards = _parse_cards_response(response_text)
```

#### 問題点

- `response.content` が空リストの場合 `IndexError` が発生する
- `content[0]` が `TextBlock` でない場合（`ToolUseBlock` など） `AttributeError` が発生する
- `quality.py:547` には `if not response.content:` のチェックがあるが、`structure.py` にはない

#### 修正後コード

```python
        response = _call_with_retry(
            client=client,
            model=model,
            max_tokens=config.max_tokens,
            messages=messages,
        )

        if not response.content:
            logger.warning("Empty API response for chunk, skipping")
            continue

        first_block = response.content[0]
        if not hasattr(first_block, "text"):
            logger.warning("Unexpected response block type: %s", type(first_block).__name__)
            continue

        response_text = first_block.text
        cards = _parse_cards_response(response_text)
```

#### 追加テスト

```python
@patch("pdf2anki.structure._call_claude_api")
def test_empty_content_response_skipped(self, mock_api: MagicMock) -> None:
    """Empty content should be skipped without error."""
    mock_response = MagicMock()
    mock_response.content = []
    mock_response.model = "claude-sonnet-4-5-20250929"
    mock_response.usage.input_tokens = 100
    mock_response.usage.output_tokens = 0
    mock_api.return_value = mock_response

    config = AppConfig()
    result, _ = extract_cards(
        text="Some text.",
        source_file="test.txt",
        config=config,
    )
    assert result.card_count == 0
```

#### 影響範囲

- `src/pdf2anki/structure.py`: `extract_cards` 関数 L175-184
- `tests/test_structure.py`: 新規テスト追加

---

## MEDIUM Issues (6 件・参考)

| ID | File | Line | Issue | Recommendation |
|----|------|------|-------|----------------|
| M-1 | `structure.py` | 151 | デフォルトモデル名ハードコード比較 | `AppConfig` に `DEFAULT_MODEL` 定数を定義し参照する |
| M-2 | `quality.py` | 280-305 | `_detect_duplicates` の O(n²) | 現行 max_cards=50 では問題なし。スケール時にハッシュベースに変更 |
| M-3 | `quality.py` | 318-366 | 文字集合ベース類似度判定の false positive | n-gram や形態素解析ベースの比較を検討 |
| M-4 | `config.py` | 57 | `cost_warn_at` が未使用 | `extract_cards` ループ内で警告表示を追加、または削除 |
| M-5 | `config.py` | 64-77 | `_flatten_yaml` が1階層のみ | 現行構造では問題なし。再帰化は必要になった時点で |
| M-6 | `test_quality.py` | 複数箇所 | 関数ローカル import の重複 | ファイル先頭にまとめる |

---

## Checklist (実装時)

- [ ] H-1: `structure.py` の `_call_with_retry` を `_RETRYABLE_ERRORS` に限定
- [ ] H-1: `test_structure.py` の例外型を `anthropic.APIConnectionError` に変更
- [ ] H-2: `quality.py` の `except` を `anthropic.APIError` のみに変更
- [ ] H-3: `structure.py` に `response.content` の空チェック・型チェック追加
- [ ] H-3: `test_structure.py` に空レスポンステスト追加
- [ ] 全テスト通過確認 (`pytest tests/ --cov=pdf2anki`)
