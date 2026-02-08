# ライセンス変更作業プラン: MIT → AGPL-3.0

## 背景

pdf2anki は `pymupdf` および `pymupdf4llm`（AGPL-3.0）に依存している。
現在のライセンスは MIT であり、AGPL のコピーレフト条項と互換性がない。
公開前にライセンスを AGPL-3.0 に統一する必要がある。

## 変更対象ファイル（5箇所）

### 1. LICENSE（新規作成）

- プロジェクトルートに `LICENSE` ファイルを作成
- AGPL-3.0 の全文を記載
- 原文: https://www.gnu.org/licenses/agpl-3.0.txt
- Copyright 表記: `Copyright (C) 2026 shimo4228`

### 2. pyproject.toml

現在 license フィールドがないので追加する。

```toml
[project]
name = "pdf2anki"
version = "0.1.0"
description = "Generate high-quality Anki flashcards from PDF/TXT/MD using Claude AI"
readme = "README.md"
license = "AGPL-3.0-or-later"          # ← 追加
requires-python = ">=3.12"
```

### 3. README.md（L157-159）

```diff
 ## License

-MIT License
+AGPL-3.0 License - See [LICENSE](LICENSE) for details.
```

### 4. README.ja.md（L157-159）

```diff
 ## ライセンス

-MIT License
+AGPL-3.0 License - 詳細は [LICENSE](LICENSE) を参照。
```

### 5. docs/RELEASE_RESEARCH.md

「1.4 PyMuPDF ライセンス（要対応）」のセクションに解決済みステータスを追記:

```diff
-**判定: ライセンス不整合あり**
+**判定: 解決済み（AGPL-3.0 に変更）**
```

アクションリストのチェックボックスも更新:

```diff
-- [ ] **ライセンス問題を解決する**（AGPL-3.0 に変更 or pymupdf を置換）
+- [x] **ライセンス問題を解決する** → AGPL-3.0 に変更済み
```

## Git コミット

```bash
git add LICENSE pyproject.toml README.md README.ja.md docs/RELEASE_RESEARCH.md
git commit -m "chore: change license from MIT to AGPL-3.0 for pymupdf compatibility"
```

## 補足

- 変更は 5 ファイルのみ。コードの変更は不要
- AGPL-3.0 全文は約 660 行（コピペ）
- テストへの影響なし
