# pdf2anki

[English version](README.md)

PDF/テキスト/Markdown から高品質な Anki フラッシュカードを自動生成する CLI ツール。Claude AI を活用。

## 特徴

- **品質保証パイプライン**: 信頼度スコアリング + LLM 批評による自動改善
- **Wozniak の知識定式化 20 原則**: 認知科学に基づくカード生成
- **8 種類のカードタイプ**: QA, 用語定義, 要約, 穴埋め, 可逆, 順序, 比較対照, 画像穴埋め
- **Bloom の分類法**: 全カードに認知レベル (remember 〜 create) を付与

## インストール

```bash
git clone https://github.com/shimo4228/Anki-QA.git
cd Anki-QA
uv sync --all-extras
```

## 使い方

```bash
# 基本変換（PDF → TSV）
pdf2anki convert input.pdf

# 出力先を指定
pdf2anki convert input.pdf -o output.tsv

# 品質保証パイプライン有効
pdf2anki convert input.pdf --quality full

# プレビュー（テキスト抽出のみ）
pdf2anki preview input.pdf
```

## アーキテクチャ

```
[入力] PDF / TXT / MD
  ↓
[Step 1] テキスト抽出 (pymupdf4llm + OCR フォールバック)
  ↓
[Step 2] LLM 構造化抽出 (Claude API Structured Outputs)
  ↓
[Step 3] 品質保証 (信頼度スコア → LLM 批評)
  ↓
[Step 4] 出力 (TSV / JSON)
```

## 必要環境

- Python 3.12+
- [uv](https://docs.astral.sh/uv/)
- Anthropic API キー

## ライセンス

MIT License
