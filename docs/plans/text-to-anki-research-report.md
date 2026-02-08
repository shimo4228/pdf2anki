# テキスト情報 → 構造化スキーマ抽出 リサーチレポート

## 目的
PDF/テキストファイルからAnkiインポート用データ（TSV/JSON）を自動生成するための、ベストプラクティスとコスト分析。

---

## 1. 推奨アーキテクチャ

竜也さんの要件（少量・個人用・最小コスト・自動実行）に最適な構成は **2段階パイプライン** です。

```
[入力ファイル] → [Step 1: テキスト抽出] → [Step 2: LLM構造化] → [Anki TSV / JSON]
```

### なぜ2段階か？

LLMに直接PDFを渡す「1段階」方式は一見シンプルですが、2026年時点のベストプラクティスでは **ハイブリッドアプローチ**（テキスト抽出ツール＋LLM）が推奨されています。理由は3つあります。

1. **コスト削減**: PDF全体をLLMに送ると大量のトークンを消費する。事前にテキスト抽出すれば入力トークンを最適化できる
2. **精度向上**: PDF解析ツールがレイアウト・テーブル構造を保持した状態でテキストを渡すため、LLMの構造化精度が上がる
3. **OCR分離**: スキャン画像PDFはOCR処理が必要だが、これをLLMに任せると高コスト。専用ツールで事前処理する方が効率的

---

## 2. Step 1: テキスト抽出 ─ ツール比較

### テキストPDF（抽出可能）

| ツール | 速度 | 品質 | コスト | 日本語対応 | 推奨度 |
|--------|------|------|--------|------------|--------|
| **PyMuPDF** | ◎ 0.05s/page | ◎ | 無料(AGPL) | ◎ UTF-8 | ★★★★★ |
| **pymupdf4llm** | ◎ 0.12s/page | ◎ Markdown出力 | 無料 | ◎ | ★★★★★ |
| pypdf | ○ 0.02s/page | △ | 無料 | ○ | ★★★ |
| pdfplumber | ○ 0.10s/page | ○ テーブル強い | 無料 | ○ | ★★★ |

**推奨: pymupdf4llm**
- PyMuPDFベースで高速、Markdown形式で出力されるため見出し・テーブル構造がLLMに最も理解しやすい
- `pip install pymupdf4llm` のみで導入可能

### スキャンPDF（OCR必要）

| ツール | 日本語OCR | コスト | 導入難度 |
|--------|-----------|--------|----------|
| **Tesseract + PyMuPDF** | ○ (jpn言語パック) | 無料 | 中 |
| **OCRmyPDF** | ○ | 無料 | 低 |
| Google Document AI | ◎ | 従量課金 | 高 |
| Claude Vision（PDF直接入力） | ◎ | API料金 | 低 |

**推奨: Tesseract + PyMuPDF**（コスト優先）
- 完全無料で日本語対応
- `tesseract-ocr` + `tesseract-ocr-jpn` をインストールし、PyMuPDFの `get_textpage_ocr()` を使用

**代替: Claude Vision**（精度優先・少量の場合）
- スキャンPDFをそのままClaude APIに画像として送信可能
- 少量なら月数百円程度で収まるが、大量処理には不向き

### プレーンテキスト / Markdown

追加ツール不要。Pythonの標準ファイル読み込みでそのまま処理可能。

---

## 3. Step 2: LLM構造化 ─ アプローチ比較

### 方式A: Claude API + Structured Outputs（推奨）

Anthropicが提供する **Structured Outputs**機能を使えば、Pydanticでスキーマを定義し、JSON Schemaに準拠した出力をLLMに強制できます。これはconstrained decodingによりスキーマ違反が「物理的に不可能」になる仕組みです。

```python
from pydantic import BaseModel
from typing import List, Literal, Optional

class AnkiCard(BaseModel):
    question: str
    answer: str
    card_type: Literal["qa", "term_definition", "cloze"]
    tags: List[str]
    category: Optional[str] = None
    source_page: Optional[int] = None

class ExtractionResult(BaseModel):
    cards: List[AnkiCard]
    summary: Optional[str] = None
```

この方式の利点:
- **スキーマ準拠が保証される**: JSON解析エラーやリトライが不要
- **Pydanticとの統合**: Pythonオブジェクトとして直接操作可能
- **バリデーション内蔵**: 型チェック、enum制約がSDK側で自動検証

### 方式B: プロンプトエンジニアリングのみ

Structured Outputsを使わず、プロンプトでJSON出力を指示する方法。安価なモデル（Haiku）で使いやすいが、出力のパースエラーリスクがある。

### 方式C: 既存OSSツール（PDF2Anki系）

GitHubに複数のプロジェクトが存在（PDF2Anki-AI、pdf2anki、Anki_FlashCard_Generator等）。いずれもOpenAI GPTベースで、以下の共通課題がある:
- チャンク分割時のコンテキスト喪失
- 出力フォーマットの不安定さ
- 日本語対応が不十分
- カスタマイズ性が低い

**→ 自作スクリプトの方が竜也さんの要件に合致**

---

## 4. コスト分析

### Claude APIモデル別料金（2026年2月時点）

| モデル | 入力/MTok | 出力/MTok | 特徴 |
|--------|-----------|-----------|------|
| **Haiku 4.5** | $1.00 | $5.00 | 最安・高速・構造化抽出に十分 |
| Sonnet 4.5 | $3.00 | $15.00 | バランス型 |
| Opus 4.5 | $5.00 | $25.00 | 最高性能・通常不要 |

### コスト最適化テクニック

| テクニック | 削減率 | 説明 |
|------------|--------|------|
| **Batch API** | 50%割引 | 24時間以内の非同期処理で全トークン半額 |
| **Prompt Caching** | 最大90%削減 | 同じシステムプロンプト・スキーマ定義を再利用 |
| **モデルルーティング** | - | 簡単なQ&AはHaiku、複雑な要約はSonnetに振り分け |

### 実コスト試算

**前提**: 月20ファイル、1ファイル平均5,000文字（≒2,500トークン）、1ファイルから平均20枚のカード生成

| 項目 | トークン数 | 料金 |
|------|-----------|------|
| 入力（テキスト+プロンプト） | 20ファイル × 3,000トークン = 60,000 | |
| 出力（JSON） | 20ファイル × 2,000トークン = 40,000 | |
| **Haiku 4.5（通常）** | | **$0.06 + $0.20 = $0.26/月** |
| **Haiku 4.5（Batch API）** | | **$0.13/月** |
| **Sonnet 4.5（通常）** | | **$0.18 + $0.60 = $0.78/月** |
| **Sonnet 4.5（Batch API）** | | **$0.39/月** |

**結論: 月額$1未満で運用可能。Haiku 4.5 + Batch APIなら月13セント程度。**

新規ユーザーは$5の無料クレジットがもらえるため、約38ヶ月分（3年以上）無料で試せます。

---

## 5. 推奨実装プラン

### 全体構成

```
pdf2anki/
├── extract.py       # Step 1: テキスト抽出（PyMuPDF / pymupdf4llm）
├── structure.py     # Step 2: LLM構造化（Claude API + Pydantic）
├── convert.py       # Step 3: 出力変換（JSON → Anki TSV / .apkg）
├── schemas.py       # Pydanticスキーマ定義
├── config.yaml      # 設定ファイル
└── main.py          # CLIエントリポイント（引数でファイル/ディレクトリ指定）
```

### 使用フロー

```bash
# 単一ファイル
python main.py input.pdf --output anki_cards.tsv --format tsv

# ディレクトリ一括処理
python main.py ./pdfs/ --output ./output/ --format both  # TSV + JSON

# OCR有効化
python main.py scanned.pdf --ocr --lang jpn+eng
```

### 依存パッケージ（すべて無料）

```
pymupdf4llm      # PDF → Markdown テキスト抽出
anthropic         # Claude API SDK
pydantic          # スキーマ定義・バリデーション
tesseract-ocr     # OCR（スキャンPDF用・システムパッケージ）
tesseract-ocr-jpn # 日本語言語パック
```

---

## 6. スキーマ設計案

竜也さんの4種類のコンテンツに対応する統一スキーマ:

```python
from pydantic import BaseModel
from typing import List, Literal, Optional

class AnkiCard(BaseModel):
    """Ankiカード1枚分"""
    front: str                    # 表面（質問 / 用語）
    back: str                     # 裏面（回答 / 定義）
    card_type: Literal[
        "qa",                     # Q&A型
        "term_definition",        # 用語・定義型
        "summary_point",          # 要約ポイント型
        "cloze"                   # 穴埋め型
    ]
    tags: List[str] = []          # Ankiタグ（科目、章など）
    category: Optional[str] = None
    difficulty: Optional[Literal["easy", "medium", "hard"]] = None
    source_file: Optional[str] = None
    source_page: Optional[int] = None

class ExtractionResult(BaseModel):
    """1ファイルの抽出結果"""
    cards: List[AnkiCard]
    metadata: dict = {}
```

### Anki TSV出力例

```
front<TAB>back<TAB>tags
機械学習における過学習とは？<TAB>訓練データに過度に適合し、未知データへの汎化性能が低下する現象。<TAB>AI::基礎 過学習
```

---

## 7. まとめ

| 観点 | 推奨 |
|------|------|
| テキスト抽出 | pymupdf4llm（無料・高速・Markdown出力） |
| OCR | Tesseract + PyMuPDF（無料・日本語対応） |
| 構造化LLM | Claude Haiku 4.5 + Structured Outputs |
| コスト最適化 | Batch API（50%割引）+ Prompt Caching |
| 月額コスト | **$0.13〜$0.78**（ほぼ無料） |
| 初期コスト | $5無料クレジットで数年分カバー |

**コストパフォーマンス評価: ◎ 極めて優秀**

月数十ファイルの個人利用であれば、実質無料に近いコストで高品質な自動変換パイプラインが構築可能です。Claude Codeでの開発に慣れている竜也さんであれば、1〜2日で実用的なツールが完成するはずです。
