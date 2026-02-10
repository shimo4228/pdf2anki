# pdf2anki

[English version](README.md)

PDF/テキスト/Markdown から高品質な Anki フラッシュカードを自動生成する CLI ツール。Claude AI を活用。

## 特徴

- **品質保証パイプライン**: 6 次元の信頼度スコアリング + LLM 批評による自動改善
- **Wozniak の知識定式化 20 原則**: 認知科学に基づくカード生成
- **8 種類のカードタイプ**: QA, 用語定義, 要約, 穴埋め, 可逆, 順序, 比較対照, 画像穴埋め
- **Bloom の分類法**: 全カードに認知レベル (remember 〜 create) を付与
- **画像認識カード生成**: PDF 内の画像を検出・抽出し、Claude Vision API で画像付きカードを生成
- **インタラクティブレビュー TUI**: エクスポート前にターミナル UI でカードを確認・承認・却下・編集
- **セクション分割処理**: 見出しベースの文書分割 + パンくずコンテキストで高品質なカード生成
- **テキスト抽出キャッシュ**: SHA-256 ハッシュで同一ファイルの再抽出をスキップ
- **Batch API**: 非同期バルク処理で 50% コスト削減
- **プロンプト評価フレームワーク**: キーワードベースのマッチングで Recall/Precision/F1 を測定
- **コスト追跡**: セッション単位の API コスト監視、予算上限設定可能
- **OCR 対応**: 画像が多い PDF 向けの OCR フォールバック (ocrmypdf)

## インストール

```bash
git clone https://github.com/shimo4228/pdf2anki.git
cd pdf2anki
uv sync --all-extras
```

### 環境設定

```bash
cp .env.example .env
# .env を編集して ANTHROPIC_API_KEY を設定
```

| 変数 | 必須 | デフォルト | 説明 |
|------|------|-----------|------|
| `ANTHROPIC_API_KEY` | Yes | - | Claude API キー |
| `PDF2ANKI_MODEL` | No | `claude-sonnet-4-5-20250929` | Claude モデルの上書き |
| `PDF2ANKI_BUDGET_LIMIT` | No | `1.00` | API コスト予算上限 (USD) |

## 使い方

### convert (変換)

```bash
# 基本変換（PDF → TSV）
pdf2anki convert input.pdf

# 出力先を指定
pdf2anki convert input.pdf -o output.tsv

# JSON 出力
pdf2anki convert input.pdf --format json

# TSV と JSON 両方
pdf2anki convert input.pdf --format both

# 品質保証パイプライン有効（フル）
pdf2anki convert input.pdf --quality full

# インタラクティブレビュー
pdf2anki convert input.pdf --review

# 画像認識カード生成（Vision API）
pdf2anki convert input.pdf --vision

# 抽出キャッシュを有効化
pdf2anki convert input.pdf --cache

# Batch API で 50% コスト削減
pdf2anki convert input.pdf --batch

# ディレクトリ一括処理
pdf2anki convert ./docs/

# タグとフォーカストピックを追加
pdf2anki convert input.pdf --tags "chapter1,important" --focus "機械学習"

# カード数と予算を制限
pdf2anki convert input.pdf --max-cards 20 --budget-limit 0.50

# OCR 有効化（画像が多い PDF 向け）
pdf2anki convert input.pdf --ocr --lang jpn+eng

# オプションの組み合わせ
pdf2anki convert input.pdf --cache --vision --review --quality full --format both
```

| オプション | デフォルト | 説明 |
|-----------|-----------|------|
| `-o, --output` | 自動 | 出力先ファイルまたはディレクトリ |
| `--format` | `tsv` | 出力形式: `tsv`, `json`, `both` |
| `--quality` | `basic` | 品質レベル: `off`, `basic`, `full` |
| `--model` | 設定値 | Claude モデル名 |
| `--max-cards` | `50` | 最大カード生成数 |
| `--tags` | - | 追加タグ (カンマ区切り) |
| `--focus` | - | フォーカストピック (カンマ区切り) |
| `--card-types` | 全 7 種 | 生成するカードタイプ (カンマ区切り) |
| `--bloom-filter` | 全て | 含める Bloom レベル (カンマ区切り) |
| `--budget-limit` | `1.00` | 予算上限 (USD) |
| `--review` | off | インタラクティブ TUI でカードレビュー |
| `--vision` | off | 画像認識カード生成を有効化 |
| `--cache` / `--no-cache` | off | 抽出キャッシュを有効化 |
| `--batch` | off | Batch API を使用 (50% 割引, 非同期) |
| `--ocr` | off | OCR を有効化 |
| `--lang` | `jpn+eng` | OCR 言語 |
| `--config` | `config.yaml` | 設定 YAML ファイルパス |
| `--verbose` | off | デバッグログ |

### preview (プレビュー)

API を呼ばずにテキスト抽出のみ実行（ドライラン）。

```bash
pdf2anki preview input.pdf
pdf2anki preview input.pdf --ocr
```

### eval (評価)

ラベル付きデータセットに対してプロンプト品質を測定。

```bash
# 評価実行
pdf2anki eval --dataset evals/dataset.yaml

# JSON レポート出力
pdf2anki eval --dataset evals/dataset.yaml --output eval-report.json
```

## 設定

設定の優先順位: **環境変数 > config.yaml > デフォルト値**

モデル、品質閾値、カードタイプ、コスト制限、キャッシュ、画像認識、OCR 設定など全オプションは [`config.yaml`](config.yaml) を参照。

## アーキテクチャ

```
[入力] PDF / TXT / MD
  ↓
[Step 1] テキスト抽出 (pymupdf4llm + OCR フォールバック + キャッシュ)
  ↓
[Step 2] セクション分割 (見出しベース + パンくずコンテキスト)
  ↓
[Step 3] LLM 構造化抽出 (Claude API + 画像は Vision API)
  ↓
[Step 4] 品質保証 (6 次元信頼度スコア → LLM 批評)
  ↓
[Step 5] クロスセクション重複検出
  ↓
[Step 6] インタラクティブレビュー TUI (オプション)
  ↓
[Step 7] 出力 (TSV / JSON)
```

### 品質パイプライン

カードは 6 次元 (front 品質, back 品質, カードタイプ適合, Bloom レベル適合, タグ品質, 原子性) でスコアリング。信頼度閾値を下回るカードは LLM 批評に送られ、改善・分割・削除される。

## プロジェクト構成

```
src/pdf2anki/
  main.py        # CLI (typer): convert, preview, eval コマンド
  config.py      # YAML + 環境変数の設定ローダー
  schemas.py     # Pydantic モデル (AnkiCard, ExtractionResult 等)
  extract.py     # テキスト抽出 (pymupdf4llm + OCR)
  section.py     # 見出しベースのセクション分割
  structure.py   # LLM 構造化カード抽出
  prompts.py     # Wozniak ベースのプロンプトテンプレート
  quality/       # 品質保証パイプライン (ヒューリスティック, 重複検出, 批評)
  convert.py     # TSV/JSON 出力変換
  cost.py        # API コスト追跡
  service.py     # サービス層オーケストレーション
  cache.py       # SHA-256 抽出キャッシュ
  image.py       # PDF 画像検出・抽出
  vision.py      # Claude Vision API 統合
  batch.py       # Batch API 対応
  tui/           # インタラクティブカードレビュー (Textual)
  eval/          # プロンプト評価フレームワーク
tests/           # 624 テスト, カバレッジ 92%+
```

## 必要環境

- Python 3.12+
- [uv](https://docs.astral.sh/uv/)
- Anthropic API キー

## ドキュメント

- [Contributing Guide](docs/CONTRIB.md) - 開発ワークフロー、テスト、CLI リファレンス
- [Runbook](docs/RUNBOOK.md) - デプロイ、トラブルシューティング、よくある問題

## ライセンス

AGPL-3.0 License - 詳細は [LICENSE](LICENSE) を参照。
