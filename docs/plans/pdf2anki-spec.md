# pdf2anki ─ 設計書 & Claude Code実装指示

## このドキュメントについて

Claude Codeに渡してプロジェクトを立ち上げるための設計書です。
「この設計書に従って実装して」と指示してください。

---

## 1. プロジェクト概要

PDF・テキスト・Markdownファイルからテキストを抽出し、Claude APIのStructured Outputsを使って学習カード（Q&A、用語定義、要約ポイント等）を構造化抽出し、Anki TSVおよびJSON形式で出力するCLIツール。

### ゴール
- PDF/テキスト/Markdown → Ankiインポート用TSV + JSON を**完全自動**で生成
- 日本語メイン＋英語混在に対応
- 最小コスト運用（Claude Haiku 4.5 + Batch API）

---

## 2. プロジェクト構成

```
pdf2anki/
├── README.md
├── pyproject.toml          # パッケージ管理（uv or pip）
├── .env.example            # ANTHROPIC_API_KEY=your-key-here
├── config.yaml             # デフォルト設定
├── src/
│   └── pdf2anki/
│       ├── __init__.py
│       ├── main.py         # CLIエントリポイント（argparse）
│       ├── extract.py      # Step 1: テキスト抽出
│       ├── structure.py    # Step 2: LLM構造化（Claude API）
│       ├── convert.py      # Step 3: 出力変換（JSON → TSV / .apkg）
│       ├── schemas.py      # Pydanticスキーマ定義
│       ├── prompts.py      # システムプロンプト定義
│       └── config.py       # 設定読み込み
├── tests/
│   ├── test_extract.py
│   ├── test_structure.py
│   ├── test_convert.py
│   └── fixtures/           # テスト用サンプルファイル
│       ├── sample.pdf
│       ├── sample_scanned.pdf
│       ├── sample.txt
│       └── sample.md
└── output/                 # デフォルト出力先（.gitignore対象）
```

---

## 3. 依存パッケージ

```toml
[project]
name = "pdf2anki"
version = "0.1.0"
requires-python = ">=3.10"
dependencies = [
    "anthropic>=0.40.0",       # Claude API SDK（Structured Outputs対応）
    "pymupdf>=1.25.0",         # PDF テキスト抽出
    "pymupdf4llm>=0.0.10",     # PDF → Markdown変換
    "pydantic>=2.0",           # スキーマ定義・バリデーション
    "pyyaml>=6.0",             # 設定ファイル
]

[project.optional-dependencies]
ocr = [
    "ocrmypdf>=16.0.0",       # OCR処理（スキャンPDF用）
]

[project.scripts]
pdf2anki = "pdf2anki.main:main"
```

### システム依存（OCR使用時のみ）
```bash
# macOS
brew install tesseract tesseract-lang

# Ubuntu
sudo apt install tesseract-ocr tesseract-ocr-jpn
```

---

## 4. スキーマ定義 ─ schemas.py

```python
"""Pydanticスキーマ定義 ─ Claude Structured Outputsで使用"""

from pydantic import BaseModel, Field, ConfigDict
from typing import List, Literal, Optional


class AnkiCard(BaseModel):
    """Ankiカード1枚分"""
    model_config = ConfigDict(extra="forbid")

    front: str = Field(
        description="カード表面。Q&Aなら質問文、用語定義なら用語名、"
                    "要約ポイントならポイントを問う質問形式にする"
    )
    back: str = Field(
        description="カード裏面。回答、定義、説明など。"
                    "簡潔だが十分な情報を含める"
    )
    card_type: Literal["qa", "term_definition", "summary_point", "cloze"] = Field(
        description="カード種別: "
                    "qa=問題と回答, "
                    "term_definition=用語と定義, "
                    "summary_point=要約ポイント, "
                    "cloze=穴埋め（{{c1::穴埋め部分}}形式）"
    )
    tags: List[str] = Field(
        default_factory=list,
        description="Ankiタグ。階層はコロン区切り（例: 'AI::機械学習::教師あり学習'）。"
                    "コンテンツの主題・分野から自動判定する"
    )
    difficulty: Optional[Literal["easy", "medium", "hard"]] = Field(
        default=None,
        description="推定難易度"
    )


class ExtractionResult(BaseModel):
    """1ファイルの抽出結果"""
    model_config = ConfigDict(extra="forbid")

    cards: List[AnkiCard] = Field(
        description="抽出されたカードのリスト"
    )
    source_summary: Optional[str] = Field(
        default=None,
        description="元テキスト全体の1-2文の要約"
    )
    detected_language: Literal["ja", "en", "mixed"] = Field(
        description="検出されたコンテンツの主要言語"
    )
    total_cards: int = Field(
        description="生成されたカードの総数"
    )
```

---

## 5. プロンプト設計 ─ prompts.py

```python
"""Claude APIに渡すプロンプト定義"""

SYSTEM_PROMPT = """あなたは教育コンテンツの構造化抽出の専門家です。
与えられたテキストから、効果的な学習カード（フラッシュカード）を生成してください。

## 抽出ルール

### カード種別の判定基準
- **qa**: 明示的な問題と回答がある場合、または重要な事実・概念から問題を生成
- **term_definition**: 専門用語、略語、概念の定義がある場合
- **summary_point**: 重要なポイント、結論、比較がある場合。問いかけ形式のfrontにする
- **cloze**: キーワードを穴埋めにすると効果的な文（公式、定義文など）。{{c1::穴埋め部分}}形式

### 品質基準
1. **1カード1概念**: 複合的な内容は分割する
2. **front（表面）は必ず質問形式または穴埋め形式**: 「〜とは何か？」「〜の特徴は？」「〜の違いは？」
3. **back（裏面）は簡潔かつ正確**: 冗長な説明は避け、要点を押さえる
4. **元テキストの専門用語はそのまま保持**: 英語の専門用語は無理に翻訳しない
5. **タグは階層構造**: 大分類::中分類::小分類（例: 'AI::深層学習::CNN'）

### 言語ルール
- 元テキストが日本語なら日本語でカードを作成
- 英語なら英語で作成
- 混在している場合は元テキストの言語をそのまま維持
- 英語の専門用語は日本語文中でもそのまま使用

### 避けるべきこと
- 曖昧な質問（「〜について述べよ」のような広すぎる問い）
- 元テキストに含まれない情報の追加（ハルシネーション）
- 自明すぎるカード（「PDFとはPDFのことである」のような無意味なもの）
- 同じ内容の重複カード
"""


def build_user_prompt(text: str, options: dict = None) -> str:
    """ユーザープロンプトを構築"""
    prompt = f"""以下のテキストから学習カードを抽出してください。

【テキスト】
{text}
"""
    if options:
        if options.get("focus_topics"):
            topics = ", ".join(options["focus_topics"])
            prompt += f"\n【重点トピック】\n{topics}\n"
        if options.get("max_cards"):
            prompt += f"\n【カード数上限】\n{options['max_cards']}枚程度\n"
        if options.get("card_types"):
            types = ", ".join(options["card_types"])
            prompt += f"\n【生成するカード種別】\n{types}\n"
        if options.get("custom_tags"):
            tags = ", ".join(options["custom_tags"])
            prompt += f"\n【追加タグ】\n以下のタグをすべてのカードに付与: {tags}\n"

    return prompt
```

---

## 6. 各モジュールの仕様

### 6.1 extract.py ─ テキスト抽出

```python
"""Step 1: ファイルからテキストを抽出"""

# 入力: ファイルパス (str | Path)
# 出力: ExtractedText (dataclass)
#   - text: str          抽出テキスト
#   - format: str        "markdown" | "plain"
#   - source_type: str   "pdf" | "pdf_ocr" | "text" | "markdown"
#   - page_count: int    ページ数（PDF）/ None
#   - char_count: int    文字数

# 処理フロー:
# 1. ファイル拡張子で分岐
#    - .pdf → pymupdf4llm.to_markdown() でMarkdown抽出
#      - テキストが空 or 極端に少ない → OCRモードにフォールバック
#      - OCRモード: PyMuPDFの get_textpage_ocr() または ocrmypdf を使用
#    - .txt → そのまま読み込み
#    - .md  → そのまま読み込み
#
# 2. 前処理
#    - 連続空行を2行に正規化
#    - 制御文字の除去
#    - BOM除去
#
# 3. テキスト長チェック
#    - Claude Haiku 4.5のコンテキスト: 200K tokens
#    - 安全マージンを考慮し、入力テキストは150K tokens以下に
#    - 超過する場合は警告を出してページ単位で分割処理

# OCRフォールバックの判定基準:
#   - 抽出テキストが空
#   - ページ数に対してテキスト量が異常に少ない（1ページ100文字未満）
#   - 抽出テキストの大部分が文字化け（chr(0xFFFD)が多い）
```

### 6.2 structure.py ─ LLM構造化

```python
"""Step 2: Claude APIで構造化抽出"""

# 入力: ExtractedText, options (dict)
# 出力: ExtractionResult (Pydanticモデル)

# 処理フロー:
# 1. anthropic.Anthropic クライアント初期化
# 2. Structured Outputs を使用してAPIコール
# 3. ExtractionResult として返却

# API呼び出しの実装:
#
# from anthropic import Anthropic
# from schemas import ExtractionResult
#
# client = Anthropic()  # ANTHROPIC_API_KEY環境変数から自動取得
#
# response = client.messages.create(
#     model=config.model,  # デフォルト: "claude-haiku-4-5-20241022"
#     max_tokens=4096,
#     system=SYSTEM_PROMPT,
#     messages=[
#         {"role": "user", "content": build_user_prompt(text, options)}
#     ],
#     output_config={
#         "format": {
#             "type": "json_schema",
#             "schema": transform_schema(ExtractionResult),
#         }
#     }
# )
#
# result = ExtractionResult.model_validate_json(response.content[0].text)

# モデル選択ガイド（config.yamlで設定）:
#   - claude-haiku-4-5-20241022  : デフォルト。最安。月数十ファイルならこれで十分
#   - claude-sonnet-4-5-20250929 : 複雑な文書、長文、高精度が必要な場合
#
# 注意: Structured Outputsの対応モデルを確認すること
#       2025年11月時点ではSonnet 4.5とOpus 4.1で利用可能
#       Haiku 4.5は対応予定とされている → 実装時にドキュメント確認
#       未対応の場合はtool_useベースのworkaroundを検討
#
# Batch API対応（オプション）:
#   大量処理時は client.beta.messages.batches.create() を使用
#   50%割引が適用される
```

### 6.3 convert.py ─ 出力変換

```python
"""Step 3: ExtractionResult → Anki TSV / JSON"""

# 入力: ExtractionResult, output_format ("tsv" | "json" | "both")
# 出力: ファイル書き出し

# --- TSV出力 ---
# Ankiインポート形式: タブ区切り、1行1カード
#
# フォーマット:
#   front<TAB>back<TAB>tags
#
# ルール:
#   - フィールド内のタブ → スペースに置換
#   - フィールド内の改行 → <br> に置換（Anki HTML対応）
#   - tags は半角スペース区切り（Anki仕様）
#   - tagsの階層はコロン2つ（::）で区切り
#   - cloze型はfrontのみ出力（backは空でOK、Ankiが自動処理）
#   - ファイル先頭に #separator:tab を付与
#   - エンコーディング: UTF-8 (BOMなし)
#
# 出力例:
#   #separator:tab
#   #html:true
#   #tags column:3
#   機械学習における過学習とは何か？<TAB>訓練データに過度に適合し...<br>汎化性能が低下する現象。<TAB>AI::機械学習 過学習
#   {{c1::勾配降下法}}は損失関数を最小化するアルゴリズムである<TAB><TAB>AI::最適化 cloze

# --- JSON出力 ---
# ExtractionResult.model_dump_json(indent=2) でそのまま書き出し
# 他アプリ連携用のためスキーマ情報もメタデータとして含める
#
# 出力例:
# {
#   "cards": [...],
#   "source_summary": "...",
#   "detected_language": "ja",
#   "total_cards": 15,
#   "_meta": {
#     "schema_version": "1.0",
#     "source_file": "input.pdf",
#     "generated_at": "2026-02-08T12:00:00Z",
#     "model": "claude-haiku-4-5-20241022"
#   }
# }
```

### 6.4 main.py ─ CLIエントリポイント

```python
"""CLIインターフェース"""

# 使用例:
#
# 基本:
#   pdf2anki input.pdf
#   pdf2anki input.pdf -o output.tsv
#   pdf2anki input.pdf --format both
#
# ディレクトリ一括:
#   pdf2anki ./documents/ -o ./output/ --format both
#
# オプション指定:
#   pdf2anki input.pdf --model claude-sonnet-4-5-20250929
#   pdf2anki input.pdf --ocr --lang jpn+eng
#   pdf2anki input.pdf --max-cards 30
#   pdf2anki input.pdf --tags "AIパスポート::2026年"
#   pdf2anki input.pdf --focus "ディープラーニング,CNN"
#   pdf2anki input.pdf --card-types qa,term_definition
#
# 引数一覧:
#   input              入力ファイルまたはディレクトリ
#   -o, --output       出力先（ファイルまたはディレクトリ）
#   --format           出力形式: tsv, json, both (default: tsv)
#   --model            Claude モデル名 (default: config.yamlから)
#   --ocr              OCR有効化
#   --lang             OCR言語 (default: jpn+eng)
#   --max-cards        1ファイルあたりの最大カード数
#   --tags             追加タグ（カンマ区切り）
#   --focus            重点トピック（カンマ区切り）
#   --card-types       生成するカード種別（カンマ区切り）
#   --config           設定ファイルパス (default: ./config.yaml)
#   --dry-run          API呼び出しなし。抽出テキストのみ出力
#   --verbose          詳細ログ出力
```

### 6.5 config.yaml

```yaml
# pdf2anki デフォルト設定

# Claude API設定
api:
  model: "claude-haiku-4-5-20241022"  # Structured Outputs対応確認後に更新
  max_tokens: 4096
  temperature: 0.0     # 構造化抽出は決定論的に

# テキスト抽出設定
extract:
  ocr_enabled: false
  ocr_language: "jpn+eng"
  # OCRフォールバック: テキスト量がこの閾値以下ならOCR試行
  ocr_fallback_threshold_chars_per_page: 100

# 出力設定
output:
  format: "tsv"        # tsv, json, both
  directory: "./output"
  # TSV設定
  tsv:
    separator: "tab"
    html_enabled: true  # 改行を<br>に変換
  # JSON設定
  json:
    indent: 2
    include_meta: true

# カード生成設定
cards:
  max_per_file: null    # null = 制限なし
  card_types:           # 生成するカード種別
    - qa
    - term_definition
    - summary_point
    - cloze
  default_tags: []      # すべてのカードに付与するタグ
```

---

## 7. 実装上の注意事項

### Structured Outputs の対応状況確認

2025年11月時点の情報では、Structured Outputs は Claude Sonnet 4.5 と Opus 4.1 で利用可能。Haiku 4.5 は「対応予定」とされている。実装時に以下を確認:

1. `https://docs.anthropic.com/en/docs/build-with-claude/structured-outputs` で最新の対応モデル一覧を確認
2. Haiku 4.5 が未対応の場合の代替案:
   - **案A**: Sonnet 4.5 を使用（コストは3倍だが月$0.78程度で許容範囲）
   - **案B**: tool_use の strict mode を使用（Structured Outputs と同等のスキーマ強制が可能）
   - **案C**: プロンプトでJSON出力を指示 + Pydanticでバリデーション + リトライ

### transform_schema の使用

```python
from anthropic import Anthropic, transform_schema
from schemas import ExtractionResult

# transform_schema() はPydanticの制約（min, max等）をdescriptionに変換し、
# Claude互換のJSON Schemaを生成する
schema = transform_schema(ExtractionResult)
```

### エラーハンドリング

```python
# API呼び出しのリトライ戦略
# - RateLimitError → exponential backoff（2, 4, 8秒）最大3回
# - APIError → 1回リトライ後、失敗したファイルをスキップしてログ記録
# - ValidationError → パース失敗。ログに生レスポンスを記録

# ファイル処理のエラー
# - PDFパースエラー → スキップしてログ
# - 空テキスト → 警告表示して次のファイルへ
# - ディレクトリ処理時は全ファイルの結果サマリーを最後に表示
```

### テスト

```python
# tests/test_extract.py
# - テキストPDFからのMarkdown抽出テスト
# - プレーンテキスト読み込みテスト
# - Markdown読み込みテスト
# - 空ファイルのハンドリングテスト

# tests/test_structure.py
# - ExtractionResultスキーマのバリデーションテスト
# - プロンプト構築テスト（オプション有無）
# - API呼び出しのモックテスト

# tests/test_convert.py
# - TSV出力の正確性テスト（タブ、改行、タグ）
# - JSON出力のスキーマ準拠テスト
# - cloze型カードの特殊処理テスト
# - 日本語テキストのエンコーディングテスト
```

---

## 8. Claude Code への指示プロンプト

以下をClaude Codeに渡してください:

---

```
このプロジェクトの設計書（pdf2anki-spec.md）に従って、pdf2ankiツールを実装してください。

実装順序:
1. pyproject.toml を作成し、依存パッケージをインストール
2. schemas.py を設計書のスキーマ定義通りに実装
3. prompts.py を設計書のプロンプト定義通りに実装
4. extract.py を実装（pymupdf4llmでのPDF抽出、テキスト/MD読み込み）
5. structure.py を実装（Claude API Structured Outputs呼び出し）
   - まずStructured Outputsの最新の対応モデルをドキュメントで確認
   - 対応状況に応じて実装方式を選択
6. convert.py を実装（JSON→TSV変換、Ankiインポート形式）
7. config.py を実装（YAML設定読み込み）
8. main.py を実装（CLIエントリポイント）
9. テストを作成・実行
10. サンプルPDFで動作確認

注意:
- Structured Outputsの対応モデルは必ず最新ドキュメントで確認してから実装すること
- Haiku 4.5 が未対応の場合は設計書の代替案を検討
- すべてのファイルにdocstringとtype hintsを付与
- エラーハンドリングは設計書の方針に従う
```

---

## 9. 将来の拡張案（優先度低・メモ）

- **Anki Connect連携**: AnkiのREST APIに直接カードを追加（.apkg不要）
- **Batch API対応**: 大量ファイル処理時の50%コスト削減
- **Prompt Caching**: システムプロンプトのキャッシュで入力トークン90%削減
- **G-testアプリとの統合**: 既存のAIパスポート学習アプリのデータソースとして活用
- **Webフロントエンド**: Streamlit等で簡易GUIを追加
- **カード品質評価**: 生成されたカードを別のLLMコールで評価・フィルタリング
