Language: [English](README.md) | 日本語

# pdf2anki

pdf2anki は、PDF・テキスト・Markdown のファイルから Anthropic の Claude で Anki のカードを作るコマンドラインツールです。文書を読みながら Anki で学ぶ人に向けています。既定ではエクスポート前に全カードを採点し、`--quality full` を付けると点の低いカードを LLM critique に回して、改善・分割・削除します。

LLM によるカード生成が生む問題（壊れた JSON、ばらつくカードの品質、見えないコスト）への防御を、LLM による構造化抽出、認知科学（SuperMemo の作者 Wozniak によるカード作りの 20 原則）、プロンプト変更をキーワード照合で測る評価を組み合わせて試すために作った実践プロジェクトです。

> [!NOTE]
> `convert` と Web UI は文書のテキスト（`--vision` のときは画像の多いページに埋め込まれた画像も）を、`eval` はラベル付きデータセットのテキストを Anthropic API に送ります。どれも自分の有料の `ANTHROPIC_API_KEY` が必要です。実行ごとに予算上限（既定 1.00 USD）で止まります。`preview` は API を呼びません。

作った経緯と、後日このコードを監査した記録は[著者のほかの仕事](#著者のほかの仕事)にあります。

## インストール

必要なのは Python 3.12+、[uv](https://docs.astral.sh/uv/)、Anthropic API キー（有料）です。OCR（`--ocr`）には、システムに Tesseract と言語パックも必要です（[Runbook](docs/RUNBOOK.md)、英語）。

```bash
git clone https://github.com/shimo4228/pdf2anki.git
cd pdf2anki
uv sync --all-extras
```

`uv sync` は `pdf2anki` コマンドを `.venv` に入れます。`uv run pdf2anki ...` で実行するか、`source .venv/bin/activate` で一度有効にしてください。以下の例は有効にした状態を前提にしています。

### 環境設定

pdf2anki は API キーと下の 2 つの `PDF2ANKI_` 変数を環境変数から読み、`.env` ファイルは読み込みません。実行するシェルでキーを export してください。

```bash
export ANTHROPIC_API_KEY=sk-ant-...
```

変数を `.env` ファイル（`.env.example` からコピー）に置く場合は、先に `set -a; source .env; set +a` でシェルに読み込むか、`uv run --env-file .env pdf2anki ...` で実行してください。

| 変数 | 必須 | デフォルト | 説明 |
|------|------|-----------|------|
| `ANTHROPIC_API_KEY` | Yes | - | Claude API キー |
| `PDF2ANKI_MODEL` | No | `claude-sonnet-4-5-20250929` | Claude モデルの上書き |
| `PDF2ANKI_BUDGET_LIMIT` | No | `1.00` | API コスト予算上限 (USD) |

### 最初の実行

```bash
pdf2anki preview input.pdf     # テキスト抽出のみ。API は呼ばない
pdf2anki convert input.pdf     # PDF と同じ場所に input.tsv を書き出す
```

`convert` は集計表（生成カード数、API 呼び出し数、合計コスト、出力先）を表示します。TSV は 1 行 1 カード（可逆カードは表→裏と裏→表の 2 行）で、表、裏、タグ、ノートタイプの列と Anki のインポート用ヘッダ（`#separator:tab`、`#html:true`、タグとノートタイプの列番号）を持つので、Anki の「ファイル > 読み込む」で取り込めます。起動中の Anki へ `--push` で直接送ることもできます（ほかのプログラムから起動中の Anki にカードを追加できるようにする AnkiConnect アドオンが必要です）。

## 特徴

- **品質保証パイプライン**: 既定のレベル（`--quality basic`）では、各カードを 6 次元（表の品質、裏の品質、カードタイプ適合、Bloom レベル適合、タグ品質、原子性 = 1 枚に 1 つの事実）で採点し、重複しているらしいカードに印を付けます。追加の API 呼び出しはしません。`--quality full` では、重み付きの点が信頼度の閾値（0〜1 の尺度で既定 0.90）を下回るカードを LLM critique（Claude をもう一度呼んでカードを見直させる段）に送り、改善・分割・削除します。`--quality off` はどちらも行いません
- **Wozniak の知識定式化 20 原則**: プロンプトが最小情報の原則を当て、1 枚のカードで 1 つの概念を問うよう求めます
- **8 種類のカードタイプ**: QA, 用語定義, 要約, 穴埋め, 可逆, 順序, 比較対照, 画像穴埋め（図の一部を隠した様子を文章で説明し、そこに何が入るかを問うカード。画像そのものはカードに埋め込まれません。`--vision` のプロンプトはこのカードを求めます）
- **Bloom の分類法**: 全カードに認知レベル (remember 〜 create) を付与し、`--bloom-filter` で選んだレベルだけを残せます
- **セクション分割処理**: 文書を見出しで分け、各セクションを見出しの階層（例: `本 > 部 > 章`）を文脈として付けて Claude に送ります。この階層はカードの階層タグにもなります

## 使い方

### convert

PDF・テキスト・Markdown をカードに変換します。

```bash
# 基本変換（PDF → TSV）
pdf2anki convert input.pdf

# 出力先を指定
pdf2anki convert input.pdf -o output.tsv

# TSV と JSON 両方
pdf2anki convert input.pdf --format both

# ディレクトリ一括処理
pdf2anki convert ./docs/

# タグとフォーカストピックを追加
pdf2anki convert input.pdf --tags "chapter1,important" --focus "機械学習"

# カード数と予算を制限
pdf2anki convert input.pdf --max-cards 20 --budget-limit 0.50

# オプションの組み合わせ
pdf2anki convert input.pdf --cache --vision --review --quality full --format both
```

| オプション | デフォルト | 説明 |
|-----------|-----------|------|
| `-o, --output` | 自動 | 出力先ファイルまたはディレクトリ |
| `--format` | `tsv` | 出力形式: `tsv`, `json`, `both` |
| `--quality` | `basic` | 品質レベル: `off`, `basic`, `full` |
| `--model` | 自動 (Sonnet、短いセクションは Haiku) | Claude モデル名 |
| `--max-cards` | `50` | 最大カード生成数 |
| `--tags` | - | 追加タグ (カンマ区切り) |
| `--focus` | - | フォーカストピック (カンマ区切り) |
| `--card-types` | `qa,term_definition,summary_point,reversible,sequence,compare_contrast` | 生成するカードタイプ (カンマ区切り)。`image_occlusion` を加えると画像穴埋めを求める。`--vision` もこのカードを求める |
| `--bloom-filter` | 全て | 含める Bloom レベル (カンマ区切り) |
| `--budget-limit` | `1.00` | 予算上限 (USD) |
| `--review` | off | エクスポート前にターミナル UI でカードを確認・承認・却下・編集 |
| `--vision` / `--no-vision` | off | 画像の多いページに埋め込まれた画像について、Claude Vision API で文章のカードを生成 (その画像を API に送る。抽出した画像を出力先の横の `_media/` フォルダにも書き出すが、TSV はそれを参照しない) |
| `--cache` / `--no-cache` | off | 内容が変わっていないファイルの再抽出を省く (SHA-256 ハッシュ) |
| `--cache-clear` | off | 処理の前に抽出キャッシュを消す |
| `--batch` | off | Batch API を使用 (50% 割引, 非同期) |
| `--ocr` | off | 画像が多い PDF 向けの OCR フォールバック (ocrmypdf、Tesseract が必要) |
| `--lang` | `jpn+eng` | OCR 言語 |
| `--push` | off | AnkiConnect アドオン経由で起動中の Anki デスクトップへカードを送る (ローカル, `127.0.0.1:8765`) |
| `--deck` | `pdf2anki` | `--push` の送り先デッキ名 |
| `--config` | なし (組み込みの設定) | 設定 YAML のパス。同梱の `config.yaml` など |
| `--verbose` | off | デバッグログ |

### preview

プレビューです。API を呼ばずにテキスト抽出だけを実行します（ドライラン）。

```bash
pdf2anki preview input.pdf
pdf2anki preview input.pdf --ocr
```

### eval

評価です。ラベル付きデータセットに対してプロンプトの品質を測ります。期待カードごとにキーワードを持ち、そのキーワードが十分に現れた生成カードを一致とみなして、Recall・Precision・F1 で集計します。測るのはキーワードの網羅で、カードが学習にどれだけ役立つかではありません。

```bash
# 評価実行
pdf2anki eval --dataset evals/dataset.yaml

# JSON レポート出力
pdf2anki eval --dataset evals/dataset.yaml --output eval-report.json
```

### Web UI

同じ変換を行う Gradio の画面です。`uv sync --all-extras`（`web` extra）で入ります。

```bash
pdf2anki web                   # http://127.0.0.1:7860
```

Web UI には認証がありません。`--host` で別のホストに公開すると、そのネットワークの誰でも Web UI を使えるので、自分の API 予算を他人に使われます。

## 設定

設定の優先順位は **CLI フラグ > 環境変数 > 設定 YAML > デフォルト値** です。設定 YAML は `--config` で渡したときだけ読まれ、渡さなければ組み込みのデフォルト値で動きます。

同梱の [`config.yaml`](config.yaml) には、モデル、品質の閾値、出力、カードタイプ（組み込みの既定に穴埋めを加えています）、コスト上限、OCR、キャッシュの設定があります。`pdf2anki convert input.pdf --config config.yaml` のように渡して使います。

## ドキュメント

- [Contributing Guide](docs/CONTRIB.md)（英語） - 開発ワークフロー、テスト
- [Runbook](docs/RUNBOOK.md)（英語） - デプロイ、トラブルシューティング、よくある問題

## 著者のほかの仕事

- **[LLM の出力は信用するな — Claude API で PDF→Anki 自動生成 CLI を作って学んだ 6 つの防御策](https://zenn.dev/shimo4228/articles/ecc-journey-part2)**（[English](https://dev.to/shimo4228/never-trust-llm-output-6-defenses-from-building-a-pdf-to-anki-cli-43mo)）: pdf2anki がカードを 1 枚ずつ検証し、LLM critique の前に採点し、コストに上限を置き、セクション単位で分割する理由を、作る途中で壊れたことから説明しています。
- **[半年前のAIコードは現世代レビューに耐えるか — 実測25件の仕分け](https://zenn.dev/shimo4228/articles/ai-code-half-year-audit)**（[English](https://dev.to/shimo4228/can-six-month-old-ai-code-survive-todays-review-a-25-bug-triage-4o6d)）: 約半年前に書かれたこのコードを、2026 年 8 月に Claude Fable 5 と GPT-5.6-Sol（Codex CLI 経由）の 2 系統で監査した記録です。設計は持ちこたえ、可逆カードの Anki への送信と大きな画像の抽出は、続く修正まで一度も正しく動いていませんでした。
- **[code-and-llm-collaboration](https://github.com/shimo4228/code-and-llm-collaboration)**（英語）: 決定論的なコードと LLM 呼び出しを 1 本のパイプラインに重ねる 4 つのパターンです。pdf2anki の「採点してから LLM critique」もその 1 つに当たります。
- **[llm-as-judge](https://github.com/shimo4228/llm-as-judge)**（英語）: 二値のチェックを証拠にして 1 つの判定名を出す LLM の判定器の設計です。critique の段をさらに作り込みたいときに。
- **[shimo4228](https://github.com/shimo4228/shimo4228)**: 著者のハブです。5 つの長期プロジェクトとその DOI、Claude Code 用のツールがまとまっています。

## ライセンス

AGPL-3.0 License です。詳細は [LICENSE](LICENSE) をご覧ください。

<details>
<summary>ツールと AI アシスタント向けの資料</summary>

pdf2anki は、PDF・プレーンテキスト・Markdown のファイルを Anthropic の Claude API で Anki のカードに変える、文書から Anki で学ぶ人向けのコマンドラインツールです。LLM によるカード生成は、壊れた JSON、ばらつくカードの品質、見えないコストを生みます。著者はそれぞれへの防御を試すためにこのツールを作りました。カードごとのスキーマ検証（部分的な成功を受け入れる）、Wozniak の知識定式化 20 原則に基づくカードの規則、品質パイプライン、予算の上限、プロンプト変更をキーワード照合で測る評価です。

基本的な事実: ライセンスは AGPL-3.0（`pyproject.toml` では `AGPL-3.0-or-later`）です。Python 3.12+ を uv で管理し、CLI は Typer と Rich、レビュー TUI は Textual、任意の Web UI は Gradio、PDF の抽出は PyMuPDF と pymupdf4llm、任意の OCR は ocrmypdf で動きます。状態: このリポジトリで著者が手で管理している実践プロジェクトで、最新の変更は 2026 年 8 月のコード監査に続く修正です。著者の長期プロジェクトのどれにも属さない単独のツールです。有料の `ANTHROPIC_API_KEY` が必要です。`convert` と Web UI は文書のテキストを、`--vision` では画像の多いページに埋め込まれた画像も、`eval` はラベル付きデータセットのテキストを Anthropic API に送り、`preview` はローカルだけで動きます。既定のモデルは `claude-sonnet-4-5-20250929` で、モデルを明示しない限り、10,000 文字未満でカードの要求数が 30 枚未満のセクションは自動で Claude Haiku に回ります。実行ごとに予算上限（設定しなければ 1.00 USD）で止まります。`--push` は `127.0.0.1:8765` の AnkiConnect を通じてローカルの Anki とだけ通信します。

各カードは、表、裏、8 種類のカードタイプ（qa、term_definition、summary_point、cloze、reversible、sequence、compare_contrast、image_occlusion）の 1 つ、Bloom の分類の認知レベル、タグを持ちます。`convert` は、テキスト抽出（pymupdf4llm、OCR フォールバックと任意のキャッシュ付き）、見出しによるセクション分割、Claude による構造化カード抽出（PDF に `--vision` を付けると、全文と画像を 1 回の Claude Vision リクエストに送る処理がセクション分割による抽出に代わり、Haiku への振り分けも行いません）、品質パイプライン、任意の TUI レビュー、TSV または JSON の出力の順に進みます。既定のレベル（`--quality basic`）では品質パイプラインが 6 次元で採点し、重複しているらしいカードに印を付けます。`--quality full` では 6 次元の重み付きの点が信頼度の閾値（既定 0.90）を下回るカードを LLM critique に送って、改善・分割・削除します。`--quality off` は品質パイプラインを通しません。

例: `pdf2anki convert input.pdf --quality full` は PDF と同じ場所に `input.tsv` を書き出します。1 行 1 カード（可逆カードは表→裏と裏→表の 2 行）で、表、裏、タグ、ノートタイプの列と Anki のインポート用ヘッダを持ち、各カードに `bloom::<level>` タグが付きます。画面には、生成カード数、API 呼び出し数、合計コスト、合格・critique・削除のカード数、出力先が表示されます。

参照先: [`config.yaml`](config.yaml) は設定の見本で、`--config` で渡したときだけ読まれ、画像認識と Batch API の項目はありません。開発とテストは [docs/CONTRIB.md](docs/CONTRIB.md)（英語）、トラブルシューティング（OCR を含む）は [docs/RUNBOOK.md](docs/RUNBOOK.md)（英語）、`pdf2anki eval` が使うラベル付きデータは [evals/dataset.yaml](evals/dataset.yaml) にあります。著者のほかの仕事の一覧は https://github.com/shimo4228/shimo4228 にあります。

</details>
