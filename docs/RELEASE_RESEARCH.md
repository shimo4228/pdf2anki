# pdf2anki 公開・告知に関する調査レポート

調査日: 2026-02-08

## 1. 規約チェック結果

### 1.1 Anthropic API 利用規約

**判定: 問題なし**

- API を使ったアプリケーションの構築・配布は許可されている
- pdf2anki は教育目的の正当なユースケース
- 禁止事項（有害コンテンツ生成、武器製造、詐欺等）に該当しない
- オープンソースプロジェクトへの明示的な制限なし
- 商用利用も Universal Usage Standards に準拠していれば許可

**注意点:**
- コンシューマー向け AI チャットボットを提供する場合は「AI であることの開示」が必要だが、pdf2anki は CLI ツールであり該当しない
- API 出力を AI モデルの学習に使う場合は Anthropic の事前承認が必要（pdf2anki では該当しない）

参照: https://www.anthropic.com/legal/aup

### 1.2 Anki / AnkiWeb 利用規約

**判定: 問題なし**

- pdf2anki は Anki アドオンではなく、TSV/JSON を生成する外部 CLI ツール
- Anki 本体の改変やリバースエンジニアリングは行っていない
- AnkiWeb の API を利用していない
- AnkiWeb の規約適用範囲外

参照: https://ankiweb.net/account/terms

### 1.3 コミュニティルール

**判定: 問題なし**

Anki Forums および Reddit r/Anki では類似ツールの告知投稿が多数存在し、コミュニティに歓迎されている。

直近の類似告知事例:
- "Fully automated Deck creating using Ai-Anki generator" (2025-01) - https://forums.ankiweb.net/t/fully-automated-deck-creatring-using-ai-anki-generator/54160
- "I Made a Tool That Turns Your Voice/lectures into High-Quality Anki Flashcards" (2024-12) - https://forums.ankiweb.net/t/i-made-a-tool-that-turns-your-voice-lectures-into-high-quality-anki-flashcards/53689
- "I Made AnkiAIUtils: Mnemonics Helper" (2024-12) - https://forums.ankiweb.net/t/i-made-ankiaiutils-mnemonics-helper-quick-mnemonic-generation-for-anki/53687
- "Use Generative AI to generate flash cards!" - https://forums.ankiweb.net/t/use-generative-ai-to-generate-flash-cards/45401

### 1.4 PyMuPDF ライセンス（要対応）

**判定: 解決済み（AGPL-3.0 に変更）**

`pymupdf` および `pymupdf4llm` は **AGPL-3.0** でライセンスされている（Artifex Software, Inc.）。

AGPL-3.0 のコピーレフト条項により、AGPL ライブラリを使用するアプリケーション全体も AGPL で公開する義務がある。pdf2anki の現在のライセンスは **MIT** であり、互換性がない。

**対応策（3択）:**

| 対応策 | メリット | デメリット |
|--------|---------|-----------|
| ライセンスを AGPL-3.0 に変更 | 最も簡単。OSS として無料配布するなら実質的デメリット少ない | 商用利用者がソースコード公開義務を負う。MIT より制約が強い |
| Artifex から商用ライセンスを購入 | MIT ライセンスを維持できる | コストが発生。個人プロジェクトには過剰 |
| pymupdf を別ライブラリに置換 | MIT ライセンスを維持。依存関係がクリーンになる | 実装の変更が必要 |

**pymupdf 代替候補（MIT ライセンス）:**

| ライブラリ | ライセンス | 特徴 |
|-----------|----------|------|
| pdfplumber | MIT | テーブル抽出に強い。テキスト抽出も可能 |
| pdfminer.six | MIT | テキスト抽出特化。レイアウト解析に強い |
| pypdf | BSD-3 | 軽量。基本的なテキスト抽出 |

参照:
- https://github.com/pymupdf/pymupdf4llm (AGPL-3.0)
- https://artifex.com/licensing

## 2. 告知チャネルと手順

### 2.1 Anki Forums（最も効果的）

**URL**: https://forums.ankiweb.net/

**投稿先カテゴリ**: "Learning Effectively" または "Development"

**手順:**
1. https://forums.ankiweb.net/ でアカウント作成
2. 適切なカテゴリで新規トピック作成
3. 以下の構成で投稿

**投稿テンプレート:**

```
タイトル: I made pdf2anki: Generate Anki cards from PDF/TXT/MD using Claude AI

本文:

Hi everyone,

I built a CLI tool called pdf2anki that automatically generates
high-quality Anki flashcards from PDF, text, and Markdown files
using Claude AI.

## What it does
- Extracts text from PDF/TXT/MD files
- Generates flashcards using Claude AI with structured outputs
- Quality assurance pipeline with 6-dimension confidence scoring
- Cards follow Wozniak's 20 Rules of Knowledge Formulation
- Every card tagged with Bloom's Taxonomy level
- Outputs Anki-importable TSV or JSON

## Card types
QA, term definition, summary, cloze, reversible, sequence,
compare & contrast

## Quick start
git clone https://github.com/shimo4228/pdf2anki.git
cd pdf2anki
uv sync --all-extras
pdf2anki convert input.pdf

## Features
- Quality pipeline: off / basic / full
- Batch processing (entire directories)
- OCR support for image-heavy PDFs
- Cost tracking with budget limits
- Customizable card types and Bloom level filtering

GitHub: https://github.com/shimo4228/pdf2anki

Feedback and contributions welcome!
```

### 2.2 Reddit r/Anki

**URL**: https://www.reddit.com/r/Anki/

**自己宣伝ルール:**
- 投稿の 80% は通常のコミュニティ参加、20% 以下が宣伝が目安
- 事前に数回コミュニティに参加（質問への回答等）してから投稿するのがベター
- 「宣伝」ではなく「共有」のトーンで書く

**手順:**
1. r/Anki に参加し、数日〜数週間は通常の参加をする
2. 投稿タイトル例: `[Tool] pdf2anki - Generate Anki cards from PDF using Claude AI (open source)`
3. Anki Forums の内容を Reddit 向けにカジュアルに調整
4. デモ出力のスクリーンショットや GIF があると効果的

### 2.3 GitHub リポジトリの最適化

**手順:**

1. **Topics を設定** (Settings > Topics):
   - `anki`
   - `flashcards`
   - `spaced-repetition`
   - `claude`
   - `pdf`
   - `education`
   - `python`
   - `cli`

2. **README にバッジを追加:**
   - Python version
   - License
   - Test status (GitHub Actions)

3. **GitHub Discussions を有効化:**
   - Settings > Features > Discussions にチェック

4. **Release を作成:**
   - `git tag v0.1.0 && git push --tags`
   - GitHub Releases でリリースノートを書く

### 2.4 その他のチャネル

| チャネル | URL | 備考 |
|---------|-----|------|
| Hacker News (Show HN) | https://news.ycombinator.com/submit | テック層にリーチ。品質パイプラインの技術的アプローチが刺さりやすい |
| X (Twitter) | - | `#Anki` `#SpacedRepetition` `#Claude` タグ |
| AnkiHub Community | https://community.ankihub.net/ | Anki 関連ツールの告知実績あり |
| Product Hunt | https://www.producthunt.com/ | OSS ツールのローンチに適している |

## 3. 推奨アクションリスト

優先順に:

- [x] **ライセンス問題を解決する** → AGPL-3.0 に変更済み
- [ ] GitHub リポジトリの Topics を設定する
- [ ] GitHub Release v0.1.0 を作成する
- [ ] CI/CD を設定する（GitHub Actions でテスト自動化）
- [ ] README にバッジを追加する
- [ ] デモ出力のスクリーンショットまたは GIF を作成する
- [ ] Anki Forums に告知投稿を書く
- [ ] Reddit r/Anki にクロスポスト
- [ ] （任意）Hacker News に Show HN 投稿
