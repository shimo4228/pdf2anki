# 長文PDF処理の機能追加リサーチ

> 調査日: 2026-02-08
> 対象プロジェクト: pdf2anki (Anki-QA)
> 背景: 572K文字の日本語PDF（正理の海全編）を処理する際、全文を1チャンクでLLMに投入する現状設計のコスト・品質問題を解決するための技術調査

---

## 目次

1. [現状の課題](#1-現状の課題)
2. [構造的チャンキング](#2-構造的チャンキング推奨)
3. [Lost in the Middle 問題](#3-lost-in-the-middle-問題)
4. [コスト最適化](#4-コスト最適化)
5. [既存OSSプロジェクトの知見](#5-既存ossプロジェクトの知見)
6. [推奨実装計画](#6-推奨実装計画)
7. [参考文献](#7-参考文献)

---

## 1. 現状の課題

### 現在のパイプライン

```
PDF → pymupdf4llm.to_markdown() → 前処理 → チャンク分割 → LLMに投入 → カード生成
```

### 問題点

| 問題 | 詳細 |
|------|------|
| **コスト過大** | 572K文字を1チャンクで送信 → 入力だけで$0.60〜$0.90 |
| **Lost in the Middle** | 長文の中間部分でLLMの注意力が低下し、カード品質が不均一 |
| **構造の無視** | `\n\n`での段落分割のみ。章・節の論理構造を活用していない |
| **再処理不可** | 特定セクションだけ再生成する手段がない |
| **並列処理なし** | チャンクを逐次処理。Batch APIも未対応 |

### コスト試算（現状: 正理の海全編）

```
入力: 572,313文字 ≈ 200K〜300Kトークン（日本語はトークン効率が低い）
Sonnet 4.5: 入力$3/MTok, 出力$15/MTok
  入力コスト: ~$0.60〜$0.90
  出力コスト: ~$0.15〜$0.30（50枚カード生成）
  合計: ~$0.75〜$1.20
```

---

## 2. 構造的チャンキング（推奨）

### 2.1 チャンキング手法の比較

| 手法 | 粒度 | 文脈保持 | コスト | 適用場面 |
|------|------|---------|--------|---------|
| 一文単位 + ID（旧来） | 極小 | 低 | 低 | 現在は不要（モデル性能が解決） |
| 固定長分割 | 中 | 低 | 中 | 構造のない文書 |
| 段落境界分割（現状） | 中〜大 | 中 | 高 | 短い文書のみ適切 |
| **Markdownヘッダー分割** | **章・節** | **高** | **低〜中** | **構造化文書に最適** |
| セマンティック分割 | 意味単位 | 高 | 中 | 構造が不明確な文書 |
| ページ単位 | ページ | 中 | 中 | レイアウト依存の文書 |

**推奨: Markdownヘッダー分割**（pymupdf4llmの出力がMarkdownなので最も自然）

### 2.2 pymupdf4llm の既存機能を活用

現在のコード（`extract.py`）:

```python
# 現状: 全体を1つの文字列として取得
result: str = pymupdf4llm.to_markdown(str(path))
```

改善案:

```python
# 改善案1: page_chunks=True でページ単位の構造付きデータを取得
chunks = pymupdf4llm.to_markdown(str(path), page_chunks=True)
# → [
#     {
#         "metadata": {...},
#         "toc_items": [[1, "序論", 3], [2, "論書の著者の偉大さ", 4], ...],
#         "text": "# 序論\n\n真如を探求しなければ...",
#         "tables": [...],
#         "images": [...],
#         "page_boxes": [...]
#     },
#     ...
# ]

# 改善案2: hdr_info で見出し検出を制御
result = pymupdf4llm.to_markdown(
    str(path),
    page_chunks=True,
    hdr_info=None,  # Noneで自動検出（フォントサイズから # レベルを推定）
)
```

#### pymupdf4llm の主要パラメータ

| パラメータ | 型 | 説明 |
|-----------|------|------|
| `page_chunks` | bool | `True`でページ単位の辞書リストを返す |
| `hdr_info` | callable/None | 見出し検出ロジック。`None`でフォントサイズ自動検出 |
| `page_separators` | bool | ページ間に`--- end of page=n ---`マーカーを挿入 |
| `margins` | float/seq | ページ余白の指定（ヘッダー・フッター除外に有用） |

### 2.3 Markdownヘッダーベースのセクション分割

pymupdf4llmが出力するMarkdownの`#`ヘッダーを利用してセクション分割:

```python
import re
from dataclasses import dataclass

@dataclass(frozen=True)
class Section:
    """文書の1セクションを表す。"""
    id: str              # "section-0", "section-1", ...
    heading: str         # "序論", "第1章 論書名の意味", ...
    level: int           # 1=H1, 2=H2, 3=H3
    breadcrumb: str      # "正理の海 > 本論 > 第1章"
    text: str            # セクション本文
    page_range: str      # "pp.3-18"

def split_by_headings(markdown_text: str, max_chars: int = 30_000) -> list[Section]:
    """Markdownヘッダーでセクション分割する。

    - H1/H2/H3 の境界で分割
    - max_chars を超えるセクションはサブ分割
    - パンくずコンテキストを自動生成
    """
    heading_pattern = re.compile(r'^(#{1,3})\s+(.+)$', re.MULTILINE)
    # ... 実装
```

### 2.4 日本語特有のヘッダー検出

日本語文書では標準的なMarkdownヘッダー以外に、以下のパターンも検出が必要:

```python
# 日本語文書で頻出するセクション区切りパターン
JAPANESE_HEADING_PATTERNS = [
    r'^第\s*\d+\s*章',          # 第1章, 第 2 章
    r'^第\s*[一二三四五六七八九十]+\s*章',  # 第一章
    r'^序論|本論|結論',           # 序論, 本論, 結論
    r'^\d+\.\s+',               # 1. セクション
    r'^（\d+）',                 # （1）サブセクション
    r'^[一二三四五六七八九十]、',  # 一、概要
]
```

### 2.5 パンくずコンテキスト（Breadcrumb）

各チャンクをLLMに送る際、文書内の位置情報を付与:

```python
def build_user_prompt_with_context(
    section: Section,
    document_title: str,
    max_cards: int = 20,
) -> str:
    """パンくずコンテキスト付きのユーザープロンプトを構築する。"""
    return (
        f"Generate up to {max_cards} Anki flashcards from the following section.\n\n"
        f"Document: {document_title}\n"
        f"Section: {section.breadcrumb}\n"
        f"Page range: {section.page_range}\n\n"
        f"Add the tag hierarchy based on the section path "
        f"(e.g., '正理の海::本論::第1章').\n\n"
        f"---\n\n{section.text}"
    )
```

これにより:
- LLMが文脈を正確に把握できる
- タグの自動生成精度が向上する
- カードに出典情報を埋め込める

### 2.6 推奨チャンクサイズ

| 用途 | 推奨サイズ | 根拠 |
|------|-----------|------|
| RAG検索 | 400〜512トークン | 検索精度の最適化 |
| **カード生成（本プロジェクト）** | **5K〜30K文字** | LLMの注意力維持 + 十分な文脈 |
| 要約 | 文書全体 or 章単位 | 全体像の把握が必要 |

---

## 3. Lost in the Middle 問題

### 3.1 問題の概要

[Liu et al. 2023](https://arxiv.org/abs/2307.03172) の研究:

- LLMは入力の**冒頭と末尾に注意が偏り**、中間部分で**30%以上の性能低下**
- Rotary Position Embedding (RoPE) の構造的な制約
- Claude, GPT-4 等の最新モデルでも完全には解消されていない

### 3.2 pdf2anki への影響

572K文字を1チャンクで送信した場合:

```
入力テキスト:
[序論]        ← LLMの注意力 高（冒頭バイアス）
[第1章〜第5章] ← LLMの注意力 低（中間部分の劣化）
[最終章]      ← LLMの注意力 高（末尾バイアス）
```

結果: 中間の章からのカード生成品質が低下する。

### 3.3 緩和策

| 手法 | 効果 | 実装コスト |
|------|------|-----------|
| **構造的チャンキング** | **最大** | **中** |
| 検索結果の並べ替え | 中 | 低（RAG向け） |
| ファインチューニング | 中 | 高 |
| アーキテクチャ変更 | 中 | 不可（API利用のため） |

**結論: チャンクを小さくすること自体が最も効果的な対策。**

---

## 4. コスト最適化

### 4.1 Anthropic Message Batches API（50%割引）

最も大きなコスト削減効果。即時レスポンス不要な場合に最適。

#### 価格比較

| モデル | 通常（入力/出力） | Batch（入力/出力） | 割引率 |
|--------|-------------------|-------------------|--------|
| Claude Opus 4.6 | $5/MTok / $25/MTok | $2.50/MTok / $12.50/MTok | 50% |
| Claude Sonnet 4.5 | $3/MTok / $15/MTok | **$1.50/MTok / $7.50/MTok** | 50% |
| Claude Haiku 4.5 | $1/MTok / $5/MTok | **$0.50/MTok / $2.50/MTok** | 50% |

#### 制約事項

- 最大100,000リクエスト or 256MBのバッチサイズ
- 24時間以内に処理（多くは1時間以内に完了）
- 結果は29日間保持
- プロンプトキャッシュとの併用可能（さらにコスト削減）

#### 実装例

```python
import anthropic
from anthropic.types.message_create_params import MessageCreateParamsNonStreaming
from anthropic.types.messages.batch_create_params import Request

client = anthropic.Anthropic()

# セクションごとにバッチリクエストを作成
requests = [
    Request(
        custom_id=f"section-{section.id}",
        params=MessageCreateParamsNonStreaming(
            model="claude-sonnet-4-5-20250929",
            max_tokens=8192,
            system=[
                {
                    "type": "text",
                    "text": SYSTEM_PROMPT,
                    "cache_control": {"type": "ephemeral"},  # キャッシュ有効化
                }
            ],
            messages=[{
                "role": "user",
                "content": build_user_prompt_with_context(section, doc_title),
            }],
        ),
    )
    for section in sections
]

# バッチ送信
batch = client.messages.batches.create(requests=requests)

# ポーリングで完了を待つ
import time
while True:
    batch = client.messages.batches.retrieve(batch.id)
    if batch.processing_status == "ended":
        break
    time.sleep(60)

# 結果を取得
for result in client.messages.batches.results(batch.id):
    if result.result.type == "succeeded":
        cards = parse_cards_response(result.result.message.content[0].text)
        # custom_id からセクションを特定して紐付け
```

### 4.2 AsyncAnthropic（リアルタイム並列処理）

即時結果が必要な場合の並列処理:

```python
import asyncio
import anthropic

async def process_section(
    client: anthropic.AsyncAnthropic,
    section: Section,
    semaphore: asyncio.Semaphore,
) -> list[AnkiCard]:
    """セクションを非同期で処理する。"""
    async with semaphore:  # 同時実行数を制限
        response = await client.messages.create(
            model="claude-sonnet-4-5-20250929",
            max_tokens=8192,
            system=[{"type": "text", "text": SYSTEM_PROMPT,
                     "cache_control": {"type": "ephemeral"}}],
            messages=[{"role": "user", "content": section.text}],
        )
        return parse_cards_response(response.content[0].text)

async def process_all_sections(sections: list[Section]) -> list[AnkiCard]:
    """全セクションを並列処理する。"""
    semaphore = asyncio.Semaphore(5)  # 最大5並列
    async with anthropic.AsyncAnthropic() as client:
        tasks = [process_section(client, s, semaphore) for s in sections]
        results = await asyncio.gather(*tasks)
    return [card for cards in results for card in cards]
```

### 4.3 モデルルーティングの改善

チャンク単位でモデルを選択:

```python
def select_model_for_section(section: Section) -> str:
    """セクションの特性に基づいてモデルを選択する。"""
    text_len = len(section.text)

    # 短い定義・用語セクション → Haiku で十分
    if text_len < 5_000:
        return "claude-haiku-4-5-20251001"

    # 複雑な論理展開・比較セクション → Sonnet
    return "claude-sonnet-4-5-20250929"
```

### 4.4 コスト試算（改善後: 正理の海全編）

```
前提: 572K文字を約20セクション（平均28K文字）に分割

Batch API + Sonnet 4.5:
  入力: 300K tokens × $1.50/MTok = $0.45
  出力:  50K tokens × $7.50/MTok = $0.375
  合計: ~$0.83

Batch API + Haiku 4.5（短いセクション）+ Sonnet 4.5（長いセクション）:
  入力: 300K tokens × ~$0.80/MTok(加重平均) = $0.24
  出力:  50K tokens × ~$4.00/MTok(加重平均) = $0.20
  合計: ~$0.44

キャッシュヒット率50%を考慮:
  合計: ~$0.30〜$0.40
```

**現状 ~$0.90 → 改善後 ~$0.35（約60%削減）**

---

## 5. 既存OSSプロジェクトの知見

### 5.1 類似プロジェクト

| プロジェクト | Stars | 特徴 | 参考URL |
|------------|-------|------|---------|
| **doc2deck** | - | Claude/Gemini/GPT対応、複数入力形式 | https://github.com/raydioactive/doc2deck |
| **AnkiGPT** | 高 | GPT-5使用、310万枚の生成実績 | https://github.com/nilsreichardt/AnkiGPT |
| **AnkiAIUtils** | - | LiteLLM経由で全プロバイダー対応、既存カード改善 | https://github.com/thiswillbeyourgithub/AnkiAIUtils |
| **Anki_FlashCard_Generator** | 高 | PromtEngineer作、PDF→ChatGPTの参考実装 | https://github.com/PromtEngineer/Anki_FlashCard_Generator |
| **PDF2Anki-AI** | - | 要約+カード生成のモード選択 | https://github.com/n-hadi/PDF2Anki-AI |

### 5.2 共通パターンと差別化ポイント

多くのOSSプロジェクトは**全文丸投げ**方式で、構造的チャンキングを実装しているものは少ない。
pdf2anki が以下を実装すれば差別化要因となる:

- 構造的チャンキング + パンくずコンテキスト
- Batch API 対応（50%コスト削減）
- 品質パイプライン（QA scoring + critique）← 既存の強み
- 日本語文書への最適化

---

## 6. 推奨実装計画

### Phase 1: 構造的チャンキング（最優先）

**変更対象:** `extract.py`

1. `pymupdf4llm.to_markdown(page_chunks=True)` に切り替え
2. TOC情報を取得してセクション境界を検出
3. Markdownヘッダー（`#`, `##`, `###`）でセクション分割
4. 日本語ヘッダーパターンのフォールバック検出
5. `Section` データクラスを定義（id, heading, level, breadcrumb, text）
6. 最大チャンクサイズ（30K文字）を超えるセクションのサブ分割

**新規ファイル:** `src/pdf2anki/section.py`（セクション分割ロジック）

### Phase 2: パンくずコンテキスト付きプロンプト

**変更対象:** `prompts.py`, `structure.py`

1. `build_user_prompt()` にセクション情報パラメータを追加
2. パンくずパスをプロンプトに含める
3. セクションパスに基づくタグ自動生成の指示を追加

### Phase 3: Batch API 対応

**変更対象:** `structure.py`, `cost.py`

1. `anthropic.Anthropic().messages.batches` を利用したバッチ送信
2. ポーリングによる完了待ち
3. `custom_id` でセクションとカードの紐付け
4. コストトラッカーのBatch API価格対応
5. CLIに `--batch` フラグを追加

### Phase 4: チャンク間の重複排除

**変更対象:** `quality.py`

1. 全セクションのカードをマージ後に重複検出
2. front/back のテキスト類似度による重複判定
3. 同一概念のカードが複数セクションから生成された場合の統合

### Phase 5: asyncio 対応（オプション）

**変更対象:** `structure.py`

1. `AsyncAnthropic` クライアントの導入
2. セマフォによる同時実行数制御
3. CLIに `--parallel N` フラグを追加

---

## 7. 参考文献

### チャンキング手法

- [Pinecone - Chunking Strategies for LLM Applications](https://www.pinecone.io/learn/chunking-strategies/)
- [Weaviate - Chunking Strategies for RAG](https://weaviate.io/blog/chunking-strategies-for-rag)
- [Firecrawl - Best Chunking Strategies for RAG 2025](https://www.firecrawl.dev/blog/best-chunking-strategies-rag-2025)
- [Redis - LLM Chunking](https://redis.io/blog/llm-chunking/)
- [Machine Learning Mastery - Essential Chunking Techniques](https://machinelearningmastery.com/essential-chunking-techniques-for-building-better-llm-applications/)

### pymupdf4llm

- [PyMuPDF4LLM 公式ドキュメント](https://pymupdf.readthedocs.io/en/latest/pymupdf4llm/)
- [PyMuPDF4LLM API リファレンス](https://pymupdf.readthedocs.io/en/latest/pymupdf4llm/api.html)
- [PyMuPDF4LLM GitHub](https://github.com/pymupdf/pymupdf4llm)
- [PyMuPDF RAG ドキュメント](https://pymupdf.readthedocs.io/en/latest/rag.html)

### Lost in the Middle 問題

- [Liu et al. 2023 - Lost in the Middle: How Language Models Use Long Contexts](https://arxiv.org/abs/2307.03172)
- [Maxim AI - Solving the Lost in the Middle Problem](https://www.getmaxim.ai/articles/solving-the-lost-in-the-middle-problem-advanced-rag-techniques-for-long-context-llms/)
- [What Works for Lost-in-the-Middle in LLMs?](https://arxiv.org/html/2511.13900v1)

### Anthropic API

- [Claude Batch Processing API](https://platform.claude.com/docs/en/build-with-claude/batch-processing)
- [Anthropic SDK Python](https://github.com/anthropics/anthropic-sdk-python)
- [anthropic-parallel-calling](https://github.com/milistu/anthropic-parallel-calling)

### 既存OSSプロジェクト

- [doc2deck](https://github.com/raydioactive/doc2deck)
- [AnkiGPT](https://github.com/nilsreichardt/AnkiGPT)
- [AnkiAIUtils](https://github.com/thiswillbeyourgithub/AnkiAIUtils)
- [Anki_FlashCard_Generator](https://github.com/PromtEngineer/Anki_FlashCard_Generator)
- [PDF2Anki-AI](https://github.com/n-hadi/PDF2Anki-AI)
