# Provider Selection Architecture Decision - Architect Briefing

> 作成日: 2026-02-09
> 対象: architect エージェント
> 目的: OpenAI API サポート追加のアーキテクチャ判断

---

## Executive Summary

**背景:** pdf2anki に OpenAI API サポートを追加し、ユーザーが Claude/OpenAI を選択できるようにする。

**現状:** Anthropic SDK のみ使用、3,348行、依存6個、456テスト、96%カバレッジ。

**重要な洞察（本セッションで発見）:**
AI時代の「Micro-Dependencies原則」を適用すると、**条件付き依存（optional-dependencies）** が最適解の可能性。

**あなたの役割:**
以下3つの重要なアーキテクチャ判断を下してください。

---

## 判断事項

### 1. 依存管理戦略 🎯 最重要

#### Option A: 条件付き依存（推奨候補）

```toml
[project]
dependencies = ["pymupdf", "pydantic", "pyyaml", "typer", "rich"]

[project.optional-dependencies]
claude = ["anthropic>=0.40.0"]
openai = ["openai>=1.0.0"]
all = ["anthropic>=0.40.0", "openai>=1.0.0"]
```

**使用例:**
```bash
pip install pdf2anki[claude]   # Claudeのみ
pip install pdf2anki[openai]   # OpenAIのみ
pip install pdf2anki[all]      # 両方（比較用）
```

**メリット:**
- ✅ Micro-Dependencies原則に完全適合（使わないSDKは含まない）
- ✅ コードベース1つ（メンテナンス容易）
- ✅ ユーザーが依存を選択可能
- ✅ FastAPI, SQLAlchemy等も採用している実績あり

**デメリット:**
- ❌ 実行時チェック実装が必要（+50行程度）
- ❌ 両方未インストール時のエラーハンドリング

**実装コスト:** Provider抽象化 500行 + optional-dependencies設定 50行 = **550行**

---

#### Option B: フル統合（元の計画）

```toml
[project]
dependencies = [
    "anthropic>=0.40.0",
    "openai>=1.0.0",        # ← 強制
    "pymupdf", "pydantic", ...
]
```

**メリット:**
- ✅ 実装がシンプル（実行時チェック不要）
- ✅ どちらも即座に使える

**デメリット:**
- ❌ **Micro-Dependencies原則に反する**
- ❌ Claude専用ユーザーも OpenAI SDK 必須（倉庫全体を輸入）
- ❌ 依存パッケージ増加（anthropic + openai）

**実装コスト:** Provider抽象化 500行のみ

---

**判断基準:**
- プロジェクトの設計思想（後述）との整合性
- ユーザー体験
- メンテナンスコスト

**あなたの判断:** Option A or B? 理由は？

---

### 2. OpenAI Batch API 対応

#### 背景

- **Anthropic Batch API:** 50%割引、24時間以内処理、既に実装済み（batch.py）
- **OpenAI Batch API:** 同様に50%割引、24時間以内処理

#### Option A: OpenAI Batch API 対応する

**メリット:**
- ✅ OpenAI使用時も50%コスト削減
- ✅ Claude版と機能パリティ

**デメリット:**
- ❌ 実装複雑化（+200-300行）
- ❌ OpenAI Batch API の仕様調査が必要
- ❌ テスト追加

**実装コスト:** +200-300行

---

#### Option B: OpenAI Batch API 対応しない（シンプル実装）

**メリット:**
- ✅ 実装がシンプル
- ✅ asyncio + Semaphore で並列処理（独自実装）
- ✅ リアルタイム処理（24時間待ちなし）

**デメリット:**
- ❌ OpenAI使用時はコスト削減なし（通常価格）
- ❌ Claude版と機能差

**実装コスト:** +100行程度（asyncio並列処理）

---

**判断基準:**
- $10クレジット活用の主目的（コスト削減 vs 試用）
- 実装複雑性 vs 機能パリティ

**あなたの判断:** 対応する/しない? 理由は？

---

### 3. Provider 抽象化の設計

#### Option A: Protocol（軽量）

```python
from typing import Protocol

class LLMProvider(Protocol):
    def generate_cards(
        self, prompt: str, model: str, max_tokens: int
    ) -> tuple[list[AnkiCard], CostRecord]: ...
```

**メリット:**
- ✅ 軽量（ダックタイピング）
- ✅ Python 3.8+ 標準
- ✅ このプロジェクトの frozen dataclass パターンと親和性高い

**デメリット:**
- ❌ 実行時までエラー検出できない

---

#### Option B: Abstract Base Class（厳密）

```python
from abc import ABC, abstractmethod

class LLMProvider(ABC):
    @abstractmethod
    def generate_cards(...) -> ...: ...
```

**メリット:**
- ✅ 継承チェックが厳密
- ✅ IDE補完が強力

**デメリット:**
- ❌ やや重い
- ❌ ダックタイピングの柔軟性を失う

---

**判断基準:**
- プロジェクトの既存パターン（frozen dataclass, 型ヒント重視）
- Pythonic さ vs 厳密性

**あなたの判断:** Protocol or ABC? 理由は？

---

## プロジェクトの設計思想（重要）

### Micro-Dependencies原則
- 大規模フレームワーク不使用（LangChain等）→ 直接SDK利用
- 依存6個のみ: anthropic, pymupdf, pydantic, pyyaml, typer, rich
- **理由:** Claude Codeなら必要機能を数時間でカスタム実装可能
- **結果:** 3,348行で完全制御、透明性最大、メンテナンス性向上

### Perfect Fit原則
- 汎用抽象化 → ドメイン特化設計（PDF→Anki変換のみ）
- プロジェクト固有要求に完全適合:
  - frozen dataclass → 不変性要求
  - セクション単位処理 → 長文最適化
  - Batch API直接制御 → 50%割引活用
  - プロンプトキャッシュ直接操作 → コスト最小化

### Full Control原則
- API呼び出し・コスト追跡・エラーハンドリングを完全制御
- 456テスト、96%カバレッジ → 全コードパス理解
- ブラックボックス依存なし → デバッグ容易

### AI時代の格言
> 従来: "Don't Reinvent the Wheel"
> AI時代: **"Don't Import the Warehouse for a Single Wheel"**
> Claude Codeがあれば、必要な機能だけを短時間で実装可能

**この思想を維持しつつ、Provider選択機能を追加する最適なアーキテクチャは？**

---

## 既に収集した情報

### OpenAI Structured Outputs
- Pydantic統合: `client.responses.parse(text_format=MyModel)` で自動変換
- JSON Schema対応: Pydanticモデルを自動変換
- エラーハンドリング: ValidationError時の挙動は調査中

### OpenAI Batch API
- 50%割引、24時間以内処理
- JSONL形式、最大50,000リクエスト、200MB
- 対応エンドポイント: `/v1/chat/completions` 等

### 既存コードの影響範囲
- API モック使用箇所: 209箇所（5ファイル）
  - `tests/test_structure.py`: 61箇所
  - `tests/test_quality.py`: 22箇所
  - `tests/test_main.py`: 55箇所
  - `tests/test_batch.py`: 31箇所
  - `tests/test_e2e.py`: 40箇所

### LiteLLM 設計パターン
- Unified Provider Architecture
- enum-based static dispatch + trait-based polymorphism
- OpenAI-compatible interface で全プロバイダーを抽象化

### 現在のコード構造
- `structure.py`: 437行 - メインAPI呼び出し（3箇所）
- `quality.py`: 692行 - 品質評価API（2箇所）
- `batch.py`: 277行 - バッチAPI（4箇所）
- `cost.py`: 140行 - コスト追跡・モデルルーティング
- `config.py`: 137行 - 設定管理（Pydantic BaseModel）

---

## 期待するアウトプット

### 1. 3つの判断事項に対する明確な回答

```markdown
## アーキテクチャ判断

### 1. 依存管理戦略
**判断:** Option A (条件付き依存) / Option B (フル統合)
**理由:** ...

### 2. OpenAI Batch API 対応
**判断:** 対応する / 対応しない
**理由:** ...

### 3. Provider 抽象化の設計
**判断:** Protocol / ABC
**理由:** ...
```

### 2. 推奨アーキテクチャ概要

```
src/pdf2anki/
  ├── providers/
  │   ├── __init__.py
  │   ├── base.py          # Protocol or ABC
  │   ├── anthropic.py     # AnthropicProvider
  │   └── openai.py        # OpenAIProvider
  ├── config.py            # + provider フィールド
  ├── cost.py              # + OpenAI 価格テーブル
  └── ...
```

### 3. 実装フェーズ計画

どの順序で実装すべきか（Phase 1, 2, 3...）

### 4. リスク・トレードオフ分析

選択したアーキテクチャのリスクと軽減策

---

## 制約条件

- **後方互換性維持必須**: 既存の Claude専用ユーザーに影響を与えない
- **テストカバレッジ 80%以上維持**
- **frozen dataclass パターン維持**: 不変性重視
- **実装時間目安**: 4-6時間（Claude Codeで）

---

## 参考リソース

### 既存ドキュメント
- `docs/plans/provider-selection-handoff.md` - 元の引き継ぎドキュメント
- `docs/plans/long-document-processing-plan.md` - Phase 1-4 実装計画
- `MEMORY.md` - プロジェクトメモリ

### 既存実装（参考）
- `src/pdf2anki/batch.py` - Anthropic Batch API 実装（277行）
- `src/pdf2anki/cost.py` - モデルルーティング + コスト追跡（140行）
- `src/pdf2anki/config.py` - Pydantic設定管理（137行）

---

## 次のステップ（architectが判断後）

1. あなたの判断を `docs/plans/provider-selection-architecture-decision.md` に保存
2. 必要なリサーチ項目を絞り込み（判断に基づく）
3. 詳細な実装計画を作成（Phase 1, 2, 3...）
4. 実装開始

---

**あなたの専門的判断を期待しています。プロジェクトの設計思想を尊重しつつ、最適なアーキテクチャを提案してください。**
