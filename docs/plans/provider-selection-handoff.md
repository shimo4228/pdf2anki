# LLM Provider Selection Feature - 新セッション引き継ぎ

> 作成日: 2026-02-09
> 目的: OpenAI API サポート追加（$10クレジット活用 + プロバイダー選択柔軟性）
> ステータス: リサーチフェーズ開始前

---

## 現在のプロジェクト状態

### 完了済み機能
- ✅ Phase 1-4: 長文ドキュメント処理パイプライン完了
  - 構造的チャンキング (section.py)
  - セクション単位カード抽出
  - Batch API 対応 (50%割引)
  - セクション間重複排除
- ✅ 型ヒント修正 (mypy --strict 合格)
- ✅ 全456テスト合格、96.49%カバレッジ
- ✅ ruff + mypy clean
- ✅ Git: main ブランチ最新、origin と同期済み

### 技術スタック
- Python 3.13
- Anthropic SDK (`anthropic` パッケージ)
- Pydantic (データモデル)
- pytest (テスト)
- Rich (CLI UI)

---

## 提案機能: LLM Provider Selection

### 背景・動機
1. **ユーザーニーズ**: OpenAI に $10 のクレジット残高あり、活用したい
2. **柔軟性**: プロバイダーを選択できる方がコスト・モデル比較が容易
3. **拡張性**: 将来的に Gemini 等の他プロバイダー追加も見据える

### 現状の API 使用箇所

| ファイル | 用途 | API呼び出し箇所 |
|---------|------|----------------|
| `structure.py` | カード抽出 | 3箇所 (`anthropic.Anthropic()` + `messages.create()`) |
| `quality.py` | 品質評価 | 2箇所 |
| `batch.py` | バッチ処理 | 4箇所 (`messages.batches.*`) |

**合計:** 9箇所の API 呼び出し

### 現在の Config 構造

```python
# config.py
class AppConfig(BaseModel, frozen=True):
    model: str = "claude-sonnet-4-5-20250929"
    max_tokens: int = 8192
    # ... (他のフィールド)
```

```yaml
# config.yaml
model: claude-sonnet-4-5-20250929
max_tokens: 8192
```

---

## 実装オプション

### オプション A: フル実装（推奨）

**設計:**
```
src/pdf2anki/
  ├── providers/
  │   ├── __init__.py
  │   ├── base.py          # Protocol/Abstract Base
  │   ├── anthropic.py     # AnthropicProvider
  │   └── openai.py        # OpenAIProvider
  ├── config.py            # + provider フィールド追加
  ├── cost.py              # + OpenAI 価格テーブル追加
  └── (既存ファイル)       # provider経由でAPI呼び出し
```

**見積もり:**
- 新規LOC: ~500-600行
- 新規ファイル: 3-4ファイル
- 複雑度: MEDIUM
- 実装時間: 4-6時間

**メリット:**
- クリーンな設計、保守性高い
- 他プロバイダー追加が容易
- テストしやすい

**デメリット:**
- 実装時間が長い

---

### オプション B: 最小限実装

**設計:**
```python
# structure.py に簡易切り替えロジック追加
if config.provider == "openai":
    client = openai.OpenAI()
else:
    client = anthropic.Anthropic()
```

**見積もり:**
- 新規LOC: ~200行
- 複雑度: LOW
- 実装時間: 2-3時間

**メリット:**
- 実装が早い

**デメリット:**
- 技術的負債
- 拡張性低い
- テストが複雑化

---

## 制約・注意点

### OpenAI の機能制限

1. **Batch API 非対応**
   - Anthropic: `messages.batches.create()` で 50%割引
   - OpenAI: 相当機能なし（独自の並列処理実装が必要）
   - → Phase 5（非同期並列処理）の再検討が必要

2. **Prompt Caching 非対応**
   - Anthropic: プロンプトキャッシュで入力コスト削減
   - OpenAI: 相当機能なし
   - → コスト削減効果が減少

3. **Structured Outputs の違い**
   - Anthropic: Tool use パターン (`tools=[...]`)
   - OpenAI: `response_format={"type": "json_schema", ...}`
   - → 両者の差異を吸収する抽象化が必要

4. **価格体系の違い**
   ```
   Anthropic Sonnet 4.5:
     Input:  $3.00 / 1M tokens
     Output: $15.00 / 1M tokens
     Batch:  50% off

   OpenAI GPT-4o:
     Input:  $2.50 / 1M tokens
     Output: $10.00 / 1M tokens
     Batch:  なし
   ```

---

## リサーチ項目

新セッションで以下をリサーチ・検証してください：

### 1. OpenAI Structured Outputs の仕様確認

**調査内容:**
- OpenAI の `response_format` による JSON Schema 対応
- Pydantic モデルから JSON Schema への変換方法
- エラーハンドリング（validation error 時の挙動）

**参考:**
- https://platform.openai.com/docs/guides/structured-outputs
- `openai.pydantic_function_tool()` の使用可否

**成果物:**
- `docs/research/openai-structured-outputs.md`

---

### 2. Provider 抽象化の設計パターン

**調査内容:**
- LiteLLM, LangChain 等の既存ライブラリの設計参考
- Protocol vs Abstract Base Class の選択
- 非同期 vs 同期 API の扱い
- エラーハンドリングの統一方法

**参考:**
- LiteLLM: https://github.com/BerriAI/litellm
- LangChain: https://github.com/langchain-ai/langchain

**成果物:**
- `docs/research/provider-abstraction-design.md`

---

### 3. OpenAI のバッチ処理代替手段

**調査内容:**
- OpenAI に Batch API 相当の機能があるか（2026年2月時点）
- ない場合: `asyncio` + `Semaphore` による独自実装の設計
- コスト比較: Anthropic Batch (50% off) vs OpenAI 並列処理

**参考:**
- Phase 5 (非同期並列処理) の延期理由を再評価
- `docs/plans/long-document-processing-plan.md` の Phase 5 セクション

**成果物:**
- `docs/research/openai-batch-processing.md`

---

### 4. 既存テストの影響範囲

**調査内容:**
- 現在の 456 テストのうち、API 呼び出しをモックしているテスト数
- Provider 抽象化により影響を受けるテストファイル
- テスト戦略: モックを provider レベルで行うか、個別に行うか

**調査コマンド:**
```bash
# API モックを使用しているテストを検索
grep -r "anthropic\|mock" tests/ | wc -l
grep -r "@patch\|@mock" tests/ | wc -l
```

**成果物:**
- `docs/research/test-impact-analysis.md`

---

### 5. コスト試算・比較

**調査内容:**
- 典型的な PDF (100ページ、日本語) の処理コスト試算
  - Anthropic Sonnet + Batch API
  - Anthropic Sonnet (通常)
  - OpenAI GPT-4o
  - OpenAI GPT-4o-mini
- $10 クレジットで処理できる PDF ページ数

**成果物:**
- `docs/research/cost-comparison.md`

---

## 実装フェーズ計画（リサーチ後）

リサーチ完了後、以下のフェーズで実装を進めることを推奨：

### Phase 1: Provider 抽象化レイヤー
- `providers/base.py` - Protocol 定義
- `providers/anthropic.py` - 既存コードのリファクタリング
- テスト追加

### Phase 2: OpenAI Provider 実装
- `providers/openai.py` - OpenAI SDK ラッパー
- Structured Outputs 対応
- テスト追加

### Phase 3: Config + Cost 統合
- `config.py` - `provider` フィールド追加
- `cost.py` - OpenAI 価格テーブル追加
- CLI に `--provider` オプション追加

### Phase 4: バッチ処理対応（オプション）
- OpenAI 用の並列処理実装
- Phase 5 延期内容の再評価

---

## 次のセッションで最初にやること

1. **このドキュメントを読む**
   ```bash
   cat docs/plans/provider-selection-handoff.md
   ```

2. **現在の git 状態を確認**
   ```bash
   git status
   git log --oneline -5
   ```

3. **リサーチ項目 1 から順に調査開始**
   - WebSearch でOpenAI API の最新仕様を確認
   - 各項目の成果物を `docs/research/` に保存

4. **リサーチ完了後、実装プラン作成**
   ```bash
   # Plan mode で詳細な実装計画を作成
   /plan
   ```

---

## 参考リソース

### 既存ドキュメント
- `docs/plans/long-document-processing-plan.md` - Phase 1-4 の実装計画
- `docs/research/long-document-processing.md` - 長文処理のリサーチ結果
- `.claude/projects/-Users-shimomoto-tatsuya-MyAI-Lab-Anki-QA/memory/MEMORY.md` - プロジェクトメモリ

### コードベース重要ファイル
- `src/pdf2anki/structure.py` - メインAPI呼び出し箇所
- `src/pdf2anki/quality.py` - 品質評価API
- `src/pdf2anki/batch.py` - バッチAPI
- `src/pdf2anki/config.py` - 設定管理
- `src/pdf2anki/cost.py` - コスト管理

### テスト
- `tests/test_structure.py` - 243行、API モック多数
- `tests/test_quality.py` - 157行
- `tests/test_batch.py` - 195行

---

## 質問・不明点

新セッションで疑問が生じた場合:
1. この引き継ぎドキュメントを再読
2. `MEMORY.md` を参照
3. コードベースを `Grep` で検索
4. WebSearch で最新情報を確認

---

## 成功の定義

リサーチフェーズ完了の条件:
- [ ] 5つのリサーチ項目すべての成果物が `docs/research/` に保存済み
- [ ] OpenAI API の技術的実現可能性が確認済み
- [ ] 実装オプション A vs B の最終判断材料が揃っている
- [ ] 実装フェーズの詳細プランが作成可能な状態

---

**Good luck with the research! 🚀**
