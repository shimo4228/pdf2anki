# Security Checklist

セキュリティベストプラクティスを実践するための再利用可能なチェックリストです。

このドキュメントは [アーキテクチャドキュメント](../../docs/architecture/05-security.md) のセクション 7 および 8 から抽出されています。

---

## 1. コミット前チェックリスト

コードをコミットする前に、以下の項目を確認してください：

- [ ] ハードコードされた API キーがないか（`ANTHROPIC_API_KEY` など）
- [ ] `.env` ファイルが `.gitignore` に含まれているか
- [ ] `credentials.json` や `*.key` ファイルがステージされていないか
- [ ] 機密データを含むログファイルがステージされていないか
- [ ] すべてのユーザー入力が検証されているか（`extract.py`、`config.py`）
- [ ] SQL インジェクション対策がされているか（該当する場合）
- [ ] XSS 対策がされているか（該当する場合）

### 確認コマンド

```bash
# ステージングエリアの確認
git status

# .env が gitignore に含まれているか確認
grep -q "^\.env$" .gitignore && echo "✅ .env is ignored" || echo "❌ .env is NOT ignored"

# ハードコードされた API キーのパターン検索
git diff --cached | grep -E "(sk-ant-|ANTHROPIC_API_KEY.*=.*sk-)" && echo "⚠️ API key found in staged changes!" || echo "✅ No API keys found"
```

---

## 2. リリース前チェックリスト

プロジェクトをリリースする前に、以下の項目を確認してください：

- [ ] 依存関係が最新か（`uv pip list --outdated` で確認）
- [ ] 既知の脆弱性がないか（`pip-audit` で確認）
- [ ] README にプライバシーポリシーセクションが含まれているか
- [ ] LICENSE ファイルが正しいか（AGPL-3.0）
- [ ] `.env.example` が最新で完全か
- [ ] すべてのセキュリティ関連テストが通過しているか
- [ ] 本番環境用の設定が適切か

### 確認コマンド

```bash
# 依存関係の確認
cd /Users/shimomoto_tatsuya/MyAI_Lab/Anki-QA
.venv/bin/uv pip list --outdated

# セキュリティ脆弱性のスキャン
.venv/bin/pip-audit

# ライセンスファイルの確認
grep -q "AGPL-3.0" LICENSE && echo "✅ License is AGPL-3.0" || echo "❌ License mismatch"

# .env.example が存在するか確認
[ -f .env.example ] && echo "✅ .env.example exists" || echo "❌ .env.example missing"
```

---

## 3. 記事公開前チェックリスト

Zenn や他のプラットフォームに記事を公開する前に、以下の項目を確認してください：

- [ ] コードスニペットに API キーが含まれていないか
- [ ] スクリーンショットに機密情報（ファイルパス、ユーザー名、API キーなど）が含まれていないか
- [ ] SpecStory ログがサニタイズされているか
- [ ] ファイルパスが匿名化されているか（`/Users/username/` を含まない）
- [ ] 実リポジトリへのリンクが正しいか
- [ ] コード例が実行可能で、セキュリティリスクがないか

### 確認コマンド

```bash
# 記事内の API キーパターン検索
grep -rE "(sk-ant-|ANTHROPIC_API_KEY.*=.*sk-)" zenn-content/articles/ && echo "⚠️ API key found!" || echo "✅ No API keys"

# ファイルパス漏洩の検索
grep -rE "/Users/[a-z_]+/" zenn-content/articles/ && echo "⚠️ File paths found!" || echo "✅ No file paths"
```

---

## 4. インシデント対応クイックリファレンス

### 4.1 API キー漏洩時の手順

セキュリティインシデントが発生した場合、以下の手順で迅速に対応してください。

#### ステップ 1: 即座に無効化

```bash
# Anthropic Console で API キーを削除
# https://console.anthropic.com/settings/keys
```

1. Anthropic Console にログイン
2. Settings > API Keys に移動
3. 漏洩したキーを削除

#### ステップ 2: 新しいキーを生成

```bash
# 新しい API キーを取得し、.env ファイルを更新
export ANTHROPIC_API_KEY=sk-ant-new-xxxxx
```

`.env` ファイルを更新：

```bash
echo "ANTHROPIC_API_KEY=sk-ant-new-xxxxx" > .env
```

#### ステップ 3: Git 履歴から削除（コミットされた場合）

```bash
# BFG Repo-Cleaner で履歴から削除
# まず secrets.txt に削除したい文字列を記載
echo "sk-ant-old-xxxxx" > secrets.txt

# BFG で履歴を書き換え
bfg --replace-text secrets.txt

# 履歴を整理
git reflog expire --expire=now --all
git gc --prune=now --aggressive

# リモートに強制プッシュ（⚠️ 慎重に実行）
git push --force
```

#### ステップ 4: 影響範囲の調査

- いつ漏洩したか
- どのリポジトリに含まれていたか
- 不正使用の痕跡がないか（Anthropic Console のログ確認）

### 4.2 機密文書誤送信時の手順

#### ステップ 1: Anthropic サポートに連絡

- Email: support@anthropic.com
- データ削除を依頼

#### ステップ 2: ローカルログの削除

```bash
# ローカルログを削除
rm -f .cache/*.log
rm -f .specstory/history/YYYY-MM-DD_*.md
```

#### ステップ 3: 再発防止

- 機密ファイル用の `.claudeignore` を強化
- README に警告を追加
- チェックリストを更新

---

## 関連ドキュメント

- [05-security.md](../../docs/architecture/05-security.md) - 完全なセキュリティガイドライン
- [.env.example](../.env.example) - 環境変数のテンプレート
- [.gitignore](../.gitignore) - Git 除外設定

---

**最終更新:** 2026-02-09
**メンテナンス頻度:** 新しいセキュリティプラクティスが追加されるたびに更新
