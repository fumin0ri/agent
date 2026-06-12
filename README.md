# LangGraph Slack MCP Agent

利用者の自然言語指示をOllamaで理解し、Slack公式MCPサーバーが実行時に公開するツールを選択・実行するLangGraphエージェントです。

論文検索機能はありません。エージェントは起動時にSlack MCPへ接続して利用可能な機能を発見し、必要に応じて複数のツールを順番に呼び出します。

```text
利用者の指示
  → Slack MCPからツール一覧を取得
  → Ollamaが次の操作を判断
  → MCPツールを実行
  → 結果をOllamaへ戻す
  → 必要なら追加ツールを実行
  → 最終回答
```

## Slack MCPでできること

利用可能な機能は、Slackアプリへ付与したOAuthスコープとワークスペース管理者の承認によって変わります。Slack公式MCPサーバーは、主に次の操作を提供します。

- メッセージ、ファイル、ユーザー、チャンネル、絵文字の検索
- チャンネル履歴・スレッドの読取
- メッセージ送信、会話作成、リアクション追加
- Canvasの作成・更新・読取
- ユーザープロフィール・チャンネルメンバーの取得

## Slack側の準備

Slack公式ドキュメントに従い、Marketplace公開アプリまたは内部アプリを作成します。

1. SlackアプリでModel Context Protocol機能を有効化
2. 必要なユーザースコープを追加
3. OAuth Redirect URLを設定
4. アプリをインストールし、OAuthでユーザートークンを取得

Slack公式MCPはStreamable HTTPのみをサポートします。

```text
https://mcp.slack.com/mcp
```

## セットアップ

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e ".[dev]"
Copy-Item .env.example .env
```

`.env`へOllama接続先を設定します。Slack OAuthで取得したユーザートークンもローカルの`.env`へ設定してください。トークンをREADMEやGit管理対象ファイルへ記載しないでください。

```dotenv
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=gpt-oss:20b
SLACK_MCP_URL=https://mcp.slack.com/mcp
```

## 使用方法

まずSlack MCPへ接続し、現在利用できるツールとREAD/WRITE分類を確認します。

```powershell
slack-agent doctor
slack-agent tools
```

検索・読取指示はそのまま実行できます。

```powershell
slack-agent ask "先週project-alphaチャンネルで決まった内容を要約して"
slack-agent ask "田中さんのプロフィールと現在のステータスを調べて"
```

メッセージ送信、チャンネル作成、リアクション、Canvas更新などの書込み操作は、誤操作防止のため`--allow-writes`が必要です。

```powershell
slack-agent ask --allow-writes "generalチャンネルに明日の会議は10時開始と投稿して"
```

## 安全設計

- ツール名・説明・入力スキーマはSlack MCPから実行時に取得
- 検索・読取系以外の未知ツールは書込み操作として保守的に扱う
- 書込み操作は初期設定でブロック
- MCPツール結果をLLMへ戻し、追加操作または最終回答を判断
- 最大ツール実行ステップ数を設定し、無限ループを防止

## テスト

```powershell
pytest
```

## 参考

- [Slack MCP Server overview](https://docs.slack.dev/ai/slack-mcp-server/)
- [Developing a sample app with the Slack MCP Server](https://docs.slack.dev/ai/slack-mcp-server/developing/)
