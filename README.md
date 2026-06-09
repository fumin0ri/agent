# LangGraph Paper Secretary

別のマシンで動くOllamaを頭脳として使い、ローカルの論文フォルダから指示に合う論文を探す秘書エージェントです。

処理はLangGraphで次の順に実行します。

1. 利用者の自然言語指示から検索語を抽出
2. PDF、Markdown、テキストのSQLite全文検索
3. 検索結果だけを根拠に、候補とファイルパスを回答

## セットアップ

Python 3.11以上を用意し、プロジェクト直下で実行します。

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e ".[dev]"
Copy-Item .env.example .env
```

`.env`を環境に合わせて変更します。

```dotenv
# Ollamaが動いている別マシン。末尾に /api は付けません。
OLLAMA_BASE_URL=http://192.168.1.20:11434
OLLAMA_MODEL=qwen3:8b

# 論文フォルダと検索索引
PAPER_LIBRARY=D:\papers
PAPER_INDEX=./work/papers.sqlite3
```

OllamaサーバーをLANから利用する場合、Ollama側で待受アドレスとファイアウォールを適切に設定してください。APIには機密性のある論文本文が送信されるため、信頼できないネットワークへ公開しないでください。

## 使い方

論文フォルダを索引化します。ファイルを追加・更新した際も同じコマンドで差分更新できます。

```powershell
paper-secretary index
```

自然言語で検索を依頼します。

```powershell
paper-secretary ask "Transformerを科学的発見に応用した最近の論文を探して"
paper-secretary ask "著者名にHintonを含み、蒸留を扱う論文はある？"
```

## 現在の検索方式

検索はSQLite FTS5による高速なキーワード全文検索です。LLMは依頼から適切な検索語を抽出し、候補を説明します。意味検索が必要になった場合は、Ollamaの埋め込みAPIとベクトルDBを検索ノードへ追加できます。

## テスト

```powershell
pytest
```
