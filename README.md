# LangGraph Paper Secretary

別のマシンで動くOllamaを頭脳として使い、ローカルの論文フォルダから指示に合う論文を探す秘書エージェントです。

処理はLangGraphで次の順に実行します。

1. PDF、Markdown、テキストをページ・チャンク単位に分割
2. Ollamaの埋め込みモデルで各チャンクをベクトル化
3. 利用者の依頼と意味的に近い本文をコサイン類似度で検索
4. 関連本文だけを根拠に、候補、理由、ページ番号、ファイルパスを回答

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
OLLAMA_EMBEDDING_MODEL=embeddinggemma

# 論文フォルダと検索索引
PAPER_LIBRARY=D:\papers
PAPER_INDEX=./work/papers.sqlite3
```

OllamaサーバーをLANから利用する場合、Ollama側で待受アドレスとファイアウォールを適切に設定してください。APIには機密性のある論文本文が送信されるため、信頼できないネットワークへ公開しないでください。

## 使い方

Ollamaサーバー側でチャットモデルと埋め込みモデルを準備します。

```powershell
ollama pull qwen3:8b
ollama pull embeddinggemma
```

論文フォルダをベクトル索引化します。ファイルを追加・更新した際も同じコマンドで差分更新できます。

```powershell
paper-secretary index
```

自然言語で検索を依頼します。

```powershell
paper-secretary ask "Transformerを科学的発見に応用した最近の論文を探して"
paper-secretary ask "著者名にHintonを含み、蒸留を扱う論文はある？"
```

## ベクトル検索について

索引化時に論文本文を約1800文字のチャンクへ分割し、Ollamaの埋め込みAPIでベクトル化します。検索時には依頼文も同じモデルでベクトル化し、コサイン類似度が高い本文をLLMへ渡します。このため、論文と依頼で表現が異なっていても、意味が近ければ候補になります。

初期設定では1論文あたり最大2チャンクを採用し、検索結果が一つの論文だけで埋まらないようにしています。回答プロンプトは、各主張に検索結果番号とページ番号を付け、根拠にない内容を補わないよう要求します。`.env`の`SEARCH_RESULT_LIMIT`と`MAX_CHUNKS_PER_PAPER`で調整できます。

埋め込みモデルを変更すると、次回の`paper-secretary index`で索引全体を自動的に再構築します。小・中規模の論文ライブラリ向けに、ベクトルはSQLiteへ保存しPythonで比較しています。数十万チャンク規模になった場合はQdrantやpgvectorへの移行が適しています。

## テスト

```powershell
pytest
```
