# graphRAG

ニュースデータからテキストを取り込み、RAG（ベクトル検索）と GraphRAG（知識グラフ活用）の 2 手法で市場コメントを自動生成する実験用プロジェクトです。日経平均（^N225）の価格推移、NewsAPI から取得した記事、Neo4j 上の知識グラフを用いて、投資家向けに簡潔な月次コメントを作成します。

## 機能概要

- **RAG**: OpenAI 埋め込み + コサイン類似度で上位文脈を抽出し、LLM でコメント生成（`src/RAG/main.py`）
- **GraphRAG**: LLM で三つ組抽出 → Neo4j へ格納 → NetworkX でコミュニティ検出 → 要約 → LLM でコメント生成（`src/GraphRAG/main.py`）
- **ニュース収集**: NewsAPI から検索クエリで記事を取得し `data/news.json` に保存（`src/data/news_data.py`、CLI: `src/cli/get_news.py`）
- **市況データ**: yfinance で ^N225 の終値時系列を取得し、直近ポイントの差分と変化率を文字列化（`src/data/yfinance_data.py`）
- **可視化**: Neo4j から抽出したグラフを GEXF でエクスポート（`outputs/graph.gexf`）。Embedding Projector 用 TSV も出力（`outputs/embedding_projector/`）

## 必要環境

- Python 3.12+
- [uv](https://github.com/astral-sh/uv)（推奨）または標準 `venv`
- OpenAI API キー
- NewsAPI API キー
- Neo4j インスタンス（Aura など）

環境変数は `.env`（UTF-8）で読み込まれます（`config/setting.py`）。

```ini
NEWSAPI_API_KEY=your_newsapi_key
OPENAI_API_KEY=your_openai_key
NEO4J_URI=bolt+s://xxxx.databases.neo4j.io
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=your_password
AURA_INSTANCENAME=optional
AURA_INSTANCEID=optional
```

## セットアップ

### uv を使う場合（推奨）

```bash
# Python バージョン確認（uv 経由）
uv run python -V
# 主要依存の読み込み確認
uv run python -c "import langchain_openai; print('ok')"
```

`pyproject.toml` に主要依存が定義済みです。

### venv を使う場合

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -U pip
# 必要パッケージを手動インストール（pyproject を参照）
pip install \
  graphrag==2.4.0 \
  "langchain>=0.3.27" \
  langchain-community==0.3.20 \
  "langchain-experimental>=0.3.4" \
  "langchain-neo4j>=0.4.0" \
  "langchain-openai>=0.3.34" \
  "neo4j>=5.25,<6" \
  "networkx>=3.5" \
  "pydantic>=2.11.9" \
  "pydantic-settings>=2.11.0" \
  "pyyaml>=6.0.3" \
  "requests>=2.32.5" \
  "typer>=0.16.1" \
  "yfinance>=0.2.66"
```

### just（任意）

`justfile` に簡易ターゲットがあります。

```bash
just hello
just generate_RAG_comment
just generate_GraphRAG_comment
```

## データ取得（NewsAPI）

`data/news.json` が無い場合は、まずニュースを取得します。

```bash
# Typer ベースの CLI（関数名がコマンド名になり、_ は - に変換されます）
uv run python -m src.cli.get_news fetch-news "日経平均 OR 日経平均株価"
# もしくは直接実行
uv run python src/cli/get_news.py fetch-news "日経平均 OR 日経平均株価"
```

出力: `data/news.json`

## 実行方法

### RAG でコメント生成

```bash
uv run python src/RAG/main.py
# or
just generate_RAG_comment
```

- **参照データ**: `data/news.json`
- **生成補助**: Embedding Projector 用 TSV を `outputs/embedding_projector/` に出力
- **終値トレンド**: `yfinance` で ^N225 を取得し、直近最大 10 ポイントを標準出力

### GraphRAG でコメント生成

```bash
export NEO4J_URI=... NEO4J_USERNAME=... NEO4J_PASSWORD=...
uv run python src/GraphRAG/main.py
# or
just graph_rag
```

処理概要:

- ニュース本文から LLM で三つ組を抽出し Neo4j へアップサート
- グラフを NetworkX へ展開しコミュニティ検出
- コミュニティの代表エッジを元に LLM で要約
- クエリ類似度で重要コミュニティを選び、コメント生成
- Gephi 向けに `outputs/graph.gexf` を出力

## CLI ラッパー

`src/cli/main.py` は Typer アプリのエントリです。

```bash
uv run python src/cli/main.py news fetch-news "日経平均 OR 日経平均株価"
uv run python src/cli/main.py rag
```

## 出力物

- `outputs/embedding_projector/news_docs_metadata.tsv`
- `outputs/embedding_projector/news_docs_vectors.tsv`
- `outputs/graph.gexf`（Gephi 等で可視化可能）
- 生成コメントは標準出力に表示されます（必要に応じてリダイレクトしてください）。

## 可視化

- GEXF は Gephi で読み込み可能です。
- `src/data/plot_chart.py` を実行すると `data/chart.png` を生成します。

チャート画像の生成コマンド:

```bash
uv run python src/data/plot_chart.py
```

## トラブルシュート

- **NEO4J_* 未設定**: GraphRAG 実行時に例外を投げます。環境変数を設定してください。
- **data/news.json が空/エラー応答**: `status` が `ok` 以外ならエラーにします（RAG）。クエリを変えて再取得してください。
- **yfinance が空**: 期間が休場に当たる場合は自動でバッファ期間で再取得します。
- **Embedding Projector TSV が空**: ニューステキストが空の場合は出力されません。

## ライセンス

本リポジトリのコードは、プロジェクト作成者の意図に従い利用してください。外部 API・データの利用規約（OpenAI, NewsAPI, Neo4j, Yahoo! 等）に従ってください。
