generate_RAG_comment:
    uv run python src/RAG/main.py

generate_GraphRAG_comment:
    uv run python src/GraphRAG/main.py

# NewsAPI からニュースを取得して data/news.json を更新
news q="日経平均 OR 日経平均株価":
    uv run python -m src.cli.get_news fetch-news {{q}}

# Typer 統合CLIのエントリ
cli:
    uv run python src/cli/main.py

# Typer経由でRAG/GraphRAGを起動
rag:
    uv run python src/cli/main.py rag

graph_rag:
    uv run python src/cli/main.py graph-rag