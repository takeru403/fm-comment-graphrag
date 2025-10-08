hello:
    echo "hello world"

generate_RAG_comment:
    uv run python src/RAG/main.py

generate_GraphRAG_comment:
    uv run python src/GraphRAG/main.py