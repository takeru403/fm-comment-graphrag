from src.GraphRAG.main import generate_comment


def graph_rag() -> None:
    comment = generate_comment()
    print(comment)

if __name__ == "__main__":
    graph_rag()