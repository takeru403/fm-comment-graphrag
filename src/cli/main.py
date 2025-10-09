from typer import Typer

from src.cli.get_news import app as news_app
from src.cli.RAG_fmcomment import rag
from src.cli.GraphRAG_fmcomment import graph_rag


def main() -> None:
    app = Typer()
    app.add_typer(news_app, name="news")

    @app.command("rag")
    def rag_cmd() -> None:
        rag()

    @app.command("graph-rag")
    def graph_rag_cmd() -> None:
        graph_rag()

    app()


if __name__ == "__main__":
    main()
