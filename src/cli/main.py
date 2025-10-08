from typer import Typer

from cli.get_news import app as news_app
from cli.RAG_fmcomment import rag

def main() -> None:
    app = Typer()
    app.add_typer(news_app, name="news")
    @app.command("rag")
    def rag_cmd() -> None:
        rag()
    app()


if __name__ == "__main__":
    main()
