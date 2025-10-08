import typer
from typer import Typer

from src.data.news_data import FetchNewsAPI

app = Typer(no_args_is_help=True)


@app.command()
def fetch_news(q: str = typer.Argument(..., help="検索クエリ")):
    fetch_news = FetchNewsAPI(q)
    fetch_news.save_news()

if __name__ == "__main__":
    app()
