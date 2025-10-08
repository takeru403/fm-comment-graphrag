from typer import Typer

from cli.get_news import app as get_news_app
from cli.get_news import fetch_news as fetch_news_command

app = Typer(no_args_is_help=True)


# サブコマンド形式: `python -m cli.main news fetch-news "キーワード"`
app.add_typer(get_news_app, name="news")


# トップレベルでも呼べるように: `python -m cli.main fetch-news "キーワード"`
app.command()(fetch_news_command)


if __name__ == "__main__":
    app()
