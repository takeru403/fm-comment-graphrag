import typer
from src.RAG.main import generate_comment

app = typer.Typer()

@app.command()
def rag() -> None:
    comment = generate_comment()
    print(comment)


if __name__ == "__main__":
    rag()