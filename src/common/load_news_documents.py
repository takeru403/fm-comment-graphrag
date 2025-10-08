import json
from pathlib import Path
from typing import List

from langchain_core.documents import Document


def load_news_documents(
    news_path: Path,
) -> List[Document]:
    with open(news_path, "r", encoding="utf-8") as f:
        news_text = json.load(f)
    return news_text
