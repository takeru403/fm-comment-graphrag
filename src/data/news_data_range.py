import json
from datetime import datetime
from pathlib import Path
from typing import Tuple, Union

def get_news_data_range(news_path: Union[str, Path] = "data/news.json") -> Tuple[str, str]:
    path_str = str(news_path)
    with open(path_str, "r", encoding="utf-8") as f:
        data = json.load(f)

    times = [
        datetime.fromisoformat(a["publishedAt"].replace("Z", "+00:00"))
        for a in data.get("articles", [])
        if a.get("publishedAt")
    ]
    if not times:
        return "", ""
    return min(times).isoformat(), max(times).isoformat()

if __name__ == "__main__":
    print(get_news_data_range())