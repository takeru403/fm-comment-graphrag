import json
from datetime import datetime

def get_news_data_range():
    with open("data/news.json", "r", encoding="utf-8") as f:
        data = json.load(f)

    times = [
        datetime.fromisoformat(a["publishedAt"].replace("Z", "+00:00"))
        for a in data.get("articles", [])
        if a.get("publishedAt")
    ]
    return min(times).isoformat(), max(times).isoformat()

if __name__ == "__main__":
    print(get_news_data_range())