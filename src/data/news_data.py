import json

import requests

from config.setting import Settings


class FetchNewsAPI:
    def __init__(self, query: str):
        settings = Settings()
        self.headers = {"X-Api-Key": settings.NEWSAPI_API_KEY}
        self.url = "https://newsapi.org/v2/everything"
        self.params = {
            "q": query,
            "sortBy": "publishedAt",
            "pageSize": 100,
        }

    def fetch_news(self):
        response = requests.get(
            self.url,
            headers=self.headers,
            params=self.params,
        )
        return response.json()

    def save_news(self):
        news = self.fetch_news()
        with open("data/news.json", "w", encoding="utf-8") as f:
            json.dump(news, f, ensure_ascii=False, indent=4)


if __name__ == "__main__":
    query = "日経平均 OR 日経平均株価"

    fetch_news = FetchNewsAPI(query)
    fetch_news.save_news()
