from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")
    NEWSAPI_API_KEY: str
    OPENAI_API_KEY: str
    NEO4J_URI: str
    NEO4J_USERNAME: str
    NEO4J_PASSWORD: str
    AURA_INSTANCENAME: str
    AURA_INSTANCEID: str


settings = Settings()

if __name__ == "__main__":
    settings = Settings()
    print(settings.NEWSAPI_API_KEY)
