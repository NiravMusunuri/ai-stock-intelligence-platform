from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    database_url: str = "postgresql://stockintel:stockintel@localhost:5432/stockintel"

    alpha_vantage_api_key: str = ""
    polygon_api_key: str = ""
    news_api_key: str = ""
    anthropic_api_key: str = ""

    class Config:
        env_file = ".env"
        extra = "ignore"


settings = Settings()
