from pathlib import Path
from typing import Optional
from functools import lru_cache

from pydantic import Field, computed_field
from pydantic_settings import BaseSettings, SettingsConfigDict


class LLMSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="OLLAMA_")

    model: str = Field(default="mistral", alias="OLLAMA_MODEL")
    base_url: str = Field(default="http://localhost:11434", alias="OLLAMA_BASE_URL")
    embedding_model: str = Field(default="nomic-embed-text", alias="OLLAMA_EMBEDDING_MODEL")
    temperature: float = 0.7
    max_tokens: int = 2000
    timeout: int = 120


class RiskSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="")

    initial_bankroll: float = Field(default=50, alias="INITIAL_BANKROLL")
    max_position_pct: float = Field(default=0.10, alias="MAX_POSITION_PCT")
    min_bet: float = Field(default=1.0, alias="MIN_BET")
    max_bet: float = Field(default=10.0, alias="MAX_BET")
    max_daily_loss_pct: float = Field(default=0.10, alias="MAX_DAILY_LOSS_PCT")
    max_open_positions: int = Field(default=8, alias="MAX_OPEN_POSITIONS")
    min_edge: float = Field(default=0.08, alias="MIN_EDGE")
    min_confidence: float = Field(default=0.65, alias="MIN_CONFIDENCE")
    min_liquidity: float = Field(default=1000, alias="MIN_LIQUIDITY")


class DataSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="")

    newsapi_key: Optional[str] = Field(default=None, alias="NEWSAPI_KEY")
    cache_ttl_news: int = 3600
    cache_ttl_market_list: int = 300
    cache_ttl_market_detail: int = 60


class PolymarketSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="POLYMARKET_")

    api_key: Optional[str] = Field(default=None, alias="POLYMARKET_API_KEY")
    api_secret: Optional[str] = Field(default=None, alias="POLYMARKET_API_SECRET")
    api_passphrase: Optional[str] = Field(default=None, alias="POLYMARKET_API_PASSPHRASE")
    clob_url: str = "https://clob.polymarket.com"
    gamma_url: str = "https://gamma-api.polymarket.com"


class DatabaseSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="")

    db_dir: Path = Field(default=Path("db"), alias="DB_DIR")

    @computed_field
    @property
    def duckdb_path(self) -> Path:
        return self.db_dir / "analytics.duckdb"

    @computed_field
    @property
    def sqlite_path(self) -> Path:
        return self.db_dir / "poly_oracle.db"

    @computed_field
    @property
    def chroma_path(self) -> Path:
        return self.db_dir / "chroma"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    log_level: str = Field(default="DEBUG", alias="LOG_LEVEL")
    paper_trading: bool = Field(default=True, alias="PAPER_TRADING")

    llm: LLMSettings = Field(default_factory=LLMSettings)
    risk: RiskSettings = Field(default_factory=RiskSettings)
    data: DataSettings = Field(default_factory=DataSettings)
    polymarket: PolymarketSettings = Field(default_factory=PolymarketSettings)
    database: DatabaseSettings = Field(default_factory=DatabaseSettings)


@lru_cache
def get_settings() -> Settings:
    return Settings()
