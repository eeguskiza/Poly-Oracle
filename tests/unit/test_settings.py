from pathlib import Path

from config.settings import (
    Settings,
    LLMSettings,
    RiskSettings,
    DataSettings,
    PolymarketSettings,
    DatabaseSettings,
)


def test_llm_settings_defaults() -> None:
    llm = LLMSettings()
    assert llm.model == "mistral"
    assert llm.base_url == "http://localhost:11434"
    assert llm.embedding_model == "nomic-embed-text"
    assert llm.temperature == 0.7
    assert llm.max_tokens == 2000
    assert llm.timeout == 120


def test_risk_settings_defaults() -> None:
    risk = RiskSettings()
    assert risk.initial_bankroll == 50
    assert risk.max_position_pct == 0.10
    assert risk.min_bet == 1.0
    assert risk.max_bet == 10.0
    assert risk.max_daily_loss_pct == 0.10
    assert risk.max_open_positions == 8
    assert risk.min_edge == 0.08
    assert risk.min_confidence == 0.65
    assert risk.min_liquidity == 1000


def test_data_settings_defaults() -> None:
    data = DataSettings()
    assert data.cache_ttl_news == 3600
    assert data.cache_ttl_market_list == 300
    assert data.cache_ttl_market_detail == 60


def test_polymarket_settings_defaults() -> None:
    poly = PolymarketSettings()
    assert poly.clob_url == "https://clob.polymarket.com"
    assert poly.gamma_url == "https://gamma-api.polymarket.com"


def test_database_settings_defaults() -> None:
    db = DatabaseSettings()
    assert db.db_dir == Path("db")
    assert db.duckdb_path == Path("db/analytics.duckdb")
    assert db.sqlite_path == Path("db/poly_oracle.db")
    assert db.chroma_path == Path("db/chroma")


def test_database_settings_computed_paths(monkeypatch) -> None:
    monkeypatch.setenv("DB_DIR", "custom")
    db = DatabaseSettings()
    assert db.duckdb_path == Path("custom/analytics.duckdb")
    assert db.sqlite_path == Path("custom/poly_oracle.db")
    assert db.chroma_path == Path("custom/chroma")


def test_settings_loads_defaults() -> None:
    settings = Settings()
    assert settings.log_level == "DEBUG"
    assert settings.paper_trading is True
    assert isinstance(settings.llm, LLMSettings)
    assert isinstance(settings.risk, RiskSettings)
    assert isinstance(settings.data, DataSettings)
    assert isinstance(settings.polymarket, PolymarketSettings)
    assert isinstance(settings.database, DatabaseSettings)


def test_risk_limits_match_spec() -> None:
    risk = RiskSettings()
    assert risk.max_position_pct == 0.10
    assert risk.min_bet == 1.0
    assert risk.max_bet == 10.0
    assert risk.max_daily_loss_pct == 0.10
    assert risk.max_open_positions == 8
    assert risk.min_edge == 0.08
    assert risk.min_confidence == 0.65
    assert risk.min_liquidity == 1000
