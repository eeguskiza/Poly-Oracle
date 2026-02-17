"""Tests for v1 configuration.

Note: These tests assert v1 defaults (small bankroll, conservative sizing).
Environment variables from conftest.py may override defaults, so tests that
check 'raw defaults' use monkeypatch to clear relevant env vars.
"""

from pathlib import Path

from bot.config.settings import (
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
    assert llm.timeout == 300


def test_risk_settings_from_env(monkeypatch) -> None:
    """v1 defaults: $1 bankroll, $0.10-$1.00 bet range, 3 max positions."""
    monkeypatch.setenv("INITIAL_BANKROLL", "1.0")
    monkeypatch.setenv("MIN_BET", "0.10")
    monkeypatch.setenv("MAX_BET", "1.0")
    monkeypatch.setenv("MAX_OPEN_POSITIONS", "3")
    risk = RiskSettings()
    assert risk.initial_bankroll == 1.0
    assert risk.max_position_pct == 0.10
    assert risk.min_bet == 0.10
    assert risk.max_bet == 1.0
    assert risk.max_daily_loss_pct == 0.10
    assert risk.max_open_positions == 3
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


def test_database_settings_computed_paths(monkeypatch) -> None:
    monkeypatch.setenv("DB_DIR", "custom")
    db = DatabaseSettings()
    assert db.duckdb_path == Path("custom/analytics.duckdb")
    assert db.sqlite_path == Path("custom/poly_oracle.db")
    assert db.chroma_path == Path("custom/chroma")


def test_settings_loads_structure() -> None:
    settings = Settings()
    assert settings.paper_trading is True
    assert isinstance(settings.llm, LLMSettings)
    assert isinstance(settings.risk, RiskSettings)
    assert isinstance(settings.data, DataSettings)
    assert isinstance(settings.polymarket, PolymarketSettings)
    assert isinstance(settings.database, DatabaseSettings)


def test_v1_specific_fields(monkeypatch) -> None:
    """v1 adds btc_market_id, loop_interval_minutes, debate_rounds."""
    monkeypatch.setenv("BTC_MARKET_ID", "test-mkt-999")
    monkeypatch.setenv("LOOP_INTERVAL_MINUTES", "30")
    monkeypatch.setenv("DEBATE_ROUNDS", "2")
    settings = Settings()
    assert settings.btc_market_id == "test-mkt-999"
    assert settings.loop_interval_minutes == 30
    assert settings.debate_rounds == 2


def test_risk_edge_thresholds() -> None:
    risk = RiskSettings()
    assert risk.max_position_pct == 0.10
    assert risk.max_daily_loss_pct == 0.10
    assert risk.min_edge == 0.08
    assert risk.min_confidence == 0.65
    assert risk.min_liquidity == 1000
