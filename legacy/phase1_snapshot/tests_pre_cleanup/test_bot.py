"""Tests for the TradingBot core logic.

These tests verify config validation, edge computation, position settlement,
and the market context builder — all without hitting real APIs or LLMs.
"""

import pytest
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

from bot.config.settings import Settings
from bot.core import TradingBot
from bot.models.forecast import EdgeAnalysis, SimpleForecast
from bot.models.market import Market
from bot.exceptions import ConfigError


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_settings(**overrides) -> Settings:
    defaults = {
        "btc_market_id": "test-market-123",
        "paper_trading": True,
        "loop_interval_minutes": 1,
        "debate_rounds": 1,
    }
    defaults.update(overrides)
    return Settings(**defaults)


def _make_market(**overrides) -> Market:
    defaults = dict(
        id="test-market-123",
        question="Will BTC exceed $100k by end of March?",
        description="Resolves YES if BTC >= $100,000 on March 31.",
        market_type="binary",
        current_price=0.65,
        volume_24h=50000.0,
        liquidity=100000.0,
        resolution_date=datetime(2026, 3, 31, tzinfo=timezone.utc),
        created_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
        outcomes=["Yes", "No"],
        token_ids={"Yes": "tok-yes", "No": "tok-no"},
    )
    defaults.update(overrides)
    return Market(**defaults)


def _make_forecast(probability: float = 0.75, **overrides) -> SimpleForecast:
    defaults = dict(
        market_id="test-market-123",
        probability=probability,
        confidence_lower=probability - 0.10,
        confidence_upper=probability + 0.10,
        reasoning="Test reasoning",
        model_name="test-model",
        debate_rounds=1,
        bull_probabilities=[probability],
        bear_probabilities=[probability - 0.05],
    )
    defaults.update(overrides)
    return SimpleForecast(**defaults)


# ---------------------------------------------------------------------------
# Config validation
# ---------------------------------------------------------------------------

class TestValidateConfig:
    def test_missing_market_id_raises(self, monkeypatch):
        monkeypatch.setenv("BTC_MARKET_ID", "")
        settings = Settings()
        bot = TradingBot(settings)
        with pytest.raises(ConfigError, match="BTC_MARKET_ID"):
            bot._validate_config()

    def test_live_without_credentials_raises(self, monkeypatch):
        monkeypatch.setenv("PAPER_TRADING", "false")
        monkeypatch.delenv("POLYMARKET_API_KEY", raising=False)
        monkeypatch.delenv("POLYMARKET_API_SECRET", raising=False)
        monkeypatch.delenv("POLYMARKET_API_PASSPHRASE", raising=False)
        settings = Settings()
        bot = TradingBot(settings)
        with pytest.raises(ConfigError, match="Live trading requires"):
            bot._validate_config()

    def test_zero_bankroll_raises(self):
        settings = _make_settings()
        settings.risk.initial_bankroll = 0
        bot = TradingBot(settings)
        with pytest.raises(ConfigError, match="INITIAL_BANKROLL"):
            bot._validate_config()

    def test_max_bet_less_than_min_raises(self):
        settings = _make_settings()
        settings.risk.min_bet = 5.0
        settings.risk.max_bet = 1.0
        bot = TradingBot(settings)
        with pytest.raises(ConfigError, match="MAX_BET"):
            bot._validate_config()

    def test_valid_config_passes(self):
        bot = TradingBot(_make_settings())
        bot._validate_config()  # Should not raise


# ---------------------------------------------------------------------------
# Edge computation
# ---------------------------------------------------------------------------

class TestComputeEdge:
    def setup_method(self):
        self.bot = TradingBot(_make_settings())

    def test_buy_yes_when_forecast_above_market(self):
        forecast = _make_forecast(probability=0.80)
        market = _make_market(current_price=0.65)
        edge = self.bot._compute_edge(forecast, market)

        assert edge.direction == "BUY_YES"
        assert edge.raw_edge == pytest.approx(0.15, abs=0.01)
        assert edge.recommended_action == "TRADE"

    def test_buy_no_when_forecast_below_market(self):
        forecast = _make_forecast(probability=0.40)
        market = _make_market(current_price=0.65)
        edge = self.bot._compute_edge(forecast, market)

        assert edge.direction == "BUY_NO"
        assert edge.raw_edge == pytest.approx(-0.25, abs=0.01)
        assert edge.recommended_action == "TRADE"

    def test_skip_when_edge_too_small(self):
        forecast = _make_forecast(probability=0.66)
        market = _make_market(current_price=0.65)
        edge = self.bot._compute_edge(forecast, market)

        assert edge.recommended_action == "SKIP"
        assert "edge" in edge.reasoning.lower()

    def test_skip_when_liquidity_too_low(self):
        forecast = _make_forecast(probability=0.80)
        market = _make_market(current_price=0.65, liquidity=100)
        edge = self.bot._compute_edge(forecast, market)

        assert edge.recommended_action == "SKIP"
        assert "liquidity" in edge.reasoning.lower()


# ---------------------------------------------------------------------------
# Position settlement
# ---------------------------------------------------------------------------

class TestSettlePosition:
    def test_buy_yes_wins(self, tmp_db):
        tmp_db.upsert_position({
            "market_id": "mkt-1",
            "direction": "BUY_YES",
            "num_shares": 2.0,
            "amount_usd": 1.0,
            "avg_entry_price": 0.50,
            "current_price": 0.50,
            "unrealized_pnl": 0.0,
            "updated_at": datetime.now(timezone.utc).isoformat(),
        })

        pnl = TradingBot._settle_position(tmp_db, "mkt-1", outcome=True)
        assert pnl == pytest.approx(1.0)  # 2 shares * $1 - $1 cost

        pos = tmp_db.get_position("mkt-1")
        assert pos["num_shares"] == 0.0

    def test_buy_yes_loses(self, tmp_db):
        tmp_db.upsert_position({
            "market_id": "mkt-2",
            "direction": "BUY_YES",
            "num_shares": 2.0,
            "amount_usd": 1.0,
            "avg_entry_price": 0.50,
            "current_price": 0.50,
            "unrealized_pnl": 0.0,
            "updated_at": datetime.now(timezone.utc).isoformat(),
        })

        pnl = TradingBot._settle_position(tmp_db, "mkt-2", outcome=False)
        assert pnl == pytest.approx(-1.0)  # lose entire cost

    def test_buy_no_wins(self, tmp_db):
        tmp_db.upsert_position({
            "market_id": "mkt-3",
            "direction": "BUY_NO",
            "num_shares": 2.0,
            "amount_usd": 1.0,
            "avg_entry_price": 0.50,
            "current_price": 0.50,
            "unrealized_pnl": 0.0,
            "updated_at": datetime.now(timezone.utc).isoformat(),
        })

        pnl = TradingBot._settle_position(tmp_db, "mkt-3", outcome=False)
        assert pnl == pytest.approx(1.0)  # NO wins -> $1/share

    def test_no_position_returns_zero(self, tmp_db):
        pnl = TradingBot._settle_position(tmp_db, "nonexistent", outcome=True)
        assert pnl == 0.0


# ---------------------------------------------------------------------------
# Context builder
# ---------------------------------------------------------------------------

class TestBuildContext:
    def test_context_contains_market_data(self):
        market = _make_market()
        context = TradingBot._build_market_context(market)

        assert "BTC" in context
        assert "$100k" in context or "100,000" in context or "100k" in context
        assert "65.0%" in context
        assert "Liquidity" in context
