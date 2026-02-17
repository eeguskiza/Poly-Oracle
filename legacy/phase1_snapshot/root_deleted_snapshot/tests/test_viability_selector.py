import pytest
from datetime import datetime, timezone, timedelta
from unittest.mock import AsyncMock, Mock

from src.data.selectors.viability import ViabilitySelector, NO_TRADE_RULES
from src.models import Market


def _make_settings(**overrides):
    s = Mock()
    s.risk = Mock()
    s.risk.min_liquidity = overrides.get("min_liquidity", 1000.0)
    return s


def _make_market(
    question: str = "Will X happen?",
    liquidity: float = 25_000,
    volume_24h: float = 5_000,
    current_price: float = 0.50,
    days_ahead: float = 30,
) -> Market:
    now = datetime.now(timezone.utc)
    return Market(
        id="test-market",
        question=question,
        description="desc",
        market_type="general",
        current_price=current_price,
        volume_24h=volume_24h,
        liquidity=liquidity,
        resolution_date=now + timedelta(days=days_ahead),
        created_at=now - timedelta(days=5),
        outcomes=["Yes", "No"],
        token_ids={"Yes": "tok_yes", "No": "tok_no"},
    )


class TestNoTradeRules:
    def test_liquidity_too_low(self):
        settings = _make_settings(min_liquidity=5000)
        m = _make_market(liquidity=100)
        sel = ViabilitySelector(settings)
        reasons = sel._check_rules(m)
        assert "liquidity_too_low" in reasons

    def test_price_extreme_high(self):
        settings = _make_settings()
        m = _make_market(current_price=0.98)
        sel = ViabilitySelector(settings)
        reasons = sel._check_rules(m)
        assert "price_extreme" in reasons

    def test_price_extreme_low(self):
        settings = _make_settings()
        m = _make_market(current_price=0.02)
        sel = ViabilitySelector(settings)
        reasons = sel._check_rules(m)
        assert "price_extreme" in reasons

    def test_resolves_too_soon(self):
        settings = _make_settings()
        m = _make_market(days_ahead=0.1)
        sel = ViabilitySelector(settings)
        reasons = sel._check_rules(m)
        assert "resolves_too_soon" in reasons

    def test_resolves_too_far(self):
        settings = _make_settings()
        m = _make_market(days_ahead=200)
        sel = ViabilitySelector(settings)
        reasons = sel._check_rules(m)
        assert "resolves_too_far" in reasons

    def test_volume_dead(self):
        settings = _make_settings()
        m = _make_market(volume_24h=50)
        sel = ViabilitySelector(settings)
        reasons = sel._check_rules(m)
        assert "volume_dead" in reasons

    def test_viable_market_has_no_exclusions(self):
        settings = _make_settings()
        m = _make_market(liquidity=10_000, volume_24h=5_000, current_price=0.50, days_ahead=30)
        sel = ViabilitySelector(settings)
        reasons = sel._check_rules(m)
        assert reasons == []


class TestScore:
    def test_perfect_score(self):
        settings = _make_settings()
        sel = ViabilitySelector(settings)
        m = _make_market(liquidity=50_000, volume_24h=10_000, current_price=0.50, days_ahead=30)
        score = sel._score(m)
        assert score > 0.9

    def test_score_between_zero_and_one(self):
        settings = _make_settings()
        sel = ViabilitySelector(settings)
        m = _make_market(liquidity=1_000, volume_24h=200, current_price=0.10, days_ahead=100)
        score = sel._score(m)
        assert 0.0 <= score <= 1.0


class TestSelectMarkets:
    @pytest.mark.asyncio
    async def test_viable_first(self):
        settings = _make_settings()
        sel = ViabilitySelector(settings)

        viable = _make_market(liquidity=20_000, volume_24h=5_000, current_price=0.50, days_ahead=30)
        excluded = _make_market(liquidity=20_000, volume_24h=50, current_price=0.50, days_ahead=30)

        mock_poly = AsyncMock()
        mock_poly.get_active_markets = AsyncMock(return_value=[excluded, viable])

        results = await sel.select_markets(mock_poly, top_n=10)
        # viable (no reasons) should come first
        assert len(results) >= 1
        first_market, first_score, first_reasons = results[0]
        assert first_reasons == []

    @pytest.mark.asyncio
    async def test_empty_markets(self):
        settings = _make_settings()
        sel = ViabilitySelector(settings)
        mock_poly = AsyncMock()
        mock_poly.get_active_markets = AsyncMock(return_value=[])

        results = await sel.select_markets(mock_poly, top_n=5)
        assert results == []
