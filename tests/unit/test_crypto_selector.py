import pytest
from datetime import datetime, timezone, timedelta
from unittest.mock import AsyncMock

from src.data.selectors.crypto import CryptoSelector
from src.models import Market


def _make_market(
    question: str = "Will BTC reach $100k?",
    description: str = "Bitcoin price prediction",
    liquidity: float = 25_000,
    volume_24h: float = 5_000,
    current_price: float = 0.50,
    days_ahead: float = 30,
) -> Market:
    now = datetime.now(timezone.utc)
    return Market(
        id="test-market-1",
        question=question,
        description=description,
        market_type="crypto",
        current_price=current_price,
        volume_24h=volume_24h,
        liquidity=liquidity,
        resolution_date=now + timedelta(days=days_ahead),
        created_at=now - timedelta(days=5),
        outcomes=["Yes", "No"],
        token_ids={"Yes": "tok_yes", "No": "tok_no"},
    )


class TestIsCrypto:
    def test_detects_bitcoin(self):
        sel = CryptoSelector()
        m = _make_market(question="Will Bitcoin reach $100k?", description="")
        assert sel._is_crypto(m) is True

    def test_detects_btc(self):
        sel = CryptoSelector()
        m = _make_market(question="BTC above 80k by March?", description="")
        assert sel._is_crypto(m) is True

    def test_detects_ethereum_in_description(self):
        sel = CryptoSelector()
        m = _make_market(question="Price above X?", description="Ethereum forecast")
        assert sel._is_crypto(m) is True

    def test_rejects_non_crypto(self):
        sel = CryptoSelector()
        m = _make_market(question="Will Biden win?", description="US politics")
        assert sel._is_crypto(m) is False

    def test_case_insensitive(self):
        sel = CryptoSelector()
        m = _make_market(question="SOLANA to moon?", description="")
        assert sel._is_crypto(m) is True


class TestViabilityScore:
    def test_perfect_market(self):
        sel = CryptoSelector()
        m = _make_market(liquidity=50_000, volume_24h=10_000, current_price=0.50, days_ahead=30)
        score = sel._viability_score(m)
        assert score == pytest.approx(1.0, abs=0.05)

    def test_low_liquidity_penalized(self):
        sel = CryptoSelector()
        m_high = _make_market(liquidity=50_000, volume_24h=10_000, current_price=0.50)
        m_low = _make_market(liquidity=5_000, volume_24h=10_000, current_price=0.50)
        assert sel._viability_score(m_high) > sel._viability_score(m_low)

    def test_extreme_price_penalized(self):
        sel = CryptoSelector()
        m_mid = _make_market(current_price=0.50)
        m_extreme = _make_market(current_price=0.95)
        assert sel._viability_score(m_mid) > sel._viability_score(m_extreme)

    def test_resolution_too_far(self):
        sel = CryptoSelector()
        m_ok = _make_market(days_ahead=30)
        m_far = _make_market(days_ahead=90)
        assert sel._viability_score(m_ok) > sel._viability_score(m_far)


class TestSelectMarkets:
    @pytest.mark.asyncio
    async def test_filters_and_ranks(self):
        sel = CryptoSelector()
        crypto_market = _make_market(
            question="Will BTC reach $100k?", liquidity=40_000, volume_24h=8_000
        )
        politics_market = _make_market(
            question="Will Biden win?", description="US politics",
            liquidity=60_000, volume_24h=15_000,
        )
        mock_poly = AsyncMock()
        mock_poly.get_active_markets = AsyncMock(return_value=[crypto_market, politics_market])

        result = await sel.select_markets(mock_poly, top_n=5)

        assert len(result) == 1
        assert result[0].question == "Will BTC reach $100k?"

    @pytest.mark.asyncio
    async def test_respects_top_n(self):
        sel = CryptoSelector()
        markets = [
            _make_market(question=f"BTC market {i}", liquidity=10_000 * (i + 1))
            for i in range(10)
        ]
        mock_poly = AsyncMock()
        mock_poly.get_active_markets = AsyncMock(return_value=markets)

        result = await sel.select_markets(mock_poly, top_n=3)
        assert len(result) == 3

    @pytest.mark.asyncio
    async def test_empty_markets(self):
        sel = CryptoSelector()
        mock_poly = AsyncMock()
        mock_poly.get_active_markets = AsyncMock(return_value=[])

        result = await sel.select_markets(mock_poly, top_n=5)
        assert result == []
