"""
Crypto market selector — filters Polymarket for BTC/ETH/SOL and crypto markets,
then ranks them by a multi-factor viability score.
"""
from __future__ import annotations

from src.data.sources.polymarket import PolymarketClient
from src.models import Market


class CryptoSelector:
    """Select and rank crypto-related prediction markets."""

    CRYPTO_KEYWORDS = [
        "bitcoin", "btc", "ethereum", "eth", "solana", "sol",
        "crypto", "cryptocurrency", "defi", "blockchain",
        "binance", "coinbase", "altcoin", "stablecoin",
    ]

    async def select_markets(
        self, polymarket: PolymarketClient, top_n: int = 5
    ) -> list[Market]:
        """Filter active markets for crypto keywords and rank by viability."""
        all_markets = await polymarket.get_active_markets(limit=200)
        crypto_markets = [m for m in all_markets if self._is_crypto(m)]
        scored = [(m, self._viability_score(m)) for m in crypto_markets]
        scored.sort(key=lambda x: x[1], reverse=True)
        return [m for m, _ in scored[:top_n]]

    def _is_crypto(self, market: Market) -> bool:
        """Return True if market question/description mentions crypto terms."""
        text = (market.question + " " + market.description).lower()
        return any(kw in text for kw in self.CRYPTO_KEYWORDS)

    def _viability_score(self, market: Market) -> float:
        """Multi-factor score: liquidity, volume, time-to-resolution, spread."""
        liq_score = min(market.liquidity / 50_000, 1.0)
        vol_score = min(market.volume_24h / 10_000, 1.0)
        days = market.days_until_resolution
        days_score = 1.0 if 1 <= days <= 60 else 0.3
        price = market.current_price
        spread_score = 1.0 - 2 * abs(price - 0.5)
        return (
            0.35 * liq_score
            + 0.25 * vol_score
            + 0.20 * days_score
            + 0.20 * spread_score
        )
