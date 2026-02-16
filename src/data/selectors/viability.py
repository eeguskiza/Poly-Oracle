"""
Generic viability selector — multi-factor scoring with no-trade rules.
Used by the ``auto`` trading mode.
"""
from __future__ import annotations

from typing import Callable

from config.settings import Settings
from src.data.sources.polymarket import PolymarketClient
from src.models import Market


# Each rule is (name, predicate).  True means the market is excluded.
_Rule = tuple[str, Callable[[Market, Settings], bool]]

NO_TRADE_RULES: list[_Rule] = [
    ("liquidity_too_low", lambda m, s: m.liquidity < s.risk.min_liquidity),
    ("price_extreme", lambda m, s: m.current_price < 0.03 or m.current_price > 0.97),
    ("resolves_too_soon", lambda m, s: m.days_until_resolution < 0.5),
    ("resolves_too_far", lambda m, s: m.days_until_resolution > 180),
    ("volume_dead", lambda m, s: m.volume_24h < 100),
]


class ViabilitySelector:
    """Rank markets by composite viability, filtering out non-tradable ones."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    async def select_markets(
        self,
        polymarket: PolymarketClient,
        top_n: int = 5,
    ) -> list[tuple[Market, float, list[str]]]:
        """Return ``(market, score, exclusion_reasons)`` tuples.

        Markets with an empty ``exclusion_reasons`` list are considered viable.
        The list is sorted by score descending, viable markets first.
        """
        all_markets = await polymarket.get_active_markets(limit=200)
        results: list[tuple[Market, float, list[str]]] = []
        for m in all_markets:
            reasons = self._check_rules(m)
            score = self._score(m)
            results.append((m, score, reasons))
        # Viable first, then by score descending
        results.sort(key=lambda t: (len(t[2]) == 0, t[1]), reverse=True)
        return results[:top_n]

    def _check_rules(self, market: Market) -> list[str]:
        return [name for name, pred in NO_TRADE_RULES if pred(market, self.settings)]

    def _score(self, market: Market) -> float:
        """Composite score: liquidity, volume, days, spread, vol/liq ratio."""
        liq_score = min(market.liquidity / 50_000, 1.0)
        vol_score = min(market.volume_24h / 10_000, 1.0)
        days = market.days_until_resolution
        days_score = 1.0 if 1 <= days <= 60 else 0.3
        price = market.current_price
        spread_score = 1.0 - 2 * abs(price - 0.5)
        ratio = market.volume_24h / max(market.liquidity, 1)
        vol_liq_score = min(ratio / 0.5, 1.0)
        return (
            0.30 * liq_score
            + 0.25 * vol_score
            + 0.20 * days_score
            + 0.15 * spread_score
            + 0.10 * vol_liq_score
        )
