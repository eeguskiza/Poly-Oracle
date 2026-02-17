"""Primary/fallback data feed placeholders for startup pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from bot.errors import DependencyInitializationError


@dataclass(frozen=True)
class TargetMarket:
    market_id: str
    symbol: str
    source: str


class PrimaryPolymarketFeed:
    """Placeholder primary feed implementation."""

    def __init__(self, base_url: str, market_id: str) -> None:
        if not base_url:
            raise DependencyInitializationError("Primary feed requires polymarket base_url.")
        self.base_url = base_url
        self.market_id = market_id

    def warmup(self, cycles: int) -> list[dict[str, Any]]:
        return [
            {
                "cycle": idx + 1,
                "market_id": self.market_id,
                "feed": "primary",
                "status": "ok",
            }
            for idx in range(cycles)
        ]


class FallbackPolymarketFeed:
    """Placeholder fallback feed implementation."""

    def __init__(self, base_url: str, market_id: str) -> None:
        if not base_url:
            raise DependencyInitializationError("Fallback feed requires polymarket base_url.")
        self.base_url = base_url
        self.market_id = market_id

    def warmup(self, cycles: int) -> list[dict[str, Any]]:
        return [
            {
                "cycle": idx + 1,
                "market_id": self.market_id,
                "feed": "fallback",
                "status": "ok",
            }
            for idx in range(cycles)
        ]
