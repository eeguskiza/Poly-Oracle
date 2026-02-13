"""
Market Resolver - Automatically resolves markets and calculates P&L.
"""
import inspect
from datetime import datetime, timezone
from typing import Any
from loguru import logger

from src.data.sources.polymarket import PolymarketClient
from src.data.storage.sqlite_client import SQLiteClient
from src.data.storage.duckdb_client import DuckDBClient
from src.calibration.feedback import FeedbackLoop


class MarketResolver:
    """
    Automatically resolves closed markets and calculates P&L.

    Workflow:
    1. Get open positions from SQLite
    2. Get unresolved forecasts from DuckDB
    3. Check resolution status via Polymarket API
    4. Calculate P&L for resolved positions
    5. Update positions and forecasts
    6. Update calibration via feedback loop
    """

    def __init__(
        self,
        polymarket: PolymarketClient,
        feedback: FeedbackLoop,
        sqlite: SQLiteClient,
        duckdb: DuckDBClient,
    ):
        """
        Initialize market resolver.

        Args:
            polymarket: Polymarket API client
            feedback: Feedback loop for calibration
            sqlite: SQLite client for positions
            duckdb: DuckDB client for forecasts
        """
        self.polymarket = polymarket
        self.feedback = feedback
        self.sqlite = sqlite
        self.duckdb = duckdb

    async def run_resolution_cycle(self) -> dict[str, Any]:
        """
        Run one resolution cycle to check and resolve markets.

        Returns:
            Dict with stats: {"checked": int, "resolved": int, "pnl": float}
        """
        # Step 1: Get market IDs to check
        market_ids = set()

        # Get from open positions
        open_positions = self.sqlite.get_open_positions()
        for pos in open_positions:
            market_ids.add(pos["market_id"])

        # Get from unresolved forecasts
        unresolved_forecasts = self.duckdb.get_unresolved_forecasts()
        for forecast in unresolved_forecasts:
            market_ids.add(forecast["market_id"])

        market_ids = list(market_ids)

        if not market_ids:
            logger.debug("No markets to check for resolution")
            return {"checked": 0, "resolved": 0, "pnl": 0.0}

        logger.info(f"Checking resolution status for {len(market_ids)} markets")

        # Step 2: Check resolutions via API
        resolved_markets = await self.polymarket.check_resolutions(market_ids)

        if not resolved_markets:
            logger.info("No markets resolved")
            return {"checked": len(market_ids), "resolved": 0, "pnl": 0.0}

        # Step 3: Process each resolved market
        total_pnl = 0.0

        for market_id, outcome in resolved_markets.items():
            pnl = await self._resolve_market(market_id, outcome)
            total_pnl += pnl

        logger.info(
            f"Resolution cycle complete: checked {len(market_ids)}, "
            f"resolved {len(resolved_markets)}, total P&L: {total_pnl:+.2f} EUR"
        )

        return {
            "checked": len(market_ids),
            "resolved": len(resolved_markets),
            "pnl": total_pnl,
        }

    async def _resolve_market(self, market_id: str, outcome: bool) -> float:
        """
        Resolve a single market and calculate P&L.

        Args:
            market_id: Market ID
            outcome: True if YES won, False if NO won

        Returns:
            Realized P&L for this market
        """
        # Check if we have a position in this market
        position = self.sqlite.get_position(market_id)

        pnl = 0.0

        if position and position["num_shares"] > 0:
            # Calculate P&L based on direction and outcome
            direction = position["direction"]
            num_shares = position["num_shares"]
            amount_usd = position["amount_usd"]

            if direction == "BUY_YES":
                if outcome:  # YES won
                    # Win $1 per share
                    pnl = (num_shares * 1.0) - amount_usd
                else:  # NO won
                    # Lose everything
                    pnl = -amount_usd
            elif direction == "BUY_NO":
                if not outcome:  # NO won
                    # Win $1 per share
                    pnl = (num_shares * 1.0) - amount_usd
                else:  # YES won
                    # Lose everything
                    pnl = -amount_usd

            outcome_str = "YES" if outcome else "NO"
            logger.info(
                f"RESOLVED: {market_id[:8]}... -> {outcome_str} | "
                f"{direction} {num_shares:.2f} shares | P&L: {pnl:+.2f} EUR"
            )

            # Update position - set num_shares to 0 to mark as closed
            updated_position = {
                **position,
                "num_shares": 0.0,
                "amount_usd": 0.0,
                "unrealized_pnl": 0.0,
                "updated_at": datetime.now(timezone.utc).isoformat(),
            }
            self.sqlite.upsert_position(updated_position)

            # Update daily stats with realized P&L
            today = datetime.now(timezone.utc).date().isoformat()
            stats = self.sqlite.get_daily_stats(today)

            if stats:
                stats["ending_bankroll"] = stats["ending_bankroll"] + pnl
                stats["net_pnl"] = stats["net_pnl"] + pnl
                stats["gross_pnl"] = stats["gross_pnl"] + pnl
                if pnl > 0:
                    stats["trades_won"] = stats["trades_won"] + 1
            else:
                # Create new daily stats
                current_bankroll = self.sqlite.get_current_bankroll()
                stats = {
                    "date": today,
                    "starting_bankroll": current_bankroll,
                    "ending_bankroll": current_bankroll + pnl,
                    "trades_executed": 1,
                    "trades_won": 1 if pnl > 0 else 0,
                    "gross_pnl": pnl,
                    "fees_paid": 0.0,
                    "net_pnl": pnl,
                }

            self.sqlite.upsert_daily_stats(stats)

        # Process resolution through feedback loop for calibration
        resolution_result = self.feedback.process_resolution(market_id, outcome)
        if inspect.isawaitable(resolution_result):
            await resolution_result

        return pnl
