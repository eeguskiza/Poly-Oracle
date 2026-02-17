"""
Live Trading Executor — executes real trades on Polymarket via py-clob-client.
"""
from __future__ import annotations

from datetime import datetime, timezone
from loguru import logger

from bot.config.settings import Settings
from bot.state.store import SQLiteClient
from bot.execution.sizer import PositionSizer
from bot.risk.manager import RiskManager
from bot.models import Market, Trade, ExecutionResult, EdgeAnalysis

try:
    from py_clob_client.client import ClobClient
    from py_clob_client.clob_types import OrderArgs, OrderType, ApiCreds

    HAS_CLOB = True
except ImportError:
    HAS_CLOB = False


class LiveTradingExecutor:
    """Execute real trades on Polymarket using the CLOB API."""

    def __init__(
        self,
        sqlite: SQLiteClient,
        sizer: PositionSizer,
        risk: RiskManager,
        settings: Settings,
    ) -> None:
        self.sqlite = sqlite
        self.sizer = sizer
        self.risk = risk
        self.settings = settings
        self._clob: ClobClient | None = None

    def _ensure_clob(self) -> ClobClient:
        """Lazily initialise the CLOB client (requires credentials)."""
        if self._clob is not None:
            return self._clob

        if not HAS_CLOB:
            raise RuntimeError(
                "py-clob-client is not installed. "
                "Run: pip install py-clob-client"
            )

        ps = self.settings.polymarket
        if not all([ps.api_key, ps.api_secret, ps.api_passphrase]):
            raise RuntimeError(
                "Polymarket API credentials are missing. "
                "Set POLYMARKET_API_KEY, POLYMARKET_API_SECRET, "
                "and POLYMARKET_API_PASSPHRASE in your .env file."
            )

        creds = ApiCreds(
            api_key=ps.api_key,
            api_secret=ps.api_secret,
            api_passphrase=ps.api_passphrase,
        )
        self._clob = ClobClient(
            host=ps.clob_url,
            key=ps.api_key,
            chain_id=137,  # Polygon
            creds=creds,
        )
        return self._clob

    async def execute(
        self,
        edge_analysis: EdgeAnalysis,
        calibrated_probability: float,
        market: Market,
        bankroll: float,
    ) -> ExecutionResult | None:
        """Execute a live trade — same interface as PaperTradingExecutor.execute()."""
        if edge_analysis.recommended_action == "SKIP":
            logger.info(f"Skipping trade for {market.id}: recommendation is SKIP")
            return None

        # Position sizing
        position_size = self.sizer.calculate(
            bankroll=bankroll,
            our_prob=calibrated_probability,
            market_prob=market.current_price,
            direction=edge_analysis.direction,
        )

        if position_size.amount_usd == 0:
            logger.info(f"Position size zero, skipping trade for {market.id}")
            return None

        # Risk check
        proposed_trade = Trade(
            id="",
            market_id=market.id,
            direction=edge_analysis.direction,
            amount_usd=position_size.amount_usd,
            num_shares=position_size.num_shares,
            entry_price=market.current_price,
            timestamp=datetime.now(timezone.utc),
            status="PENDING",
        )

        current_positions = self.sqlite.get_open_positions()
        daily_stats = self.sqlite.get_daily_stats(datetime.now(timezone.utc).date())
        daily_pnl = daily_stats.get("realized_pnl", 0.0) if daily_stats else 0.0

        risk_check = self.risk.check(
            proposed_trade=proposed_trade,
            current_positions=current_positions,
            daily_pnl=daily_pnl,
            bankroll=bankroll,
        )

        if not risk_check.passed:
            logger.warning(
                f"Live trade rejected by risk manager for {market.id}. "
                f"Violations: {', '.join(risk_check.violations)}"
            )
            return ExecutionResult(
                success=False,
                trade_id=None,
                message=f"Risk check failed: {'; '.join(risk_check.violations)}",
                risk_check=risk_check,
            )

        # Submit order to CLOB
        clob = self._ensure_clob()

        # Determine token_id based on direction
        if edge_analysis.direction == "BUY_YES":
            token_id = market.token_ids.get("Yes", "")
        else:
            token_id = market.token_ids.get("No", "")

        if not token_id:
            return ExecutionResult(
                success=False,
                message=f"No token_id found for {edge_analysis.direction} on {market.id}",
            )

        try:
            order_kwargs = dict(
                price=round(market.current_price, 2),
                size=round(position_size.num_shares, 2),
                side="BUY",
                token_id=token_id,
            )

            if HAS_CLOB:
                order_args = OrderArgs(**order_kwargs)
            else:
                # Fallback for environments without py-clob-client
                # (only reachable if _clob was injected externally, e.g. tests)
                order_args = order_kwargs

            resp = clob.create_and_post_order(order_args)
            order_id = resp.get("orderID", resp.get("order_id", ""))

            logger.info(
                f"Live order submitted: {edge_analysis.direction} "
                f"{position_size.num_shares:.2f} shares of {market.id} "
                f"@ {market.current_price:.2f} — order_id={order_id}"
            )

            # Record in SQLite as PENDING
            proposed_trade.status = "PENDING"
            trade_id = self.sqlite.insert_trade(proposed_trade.to_db_dict())

            return ExecutionResult(
                success=True,
                trade_id=trade_id,
                message=f"Live order submitted: {order_id}",
                risk_check=risk_check,
            )

        except Exception as exc:
            logger.error(f"Live order failed for {market.id}: {exc}")
            return ExecutionResult(
                success=False,
                message=f"Order submission failed: {exc}",
                risk_check=risk_check,
            )
