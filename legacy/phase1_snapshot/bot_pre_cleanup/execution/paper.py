"""
Paper Trading Executor - Simulates trade execution without real money.
"""
from datetime import datetime, timezone
from typing import Literal
from loguru import logger

from bot.state.store import SQLiteClient
from bot.execution.sizer import PositionSizer
from bot.risk.manager import RiskManager
from bot.models import SimpleForecast, Market, Trade, ExecutionResult, EdgeAnalysis


class PaperTradingExecutor:
    """
    Simulates trade execution for paper trading.

    Manages the full execution flow:
    1. Position sizing
    2. Risk checks
    3. Trade execution (simulated)
    4. Position tracking
    """

    def __init__(
        self,
        sqlite: SQLiteClient,
        sizer: PositionSizer,
        risk: RiskManager,
    ):
        """
        Initialize paper trading executor.

        Args:
            sqlite: SQLite client for storing trades/positions
            sizer: Position sizer for calculating bet sizes
            risk: Risk manager for enforcing limits
        """
        self.sqlite = sqlite
        self.sizer = sizer
        self.risk = risk

    async def execute(
        self,
        edge_analysis: EdgeAnalysis,
        calibrated_probability: float,
        market: Market,
        bankroll: float,
    ) -> ExecutionResult | None:
        """
        Execute a paper trade based on forecast and edge analysis.

        Args:
            edge_analysis: Edge analysis with recommendation
            calibrated_probability: Our calibrated forecast
            market: Market to trade
            bankroll: Current bankroll

        Returns:
            ExecutionResult if trade executed, None if skipped
        """
        # Check if we should skip
        if edge_analysis.recommended_action == "SKIP":
            logger.info(f"Skipping trade for {market.id}: recommendation is SKIP")
            return None

        logger.info(
            f"Attempting to execute {edge_analysis.direction} trade for {market.id}"
        )

        # Calculate position size
        position_size = self.sizer.calculate(
            bankroll=bankroll,
            our_prob=calibrated_probability,
            market_prob=market.current_price,
            direction=edge_analysis.direction,
        )

        # Check if amount is zero (below minimum)
        if position_size.amount_usd == 0:
            logger.info(
                f"Position size is zero (below minimum), skipping trade for {market.id}"
            )
            return None

        # Create proposed trade
        proposed_trade = Trade(
            id="",  # Will be set by SQLite
            market_id=market.id,
            direction=edge_analysis.direction,
            amount_usd=position_size.amount_usd,
            num_shares=position_size.num_shares,
            entry_price=market.current_price,
            timestamp=datetime.now(timezone.utc),
            status="PENDING",
        )

        # Get current positions and daily stats for risk check
        current_positions = self.sqlite.get_open_positions()
        daily_stats = self.sqlite.get_daily_stats(datetime.now(timezone.utc).date())
        daily_pnl = daily_stats.get("realized_pnl", 0.0) if daily_stats else 0.0

        # Risk check
        risk_check = self.risk.check(
            proposed_trade=proposed_trade,
            current_positions=current_positions,
            daily_pnl=daily_pnl,
            bankroll=bankroll,
        )

        if not risk_check.passed:
            logger.warning(
                f"Trade rejected by risk manager for {market.id}. "
                f"Violations: {', '.join(risk_check.violations)}"
            )
            return ExecutionResult(
                success=False,
                trade_id=None,
                message=f"Risk check failed: {'; '.join(risk_check.violations)}",
                risk_check=risk_check,
            )

        # Execute trade (paper)
        # In paper trading, we immediately "fill" at current market price
        proposed_trade.status = "FILLED"

        # Insert trade into database
        trade_id = self.sqlite.insert_trade(proposed_trade.to_db_dict())
        proposed_trade.id = trade_id

        logger.info(
            f"Paper trade executed: {edge_analysis.direction} {position_size.num_shares:.2f} shares "
            f"of {market.id} at ${market.current_price:.2f} for ${position_size.amount_usd:.2f}"
        )

        # Update or create position
        existing_position = None
        for pos in current_positions:
            if pos.market_id == market.id:
                existing_position = pos
                break

        if existing_position:
            # Add to existing position
            new_shares = existing_position.num_shares + position_size.num_shares
            new_amount = existing_position.amount_usd + position_size.amount_usd
            new_avg_price = new_amount / new_shares if new_shares > 0 else market.current_price

            self.sqlite.upsert_position({
                "market_id": market.id,
                "direction": edge_analysis.direction,
                "num_shares": new_shares,
                "amount_usd": new_amount,
                "avg_entry_price": new_avg_price,
                "current_price": market.current_price,
                "unrealized_pnl": (market.current_price - new_avg_price) * new_shares
                if edge_analysis.direction == "BUY_YES"
                else (new_avg_price - market.current_price) * new_shares,
                "updated_at": datetime.now(timezone.utc).isoformat(),
            })

            logger.info(
                f"Updated position for {market.id}: "
                f"{new_shares:.2f} shares @ ${new_avg_price:.2f}"
            )
        else:
            # Create new position
            unrealized_pnl = 0.0  # No P&L yet at entry

            self.sqlite.upsert_position({
                "market_id": market.id,
                "direction": edge_analysis.direction,
                "num_shares": position_size.num_shares,
                "amount_usd": position_size.amount_usd,
                "avg_entry_price": market.current_price,
                "current_price": market.current_price,
                "unrealized_pnl": unrealized_pnl,
                "updated_at": datetime.now(timezone.utc).isoformat(),
            })

            logger.info(
                f"Created new position for {market.id}: "
                f"{position_size.num_shares:.2f} shares @ ${market.current_price:.2f}"
            )

        return ExecutionResult(
            success=True,
            trade_id=trade_id,
            message=f"Trade executed successfully: {edge_analysis.direction} {position_size.num_shares:.2f} shares",
            risk_check=risk_check,
        )

    async def get_portfolio_summary(self, current_prices: dict[str, float] | None = None) -> dict:
        """
        Get summary of current portfolio state.

        Args:
            current_prices: Optional dict of market_id -> current_price for P&L calculation

        Returns:
            Dict with positions, total P&L, and bankroll info
        """
        positions = self.sqlite.get_open_positions()

        # Update unrealized P&L with current prices if provided
        total_unrealized_pnl = 0.0
        position_summaries = []

        for pos in positions:
            current_price = (
                current_prices.get(pos.market_id, pos.current_price)
                if current_prices
                else pos.current_price
            )

            # Calculate unrealized P&L
            if pos.direction == "BUY_YES":
                unrealized_pnl = (current_price - pos.avg_entry_price) * pos.num_shares
            else:  # BUY_NO
                unrealized_pnl = (pos.avg_entry_price - current_price) * pos.num_shares

            total_unrealized_pnl += unrealized_pnl

            position_summaries.append({
                "market_id": pos.market_id,
                "direction": pos.direction,
                "num_shares": pos.num_shares,
                "avg_entry_price": pos.avg_entry_price,
                "current_price": current_price,
                "unrealized_pnl": unrealized_pnl,
                "unrealized_pnl_pct": (unrealized_pnl / pos.amount_usd * 100)
                if pos.amount_usd > 0
                else 0,
            })

        # Get current bankroll
        current_bankroll = self.sqlite.get_current_bankroll()

        # Get daily stats
        today_stats = self.sqlite.get_daily_stats(datetime.now(timezone.utc).date())
        realized_pnl_today = today_stats.get("realized_pnl", 0.0) if today_stats else 0.0

        return {
            "num_positions": len(positions),
            "positions": position_summaries,
            "total_unrealized_pnl": total_unrealized_pnl,
            "realized_pnl_today": realized_pnl_today,
            "current_bankroll": current_bankroll,
            "total_value": current_bankroll + total_unrealized_pnl,
        }
