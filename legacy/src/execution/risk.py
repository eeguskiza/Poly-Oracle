"""
Risk Manager - Enforces risk limits before trade execution.
"""
from loguru import logger

from src.models import Trade, Position, RiskCheck
from config.settings import RiskSettings


class RiskManager:
    """
    Enforces risk management rules before allowing trades.

    Checks:
    - Daily loss limit
    - Maximum open positions
    - Single market exposure
    """

    def __init__(self, risk_settings: RiskSettings):
        """
        Initialize risk manager with settings.

        Args:
            risk_settings: Risk management settings
        """
        self.max_daily_loss = risk_settings.max_daily_loss
        self.max_open_positions = risk_settings.max_open_positions
        self.max_single_market_exposure = risk_settings.max_single_market_exposure

    def check(
        self,
        proposed_trade: Trade,
        current_positions: list[Position],
        daily_pnl: float,
        bankroll: float,
    ) -> RiskCheck:
        """
        Check if proposed trade passes all risk limits.

        Args:
            proposed_trade: Trade to evaluate
            current_positions: List of current open positions
            daily_pnl: Today's realized P&L (negative if loss)
            bankroll: Current bankroll

        Returns:
            RiskCheck with passed status and any violations
        """
        violations = []

        # Check 1: Daily loss limit
        if daily_pnl < 0:
            daily_loss_pct = abs(daily_pnl) / bankroll if bankroll > 0 else 0
            if daily_loss_pct >= self.max_daily_loss:
                violation = (
                    f"Daily loss limit exceeded: {daily_loss_pct:.1%} >= "
                    f"{self.max_daily_loss:.1%}"
                )
                violations.append(violation)
                logger.warning(violation)

        # Check 2: Max open positions
        if len(current_positions) >= self.max_open_positions:
            violation = (
                f"Max open positions reached: {len(current_positions)} >= "
                f"{self.max_open_positions}"
            )
            violations.append(violation)
            logger.warning(violation)

        # Check 3: Single market exposure
        # Check if we already have a position in this market
        existing_position = None
        for pos in current_positions:
            if pos.market_id == proposed_trade.market_id:
                existing_position = pos
                break

        if existing_position:
            # Calculate total exposure (existing + proposed)
            existing_exposure = abs(existing_position.amount_usd)
            proposed_exposure = abs(proposed_trade.amount_usd)
            total_exposure = existing_exposure + proposed_exposure

            exposure_pct = total_exposure / bankroll if bankroll > 0 else 0

            if exposure_pct > self.max_single_market_exposure:
                violation = (
                    f"Single market exposure limit exceeded: "
                    f"${total_exposure:.2f} ({exposure_pct:.1%}) > "
                    f"{self.max_single_market_exposure:.1%} of bankroll"
                )
                violations.append(violation)
                logger.warning(violation)

        # Determine if checks passed
        passed = len(violations) == 0

        if passed:
            logger.info("All risk checks passed")
        else:
            logger.warning(f"Risk checks failed: {len(violations)} violations")

        return RiskCheck(
            passed=passed,
            violations=violations,
            daily_loss_pct=abs(daily_pnl) / bankroll if bankroll > 0 and daily_pnl < 0 else 0,
            num_open_positions=len(current_positions),
            proposed_market_exposure=(
                existing_position.amount_usd + proposed_trade.amount_usd
                if existing_position
                else proposed_trade.amount_usd
            ),
        )
