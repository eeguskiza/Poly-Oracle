"""
Position Sizer - Calculates optimal position sizes using Kelly criterion.
"""
from typing import Literal
from loguru import logger

from bot.models import PositionSize
from bot.config.settings import RiskSettings


class PositionSizer:
    """
    Calculates position sizes using Kelly criterion with conservative fraction.

    Uses fractional Kelly (15% of full Kelly) for risk management with small bankroll.
    """

    def __init__(self, risk_settings: RiskSettings):
        """
        Initialize position sizer with risk parameters.

        Args:
            risk_settings: Risk management settings
        """
        self.min_bet = risk_settings.min_bet
        self.max_bet = risk_settings.max_bet
        self.max_bankroll_pct = risk_settings.max_position_pct
        self.kelly_fraction = 0.15  # Very conservative for 50 EUR bankroll

    def calculate(
        self,
        bankroll: float,
        our_prob: float,
        market_prob: float,
        direction: Literal["BUY_YES", "BUY_NO"],
    ) -> PositionSize:
        """
        Calculate optimal position size using Kelly criterion.

        Kelly formula: f* = (bp - q) / b
        where:
        - b = (1/market_prob) - 1  (odds)
        - p = our_prob  (win probability)
        - q = 1 - p  (loss probability)

        For BUY_NO, we invert probabilities.

        Args:
            bankroll: Current bankroll in USD
            our_prob: Our forecast probability (0-1)
            market_prob: Market price probability (0-1)
            direction: "BUY_YES" or "BUY_NO"

        Returns:
            PositionSize with calculated amount and metadata
        """
        # Validate inputs
        if bankroll <= 0:
            raise ValueError(f"Bankroll must be positive, got {bankroll}")
        if not 0 < our_prob < 1:
            raise ValueError(f"our_prob must be in (0, 1), got {our_prob}")
        if not 0 < market_prob < 1:
            raise ValueError(f"market_prob must be in (0, 1), got {market_prob}")

        # Adjust probabilities for BUY_NO
        if direction == "BUY_NO":
            # When buying NO, we win if outcome is NO (1 - prob)
            p = 1 - our_prob  # Our belief that NO wins
            market_p = 1 - market_prob  # Market price of NO
        else:
            # When buying YES, we win if outcome is YES
            p = our_prob
            market_p = market_prob

        # Calculate Kelly fraction
        # b = odds = (1 / market_p) - 1
        # f* = (b * p - q) / b where q = 1 - p
        # Simplified: f* = p - q/b = p - (1-p)/b

        if market_p >= 0.99:  # Avoid division issues
            kelly = 0.0
        else:
            b = (1 / market_p) - 1  # Odds
            q = 1 - p
            kelly = (b * p - q) / b

        # Apply fractional Kelly
        kelly_frac = kelly * self.kelly_fraction

        logger.debug(
            f"Kelly calculation: p={p:.3f}, market_p={market_p:.3f}, "
            f"kelly={kelly:.3f}, fractional={kelly_frac:.3f}"
        )

        # Calculate raw amount
        raw_amount = bankroll * kelly_frac

        # Apply constraints
        max_from_bankroll = bankroll * self.max_bankroll_pct
        capped_amount = min(raw_amount, max_from_bankroll, self.max_bet)

        # Check minimum
        if capped_amount < self.min_bet:
            logger.info(
                f"Position size {capped_amount:.2f} below minimum {self.min_bet:.2f}, "
                f"not placing bet"
            )
            final_amount = 0.0
            num_shares = 0.0
        else:
            final_amount = capped_amount
            # Calculate shares based on market price
            # For YES: shares = amount / market_prob
            # For NO: shares = amount / (1 - market_prob)
            if direction == "BUY_YES":
                num_shares = final_amount / market_prob
            else:
                num_shares = final_amount / (1 - market_prob)

        logger.info(
            f"Position sizing for {direction}: "
            f"bankroll=${bankroll:.2f}, edge={(our_prob - market_prob):+.1%}, "
            f"kelly={kelly:.3f}, fractional_kelly={kelly_frac:.3f}, "
            f"amount=${final_amount:.2f}, shares={num_shares:.2f}"
        )

        return PositionSize(
            amount_usd=final_amount,
            num_shares=num_shares,
            kelly_fraction=kelly,
            applied_fraction=self.kelly_fraction,
            constraints_applied={
                "min_bet": self.min_bet,
                "max_bet": self.max_bet,
                "max_bankroll_pct": self.max_bankroll_pct,
            },
        )
