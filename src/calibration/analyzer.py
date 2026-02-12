"""
Meta Analyzer - Analyzes edge and makes trade recommendations.
"""
from typing import Literal
from loguru import logger

from src.models import CalibratedForecast, EdgeAnalysis


class MetaAnalyzer:
    """
    Analyzes edge between our forecast and market price.

    Determines whether to trade based on edge size, confidence,
    and liquidity thresholds.
    """

    def __init__(
        self,
        min_edge: float = 0.05,
        min_confidence: float = 0.6,
        min_liquidity: float = 10000.0,
    ):
        """
        Initialize analyzer with thresholds.

        Args:
            min_edge: Minimum absolute edge required to trade (default: 5%)
            min_confidence: Minimum confidence required to trade (default: 0.6)
            min_liquidity: Minimum market liquidity required (default: $10k)
        """
        self.min_edge = min_edge
        self.min_confidence = min_confidence
        self.min_liquidity = min_liquidity

    def analyze(
        self,
        our_forecast: CalibratedForecast,
        market_price: float,
        liquidity: float,
        historical_accuracy: float = 0.5,
    ) -> EdgeAnalysis:
        """
        Analyze edge and generate trade recommendation.

        Args:
            our_forecast: Our calibrated forecast
            market_price: Current market price (probability)
            liquidity: Market liquidity in USD
            historical_accuracy: Historical accuracy of our forecasts (0-1)

        Returns:
            EdgeAnalysis with edge metrics and recommendation
        """
        # Validate inputs
        if not 0.0 <= market_price <= 1.0:
            raise ValueError(f"market_price must be in [0, 1], got {market_price}")
        if liquidity < 0:
            raise ValueError(f"liquidity must be non-negative, got {liquidity}")

        # Calculate edge
        raw_edge = our_forecast.calibrated - market_price
        abs_edge = abs(raw_edge)

        # Weight edge by confidence
        weighted_edge = raw_edge * our_forecast.confidence

        # Determine direction
        direction: Literal["BUY_YES", "BUY_NO"] = "BUY_YES" if raw_edge > 0 else "BUY_NO"

        # Check if edge is actionable
        has_actionable_edge = self._check_actionable_edge(
            abs_edge=abs_edge,
            confidence=our_forecast.confidence,
            liquidity=liquidity,
        )

        # Determine recommendation
        if has_actionable_edge:
            recommended_action: Literal["TRADE", "SKIP"] = "TRADE"
        else:
            recommended_action = "SKIP"

        # Generate reasoning
        reasoning = self._generate_reasoning(
            our_forecast=our_forecast,
            market_price=market_price,
            abs_edge=abs_edge,
            confidence=our_forecast.confidence,
            liquidity=liquidity,
            has_actionable_edge=has_actionable_edge,
            direction=direction,
        )

        logger.info(
            f"Edge analysis: {abs_edge:.1%} edge, "
            f"confidence {our_forecast.confidence:.1%}, "
            f"liquidity ${liquidity:,.0f} -> {recommended_action}"
        )

        return EdgeAnalysis(
            our_forecast=our_forecast.calibrated,
            market_price=market_price,
            raw_edge=raw_edge,
            abs_edge=abs_edge,
            weighted_edge=weighted_edge,
            direction=direction,
            has_actionable_edge=has_actionable_edge,
            recommended_action=recommended_action,
            reasoning=reasoning,
        )

    def _check_actionable_edge(
        self,
        abs_edge: float,
        confidence: float,
        liquidity: float,
    ) -> bool:
        """
        Check if edge meets all thresholds for trading.

        Args:
            abs_edge: Absolute edge value
            confidence: Confidence score
            liquidity: Market liquidity

        Returns:
            True if edge is actionable
        """
        edge_sufficient = abs_edge >= self.min_edge
        confidence_sufficient = confidence >= self.min_confidence
        liquidity_sufficient = liquidity >= self.min_liquidity

        return edge_sufficient and confidence_sufficient and liquidity_sufficient

    def _generate_reasoning(
        self,
        our_forecast: CalibratedForecast,
        market_price: float,
        abs_edge: float,
        confidence: float,
        liquidity: float,
        has_actionable_edge: bool,
        direction: Literal["BUY_YES", "BUY_NO"],
    ) -> str:
        """
        Generate human-readable reasoning for the recommendation.

        Args:
            our_forecast: Our forecast
            market_price: Market price
            abs_edge: Absolute edge
            confidence: Confidence
            liquidity: Liquidity
            has_actionable_edge: Whether edge is actionable
            direction: Trade direction

        Returns:
            Formatted reasoning string
        """
        reasoning_parts = [
            f"Our calibrated forecast: {our_forecast.calibrated:.1%}",
            f"Market price: {market_price:.1%}",
            f"Edge: {abs_edge:+.1%} ({direction})",
            f"Confidence: {confidence:.1%}",
            f"Market liquidity: ${liquidity:,.0f}",
            "",
        ]

        if has_actionable_edge:
            reasoning_parts.extend([
                f"✓ Edge {abs_edge:.1%} >= threshold {self.min_edge:.1%}",
                f"✓ Confidence {confidence:.1%} >= threshold {self.min_confidence:.1%}",
                f"✓ Liquidity ${liquidity:,.0f} >= threshold ${self.min_liquidity:,.0f}",
                "",
                f"RECOMMENDATION: {direction}",
                f"This market shows a {abs_edge:.1%} mispricing opportunity.",
            ])

            # Add calibration info if available
            if our_forecast.calibration_method != "identity":
                improvement = abs(our_forecast.calibrated - our_forecast.raw)
                reasoning_parts.append(
                    f"Calibration adjusted forecast by {improvement:.1%} "
                    f"based on {our_forecast.historical_samples} historical forecasts."
                )
        else:
            # Explain why we're skipping
            reasons = []

            if abs_edge < self.min_edge:
                reasons.append(
                    f"✗ Edge {abs_edge:.1%} < threshold {self.min_edge:.1%}"
                )
            else:
                reasons.append(
                    f"✓ Edge {abs_edge:.1%} >= threshold {self.min_edge:.1%}"
                )

            if confidence < self.min_confidence:
                reasons.append(
                    f"✗ Confidence {confidence:.1%} < threshold {self.min_confidence:.1%}"
                )
            else:
                reasons.append(
                    f"✓ Confidence {confidence:.1%} >= threshold {self.min_confidence:.1%}"
                )

            if liquidity < self.min_liquidity:
                reasons.append(
                    f"✗ Liquidity ${liquidity:,.0f} < threshold ${self.min_liquidity:,.0f}"
                )
            else:
                reasons.append(
                    f"✓ Liquidity ${liquidity:,.0f} >= threshold ${self.min_liquidity:,.0f}"
                )

            reasoning_parts.extend(reasons)
            reasoning_parts.extend([
                "",
                "RECOMMENDATION: SKIP",
                "One or more criteria not met for trading.",
            ])

        return "\n".join(reasoning_parts)
