import pytest
from unittest.mock import Mock

from src.execution.sizer import PositionSizer
from src.models import PositionSize
from config.settings import RiskSettings


@pytest.fixture
def risk_settings():
    """Create test risk settings."""
    return RiskSettings.model_construct(
        min_bet=1.0,
        max_bet=10.0,
        max_position_pct=0.10,
        max_daily_loss=0.20,
        max_open_positions=5,
        max_single_market_exposure=0.15,
        min_edge=0.05,
        min_liquidity=10000.0,
        initial_bankroll=50.0,
        max_daily_loss_pct=0.20,
        min_confidence=0.65,
    )


@pytest.fixture
def sizer(risk_settings):
    """Create PositionSizer with test settings."""
    return PositionSizer(risk_settings=risk_settings)


def test_calculate_buy_yes_with_edge(sizer):
    """Test position sizing for BUY_YES with positive edge."""
    bankroll = 50.0
    our_prob = 0.65  # We think 65%
    market_prob = 0.50  # Market thinks 50%
    direction = "BUY_YES"

    result = sizer.calculate(
        bankroll=bankroll,
        our_prob=our_prob,
        market_prob=market_prob,
        direction=direction,
    )

    assert isinstance(result, PositionSize)
    assert result.amount_usd > 0
    assert result.amount_usd <= sizer.max_bet
    assert result.amount_usd <= bankroll * sizer.max_bankroll_pct
    assert result.num_shares > 0


def test_calculate_buy_no_with_edge(sizer):
    """Test position sizing for BUY_NO with positive edge."""
    bankroll = 50.0
    our_prob = 0.35  # We think 35% chance of YES (65% chance of NO)
    market_prob = 0.50  # Market thinks 50% chance of YES
    direction = "BUY_NO"

    result = sizer.calculate(
        bankroll=bankroll,
        our_prob=our_prob,
        market_prob=market_prob,
        direction=direction,
    )

    assert isinstance(result, PositionSize)
    assert result.amount_usd > 0
    assert result.num_shares > 0


def test_calculate_respects_min_bet(sizer):
    """Test that position size below minimum returns zero."""
    bankroll = 50.0
    our_prob = 0.51  # Very small edge
    market_prob = 0.50
    direction = "BUY_YES"

    result = sizer.calculate(
        bankroll=bankroll,
        our_prob=our_prob,
        market_prob=market_prob,
        direction=direction,
    )

    # With very small edge, Kelly should be tiny, below min_bet
    # So amount should be 0
    assert result.amount_usd == 0.0
    assert result.num_shares == 0.0


def test_calculate_respects_max_bet(sizer):
    """Test that position size is capped at max_bet."""
    bankroll = 1000.0  # Large bankroll
    our_prob = 0.80  # Strong edge
    market_prob = 0.50
    direction = "BUY_YES"

    result = sizer.calculate(
        bankroll=bankroll,
        our_prob=our_prob,
        market_prob=market_prob,
        direction=direction,
    )

    # Should be capped at max_bet
    assert result.amount_usd <= sizer.max_bet


def test_calculate_respects_max_bankroll_pct(sizer):
    """Test that position size respects max bankroll percentage."""
    bankroll = 50.0
    our_prob = 0.90  # Very strong edge
    market_prob = 0.50
    direction = "BUY_YES"

    result = sizer.calculate(
        bankroll=bankroll,
        our_prob=our_prob,
        market_prob=market_prob,
        direction=direction,
    )

    # Should not exceed max_bankroll_pct
    assert result.amount_usd <= bankroll * sizer.max_bankroll_pct


def test_calculate_negative_edge_returns_zero(sizer):
    """Test that negative edge returns zero position size."""
    bankroll = 50.0
    our_prob = 0.40  # We think 40%
    market_prob = 0.50  # Market thinks 50% - we have negative edge
    direction = "BUY_YES"

    result = sizer.calculate(
        bankroll=bankroll,
        our_prob=our_prob,
        market_prob=market_prob,
        direction=direction,
    )

    # Negative edge should result in zero bet
    assert result.amount_usd == 0.0
    assert result.num_shares == 0.0


def test_calculate_kelly_formula(sizer):
    """Test Kelly formula calculation."""
    bankroll = 100.0
    our_prob = 0.60
    market_prob = 0.50
    direction = "BUY_YES"

    result = sizer.calculate(
        bankroll=bankroll,
        our_prob=our_prob,
        market_prob=market_prob,
        direction=direction,
    )

    # Kelly formula: f* = (bp - q) / b
    # b = (1/0.5) - 1 = 1
    # p = 0.60, q = 0.40
    # f* = (1*0.6 - 0.4) / 1 = 0.2
    # Fractional Kelly = 0.2 * 0.15 = 0.03
    # Amount = 100 * 0.03 = 3.0

    expected_kelly = 0.2
    assert result.kelly_fraction == pytest.approx(expected_kelly, abs=0.01)

    expected_amount = 100 * expected_kelly * sizer.kelly_fraction
    # Should be close to expected, might be capped
    assert result.amount_usd == pytest.approx(expected_amount, abs=0.1)


def test_calculate_shares_for_buy_yes(sizer):
    """Test share calculation for BUY_YES."""
    bankroll = 50.0
    our_prob = 0.70
    market_prob = 0.50
    direction = "BUY_YES"

    result = sizer.calculate(
        bankroll=bankroll,
        our_prob=our_prob,
        market_prob=market_prob,
        direction=direction,
    )

    if result.amount_usd > 0:
        # For BUY_YES: shares = amount / market_prob
        expected_shares = result.amount_usd / market_prob
        assert result.num_shares == pytest.approx(expected_shares, abs=0.01)


def test_calculate_shares_for_buy_no(sizer):
    """Test share calculation for BUY_NO."""
    bankroll = 50.0
    our_prob = 0.30  # 30% YES = 70% NO
    market_prob = 0.50
    direction = "BUY_NO"

    result = sizer.calculate(
        bankroll=bankroll,
        our_prob=our_prob,
        market_prob=market_prob,
        direction=direction,
    )

    if result.amount_usd > 0:
        # For BUY_NO: shares = amount / (1 - market_prob)
        expected_shares = result.amount_usd / (1 - market_prob)
        assert result.num_shares == pytest.approx(expected_shares, abs=0.01)


def test_calculate_invalid_bankroll(sizer):
    """Test that invalid bankroll raises error."""
    with pytest.raises(ValueError, match="Bankroll must be positive"):
        sizer.calculate(
            bankroll=0.0,
            our_prob=0.60,
            market_prob=0.50,
            direction="BUY_YES",
        )


def test_calculate_invalid_our_prob(sizer):
    """Test that invalid our_prob raises error."""
    with pytest.raises(ValueError, match="our_prob must be in"):
        sizer.calculate(
            bankroll=50.0,
            our_prob=1.5,
            market_prob=0.50,
            direction="BUY_YES",
        )


def test_calculate_invalid_market_prob(sizer):
    """Test that invalid market_prob raises error."""
    with pytest.raises(ValueError, match="market_prob must be in"):
        sizer.calculate(
            bankroll=50.0,
            our_prob=0.60,
            market_prob=0.0,
            direction="BUY_YES",
        )


def test_calculate_extreme_market_price(sizer):
    """Test calculation with extreme market price."""
    bankroll = 50.0
    our_prob = 0.95
    market_prob = 0.99  # Very high market price
    direction = "BUY_YES"

    result = sizer.calculate(
        bankroll=bankroll,
        our_prob=our_prob,
        market_prob=market_prob,
        direction=direction,
    )

    # Should handle gracefully without errors
    assert result.amount_usd >= 0


def test_calculate_constraints_metadata(sizer):
    """Test that constraints metadata is included in result."""
    result = sizer.calculate(
        bankroll=50.0,
        our_prob=0.65,
        market_prob=0.50,
        direction="BUY_YES",
    )

    assert "min_bet" in result.constraints_applied
    assert "max_bet" in result.constraints_applied
    assert "max_bankroll_pct" in result.constraints_applied
    assert result.constraints_applied["min_bet"] == sizer.min_bet
    assert result.constraints_applied["max_bet"] == sizer.max_bet
