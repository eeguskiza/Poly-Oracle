import pytest
from datetime import datetime, timezone

from src.execution.risk import RiskManager
from src.models import Trade, Position, RiskCheck
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
def risk_manager(risk_settings):
    """Create RiskManager with test settings."""
    return RiskManager(risk_settings=risk_settings)


@pytest.fixture
def sample_trade():
    """Create a sample trade."""
    return Trade(
        id="trade_123",
        market_id="market_abc",
        direction="BUY_YES",
        amount_usd=5.0,
        num_shares=10.0,
        entry_price=0.50,
        timestamp=datetime.now(timezone.utc),
        status="PENDING",
    )


@pytest.fixture
def sample_position():
    """Create a sample position."""
    return Position(
        market_id="market_xyz",
        direction="BUY_YES",
        num_shares=20.0,
        amount_usd=10.0,
        avg_entry_price=0.50,
        current_price=0.55,
        unrealized_pnl=1.0,
        updated_at=datetime.now(timezone.utc),
    )


def test_check_passes_when_all_ok(risk_manager, sample_trade):
    """Test that check passes when all conditions are met."""
    current_positions = []
    daily_pnl = 0.0
    bankroll = 50.0

    result = risk_manager.check(
        proposed_trade=sample_trade,
        current_positions=current_positions,
        daily_pnl=daily_pnl,
        bankroll=bankroll,
    )

    assert isinstance(result, RiskCheck)
    assert result.passed is True
    assert len(result.violations) == 0


def test_check_fails_daily_loss_limit(risk_manager, sample_trade):
    """Test that check fails when daily loss limit exceeded."""
    current_positions = []
    bankroll = 50.0
    daily_pnl = -15.0  # Lost $15 out of $50 = 30% > 20% limit

    result = risk_manager.check(
        proposed_trade=sample_trade,
        current_positions=current_positions,
        daily_pnl=daily_pnl,
        bankroll=bankroll,
    )

    assert result.passed is False
    assert len(result.violations) > 0
    assert any("Daily loss limit" in v for v in result.violations)
    assert result.daily_loss_pct == pytest.approx(0.30, abs=0.01)


def test_check_fails_max_open_positions(risk_manager, sample_trade, sample_position):
    """Test that check fails when max open positions exceeded."""
    # Create 5 positions (at the limit)
    current_positions = [sample_position for _ in range(5)]
    daily_pnl = 0.0
    bankroll = 50.0

    result = risk_manager.check(
        proposed_trade=sample_trade,
        current_positions=current_positions,
        daily_pnl=daily_pnl,
        bankroll=bankroll,
    )

    assert result.passed is False
    assert len(result.violations) > 0
    assert any("Max open positions" in v for v in result.violations)
    assert result.num_open_positions == 5


def test_check_fails_single_market_exposure(risk_manager, sample_trade):
    """Test that check fails when single market exposure exceeded."""
    # Create existing position in same market
    existing_position = Position(
        market_id="market_abc",  # Same as sample_trade
        direction="BUY_YES",
        num_shares=20.0,
        amount_usd=8.0,  # Already have $8 in this market
        avg_entry_price=0.40,
        current_price=0.50,
        unrealized_pnl=2.0,
        updated_at=datetime.now(timezone.utc),
    )

    current_positions = [existing_position]
    daily_pnl = 0.0
    bankroll = 50.0

    result = risk_manager.check(
        proposed_trade=sample_trade,  # Proposed $5 more
        current_positions=current_positions,
        daily_pnl=daily_pnl,
        bankroll=bankroll,
    )

    # Total would be $13 out of $50 = 26% > 15% limit
    assert result.passed is False
    assert len(result.violations) > 0
    assert any("Single market exposure" in v for v in result.violations)


def test_check_passes_different_markets(risk_manager, sample_trade, sample_position):
    """Test that check passes when positions are in different markets."""
    # Position in different market
    current_positions = [sample_position]  # market_xyz
    daily_pnl = 0.0
    bankroll = 50.0

    result = risk_manager.check(
        proposed_trade=sample_trade,  # market_abc (different)
        current_positions=current_positions,
        daily_pnl=daily_pnl,
        bankroll=bankroll,
    )

    # Should pass since markets are different
    assert result.passed is True


def test_check_multiple_violations(risk_manager, sample_trade, sample_position):
    """Test check with multiple violations."""
    # Max positions
    current_positions = [sample_position for _ in range(5)]
    # Daily loss
    daily_pnl = -15.0
    bankroll = 50.0

    result = risk_manager.check(
        proposed_trade=sample_trade,
        current_positions=current_positions,
        daily_pnl=daily_pnl,
        bankroll=bankroll,
    )

    assert result.passed is False
    assert len(result.violations) >= 2
    assert any("Daily loss" in v for v in result.violations)
    assert any("Max open positions" in v for v in result.violations)


def test_check_daily_loss_at_threshold(risk_manager, sample_trade):
    """Test check when daily loss is exactly at threshold."""
    current_positions = []
    bankroll = 50.0
    daily_pnl = -10.0  # Exactly 20% of $50

    result = risk_manager.check(
        proposed_trade=sample_trade,
        current_positions=current_positions,
        daily_pnl=daily_pnl,
        bankroll=bankroll,
    )

    # At threshold should fail (>=)
    assert result.passed is False
    assert any("Daily loss" in v for v in result.violations)


def test_check_open_positions_at_threshold(risk_manager, sample_trade, sample_position):
    """Test check when open positions exactly at threshold."""
    # Exactly 5 positions (at limit)
    current_positions = [sample_position for _ in range(5)]
    daily_pnl = 0.0
    bankroll = 50.0

    result = risk_manager.check(
        proposed_trade=sample_trade,
        current_positions=current_positions,
        daily_pnl=daily_pnl,
        bankroll=bankroll,
    )

    # At threshold should fail (>=)
    assert result.passed is False


def test_check_single_market_exposure_at_threshold(risk_manager, sample_trade):
    """Test check when single market exposure exactly at threshold."""
    # Existing position brings us exactly to threshold
    existing_position = Position(
        market_id="market_abc",
        direction="BUY_YES",
        num_shares=10.0,
        amount_usd=2.5,  # $2.5 existing + $5 proposed = $7.5 = 15% of $50
        avg_entry_price=0.25,
        current_price=0.50,
        unrealized_pnl=0.0,
        updated_at=datetime.now(timezone.utc),
    )

    current_positions = [existing_position]
    daily_pnl = 0.0
    bankroll = 50.0

    result = risk_manager.check(
        proposed_trade=sample_trade,
        current_positions=current_positions,
        daily_pnl=daily_pnl,
        bankroll=bankroll,
    )

    # At threshold should pass (> not >=)
    # $7.5 / $50 = 0.15 exactly, so should NOT exceed
    assert result.passed is True


def test_check_positive_daily_pnl(risk_manager, sample_trade):
    """Test check with positive daily P&L."""
    current_positions = []
    daily_pnl = 5.0  # Positive P&L
    bankroll = 50.0

    result = risk_manager.check(
        proposed_trade=sample_trade,
        current_positions=current_positions,
        daily_pnl=daily_pnl,
        bankroll=bankroll,
    )

    # Should pass - no daily loss violation
    assert result.passed is True
    assert result.daily_loss_pct == 0.0


def test_check_zero_bankroll(risk_manager, sample_trade):
    """Test check with zero bankroll."""
    current_positions = []
    daily_pnl = 0.0
    bankroll = 0.0

    result = risk_manager.check(
        proposed_trade=sample_trade,
        current_positions=current_positions,
        daily_pnl=daily_pnl,
        bankroll=bankroll,
    )

    # Should handle gracefully
    assert result.daily_loss_pct == 0.0


def test_check_exposure_metadata(risk_manager, sample_trade):
    """Test that exposure metadata is included in result."""
    existing_position = Position(
        market_id="market_abc",
        direction="BUY_YES",
        num_shares=10.0,
        amount_usd=5.0,
        avg_entry_price=0.50,
        current_price=0.50,
        unrealized_pnl=0.0,
        updated_at=datetime.now(timezone.utc),
    )

    current_positions = [existing_position]
    daily_pnl = 0.0
    bankroll = 50.0

    result = risk_manager.check(
        proposed_trade=sample_trade,
        current_positions=current_positions,
        daily_pnl=daily_pnl,
        bankroll=bankroll,
    )

    # Should include exposure for same market
    assert result.proposed_market_exposure == 10.0  # $5 + $5
