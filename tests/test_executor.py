import pytest
from datetime import datetime, timezone
from unittest.mock import Mock, AsyncMock

from src.execution.executor import PaperTradingExecutor
from src.models import SimpleForecast, Market, EdgeAnalysis, PositionSize, RiskCheck, ExecutionResult


@pytest.fixture
def mock_sqlite():
    """Create a mock SQLite client."""
    mock = Mock()
    mock.insert_trade = Mock(return_value="trade_123")
    mock.upsert_position = Mock()
    mock.get_open_positions = Mock(return_value=[])
    mock.get_daily_stats = Mock(return_value={"realized_pnl": 0.0})
    mock.get_current_bankroll = Mock(return_value=50.0)
    return mock


@pytest.fixture
def mock_sizer():
    """Create a mock Position Sizer."""
    mock = Mock()
    mock.calculate = Mock(return_value=PositionSize(
        amount_usd=5.0,
        num_shares=10.0,
        kelly_fraction=0.20,
        applied_fraction=0.15,
        constraints_applied={},
    ))
    return mock


@pytest.fixture
def mock_risk():
    """Create a mock Risk Manager."""
    mock = Mock()
    mock.check = Mock(return_value=RiskCheck(
        passed=True,
        violations=[],
        daily_loss_pct=0.0,
        num_open_positions=0,
        proposed_market_exposure=5.0,
    ))
    return mock


@pytest.fixture
def executor(mock_sqlite, mock_sizer, mock_risk):
    """Create PaperTradingExecutor with mocks."""
    return PaperTradingExecutor(
        sqlite=mock_sqlite,
        sizer=mock_sizer,
        risk=mock_risk,
    )


@pytest.fixture
def sample_edge_analysis():
    """Create a sample EdgeAnalysis for TRADE."""
    return EdgeAnalysis(
        our_forecast=0.65,
        market_price=0.50,
        raw_edge=0.15,
        abs_edge=0.15,
        weighted_edge=0.12,
        direction="BUY_YES",
        has_actionable_edge=True,
        recommended_action="TRADE",
        reasoning="Test reasoning",
    )


@pytest.fixture
def sample_market():
    """Create a sample Market."""
    return Market(
        id="market_abc",
        question="Will this test pass?",
        description="A test market",
        current_price=0.50,
        volume_24h=50000.0,
        liquidity=100000.0,
        end_date="2026-12-31T23:59:59Z",
        resolution_date="2026-12-31T23:59:59Z",
        created_at="2026-01-01T00:00:00Z",
        market_type="binary",
        outcomes=["YES", "NO"],
        token_ids={"YES": "123", "NO": "456"},
    )


@pytest.mark.asyncio
async def test_execute_successful_trade(executor, mock_sqlite, mock_sizer, mock_risk, sample_edge_analysis, sample_market):
    """Test successful trade execution."""
    result = await executor.execute(
        edge_analysis=sample_edge_analysis,
        calibrated_probability=0.65,
        market=sample_market,
        bankroll=50.0,
    )

    # Should return success
    assert isinstance(result, ExecutionResult)
    assert result.success is True
    assert result.trade_id == "trade_123"

    # Should have called position sizer
    mock_sizer.calculate.assert_called_once()

    # Should have called risk manager
    mock_risk.check.assert_called_once()

    # Should have inserted trade
    mock_sqlite.insert_trade.assert_called_once()

    # Should have created position
    mock_sqlite.upsert_position.assert_called_once()


@pytest.mark.asyncio
async def test_execute_skip_when_recommendation_skip(executor, sample_market):
    """Test that execution is skipped when recommendation is SKIP."""
    skip_edge_analysis = EdgeAnalysis(
        our_forecast=0.52,
        market_price=0.50,
        raw_edge=0.02,
        abs_edge=0.02,
        weighted_edge=0.01,
        direction="BUY_YES",
        has_actionable_edge=False,
        recommended_action="SKIP",
        reasoning="Edge too small",
    )

    result = await executor.execute(
        edge_analysis=skip_edge_analysis,
        calibrated_probability=0.52,
        market=sample_market,
        bankroll=50.0,
    )

    # Should return None
    assert result is None


@pytest.mark.asyncio
async def test_execute_skip_when_amount_zero(executor, mock_sizer, sample_edge_analysis, sample_market):
    """Test that execution is skipped when position size is zero."""
    # Make sizer return zero amount
    mock_sizer.calculate.return_value = PositionSize(
        amount_usd=0.0,  # Below minimum
        num_shares=0.0,
        kelly_fraction=0.01,
        applied_fraction=0.15,
        constraints_applied={},
    )

    result = await executor.execute(
        edge_analysis=sample_edge_analysis,
        calibrated_probability=0.65,
        market=sample_market,
        bankroll=50.0,
    )

    # Should return None
    assert result is None


@pytest.mark.asyncio
async def test_execute_rejected_by_risk(executor, mock_risk, sample_edge_analysis, sample_market):
    """Test that execution is rejected when risk check fails."""
    # Make risk check fail
    mock_risk.check.return_value = RiskCheck(
        passed=False,
        violations=["Daily loss limit exceeded"],
        daily_loss_pct=0.25,
        num_open_positions=0,
        proposed_market_exposure=5.0,
    )

    result = await executor.execute(
        edge_analysis=sample_edge_analysis,
        calibrated_probability=0.65,
        market=sample_market,
        bankroll=50.0,
    )

    # Should return failure result
    assert isinstance(result, ExecutionResult)
    assert result.success is False
    assert result.trade_id is None
    assert "Risk check failed" in result.message


@pytest.mark.asyncio
async def test_execute_creates_new_position(executor, mock_sqlite, sample_edge_analysis, sample_market):
    """Test that new position is created when none exists."""
    # No existing positions
    mock_sqlite.get_open_positions.return_value = []

    await executor.execute(
        edge_analysis=sample_edge_analysis,
        calibrated_probability=0.65,
        market=sample_market,
        bankroll=50.0,
    )

    # Should create new position
    mock_sqlite.upsert_position.assert_called_once()
    call_args = mock_sqlite.upsert_position.call_args[0][0]

    assert call_args["market_id"] == "market_abc"
    assert call_args["direction"] == "BUY_YES"
    assert call_args["num_shares"] == 10.0
    assert call_args["amount_usd"] == 5.0


@pytest.mark.asyncio
async def test_execute_updates_existing_position(executor, mock_sqlite, mock_sizer, sample_edge_analysis, sample_market):
    """Test that existing position is updated when it exists."""
    from src.models import Position

    # Existing position in same market
    existing_position = Position(
        market_id="market_abc",
        direction="BUY_YES",
        num_shares=20.0,
        amount_usd=10.0,
        avg_entry_price=0.50,
        current_price=0.50,
        unrealized_pnl=0.0,
        updated_at=datetime.now(timezone.utc),
    )

    mock_sqlite.get_open_positions.return_value = [existing_position]

    await executor.execute(
        edge_analysis=sample_edge_analysis,
        calibrated_probability=0.65,
        market=sample_market,
        bankroll=50.0,
    )

    # Should update position
    mock_sqlite.upsert_position.assert_called_once()
    call_args = mock_sqlite.upsert_position.call_args[0][0]

    # New shares = 20 + 10 = 30
    assert call_args["num_shares"] == 30.0
    # New amount = 10 + 5 = 15
    assert call_args["amount_usd"] == 15.0


@pytest.mark.asyncio
async def test_execute_buy_no_direction(executor, mock_sizer, sample_market):
    """Test execution with BUY_NO direction."""
    buy_no_edge = EdgeAnalysis(
        our_forecast=0.35,
        market_price=0.50,
        raw_edge=-0.15,
        abs_edge=0.15,
        weighted_edge=0.12,
        direction="BUY_NO",
        has_actionable_edge=True,
        recommended_action="TRADE",
        reasoning="Test reasoning",
    )

    result = await executor.execute(
        edge_analysis=buy_no_edge,
        calibrated_probability=0.35,
        market=sample_market,
        bankroll=50.0,
    )

    # Should succeed with BUY_NO
    assert result.success is True

    # Should have called sizer with BUY_NO
    call_args = mock_sizer.calculate.call_args
    assert call_args[1]["direction"] == "BUY_NO"


@pytest.mark.asyncio
async def test_get_portfolio_summary_no_positions(executor, mock_sqlite):
    """Test portfolio summary with no positions."""
    mock_sqlite.get_open_positions.return_value = []
    mock_sqlite.get_current_bankroll.return_value = 50.0
    mock_sqlite.get_daily_stats.return_value = {"realized_pnl": 0.0}

    summary = await executor.get_portfolio_summary()

    assert summary["num_positions"] == 0
    assert summary["total_unrealized_pnl"] == 0.0
    assert summary["current_bankroll"] == 50.0


@pytest.mark.asyncio
async def test_get_portfolio_summary_with_positions(executor, mock_sqlite):
    """Test portfolio summary with open positions."""
    from src.models import Position

    positions = [
        Position(
            market_id="market_1",
            direction="BUY_YES",
            num_shares=10.0,
            amount_usd=5.0,
            avg_entry_price=0.50,
            current_price=0.60,  # Price went up
            unrealized_pnl=1.0,
            updated_at=datetime.now(timezone.utc),
        ),
        Position(
            market_id="market_2",
            direction="BUY_NO",
            num_shares=20.0,
            amount_usd=10.0,
            avg_entry_price=0.40,
            current_price=0.30,  # Price went down (good for NO)
            unrealized_pnl=2.0,
            updated_at=datetime.now(timezone.utc),
        ),
    ]

    mock_sqlite.get_open_positions.return_value = positions
    mock_sqlite.get_current_bankroll.return_value = 50.0
    mock_sqlite.get_daily_stats.return_value = {"realized_pnl": 1.5}

    summary = await executor.get_portfolio_summary()

    assert summary["num_positions"] == 2
    assert len(summary["positions"]) == 2
    assert summary["current_bankroll"] == 50.0
    assert summary["realized_pnl_today"] == 1.5


@pytest.mark.asyncio
async def test_get_portfolio_summary_with_current_prices(executor, mock_sqlite):
    """Test portfolio summary with updated current prices."""
    from src.models import Position

    position = Position(
        market_id="market_1",
        direction="BUY_YES",
        num_shares=10.0,
        amount_usd=5.0,
        avg_entry_price=0.50,
        current_price=0.50,  # Stale price
        unrealized_pnl=0.0,
        updated_at=datetime.now(timezone.utc),
    )

    mock_sqlite.get_open_positions.return_value = [position]
    mock_sqlite.get_current_bankroll.return_value = 50.0
    mock_sqlite.get_daily_stats.return_value = None

    # Provide updated current prices
    current_prices = {"market_1": 0.60}

    summary = await executor.get_portfolio_summary(current_prices=current_prices)

    # Should use updated price for P&L calculation
    pos_summary = summary["positions"][0]
    assert pos_summary["current_price"] == 0.60
    # Unrealized P&L = (0.60 - 0.50) * 10 = 1.0
    assert pos_summary["unrealized_pnl"] == pytest.approx(1.0, abs=0.01)


@pytest.mark.asyncio
async def test_execute_fills_trade_immediately(executor, mock_sqlite, sample_edge_analysis, sample_market):
    """Test that paper trade is filled immediately."""
    await executor.execute(
        edge_analysis=sample_edge_analysis,
        calibrated_probability=0.65,
        market=sample_market,
        bankroll=50.0,
    )

    # Should insert trade with FILLED status
    call_args = mock_sqlite.insert_trade.call_args[0][0]
    assert call_args["status"] == "FILLED"
