import pytest
from datetime import datetime, timezone
from unittest.mock import Mock, AsyncMock, patch

from src.calibration.resolver import MarketResolver


@pytest.fixture
def mock_polymarket():
    """Create mock Polymarket client."""
    mock = Mock()
    mock.check_resolutions = AsyncMock(return_value={})
    mock.get_market_resolution = AsyncMock(return_value=None)
    return mock


@pytest.fixture
def mock_feedback():
    """Create mock FeedbackLoop."""
    mock = Mock()
    mock.process_resolution = AsyncMock(return_value={"success": True})
    return mock


@pytest.fixture
def mock_sqlite():
    """Create mock SQLite client."""
    mock = Mock()
    mock.get_open_positions = Mock(return_value=[])
    mock.get_position = Mock(return_value=None)
    mock.upsert_position = Mock()
    mock.get_daily_stats = Mock(return_value=None)
    mock.upsert_daily_stats = Mock()
    mock.get_current_bankroll = Mock(return_value=50.0)
    return mock


@pytest.fixture
def mock_duckdb():
    """Create mock DuckDB client."""
    mock = Mock()
    mock.get_unresolved_forecasts = Mock(return_value=[])
    return mock


@pytest.fixture
def resolver(mock_polymarket, mock_feedback, mock_sqlite, mock_duckdb):
    """Create MarketResolver with mocks."""
    return MarketResolver(
        polymarket=mock_polymarket,
        feedback=mock_feedback,
        sqlite=mock_sqlite,
        duckdb=mock_duckdb,
    )


@pytest.mark.asyncio
async def test_run_resolution_cycle_no_markets(resolver, mock_polymarket):
    """Test resolution cycle with no markets to check."""
    result = await resolver.run_resolution_cycle()

    assert result["checked"] == 0
    assert result["resolved"] == 0
    assert result["pnl"] == 0.0
    mock_polymarket.check_resolutions.assert_not_called()


@pytest.mark.asyncio
async def test_run_resolution_cycle_no_resolutions(resolver, mock_sqlite, mock_polymarket):
    """Test resolution cycle with markets but no resolutions."""
    # Mock open position
    mock_sqlite.get_open_positions.return_value = [
        {
            "market_id": "market_1",
            "direction": "BUY_YES",
            "num_shares": 10.0,
            "amount_usd": 5.0,
            "avg_entry_price": 0.5,
            "current_price": 0.5,
            "unrealized_pnl": 0.0,
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }
    ]

    # No resolutions
    mock_polymarket.check_resolutions.return_value = {}

    result = await resolver.run_resolution_cycle()

    assert result["checked"] == 1
    assert result["resolved"] == 0
    assert result["pnl"] == 0.0


@pytest.mark.asyncio
async def test_resolve_market_buy_yes_wins(resolver, mock_sqlite, mock_polymarket, mock_feedback):
    """Test resolving BUY_YES position when YES wins."""
    market_id = "market_1"

    # Mock position
    mock_sqlite.get_position.return_value = {
        "market_id": market_id,
        "direction": "BUY_YES",
        "num_shares": 10.0,
        "amount_usd": 5.0,
        "avg_entry_price": 0.5,
        "current_price": 0.5,
        "unrealized_pnl": 0.0,
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }

    # Mock resolution
    mock_polymarket.check_resolutions.return_value = {market_id: True}  # YES won

    # Mock open positions
    mock_sqlite.get_open_positions.return_value = [mock_sqlite.get_position.return_value]

    result = await resolver.run_resolution_cycle()

    # P&L = (10 shares * $1) - $5 cost = +$5
    assert result["pnl"] == pytest.approx(5.0, abs=0.01)
    assert result["resolved"] == 1

    # Verify position was closed
    mock_sqlite.upsert_position.assert_called_once()
    call_args = mock_sqlite.upsert_position.call_args[0][0]
    assert call_args["num_shares"] == 0.0

    # Verify feedback was called
    mock_feedback.process_resolution.assert_called_once_with(market_id, True)


@pytest.mark.asyncio
async def test_resolve_market_buy_yes_loses(resolver, mock_sqlite, mock_polymarket):
    """Test resolving BUY_YES position when NO wins."""
    market_id = "market_1"

    # Mock position
    mock_sqlite.get_position.return_value = {
        "market_id": market_id,
        "direction": "BUY_YES",
        "num_shares": 10.0,
        "amount_usd": 5.0,
        "avg_entry_price": 0.5,
        "current_price": 0.5,
        "unrealized_pnl": 0.0,
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }

    # Mock resolution: NO won
    mock_polymarket.check_resolutions.return_value = {market_id: False}

    # Mock open positions
    mock_sqlite.get_open_positions.return_value = [mock_sqlite.get_position.return_value]

    result = await resolver.run_resolution_cycle()

    # P&L = -$5 (lost everything)
    assert result["pnl"] == pytest.approx(-5.0, abs=0.01)
    assert result["resolved"] == 1


@pytest.mark.asyncio
async def test_resolve_market_buy_no_wins(resolver, mock_sqlite, mock_polymarket):
    """Test resolving BUY_NO position when NO wins."""
    market_id = "market_1"

    # Mock position
    mock_sqlite.get_position.return_value = {
        "market_id": market_id,
        "direction": "BUY_NO",
        "num_shares": 10.0,
        "amount_usd": 5.0,
        "avg_entry_price": 0.5,
        "current_price": 0.5,
        "unrealized_pnl": 0.0,
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }

    # Mock resolution: NO won
    mock_polymarket.check_resolutions.return_value = {market_id: False}

    # Mock open positions
    mock_sqlite.get_open_positions.return_value = [mock_sqlite.get_position.return_value]

    result = await resolver.run_resolution_cycle()

    # P&L = (10 shares * $1) - $5 cost = +$5
    assert result["pnl"] == pytest.approx(5.0, abs=0.01)
    assert result["resolved"] == 1


@pytest.mark.asyncio
async def test_resolve_market_buy_no_loses(resolver, mock_sqlite, mock_polymarket):
    """Test resolving BUY_NO position when YES wins."""
    market_id = "market_1"

    # Mock position
    mock_sqlite.get_position.return_value = {
        "market_id": market_id,
        "direction": "BUY_NO",
        "num_shares": 10.0,
        "amount_usd": 5.0,
        "avg_entry_price": 0.5,
        "current_price": 0.5,
        "unrealized_pnl": 0.0,
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }

    # Mock resolution: YES won
    mock_polymarket.check_resolutions.return_value = {market_id: True}

    # Mock open positions
    mock_sqlite.get_open_positions.return_value = [mock_sqlite.get_position.return_value]

    result = await resolver.run_resolution_cycle()

    # P&L = -$5 (lost everything)
    assert result["pnl"] == pytest.approx(-5.0, abs=0.01)
    assert result["resolved"] == 1


@pytest.mark.asyncio
async def test_resolve_market_no_position(resolver, mock_sqlite, mock_polymarket, mock_duckdb):
    """Test resolving market with no position (forecast only)."""
    market_id = "market_1"

    # No position
    mock_sqlite.get_position.return_value = None

    # Mock unresolved forecast
    mock_duckdb.get_unresolved_forecasts.return_value = [
        {"market_id": market_id}
    ]

    # Mock resolution
    mock_polymarket.check_resolutions.return_value = {market_id: True}

    result = await resolver.run_resolution_cycle()

    # No P&L since no position
    assert result["pnl"] == 0.0
    assert result["resolved"] == 1

    # Position not updated (no position)
    mock_sqlite.upsert_position.assert_not_called()


@pytest.mark.asyncio
async def test_resolve_market_updates_daily_stats(resolver, mock_sqlite, mock_polymarket):
    """Test that resolving market updates daily stats."""
    market_id = "market_1"

    # Mock position
    mock_sqlite.get_position.return_value = {
        "market_id": market_id,
        "direction": "BUY_YES",
        "num_shares": 10.0,
        "amount_usd": 5.0,
        "avg_entry_price": 0.5,
        "current_price": 0.5,
        "unrealized_pnl": 0.0,
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }

    # Mock resolution
    mock_polymarket.check_resolutions.return_value = {market_id: True}

    # Mock open positions
    mock_sqlite.get_open_positions.return_value = [mock_sqlite.get_position.return_value]

    # No existing stats
    mock_sqlite.get_daily_stats.return_value = None

    await resolver.run_resolution_cycle()

    # Verify daily stats were created
    mock_sqlite.upsert_daily_stats.assert_called_once()
    call_args = mock_sqlite.upsert_daily_stats.call_args[0][0]
    assert call_args["net_pnl"] == pytest.approx(5.0, abs=0.01)
    assert call_args["trades_won"] == 1
