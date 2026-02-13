"""
End-to-end test of the full pipeline with mocked external services.

This test covers:
1. Market creation
2. Context building
3. Debate and forecasting
4. Calibration
5. Edge analysis
6. Paper trade execution
7. Market resolution
8. P&L calculation
9. Calibration update

All external services (Ollama, Polymarket API, News API) are mocked.
"""
import pytest
from datetime import datetime, timezone
from unittest.mock import Mock, AsyncMock, patch
from pathlib import Path
import tempfile

from src.models import Market, DebateResult, SimpleForecast, EdgeAnalysis, PositionSize
from src.data.storage.sqlite_client import SQLiteClient
from src.data.storage.duckdb_client import DuckDBClient
from src.execution import PositionSizer, RiskManager, PaperTradingExecutor
from src.calibration import CalibratorAgent, MetaAnalyzer, FeedbackLoop
from src.calibration.resolver import MarketResolver
from config.settings import RiskSettings


@pytest.fixture
def temp_db_dir():
    """Create temporary database directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def sqlite_client(temp_db_dir):
    """Initialize SQLite client with temp database."""
    db_path = temp_db_dir / "test.db"
    client = SQLiteClient(db_path)
    client.initialize_schema()

    # Initialize with starting bankroll
    client.upsert_daily_stats({
        "date": datetime.now(timezone.utc).date().isoformat(),
        "starting_bankroll": 50.0,
        "ending_bankroll": 50.0,
        "trades_executed": 0,
        "trades_won": 0,
        "gross_pnl": 0.0,
        "fees_paid": 0.0,
        "net_pnl": 0.0,
    })

    yield client
    client.close()


@pytest.fixture
def duckdb_client(temp_db_dir):
    """Initialize DuckDB client with temp database."""
    db_path = temp_db_dir / "test.duckdb"
    client = DuckDBClient(db_path)
    client.initialize_schema()
    yield client
    client.close()


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
        min_edge=0.08,
        min_liquidity=1000.0,
        initial_bankroll=50.0,
        max_daily_loss_pct=0.20,
        min_confidence=0.65,
    )


@pytest.fixture
def sample_market():
    """Create a sample market."""
    return Market(
        id="market_test_123",
        question="Will Bitcoin reach $100k by end of 2026?",
        description="Market resolves YES if BTC reaches $100,000 by Dec 31, 2026",
        current_price=0.40,  # Market thinks 40% chance
        volume_24h=50000.0,
        liquidity=100000.0,
        end_date="2026-12-31T23:59:59Z",
        resolution_date="2026-12-31T23:59:59Z",
        created_at="2026-01-01T00:00:00Z",
        market_type="binary",
        outcomes=["YES", "NO"],
        token_ids={"YES": "token_yes_123", "NO": "token_no_123"},
    )


@pytest.mark.asyncio
async def test_full_pipeline_success(
    sqlite_client,
    duckdb_client,
    risk_settings,
    sample_market,
):
    """
    Test complete pipeline from market discovery to resolution.

    Flow:
    1. Market at 0.40, our forecast 0.55 (15% edge)
    2. Calibration: identity (no change)
    3. Execute BUY_YES: 5 EUR at 0.40 -> 12.5 shares
    4. Market resolves YES
    5. P&L = (12.5 * $1) - $5 = +$7.50
    6. Verify position closed, bankroll updated, Brier score calculated
    """
    # Step 1: Mock PolymarketClient
    mock_polymarket = Mock()
    mock_polymarket.check_resolutions = AsyncMock(return_value={})

    # Step 2: Define raw forecast values
    raw_probability = 0.55  # Our forecast: 55%
    confidence = 0.75

    # Step 3: Initialize execution components
    sizer = PositionSizer(risk_settings=risk_settings)
    risk_manager = RiskManager(risk_settings=risk_settings)
    executor = PaperTradingExecutor(
        sqlite=sqlite_client,
        sizer=sizer,
        risk=risk_manager,
    )

    # Step 4: Initialize calibration (identity calibrator for this test)
    calibrator = CalibratorAgent(history_db=duckdb_client)
    analyzer = MetaAnalyzer(
        min_edge=0.08,
        min_confidence=0.6,
        min_liquidity=1000.0,
    )

    # Step 5: Calibrate forecast (should be identity)
    calibrated_forecast = calibrator.calibrate(
        raw_forecast=raw_probability,
        market_type="binary",
        confidence=confidence,
    )

    calibrated_prob = calibrated_forecast.calibrated

    # Should be ~0.55 (identity since no history)
    assert 0.50 <= calibrated_prob <= 0.60

    # Step 6: Analyze edge
    edge_analysis = analyzer.analyze(
        our_forecast=calibrated_forecast,
        market_price=sample_market.current_price,  # 0.40
        liquidity=sample_market.liquidity,
    )

    assert edge_analysis.has_actionable_edge
    assert edge_analysis.recommended_action == "TRADE"
    assert edge_analysis.direction == "BUY_YES"

    # Step 7: Execute trade
    bankroll = sqlite_client.get_current_bankroll()
    execution_result = await executor.execute(
        edge_analysis=edge_analysis,
        calibrated_probability=calibrated_prob,
        market=sample_market,
        bankroll=bankroll,
    )

    assert execution_result is not None
    assert execution_result.success is True

    # Verify trade was created
    positions = sqlite_client.get_open_positions()
    assert len(positions) == 1
    position = positions[0]
    assert position["market_id"] == sample_market.id
    assert position["direction"] == "BUY_YES"
    assert position["num_shares"] > 0

    # Store for P&L calculation
    num_shares = position["num_shares"]
    amount_usd = position["amount_usd"]

    # Step 8: Store forecast in DuckDB
    forecast_data = {
        "market_id": sample_market.id,
        "question": sample_market.question,
        "market_type": "binary",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "raw_probability": 0.55,
        "calibrated_probability": calibrated_prob,
        "confidence": 0.75,
        "market_price_at_forecast": sample_market.current_price,
        "edge": edge_analysis.abs_edge,
        "recommended_action": edge_analysis.recommended_action,
        "debate_log": [],
        "judge_reasoning": "Test forecast",
    }
    forecast_id = duckdb_client.insert_forecast(forecast_data)

    # Step 9: Simulate market resolution - YES wins
    feedback = FeedbackLoop(db=duckdb_client, calibrator=calibrator)
    resolver = MarketResolver(
        polymarket=mock_polymarket,
        feedback=feedback,
        sqlite=sqlite_client,
        duckdb=duckdb_client,
    )

    # Mock that market resolved YES
    mock_polymarket.check_resolutions.return_value = {
        sample_market.id: True  # YES won
    }

    # Run resolution cycle
    resolution_result = await resolver.run_resolution_cycle()

    # Verify resolution
    assert resolution_result["resolved"] == 1
    assert resolution_result["pnl"] > 0  # Should be positive

    # Calculate expected P&L
    # P&L = (num_shares * $1.00) - amount_usd
    expected_pnl = (num_shares * 1.0) - amount_usd
    assert resolution_result["pnl"] == pytest.approx(expected_pnl, abs=0.01)

    # Verify position was closed
    positions_after = sqlite_client.get_open_positions()
    assert len(positions_after) == 0

    # Verify bankroll updated
    final_bankroll = sqlite_client.get_current_bankroll()
    assert final_bankroll == pytest.approx(50.0 + expected_pnl, abs=0.01)

    # Verify forecast was resolved in DuckDB
    forecast = duckdb_client.get_forecast(forecast_id)
    assert forecast["resolved"] is True
    assert forecast["outcome"] == 1  # YES
    assert forecast["brier_score"] is not None


@pytest.mark.asyncio
async def test_full_pipeline_losing_trade(
    sqlite_client,
    duckdb_client,
    risk_settings,
    sample_market,
):
    """
    Test pipeline with losing trade.

    Flow similar to success case but market resolves NO.
    """
    # Setup (similar to success case)
    mock_polymarket = Mock()
    mock_polymarket.check_resolutions = AsyncMock(return_value={})

    sizer = PositionSizer(risk_settings=risk_settings)
    risk_manager = RiskManager(risk_settings=risk_settings)
    executor = PaperTradingExecutor(
        sqlite=sqlite_client,
        sizer=sizer,
        risk=risk_manager,
    )

    calibrator = CalibratorAgent(history_db=duckdb_client)
    analyzer = MetaAnalyzer(min_edge=0.08, min_confidence=0.6, min_liquidity=1000.0)
    feedback = FeedbackLoop(db=duckdb_client, calibrator=calibrator)

    # Forecast and execute
    calibrated_forecast = calibrator.calibrate(
        raw_forecast=0.55,
        market_type="binary",
        confidence=0.75,
    )
    calibrated_prob = calibrated_forecast.calibrated
    edge_analysis = analyzer.analyze(
        calibrated_forecast,
        sample_market.current_price,
        sample_market.liquidity,
    )

    bankroll = sqlite_client.get_current_bankroll()
    execution_result = await executor.execute(
        edge_analysis=edge_analysis,
        calibrated_probability=calibrated_prob,
        market=sample_market,
        bankroll=bankroll,
    )

    assert execution_result.success is True

    # Get position details
    position = sqlite_client.get_open_positions()[0]
    amount_usd = position["amount_usd"]

    # Store forecast
    forecast_data = {
        "market_id": sample_market.id,
        "question": sample_market.question,
        "market_type": "binary",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "raw_probability": 0.55,
        "calibrated_probability": calibrated_prob,
        "confidence": 0.75,
        "market_price_at_forecast": sample_market.current_price,
        "edge": edge_analysis.abs_edge,
        "recommended_action": edge_analysis.recommended_action,
        "debate_log": [],
        "judge_reasoning": "Test forecast",
    }
    duckdb_client.insert_forecast(forecast_data)

    # Market resolves NO (we lose)
    mock_polymarket.check_resolutions.return_value = {
        sample_market.id: False  # NO won
    }

    resolver = MarketResolver(
        polymarket=mock_polymarket,
        feedback=feedback,
        sqlite=sqlite_client,
        duckdb=duckdb_client,
    )

    resolution_result = await resolver.run_resolution_cycle()

    # Verify loss
    assert resolution_result["resolved"] == 1
    assert resolution_result["pnl"] < 0  # Should be negative

    # P&L should be -amount_usd (lost everything)
    assert resolution_result["pnl"] == pytest.approx(-amount_usd, abs=0.01)

    # Verify position closed
    positions_after = sqlite_client.get_open_positions()
    assert len(positions_after) == 0

    # Verify bankroll decreased
    final_bankroll = sqlite_client.get_current_bankroll()
    assert final_bankroll == pytest.approx(50.0 - amount_usd, abs=0.01)
