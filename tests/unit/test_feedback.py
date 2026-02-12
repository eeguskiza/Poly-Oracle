from datetime import datetime, timezone
from unittest.mock import Mock, MagicMock
import pytest

from src.calibration.feedback import FeedbackLoop
from src.calibration.calibrator import CalibratorAgent
from src.models import SimpleForecast, Market


@pytest.fixture
def mock_duckdb_client():
    """Create a mock DuckDB client."""
    mock_client = Mock()
    mock_client.conn = Mock()
    mock_client.insert_forecast = Mock()
    mock_client.get_forecast = Mock()
    return mock_client


@pytest.fixture
def mock_calibrator():
    """Create a mock CalibratorAgent."""
    return Mock(spec=CalibratorAgent)


@pytest.fixture
def feedback_loop(mock_duckdb_client, mock_calibrator):
    """Create FeedbackLoop with mocks."""
    return FeedbackLoop(db=mock_duckdb_client, calibrator=mock_calibrator)


@pytest.fixture
def sample_forecast():
    """Create a sample SimpleForecast."""
    return SimpleForecast(
        market_id="test_market_123",
        probability=0.65,
        confidence_lower=0.55,
        confidence_upper=0.75,
        reasoning="Test reasoning",
        created_at=datetime(2026, 2, 12, 12, 0, 0, tzinfo=timezone.utc),
        model_name="mistral",
        debate_rounds=2,
        bull_probabilities=[0.70, 0.68],
        bear_probabilities=[0.40, 0.42],
    )


@pytest.fixture
def sample_market():
    """Create a sample Market."""
    return Market(
        id="test_market_123",
        question="Will this test pass?",
        description="A test market",
        current_price=0.60,
        volume_24h=50000.0,
        liquidity=100000.0,
        end_date="2026-12-31T23:59:59Z",
        resolution_date="2026-12-31T23:59:59Z",
        created_at="2026-01-01T00:00:00Z",
        market_type="binary",
        outcomes=["YES", "NO"],
        token_ids={"YES": "123", "NO": "456"},
    )


def test_record_forecast(feedback_loop, mock_duckdb_client, sample_forecast, sample_market):
    """Test recording a forecast."""
    feedback_loop.record_forecast(
        forecast=sample_forecast,
        market=sample_market,
        calibrated_probability=0.62,
        edge=0.02,
        recommended_action="SKIP",
    )

    # Should call insert_forecast
    mock_duckdb_client.insert_forecast.assert_called_once()
    call_args = mock_duckdb_client.insert_forecast.call_args[0][0]

    # Verify forecast data
    assert call_args["market_id"] == "test_market_123"
    assert call_args["raw_probability"] == 0.65
    assert call_args["calibrated_probability"] == 0.62
    assert call_args["market_price_at_forecast"] == 0.60
    assert call_args["edge"] == 0.02
    assert call_args["recommended_action"] == "SKIP"
    assert call_args["outcome"] is None


def test_record_forecast_error_handling(feedback_loop, mock_duckdb_client, sample_forecast, sample_market):
    """Test error handling in record_forecast."""
    # Mock error
    mock_duckdb_client.insert_forecast.side_effect = Exception("Database error")

    with pytest.raises(Exception, match="Database error"):
        feedback_loop.record_forecast(
            forecast=sample_forecast,
            market=sample_market,
            calibrated_probability=0.62,
        )


def test_process_resolution_success(feedback_loop, mock_duckdb_client):
    """Test successful resolution processing."""
    # Mock forecast data
    mock_forecast = {
        "market_id": "test_market_123",
        "raw_probability": 0.65,
        "calibrated_probability": 0.62,
        "market_type": "binary",
    }
    mock_duckdb_client.get_forecast.return_value = mock_forecast

    # Mock execute for UPDATE
    mock_duckdb_client.conn.execute = Mock()

    result = feedback_loop.process_resolution(
        market_id="test_market_123",
        outcome=True,  # YES
    )

    assert result["success"] is True
    assert result["market_id"] == "test_market_123"
    assert result["outcome"] is True

    # Brier scores should be calculated
    # Raw: (0.65 - 1)^2 = 0.1225
    # Calibrated: (0.62 - 1)^2 = 0.1444
    assert result["brier_score_raw"] == pytest.approx(0.1225)
    assert result["brier_score_calibrated"] == pytest.approx(0.1444)
    assert "improvement" in result

    # Should have called execute to update
    mock_duckdb_client.conn.execute.assert_called_once()


def test_process_resolution_outcome_no(feedback_loop, mock_duckdb_client):
    """Test resolution with NO outcome."""
    mock_forecast = {
        "market_id": "test_market_123",
        "raw_probability": 0.30,
        "calibrated_probability": 0.28,
        "market_type": "binary",
    }
    mock_duckdb_client.get_forecast.return_value = mock_forecast
    mock_duckdb_client.conn.execute = Mock()

    result = feedback_loop.process_resolution(
        market_id="test_market_123",
        outcome=False,  # NO
    )

    assert result["success"] is True
    assert result["outcome"] is False

    # Brier scores for NO outcome (outcome = 0)
    # Raw: (0.30 - 0)^2 = 0.09
    # Calibrated: (0.28 - 0)^2 = 0.0784
    assert result["brier_score_raw"] == pytest.approx(0.09)
    assert result["brier_score_calibrated"] == pytest.approx(0.0784)


def test_process_resolution_forecast_not_found(feedback_loop, mock_duckdb_client):
    """Test resolution when forecast not found."""
    mock_duckdb_client.get_forecast.return_value = None

    result = feedback_loop.process_resolution(
        market_id="nonexistent_market",
        outcome=True,
    )

    assert result["success"] is False
    assert "error" in result
    assert "not found" in result["error"].lower()


def test_process_resolution_recalibration_threshold(feedback_loop, mock_duckdb_client):
    """Test that recalibration threshold is tracked."""
    mock_forecast = {
        "market_id": f"test_market",
        "raw_probability": 0.65,
        "calibrated_probability": 0.62,
        "market_type": "binary",
    }
    mock_duckdb_client.get_forecast.return_value = mock_forecast
    mock_duckdb_client.conn.execute = Mock()

    # Process multiple resolutions
    for i in range(12):  # Above threshold of 10
        result = feedback_loop.process_resolution(
            market_id=f"test_market",
            outcome=True,
        )
        assert result["success"] is True

    # Counter should have been reset after hitting threshold
    assert feedback_loop._forecasts_since_last_calibration.get("binary", 0) < 12


def test_get_performance_summary_no_data(feedback_loop, mock_duckdb_client):
    """Test performance summary with no data."""
    # Mock empty database
    def mock_execute(query):
        result = Mock()
        if "SELECT COUNT(*) FROM forecasts" in query and "WHERE" not in query:
            result.fetchone.return_value = (0,)
        elif "WHERE outcome IS NOT NULL" in query and "SELECT COUNT" in query:
            result.fetchone.return_value = (0,)
        else:
            result.fetchone.return_value = (0, 0)
            result.fetchall.return_value = []
        return result

    mock_duckdb_client.conn.execute = mock_execute

    summary = feedback_loop.get_performance_summary()

    assert summary["total_forecasts"] == 0
    assert summary["resolved_forecasts"] == 0
    assert "message" in summary


def test_get_performance_summary_with_data(feedback_loop, mock_duckdb_client):
    """Test performance summary with resolved forecasts."""
    def mock_execute(query):
        result = Mock()

        # Total forecasts
        if "SELECT COUNT(*) FROM forecasts" in query and "WHERE" not in query:
            result.fetchone.return_value = (10,)
            result.fetchall.return_value = []
        # Resolved forecasts
        elif "WHERE outcome IS NOT NULL" in query and "SELECT COUNT" in query:
            result.fetchone.return_value = (5,)
            result.fetchall.return_value = []
        # Brier scores
        elif "AVG(brier_score" in query and "GROUP BY" not in query:
            result.fetchone.return_value = (0.15, 0.12)  # raw, calibrated
            result.fetchall.return_value = []
        # Brier by type
        elif "GROUP BY market_type" in query:
            result.fetchone.return_value = None
            result.fetchall.return_value = [
                ("binary", 5, 0.15, 0.12),
            ]
        # Win rate
        elif "win_rate" in query.lower():
            result.fetchone.return_value = (0.6,)
            result.fetchall.return_value = []
        # Avg edge
        elif "avg_edge" in query.lower():
            result.fetchone.return_value = (0.08,)
            result.fetchall.return_value = []
        # Market Brier
        elif "market_brier" in query.lower():
            result.fetchone.return_value = (0.18,)
            result.fetchall.return_value = []
        else:
            result.fetchone.return_value = (0,)
            result.fetchall.return_value = []

        return result

    mock_duckdb_client.conn.execute = mock_execute

    summary = feedback_loop.get_performance_summary()

    assert summary["total_forecasts"] == 10
    assert summary["resolved_forecasts"] == 5
    assert summary["pending_forecasts"] == 5
    assert summary["overall_brier_raw"] == 0.15
    assert summary["overall_brier_calibrated"] == 0.12
    assert summary["calibration_improvement"] == pytest.approx(0.03)
    assert "brier_by_type" in summary
    assert summary["win_rate"] == 0.6
    assert summary["avg_edge"] == 0.08
    assert summary["market_brier"] == 0.18


def test_get_performance_summary_value_added(feedback_loop, mock_duckdb_client):
    """Test value added calculation in performance summary."""
    def mock_execute(query):
        result = Mock()

        if "SELECT COUNT(*) FROM forecasts" in query and "WHERE" not in query:
            result.fetchone.return_value = (10,)
            result.fetchall.return_value = []
        elif "WHERE outcome IS NOT NULL" in query and "SELECT COUNT" in query:
            result.fetchone.return_value = (5,)
            result.fetchall.return_value = []
        elif "AVG(brier_score" in query and "GROUP BY" not in query:
            result.fetchone.return_value = (0.15, 0.12)
            result.fetchall.return_value = []
        elif "market_brier" in query.lower():
            result.fetchone.return_value = (0.20,)  # Market Brier worse than ours
            result.fetchall.return_value = []
        elif "GROUP BY" in query:
            result.fetchone.return_value = None
            result.fetchall.return_value = []
        else:
            result.fetchone.return_value = (0,)
            result.fetchall.return_value = []

        return result

    mock_duckdb_client.conn.execute = mock_execute

    summary = feedback_loop.get_performance_summary()

    # Value added = market_brier - our_brier = 0.20 - 0.12 = 0.08
    assert summary["value_added_vs_market"] == pytest.approx(0.08)


def test_get_performance_summary_error_handling(feedback_loop, mock_duckdb_client):
    """Test error handling in performance summary."""
    mock_duckdb_client.conn.execute.side_effect = Exception("Database error")

    summary = feedback_loop.get_performance_summary()

    assert "error" in summary


def test_brier_score_calculation_perfect(feedback_loop, mock_duckdb_client):
    """Test Brier score for perfect prediction."""
    mock_forecast = {
        "market_id": "test_market",
        "raw_probability": 1.0,
        "calibrated_probability": 1.0,
        "market_type": "binary",
    }
    mock_duckdb_client.get_forecast.return_value = mock_forecast
    mock_duckdb_client.conn.execute = Mock()

    result = feedback_loop.process_resolution(
        market_id="test_market",
        outcome=True,  # Predicted 100%, outcome YES
    )

    # Perfect prediction: Brier = (1.0 - 1.0)^2 = 0
    assert result["brier_score_raw"] == 0.0
    assert result["brier_score_calibrated"] == 0.0


def test_brier_score_calculation_worst(feedback_loop, mock_duckdb_client):
    """Test Brier score for worst prediction."""
    mock_forecast = {
        "market_id": "test_market",
        "raw_probability": 0.0,
        "calibrated_probability": 0.0,
        "market_type": "binary",
    }
    mock_duckdb_client.get_forecast.return_value = mock_forecast
    mock_duckdb_client.conn.execute = Mock()

    result = feedback_loop.process_resolution(
        market_id="test_market",
        outcome=True,  # Predicted 0%, outcome YES
    )

    # Worst prediction: Brier = (0.0 - 1.0)^2 = 1.0
    assert result["brier_score_raw"] == 1.0
    assert result["brier_score_calibrated"] == 1.0
