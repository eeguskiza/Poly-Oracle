from unittest.mock import Mock, MagicMock
import pytest
import numpy as np

from src.calibration.calibrator import CalibratorAgent
from src.models import CalibratedForecast


@pytest.fixture
def mock_duckdb_client():
    """Create a mock DuckDB client."""
    mock_client = Mock()
    mock_client.conn = Mock()
    return mock_client


@pytest.fixture
def calibrator(mock_duckdb_client):
    """Create a CalibratorAgent with mock database."""
    return CalibratorAgent(history_db=mock_duckdb_client)


def test_calibrate_identity_insufficient_data(calibrator, mock_duckdb_client):
    """Test that identity calibration is used when insufficient data."""
    # Mock query to return few samples
    mock_result = [
        (0.6, 1),
        (0.7, 1),
        (0.4, 0),
    ]
    mock_duckdb_client.conn.execute.return_value.fetchall.return_value = mock_result

    # Calibrate
    result = calibrator.calibrate(
        raw_forecast=0.65,
        market_type="binary",
        confidence=0.8,
    )

    # Should use identity calibration
    assert isinstance(result, CalibratedForecast)
    assert result.raw == 0.65
    assert result.calibrated == 0.65  # No adjustment
    assert result.calibration_method == "identity"
    assert result.historical_samples == 3


def test_calibrate_with_sufficient_data(calibrator, mock_duckdb_client):
    """Test calibration with sufficient historical data."""
    # Create synthetic data: overconfident forecasts
    # High predictions should be calibrated down
    mock_result = [
        (0.9, 0) for _ in range(20)  # 20 wrong predictions at 0.9
    ] + [
        (0.9, 1) for _ in range(30)  # 30 correct predictions at 0.9
    ] + [
        (0.1, 1) for _ in range(20)  # 20 wrong predictions at 0.1
    ] + [
        (0.1, 0) for _ in range(30)  # 30 correct predictions at 0.1
    ]

    mock_duckdb_client.conn.execute.return_value.fetchall.return_value = mock_result

    # Calibrate a high confidence prediction
    result = calibrator.calibrate(
        raw_forecast=0.9,
        market_type="binary",
        confidence=0.8,
    )

    # Should use isotonic regression
    assert result.calibration_method == "isotonic_regression"
    assert result.historical_samples == 100
    # Calibrated value should exist and be different from raw
    assert result.calibrated is not None
    # With this data, 0.9 predicts 60% accuracy, so might be calibrated down
    assert 0.0 <= result.calibrated <= 1.0


def test_shrink_extremes_high(calibrator):
    """Test shrinking of extreme high predictions."""
    # High probability, low confidence should shrink more
    shrunk = calibrator._shrink_extremes(probability=0.95, confidence=0.5)
    # Should be pulled toward 0.5
    assert shrunk < 0.95
    assert shrunk > 0.5

    # High probability, high confidence should shrink less
    shrunk_less = calibrator._shrink_extremes(probability=0.95, confidence=0.9)
    assert shrunk_less < 0.95
    assert shrunk_less > shrunk  # Less shrinkage with higher confidence


def test_shrink_extremes_low(calibrator):
    """Test shrinking of extreme low predictions."""
    # Low probability, low confidence should shrink more
    shrunk = calibrator._shrink_extremes(probability=0.05, confidence=0.5)
    # Should be pulled toward 0.5
    assert shrunk > 0.05
    assert shrunk < 0.5

    # Low probability, high confidence should shrink less
    shrunk_less = calibrator._shrink_extremes(probability=0.05, confidence=0.9)
    assert shrunk_less > 0.05
    assert shrunk_less < shrunk  # Less shrinkage


def test_shrink_extremes_no_effect_middle(calibrator):
    """Test that middle predictions are not shrunk."""
    # Middle predictions should not be affected
    for prob in [0.3, 0.5, 0.7]:
        shrunk = calibrator._shrink_extremes(probability=prob, confidence=0.5)
        assert shrunk == prob


def test_calculate_brier_score(calibrator):
    """Test Brier score calculation."""
    # Perfect predictions
    predictions = [0.0, 1.0, 0.0, 1.0]
    outcomes = [0, 1, 0, 1]
    brier = calibrator._calculate_brier_score(predictions, outcomes)
    assert brier == 0.0

    # Worst predictions
    predictions = [1.0, 0.0, 1.0, 0.0]
    outcomes = [0, 1, 0, 1]
    brier = calibrator._calculate_brier_score(predictions, outcomes)
    assert brier == 1.0

    # Mixed predictions
    predictions = [0.7, 0.3]
    outcomes = [1, 0]
    brier = calibrator._calculate_brier_score(predictions, outcomes)
    # (0.7-1)^2 + (0.3-0)^2 = 0.09 + 0.09 = 0.18 / 2 = 0.09
    assert brier == pytest.approx(0.09)


def test_build_calibration_curve_data(calibrator):
    """Test calibration curve data building."""
    # Create synthetic data
    predictions = [0.1, 0.15, 0.2, 0.5, 0.5, 0.6, 0.9, 0.95]
    outcomes = [0, 0, 0, 1, 0, 1, 1, 1]

    curve_data = calibrator._build_calibration_curve_data(
        predictions=predictions,
        outcomes=outcomes,
        num_buckets=5,
    )

    # Should have some buckets with data
    assert len(curve_data) > 0
    # Each point should have required fields
    for point in curve_data:
        assert "predicted_prob" in point
        assert "actual_freq" in point
        assert "count" in point
        assert "bucket_range" in point
        assert 0 <= point["predicted_prob"] <= 1
        assert 0 <= point["actual_freq"] <= 1


def test_get_calibration_report_no_data(calibrator, mock_duckdb_client):
    """Test calibration report with no data."""
    # Mock empty result
    mock_duckdb_client.conn.execute.return_value.fetchall.return_value = []
    mock_duckdb_client.conn.execute.return_value.fetchone.return_value = (0,)

    report = calibrator.get_calibration_report()

    assert report["total_forecasts"] == 0
    assert report["resolved_forecasts"] == 0
    assert report["brier_score_raw"] is None


def test_get_calibration_report_with_data(calibrator, mock_duckdb_client):
    """Test calibration report with resolved forecasts."""
    # Mock data for report
    # First query: get resolved forecasts
    mock_resolved = [
        ("binary", 0.7, 0.65, 1),
        ("binary", 0.3, 0.35, 0),
        ("binary", 0.8, 0.75, 1),
    ]

    # Mock query execution
    call_count = [0]

    def mock_execute(query, params=None):
        result = Mock()
        call_count[0] += 1

        # First call: main query for resolved forecasts
        if "market_type" in query and "outcome IS NOT NULL" in query:
            result.fetchall.return_value = mock_resolved
        # Second call: total count
        elif "SELECT COUNT" in query and "WHERE" not in query:
            result.fetchone.return_value = (5,)  # 5 total forecasts
        # Other queries for Brier by type etc
        else:
            result.fetchall.return_value = []
            result.fetchone.return_value = (None,)

        return result

    mock_duckdb_client.conn.execute = mock_execute

    report = calibrator.get_calibration_report()

    assert report["total_forecasts"] == 5
    assert report["resolved_forecasts"] == 3
    assert "brier_score_raw" in report
    assert "brier_score_calibrated" in report


def test_calibrate_invalid_inputs(calibrator, mock_duckdb_client):
    """Test that calibrate validates inputs."""
    mock_duckdb_client.conn.execute.return_value.fetchall.return_value = []

    # Invalid raw_forecast
    with pytest.raises(ValueError, match="raw_forecast must be in"):
        calibrator.calibrate(
            raw_forecast=1.5,
            market_type="binary",
            confidence=0.8,
        )

    # Invalid confidence
    with pytest.raises(ValueError, match="confidence must be in"):
        calibrator.calibrate(
            raw_forecast=0.5,
            market_type="binary",
            confidence=-0.1,
        )


def test_calibration_curve_caching(calibrator, mock_duckdb_client):
    """Test that calibration curves are cached."""
    # Create sufficient data
    mock_result = [(0.5 + i * 0.01, i % 2) for i in range(50)]
    mock_duckdb_client.conn.execute.return_value.fetchall.return_value = mock_result

    # First calibration should build curve
    result1 = calibrator.calibrate(
        raw_forecast=0.6,
        market_type="binary",
        confidence=0.8,
    )

    # Check curve was cached
    assert "binary" in calibrator._calibration_curves
    assert "binary" in calibrator._last_calibration_sample_count
    assert calibrator._last_calibration_sample_count["binary"] == 50

    # Second calibration with same sample count should use cache
    result2 = calibrator.calibrate(
        raw_forecast=0.7,
        market_type="binary",
        confidence=0.8,
    )

    # Should have used cached curve
    assert result2.calibration_method == "isotonic_regression"


def test_get_historical_forecasts(calibrator, mock_duckdb_client):
    """Test fetching historical forecasts."""
    # Mock data
    mock_result = [
        (0.6, 1),
        (0.7, 1),
        (0.4, 0),
    ]
    mock_duckdb_client.conn.execute.return_value.fetchall.return_value = mock_result

    historical = calibrator._get_historical_forecasts("binary")

    assert len(historical) == 3
    assert historical[0] == {"prediction": 0.6, "outcome": 1}
    assert historical[1] == {"prediction": 0.7, "outcome": 1}
    assert historical[2] == {"prediction": 0.4, "outcome": 0}


def test_get_historical_forecasts_error_handling(calibrator, mock_duckdb_client):
    """Test error handling in fetching historical forecasts."""
    # Mock error
    mock_duckdb_client.conn.execute.side_effect = Exception("Database error")

    historical = calibrator._get_historical_forecasts("binary")

    # Should return empty list on error
    assert historical == []
