import pytest

from src.calibration.analyzer import MetaAnalyzer
from src.models import CalibratedForecast, EdgeAnalysis


@pytest.fixture
def analyzer():
    """Create MetaAnalyzer with default thresholds."""
    return MetaAnalyzer(
        min_edge=0.05,
        min_confidence=0.6,
        min_liquidity=10000.0,
    )


@pytest.fixture
def calibrated_forecast():
    """Create a sample calibrated forecast."""
    return CalibratedForecast(
        raw=0.65,
        calibrated=0.62,
        confidence=0.75,
        calibration_method="isotonic_regression",
        historical_samples=100,
    )


def test_analyze_actionable_edge(analyzer, calibrated_forecast):
    """Test edge analysis with actionable edge."""
    # Our forecast: 62%, Market: 50%, Edge: +12%
    analysis = analyzer.analyze(
        our_forecast=calibrated_forecast,
        market_price=0.50,
        liquidity=20000.0,
    )

    assert isinstance(analysis, EdgeAnalysis)
    assert analysis.our_forecast == 0.62
    assert analysis.market_price == 0.50
    assert analysis.raw_edge == pytest.approx(0.12)
    assert analysis.abs_edge == pytest.approx(0.12)
    assert analysis.direction == "BUY_YES"
    assert analysis.has_actionable_edge is True
    assert analysis.recommended_action == "TRADE"
    assert ("BUY_YES" in analysis.reasoning or "RECOMMENDATION" in analysis.reasoning)


def test_analyze_skip_insufficient_edge(analyzer, calibrated_forecast):
    """Test that small edge results in SKIP."""
    # Our forecast: 62%, Market: 60%, Edge: +2%
    analysis = analyzer.analyze(
        our_forecast=calibrated_forecast,
        market_price=0.60,
        liquidity=20000.0,
    )

    assert analysis.abs_edge == pytest.approx(0.02)
    assert analysis.has_actionable_edge is False
    assert analysis.recommended_action == "SKIP"
    assert "SKIP" in analysis.reasoning
    assert "Edge" in analysis.reasoning


def test_analyze_skip_low_confidence(analyzer):
    """Test that low confidence results in SKIP."""
    # Low confidence forecast
    low_conf_forecast = CalibratedForecast(
        raw=0.65,
        calibrated=0.62,
        confidence=0.4,  # Below threshold
        calibration_method="identity",
        historical_samples=10,
    )

    # Large edge but low confidence
    analysis = analyzer.analyze(
        our_forecast=low_conf_forecast,
        market_price=0.50,
        liquidity=20000.0,
    )

    assert analysis.abs_edge == pytest.approx(0.12)  # Edge is large
    assert analysis.has_actionable_edge is False  # But confidence too low
    assert analysis.recommended_action == "SKIP"
    assert "Confidence" in analysis.reasoning


def test_analyze_skip_low_liquidity(analyzer, calibrated_forecast):
    """Test that low liquidity results in SKIP."""
    # Large edge and good confidence but low liquidity
    analysis = analyzer.analyze(
        our_forecast=calibrated_forecast,
        market_price=0.50,
        liquidity=5000.0,  # Below threshold
    )

    assert analysis.abs_edge == pytest.approx(0.12)
    assert analysis.has_actionable_edge is False
    assert analysis.recommended_action == "SKIP"
    assert "Liquidity" in analysis.reasoning


def test_analyze_buy_no_direction(analyzer, calibrated_forecast):
    """Test BUY_NO direction when market overpriced."""
    # Our forecast: 62%, Market: 75%, Edge: -13%
    analysis = analyzer.analyze(
        our_forecast=calibrated_forecast,
        market_price=0.75,
        liquidity=20000.0,
    )

    assert analysis.raw_edge == pytest.approx(-0.13)
    assert analysis.abs_edge == pytest.approx(0.13)
    assert analysis.direction == "BUY_NO"
    assert analysis.has_actionable_edge is True
    assert analysis.recommended_action == "TRADE"


def test_analyze_weighted_edge(analyzer, calibrated_forecast):
    """Test weighted edge calculation."""
    analysis = analyzer.analyze(
        our_forecast=calibrated_forecast,
        market_price=0.50,
        liquidity=20000.0,
    )

    # Weighted edge = raw_edge * confidence
    expected_weighted = 0.12 * 0.75
    assert analysis.weighted_edge == pytest.approx(expected_weighted)


def test_analyze_invalid_market_price(analyzer, calibrated_forecast):
    """Test that invalid market price raises error."""
    with pytest.raises(ValueError, match="market_price must be in"):
        analyzer.analyze(
            our_forecast=calibrated_forecast,
            market_price=1.5,
            liquidity=20000.0,
        )


def test_analyze_invalid_liquidity(analyzer, calibrated_forecast):
    """Test that negative liquidity raises error."""
    with pytest.raises(ValueError, match="liquidity must be non-negative"):
        analyzer.analyze(
            our_forecast=calibrated_forecast,
            market_price=0.50,
            liquidity=-100.0,
        )


def test_check_actionable_edge_all_pass(analyzer):
    """Test actionable edge check when all criteria pass."""
    result = analyzer._check_actionable_edge(
        abs_edge=0.10,
        confidence=0.8,
        liquidity=20000.0,
    )
    assert result is True


def test_check_actionable_edge_edge_fail(analyzer):
    """Test actionable edge check when edge too small."""
    result = analyzer._check_actionable_edge(
        abs_edge=0.02,  # Below threshold
        confidence=0.8,
        liquidity=20000.0,
    )
    assert result is False


def test_check_actionable_edge_confidence_fail(analyzer):
    """Test actionable edge check when confidence too low."""
    result = analyzer._check_actionable_edge(
        abs_edge=0.10,
        confidence=0.4,  # Below threshold
        liquidity=20000.0,
    )
    assert result is False


def test_check_actionable_edge_liquidity_fail(analyzer):
    """Test actionable edge check when liquidity too low."""
    result = analyzer._check_actionable_edge(
        abs_edge=0.10,
        confidence=0.8,
        liquidity=5000.0,  # Below threshold
    )
    assert result is False


def test_check_actionable_edge_multiple_fail(analyzer):
    """Test actionable edge check when multiple criteria fail."""
    result = analyzer._check_actionable_edge(
        abs_edge=0.02,  # Fail
        confidence=0.4,  # Fail
        liquidity=5000.0,  # Fail
    )
    assert result is False


def test_generate_reasoning_trade(analyzer, calibrated_forecast):
    """Test reasoning generation for TRADE recommendation."""
    reasoning = analyzer._generate_reasoning(
        our_forecast=calibrated_forecast,
        market_price=0.50,
        abs_edge=0.12,
        confidence=0.75,
        liquidity=20000.0,
        has_actionable_edge=True,
        direction="BUY_YES",
    )

    # Should contain key information
    assert "62.0%" in reasoning or "62%" in reasoning  # Our forecast
    assert "50.0%" in reasoning or "50%" in reasoning  # Market price
    assert "BUY_YES" in reasoning
    assert "RECOMMENDATION" in reasoning
    assert "✓" in reasoning  # Checkmarks for passing criteria


def test_generate_reasoning_skip(analyzer, calibrated_forecast):
    """Test reasoning generation for SKIP recommendation."""
    reasoning = analyzer._generate_reasoning(
        our_forecast=calibrated_forecast,
        market_price=0.60,
        abs_edge=0.02,
        confidence=0.75,
        liquidity=20000.0,
        has_actionable_edge=False,
        direction="BUY_YES",
    )

    # Should explain why skipping
    assert "SKIP" in reasoning
    assert "✗" in reasoning or "not met" in reasoning.lower()


def test_custom_thresholds():
    """Test MetaAnalyzer with custom thresholds."""
    custom_analyzer = MetaAnalyzer(
        min_edge=0.10,  # Higher edge requirement
        min_confidence=0.8,  # Higher confidence requirement
        min_liquidity=50000.0,  # Higher liquidity requirement
    )

    forecast = CalibratedForecast(
        raw=0.65,
        calibrated=0.62,
        confidence=0.75,  # Below custom threshold
        calibration_method="identity",
        historical_samples=10,
    )

    analysis = custom_analyzer.analyze(
        our_forecast=forecast,
        market_price=0.50,
        liquidity=20000.0,  # Below custom threshold
    )

    # Should SKIP due to custom thresholds
    assert analysis.recommended_action == "SKIP"


def test_edge_at_threshold_boundary(analyzer, calibrated_forecast):
    """Test edge detection at exact threshold."""
    # Edge exactly at threshold
    # Our forecast: 62%, Market: 57%, Edge: 5%
    analysis = analyzer.analyze(
        our_forecast=calibrated_forecast,
        market_price=0.57,
        liquidity=20000.0,
    )

    # At threshold should pass
    assert analysis.abs_edge == pytest.approx(0.05)
    assert analysis.has_actionable_edge is True
    assert analysis.recommended_action == "TRADE"


def test_confidence_at_threshold_boundary(analyzer):
    """Test confidence at exact threshold."""
    forecast = CalibratedForecast(
        raw=0.65,
        calibrated=0.62,
        confidence=0.6,  # Exactly at threshold
        calibration_method="identity",
        historical_samples=10,
    )

    analysis = analyzer.analyze(
        our_forecast=forecast,
        market_price=0.50,
        liquidity=20000.0,
    )

    # At threshold should pass
    assert analysis.has_actionable_edge is True
    assert analysis.recommended_action == "TRADE"
