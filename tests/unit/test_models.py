from datetime import datetime, timezone, timedelta

import pytest
from pydantic import ValidationError

from src.models import (
    Market,
    MarketSnapshot,
    MarketFilter,
    DebateRound,
    RawForecast,
    CalibratedForecast,
    Forecast,
    EdgeAnalysis,
    PositionSize,
    Trade,
    Position,
    RiskCheck,
    ExecutionResult,
    AgentRole,
    DebateConfig,
    AgentResponse,
    DebateResult,
)


def test_market_price_validator() -> None:
    with pytest.raises(ValidationError):
        Market(
            id="market_1",
            question="Test question",
            description="Test description",
            market_type="binary",
            current_price=1.5,
            volume_24h=1000,
            liquidity=5000,
            resolution_date=datetime.now(timezone.utc),
            created_at=datetime.now(timezone.utc),
            outcomes=["YES", "NO"],
            token_ids={"YES": "token_yes", "NO": "token_no"},
        )


def test_market_days_until_resolution() -> None:
    future_date = datetime.now(timezone.utc) + timedelta(days=10)
    market = Market(
        id="market_1",
        question="Test question",
        description="Test description",
        market_type="binary",
        current_price=0.5,
        volume_24h=1000,
        liquidity=5000,
        resolution_date=future_date,
        created_at=datetime.now(timezone.utc),
        outcomes=["YES", "NO"],
        token_ids={"YES": "token_yes", "NO": "token_no"},
    )
    assert 9 <= market.days_until_resolution <= 11


def test_market_to_db_dict() -> None:
    now = datetime.now(timezone.utc)
    market = Market(
        id="market_1",
        question="Test question",
        description="Test description",
        market_type="binary",
        current_price=0.6,
        volume_24h=1000,
        liquidity=5000,
        resolution_date=now,
        created_at=now,
        outcomes=["YES", "NO"],
        token_ids={"YES": "token_yes", "NO": "token_no"},
    )
    db_dict = market.to_db_dict()
    assert db_dict["id"] == "market_1"
    assert db_dict["current_price"] == 0.6
    assert isinstance(db_dict["resolution_date"], str)
    assert isinstance(db_dict["created_at"], str)


def test_raw_forecast_probability_validator() -> None:
    with pytest.raises(ValidationError):
        RawForecast(
            probability=1.5,
            confidence=0.8,
            reasoning="Test reasoning",
        )


def test_raw_forecast_confidence_validator() -> None:
    with pytest.raises(ValidationError):
        RawForecast(
            probability=0.7,
            confidence=1.2,
            reasoning="Test reasoning",
        )


def test_calibrated_forecast_validator() -> None:
    with pytest.raises(ValidationError):
        CalibratedForecast(
            raw=0.7,
            calibrated=1.1,
            confidence=0.8,
            calibration_method="isotonic",
            historical_samples=100,
        )


def test_forecast_to_db_dict() -> None:
    now = datetime.now(timezone.utc)
    debate_rounds = [
        DebateRound(
            round_number=1,
            bull_argument="Bullish",
            bear_argument="Bearish",
            devil_critique="Critique",
        )
    ]
    forecast = Forecast(
        market_id="market_1",
        timestamp=now,
        raw_probability=0.7,
        calibrated_probability=0.65,
        confidence=0.8,
        debate_rounds=debate_rounds,
        judge_reasoning="Test reasoning",
        market_price_at_forecast=0.6,
        edge=0.05,
        recommended_action="TRADE",
    )
    db_dict = forecast.to_db_dict()
    assert db_dict["market_id"] == "market_1"
    assert isinstance(db_dict["timestamp"], str)
    assert len(db_dict["debate_log"]) == 1
    assert db_dict["debate_log"][0]["round_number"] == 1


def test_position_size_amount_validator() -> None:
    with pytest.raises(ValidationError):
        PositionSize(
            full_kelly=0.05,
            fractional_kelly=0.025,
            constrained_pct=0.02,
            amount_usd=-5.0,
            direction="BUY_YES",
        )


def test_trade_amount_validator() -> None:
    with pytest.raises(ValidationError):
        Trade(
            id="trade_1",
            market_id="market_1",
            timestamp=datetime.now(timezone.utc),
            direction="BUY_YES",
            amount_usd=0,
            num_shares=10.0,
            entry_price=0.6,
            status="PENDING",
        )


def test_trade_to_db_dict() -> None:
    now = datetime.now(timezone.utc)
    trade = Trade(
        id="trade_1",
        market_id="market_1",
        timestamp=now,
        direction="BUY_YES",
        amount_usd=5.0,
        num_shares=8.33,
        entry_price=0.6,
        status="PENDING",
    )
    db_dict = trade.to_db_dict()
    assert db_dict["id"] == "trade_1"
    assert db_dict["direction"] == "BUY_YES"
    assert isinstance(db_dict["timestamp"], str)


def test_position_unrealized_pnl_computed() -> None:
    position = Position(
        market_id="market_1",
        direction="BUY_YES",
        num_shares=10.0,
        amount_usd=5.0,
        avg_entry_price=0.5,
        current_price=0.6,
        unrealized_pnl=1.0,  # (0.6 - 0.5) * 10 = 1.0
        updated_at=datetime.now(timezone.utc),
    )
    assert position.unrealized_pnl == pytest.approx(1.0, abs=0.01)


def test_position_unrealized_pnl_no_current_price() -> None:
    position = Position(
        market_id="market_1",
        direction="BUY_YES",
        num_shares=10.0,
        amount_usd=5.0,
        avg_entry_price=0.5,
        current_price=0.5,  # No change in price
        unrealized_pnl=0.0,
        updated_at=datetime.now(timezone.utc),
    )
    assert position.unrealized_pnl == 0.0


def test_position_to_db_dict() -> None:
    position = Position(
        market_id="market_1",
        direction="BUY_YES",
        num_shares=10.0,
        amount_usd=5.0,
        avg_entry_price=0.5,
        current_price=0.6,
        unrealized_pnl=1.0,
        updated_at=datetime.now(timezone.utc),
    )
    db_dict = position.to_db_dict()
    assert db_dict["market_id"] == "market_1"
    assert db_dict["num_shares"] == 10.0
    assert db_dict["amount_usd"] == 5.0
    assert db_dict["unrealized_pnl"] == 1.0


def test_agent_role_enum() -> None:
    assert AgentRole.BULL == "BULL"
    assert AgentRole.BEAR == "BEAR"
    assert AgentRole.DEVIL == "DEVIL"
    assert AgentRole.JUDGE == "JUDGE"


def test_debate_config_defaults() -> None:
    config = DebateConfig()
    assert config.num_rounds == 3
    assert config.temperature == 0.7
    assert config.max_tokens_per_turn == 500


def test_debate_result_to_db_dict() -> None:
    now = datetime.now(timezone.utc)
    debate_rounds = [
        DebateRound(
            round_number=1,
            bull_argument="Bull",
            bear_argument="Bear",
            devil_critique="Devil",
        )
    ]
    forecast = RawForecast(
        probability=0.7,
        confidence=0.8,
        reasoning="Test",
    )
    config = DebateConfig()
    result = DebateResult(
        market_id="market_1",
        debate_rounds=debate_rounds,
        forecast=forecast,
        config=config,
        total_tokens=500,
        duration_seconds=10.5,
        timestamp=now,
    )
    db_dict = result.to_db_dict()
    assert db_dict["market_id"] == "market_1"
    assert db_dict["total_tokens"] == 500
    assert isinstance(db_dict["timestamp"], str)
    assert len(db_dict["debate_rounds"]) == 1


def test_execution_result_creation() -> None:
    result = ExecutionResult(
        success=True,
        trade_id="trade_123",
        message="Trade executed successfully",
        risk_check=None,
    )
    assert result.success is True
    assert result.trade_id == "trade_123"
    assert result.message == "Trade executed successfully"
    assert result.risk_check is None


def test_market_filter_optional_fields() -> None:
    filter1 = MarketFilter()
    assert filter1.min_liquidity is None
    assert filter1.market_types is None

    filter2 = MarketFilter(
        min_liquidity=1000,
        max_days_to_resolution=30,
        market_types=["binary"],
    )
    assert filter2.min_liquidity == 1000
    assert filter2.max_days_to_resolution == 30
    assert filter2.market_types == ["binary"]


def test_edge_analysis_creation() -> None:
    edge = EdgeAnalysis(
        our_forecast=0.7,
        market_price=0.6,
        raw_edge=0.1,
        abs_edge=0.1,
        weighted_edge=0.08,
        direction="BUY_YES",
        has_actionable_edge=True,
        recommended_action="TRADE",
        reasoning="Market underpriced",
    )
    assert edge.our_forecast == 0.7
    assert edge.has_actionable_edge is True


def test_risk_check_creation() -> None:
    check = RiskCheck(
        passed=True,
        violations=[],
        daily_loss_pct=0.0,
        num_open_positions=3,
        proposed_market_exposure=5.0,
    )
    assert check.passed is True
    assert len(check.violations) == 0
    assert check.num_open_positions == 3
