from datetime import datetime, timezone
from unittest.mock import AsyncMock, Mock, patch
import pytest

from bot.strategy.orchestrator import DebateOrchestrator
from bot.strategy.bull import BullAgent
from bot.strategy.bear import BearAgent
from bot.strategy.devil import DevilsAdvocateAgent
from bot.strategy.judge import JudgeAgent
from bot.models import AgentRole, AgentResponse, SimpleForecast


@pytest.fixture
def mock_ollama():
    """Create a mock OllamaClient."""
    ollama = Mock()
    ollama.model = "mistral"
    ollama.generate = AsyncMock(return_value="Test response")
    ollama.close = AsyncMock()
    return ollama


@pytest.fixture
def mock_agents(mock_ollama):
    """Create mock agents."""
    bull = Mock(spec=BullAgent)
    bull.generate = AsyncMock()
    bull.ollama = mock_ollama

    bear = Mock(spec=BearAgent)
    bear.generate = AsyncMock()
    bear.ollama = mock_ollama

    devil = Mock(spec=DevilsAdvocateAgent)
    devil.generate = AsyncMock()
    devil.ollama = mock_ollama

    judge = Mock(spec=JudgeAgent)
    judge.generate = AsyncMock()
    judge.parse_forecast = Mock()
    judge.ollama = mock_ollama

    return bull, bear, devil, judge


@pytest.mark.asyncio
async def test_orchestrator_single_round(mock_agents):
    """Test orchestrator with a single debate round."""
    bull, bear, devil, judge = mock_agents

    # Setup mock responses
    bull_response = AgentResponse(
        role=AgentRole.BULL,
        content="Bull argument: YES will happen.\n\nP(YES) = 65%",
        round_number=1,
        probability=0.65,
        timestamp=datetime.now(timezone.utc),
    )

    bear_response = AgentResponse(
        role=AgentRole.BEAR,
        content="Bear argument: NO will happen.\n\nP(YES) = 30%",
        round_number=1,
        probability=0.30,
        timestamp=datetime.now(timezone.utc),
    )

    devil_response = AgentResponse(
        role=AgentRole.DEVIL,
        content="Critique: Both arguments have flaws.",
        round_number=1,
        probability=None,
        timestamp=datetime.now(timezone.utc),
    )

    bull_rebuttal = AgentResponse(
        role=AgentRole.BULL,
        content="Rebuttal: My position stands.\n\nP(YES) = 63%",
        round_number=1,
        probability=0.63,
        timestamp=datetime.now(timezone.utc),
    )

    bear_rebuttal = AgentResponse(
        role=AgentRole.BEAR,
        content="Rebuttal: NO is still likely.\n\nP(YES) = 32%",
        round_number=1,
        probability=0.32,
        timestamp=datetime.now(timezone.utc),
    )

    judge_response = AgentResponse(
        role=AgentRole.JUDGE,
        content="Final Forecast\n\nP(YES) = 48%\n\nConfidence Interval: 40% to 55%",
        round_number=2,
        probability=0.48,
        timestamp=datetime.now(timezone.utc),
    )

    # Configure mocks
    bull.generate.side_effect = [bull_response, bull_rebuttal]
    bear.generate.side_effect = [bear_response, bear_rebuttal]
    devil.generate.return_value = devil_response
    judge.generate.return_value = judge_response
    judge.parse_forecast.return_value = {
        "probability": 0.48,
        "lower_bound": 0.40,
        "upper_bound": 0.55,
    }

    # Create orchestrator and run debate
    orchestrator = DebateOrchestrator(
        bull=bull,
        bear=bear,
        devil=devil,
        judge=judge,
    )

    forecast = await orchestrator.run_debate(
        market_id="test_market",
        context="Test context",
        rounds=1,
    )

    # Verify forecast
    assert isinstance(forecast, SimpleForecast)
    assert forecast.market_id == "test_market"
    assert forecast.probability == 0.48
    assert forecast.confidence_lower == 0.40
    assert forecast.confidence_upper == 0.55
    assert forecast.debate_rounds == 1
    assert len(forecast.bull_probabilities) == 2  # Initial + rebuttal
    assert len(forecast.bear_probabilities) == 2
    assert forecast.bull_probabilities == [0.65, 0.63]
    assert forecast.bear_probabilities == [0.30, 0.32]

    # Verify agent calls
    assert bull.generate.call_count == 2  # Initial + rebuttal
    assert bear.generate.call_count == 2
    assert devil.generate.call_count == 1
    assert judge.generate.call_count == 1


@pytest.mark.asyncio
async def test_orchestrator_multiple_rounds(mock_agents):
    """Test orchestrator with multiple debate rounds."""
    bull, bear, devil, judge = mock_agents

    # Setup mock responses for 2 rounds
    bull_responses = [
        AgentResponse(
            role=AgentRole.BULL,
            content=f"Round {r} bull.\n\nP(YES) = {60+r}%",
            round_number=r,
            probability=0.60 + r * 0.01,
            timestamp=datetime.now(timezone.utc),
        )
        for r in [1, 1, 2, 2]  # Initial + rebuttal for each round
    ]

    bear_responses = [
        AgentResponse(
            role=AgentRole.BEAR,
            content=f"Round {r} bear.\n\nP(YES) = {35+r}%",
            round_number=r,
            probability=0.35 + r * 0.01,
            timestamp=datetime.now(timezone.utc),
        )
        for r in [1, 1, 2, 2]
    ]

    devil_responses = [
        AgentResponse(
            role=AgentRole.DEVIL,
            content=f"Round {r} critique.",
            round_number=r,
            probability=None,
            timestamp=datetime.now(timezone.utc),
        )
        for r in [1, 2]
    ]

    judge_response = AgentResponse(
        role=AgentRole.JUDGE,
        content="Final P(YES) = 50%",
        round_number=3,
        probability=0.50,
        timestamp=datetime.now(timezone.utc),
    )

    bull.generate.side_effect = bull_responses
    bear.generate.side_effect = bear_responses
    devil.generate.side_effect = devil_responses
    judge.generate.return_value = judge_response
    judge.parse_forecast.return_value = {
        "probability": 0.50,
        "lower_bound": None,
        "upper_bound": None,
    }

    orchestrator = DebateOrchestrator(
        bull=bull,
        bear=bear,
        devil=devil,
        judge=judge,
    )

    forecast = await orchestrator.run_debate(
        market_id="test_market",
        context="Test context",
        rounds=2,
    )

    # Verify
    assert forecast.debate_rounds == 2
    assert len(forecast.bull_probabilities) == 4  # 2 rounds * 2 phases
    assert len(forecast.bear_probabilities) == 4
    assert bull.generate.call_count == 4
    assert bear.generate.call_count == 4
    assert devil.generate.call_count == 2
    assert judge.generate.call_count == 1


@pytest.mark.asyncio
async def test_orchestrator_no_confidence_interval(mock_agents):
    """Test orchestrator when judge doesn't provide confidence interval."""
    bull, bear, devil, judge = mock_agents

    # Setup minimal responses
    bull.generate.return_value = AgentResponse(
        role=AgentRole.BULL,
        content="Bull.\n\nP(YES) = 60%",
        round_number=1,
        probability=0.60,
        timestamp=datetime.now(timezone.utc),
    )

    bear.generate.return_value = AgentResponse(
        role=AgentRole.BEAR,
        content="Bear.\n\nP(YES) = 40%",
        round_number=1,
        probability=0.40,
        timestamp=datetime.now(timezone.utc),
    )

    devil.generate.return_value = AgentResponse(
        role=AgentRole.DEVIL,
        content="Critique.",
        round_number=1,
        probability=None,
        timestamp=datetime.now(timezone.utc),
    )

    judge.generate.return_value = AgentResponse(
        role=AgentRole.JUDGE,
        content="Final P(YES) = 52%",
        round_number=2,
        probability=0.52,
        timestamp=datetime.now(timezone.utc),
    )

    judge.parse_forecast.return_value = {
        "probability": 0.52,
        "lower_bound": None,
        "upper_bound": None,
    }

    orchestrator = DebateOrchestrator(
        bull=bull,
        bear=bear,
        devil=devil,
        judge=judge,
    )

    forecast = await orchestrator.run_debate(
        market_id="test_market",
        context="Test context",
        rounds=1,
    )

    assert forecast.probability == 0.52
    assert forecast.confidence_lower is None
    assert forecast.confidence_upper is None


@pytest.mark.asyncio
async def test_orchestrator_with_verbose(mock_agents):
    """Test orchestrator verbose mode runs without error."""
    bull, bear, devil, judge = mock_agents

    # Setup minimal responses
    bull.generate.return_value = AgentResponse(
        role=AgentRole.BULL,
        content="Bull.\n\nP(YES) = 60%",
        round_number=1,
        probability=0.60,
        timestamp=datetime.now(timezone.utc),
    )

    bear.generate.return_value = AgentResponse(
        role=AgentRole.BEAR,
        content="Bear.\n\nP(YES) = 40%",
        round_number=1,
        probability=0.40,
        timestamp=datetime.now(timezone.utc),
    )

    devil.generate.return_value = AgentResponse(
        role=AgentRole.DEVIL,
        content="Critique.",
        round_number=1,
        probability=None,
        timestamp=datetime.now(timezone.utc),
    )

    judge.generate.return_value = AgentResponse(
        role=AgentRole.JUDGE,
        content="Final P(YES) = 50%",
        round_number=2,
        probability=0.50,
        timestamp=datetime.now(timezone.utc),
    )

    judge.parse_forecast.return_value = {
        "probability": 0.50,
        "lower_bound": None,
        "upper_bound": None,
    }

    orchestrator = DebateOrchestrator(
        bull=bull,
        bear=bear,
        devil=devil,
        judge=judge,
    )

    # Should complete without error
    forecast = await orchestrator.run_debate(
        market_id="test_market",
        context="Test context",
        rounds=1,
        verbose=True,
    )

    assert forecast is not None
    assert forecast.probability == 0.50


@pytest.mark.asyncio
async def test_orchestrator_preserves_arguments_for_judge(mock_agents):
    """Test that judge receives all debate arguments."""
    bull, bear, devil, judge = mock_agents

    bull.generate.return_value = AgentResponse(
        role=AgentRole.BULL,
        content="Bull argument",
        round_number=1,
        probability=0.60,
        timestamp=datetime.now(timezone.utc),
    )

    bear.generate.return_value = AgentResponse(
        role=AgentRole.BEAR,
        content="Bear argument",
        round_number=1,
        probability=0.40,
        timestamp=datetime.now(timezone.utc),
    )

    devil.generate.return_value = AgentResponse(
        role=AgentRole.DEVIL,
        content="Devil critique",
        round_number=1,
        probability=None,
        timestamp=datetime.now(timezone.utc),
    )

    judge.generate.return_value = AgentResponse(
        role=AgentRole.JUDGE,
        content="Final forecast",
        round_number=2,
        probability=0.50,
        timestamp=datetime.now(timezone.utc),
    )

    judge.parse_forecast.return_value = {
        "probability": 0.50,
        "lower_bound": None,
        "upper_bound": None,
    }

    orchestrator = DebateOrchestrator(
        bull=bull,
        bear=bear,
        devil=devil,
        judge=judge,
    )

    await orchestrator.run_debate(
        market_id="test_market",
        context="Test context",
        rounds=1,
    )

    # Check judge was called with previous_arguments
    judge.generate.assert_called_once()
    call_kwargs = judge.generate.call_args[1]
    assert "previous_arguments" in call_kwargs
    previous_args = call_kwargs["previous_arguments"]

    # Should have: Bull initial, Bear initial, Devil, Bull rebuttal, Bear rebuttal
    assert len(previous_args) == 5
    assert previous_args[0]["role"] == "BULL"
    assert previous_args[1]["role"] == "BEAR"
    assert previous_args[2]["role"] == "DEVIL"
    assert previous_args[3]["role"] == "BULL_REBUTTAL"
    assert previous_args[4]["role"] == "BEAR_REBUTTAL"


def test_get_debate_summary():
    """Test debate summary generation."""
    mock_ollama = Mock()
    mock_ollama.model = "mistral"

    orchestrator = DebateOrchestrator(
        bull=Mock(),
        bear=Mock(),
        devil=Mock(),
        judge=Mock(),
    )

    forecast = SimpleForecast(
        market_id="test_market",
        probability=0.55,
        confidence_lower=0.45,
        confidence_upper=0.65,
        reasoning="Test reasoning",
        created_at=datetime(2026, 2, 12, 12, 0, 0, tzinfo=timezone.utc),
        model_name="mistral",
        debate_rounds=2,
        bull_probabilities=[0.60, 0.58],
        bear_probabilities=[0.40, 0.42],
    )

    summary = orchestrator.get_debate_summary(forecast)

    assert "test_market" in summary
    assert "55.0%" in summary or "55%" in summary
    assert "45.0%" in summary or "45%" in summary
    assert "65.0%" in summary or "65%" in summary
    assert "Test reasoning" in summary
    assert "mistral" in summary
    assert "Round 1" in summary
    assert "Round 2" in summary


def test_get_debate_summary_no_confidence():
    """Test debate summary without confidence intervals."""
    orchestrator = DebateOrchestrator(
        bull=Mock(),
        bear=Mock(),
        devil=Mock(),
        judge=Mock(),
    )

    forecast = SimpleForecast(
        market_id="test_market",
        probability=0.55,
        confidence_lower=None,
        confidence_upper=None,
        reasoning="Test reasoning",
        created_at=datetime(2026, 2, 12, 12, 0, 0, tzinfo=timezone.utc),
        model_name="mistral",
        debate_rounds=1,
        bull_probabilities=[0.60],
        bear_probabilities=[0.40],
    )

    summary = orchestrator.get_debate_summary(forecast)

    assert "test_market" in summary
    assert "55.0%" in summary or "55%" in summary
    assert "Confidence Interval" not in summary
