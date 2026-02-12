from datetime import datetime, timezone
from unittest.mock import AsyncMock, Mock, patch
from pathlib import Path

import pytest

from src.agents.base import OllamaClient, BaseAgent, load_prompt
from src.models import AgentRole, AgentResponse
from src.utils.exceptions import LLMError


@pytest.fixture
def mock_ollama_response() -> dict:
    return {
        "model": "mistral",
        "response": "This is a test response from Ollama.\n\nP(YES) = 65%\n\nJustification: Based on the evidence.",
        "total_duration": 1000000000,
        "prompt_eval_count": 50,
        "eval_count": 100,
    }


@pytest.mark.asyncio
async def test_ollama_generate(mock_ollama_response: dict) -> None:
    with patch("httpx.AsyncClient") as mock_client_class:
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = mock_ollama_response
        mock_response.raise_for_status = Mock()

        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=mock_response)
        mock_client.aclose = AsyncMock()
        mock_client_class.return_value = mock_client

        ollama = OllamaClient(base_url="http://localhost:11434", model="mistral")

        result = await ollama.generate(
            prompt="Test prompt",
            system="Test system",
            temperature=0.7,
            max_tokens=500,
        )

        assert result == mock_ollama_response["response"]
        assert mock_client.post.called


@pytest.mark.asyncio
async def test_ollama_is_available() -> None:
    with patch("httpx.AsyncClient") as mock_client_class:
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "models": [
                {"name": "mistral:latest"},
                {"name": "llama2:latest"},
            ]
        }
        mock_response.raise_for_status = Mock()

        mock_client = AsyncMock()
        mock_client.get = AsyncMock(return_value=mock_response)
        mock_client.aclose = AsyncMock()
        mock_client_class.return_value = mock_client

        ollama = OllamaClient(base_url="http://localhost:11434", model="mistral")

        is_available = await ollama.is_available()

        assert is_available is True


@pytest.mark.asyncio
async def test_ollama_is_not_available() -> None:
    with patch("httpx.AsyncClient") as mock_client_class:
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "models": [
                {"name": "llama2:latest"},
            ]
        }
        mock_response.raise_for_status = Mock()

        mock_client = AsyncMock()
        mock_client.get = AsyncMock(return_value=mock_response)
        mock_client.aclose = AsyncMock()
        mock_client_class.return_value = mock_client

        ollama = OllamaClient(base_url="http://localhost:11434", model="mistral")

        is_available = await ollama.is_available()

        assert is_available is False


@pytest.mark.asyncio
async def test_ollama_list_models() -> None:
    with patch("httpx.AsyncClient") as mock_client_class:
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "models": [
                {"name": "mistral:latest"},
                {"name": "llama2:latest"},
                {"name": "codellama:latest"},
            ]
        }
        mock_response.raise_for_status = Mock()

        mock_client = AsyncMock()
        mock_client.get = AsyncMock(return_value=mock_response)
        mock_client.aclose = AsyncMock()
        mock_client_class.return_value = mock_client

        ollama = OllamaClient(base_url="http://localhost:11434", model="mistral")

        models = await ollama.list_models()

        assert len(models) == 3
        assert "mistral:latest" in models
        assert "llama2:latest" in models


@pytest.mark.asyncio
async def test_ollama_generate_error() -> None:
    with patch("httpx.AsyncClient") as mock_client_class:
        mock_client = AsyncMock()
        mock_client.post = AsyncMock(side_effect=Exception("Connection error"))
        mock_client.aclose = AsyncMock()
        mock_client_class.return_value = mock_client

        ollama = OllamaClient(base_url="http://localhost:11434", model="mistral")

        with pytest.raises(LLMError):
            await ollama.generate(prompt="Test")


def test_load_prompt_bull() -> None:
    prompt = load_prompt("bull_agent")

    assert len(prompt) > 100
    assert "Bull Agent" in prompt or "YES" in prompt


def test_load_prompt_bear() -> None:
    prompt = load_prompt("bear_agent")

    assert len(prompt) > 100
    assert "Bear Agent" in prompt or "NO" in prompt


def test_load_prompt_devil() -> None:
    prompt = load_prompt("devil_agent")

    assert len(prompt) > 100
    assert "Devil" in prompt or "critique" in prompt.lower()


def test_load_prompt_judge() -> None:
    prompt = load_prompt("judge_agent")

    assert len(prompt) > 100
    assert "Judge" in prompt or "forecast" in prompt.lower()


def test_load_prompt_not_found() -> None:
    with pytest.raises(FileNotFoundError):
        load_prompt("nonexistent_agent")


@pytest.mark.asyncio
async def test_base_agent_generate(mock_ollama_response: dict) -> None:
    with patch("httpx.AsyncClient") as mock_client_class:
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = mock_ollama_response
        mock_response.raise_for_status = Mock()

        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=mock_response)
        mock_client.aclose = AsyncMock()
        mock_client_class.return_value = mock_client

        ollama = OllamaClient(base_url="http://localhost:11434", model="mistral")
        agent = BaseAgent(
            role=AgentRole.BULL,
            ollama=ollama,
            system_prompt="You are a test agent.",
        )

        response = await agent.generate(
            context="Test context",
            instruction="Test instruction",
            round_number=1,
        )

        assert isinstance(response, AgentResponse)
        assert response.role == AgentRole.BULL
        assert response.round_number == 1
        assert response.content == mock_ollama_response["response"]
        assert response.probability == 0.65


def test_base_agent_build_prompt() -> None:
    mock_ollama = Mock()
    agent = BaseAgent(
        role=AgentRole.BULL,
        ollama=mock_ollama,
        system_prompt="Test system",
    )

    prompt = agent._build_prompt(
        context="Market context here",
        instruction="Make your case",
    )

    assert "# Context" in prompt
    assert "Market context here" in prompt
    assert "# Your Task" in prompt
    assert "Make your case" in prompt


def test_base_agent_build_prompt_with_previous_args() -> None:
    mock_ollama = Mock()
    agent = BaseAgent(
        role=AgentRole.DEVIL,
        ollama=mock_ollama,
        system_prompt="Test system",
    )

    previous_args = [
        {"role": "BULL", "content": "Bull argument"},
        {"role": "BEAR", "content": "Bear argument"},
    ]

    prompt = agent._build_prompt(
        context="Market context",
        instruction="Critique both sides",
        previous_arguments=previous_args,
    )

    assert "# Previous Arguments" in prompt
    assert "BULL" in prompt
    assert "Bull argument" in prompt
    assert "BEAR" in prompt
    assert "Bear argument" in prompt


def test_extract_probability() -> None:
    mock_ollama = Mock()
    agent = BaseAgent(
        role=AgentRole.BULL,
        ollama=mock_ollama,
        system_prompt="Test",
    )

    text1 = "My analysis shows P(YES) = 0.65 based on evidence."
    prob1 = agent._extract_probability(text1)
    assert prob1 == 0.65

    text2 = "Probability Assessment\nP(YES) = 75%"
    prob2 = agent._extract_probability(text2)
    assert prob2 == 0.75

    text3 = "No probability mentioned here."
    prob3 = agent._extract_probability(text3)
    assert prob3 is None


def test_extract_probability_percentage() -> None:
    mock_ollama = Mock()
    agent = BaseAgent(
        role=AgentRole.JUDGE,
        ollama=mock_ollama,
        system_prompt="Test",
    )

    text = "Final Forecast\nP(YES) = 42%\nConfidence interval: 35% to 50%"
    prob = agent._extract_probability(text)
    assert prob == 0.42
