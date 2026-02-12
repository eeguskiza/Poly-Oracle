from src.agents.base import OllamaClient, BaseAgent, load_prompt
from src.agents.bull import BullAgent
from src.agents.bear import BearAgent
from src.agents.devil import DevilsAdvocateAgent
from src.agents.judge import JudgeAgent
from src.agents.orchestrator import DebateOrchestrator


def create_debate_system(
    base_url: str = "http://localhost:11434",
    model: str = "mistral",
    timeout: int = 120,
) -> tuple[DebateOrchestrator, OllamaClient]:
    """
    Create a complete debate system with all agents.

    Factory function that initializes:
    - OllamaClient
    - BullAgent
    - BearAgent
    - DevilsAdvocateAgent
    - JudgeAgent
    - DebateOrchestrator

    Args:
        base_url: Ollama API base URL
        model: Model name to use
        timeout: Request timeout in seconds

    Returns:
        Tuple of (orchestrator, ollama_client)

    Example:
        >>> orchestrator, ollama = create_debate_system()
        >>> forecast = await orchestrator.run_debate(
        ...     market_id="0x123",
        ...     context=context_string,
        ...     rounds=2
        ... )
    """
    # Create Ollama client
    ollama = OllamaClient(
        base_url=base_url,
        model=model,
        timeout=timeout,
    )

    # Create agents
    bull = BullAgent(ollama=ollama)
    bear = BearAgent(ollama=ollama)
    devil = DevilsAdvocateAgent(ollama=ollama)
    judge = JudgeAgent(ollama=ollama)

    # Create orchestrator
    orchestrator = DebateOrchestrator(
        bull=bull,
        bear=bear,
        devil=devil,
        judge=judge,
    )

    return orchestrator, ollama


__all__ = [
    "OllamaClient",
    "BaseAgent",
    "load_prompt",
    "BullAgent",
    "BearAgent",
    "DevilsAdvocateAgent",
    "JudgeAgent",
    "DebateOrchestrator",
    "create_debate_system",
]
