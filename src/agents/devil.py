"""
Devil's Advocate Agent - Critiques both Bull and Bear arguments.
"""
from src.agents.base import BaseAgent, load_prompt
from src.models import AgentRole


class DevilsAdvocateAgent(BaseAgent):
    """Agent that critiques arguments from both sides."""

    def __init__(self, ollama, system_prompt: str | None = None):
        """
        Initialize Devil's Advocate Agent.

        Args:
            ollama: OllamaClient instance
            system_prompt: Optional custom system prompt (defaults to devil_agent.yaml)
        """
        if system_prompt is None:
            system_prompt = load_prompt("devil_agent")

        super().__init__(
            role=AgentRole.DEVIL,
            ollama=ollama,
            system_prompt=system_prompt,
        )

    def _build_prompt(
        self,
        context: str,
        instruction: str,
        previous_arguments: list[dict[str, str]] | None = None,
    ) -> str:
        """
        Build prompt for critique.

        Devil's Advocate always needs previous arguments to critique.

        Args:
            context: Market analysis context
            instruction: Base instruction
            previous_arguments: List of arguments from Bull and Bear

        Returns:
            Formatted prompt string
        """
        prompt_parts = [
            "# Context",
            context,
            "",
        ]

        # Add arguments to critique
        if previous_arguments:
            prompt_parts.extend([
                "# Arguments to Critique",
                "",
            ])
            for arg in previous_arguments:
                prompt_parts.extend([
                    f"## {arg['role']}",
                    arg['content'],
                    "",
                ])

        # Add task instruction
        prompt_parts.extend([
            "# Your Task",
            (
                "Critique both the Bull and Bear arguments. Identify weaknesses, "
                "logical fallacies, cognitive biases, and information gaps in BOTH positions. "
                "Your goal is to stress-test the reasoning and uncover blind spots. "
                "Be thorough but fair - point out what each side is getting wrong or overlooking."
            ),
            "",
            instruction,
        ])

        return "\n".join(prompt_parts)

    def parse_probability(self, response: str) -> float | None:
        """
        Parse probability from agent response.

        Args:
            response: Agent's text response

        Returns:
            Probability as float between 0 and 1, or None if not found
        """
        return self._extract_probability(response)
