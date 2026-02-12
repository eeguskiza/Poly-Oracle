"""
Bull Agent - Argues for YES resolution.
"""
from src.agents.base import BaseAgent, load_prompt
from src.models import AgentRole


class BullAgent(BaseAgent):
    """Agent that argues the case for YES resolution."""

    def __init__(self, ollama, system_prompt: str | None = None):
        """
        Initialize Bull Agent.

        Args:
            ollama: OllamaClient instance
            system_prompt: Optional custom system prompt (defaults to bull_agent.yaml)
        """
        if system_prompt is None:
            system_prompt = load_prompt("bull_agent")

        super().__init__(
            role=AgentRole.BULL,
            ollama=ollama,
            system_prompt=system_prompt,
        )

    def _build_prompt(
        self,
        context: str,
        instruction: str,
        phase: str = "initial_argument",
        previous_arguments: list[dict[str, str]] | None = None,
    ) -> str:
        """
        Build prompt with phase-specific instructions.

        Args:
            context: Market analysis context
            instruction: Base instruction
            phase: Either "initial_argument" or "rebuttal"
            previous_arguments: List of previous arguments from other agents

        Returns:
            Formatted prompt string
        """
        # Build base prompt
        prompt_parts = [
            "# Context",
            context,
            "",
        ]

        # Add previous arguments if in rebuttal phase
        if phase == "rebuttal" and previous_arguments:
            prompt_parts.extend([
                "# Previous Arguments",
                "",
            ])
            for arg in previous_arguments:
                prompt_parts.extend([
                    f"## {arg['role']}",
                    arg['content'],
                    "",
                ])

        # Add phase-specific instruction
        if phase == "initial_argument":
            phase_instruction = (
                "Present your opening argument for why this market will resolve YES. "
                "Build your case from first principles using the context provided. "
                "Focus on the strongest evidence and reasoning that supports a YES resolution."
            )
        elif phase == "rebuttal":
            phase_instruction = (
                "Respond to the Bear's argument and address the Devil's Advocate critique. "
                "Defend your position, acknowledge valid concerns, but reinforce why "
                "the evidence still supports a YES resolution. Update your probability if needed."
            )
        else:
            phase_instruction = instruction

        prompt_parts.extend([
            "# Your Task",
            phase_instruction,
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
