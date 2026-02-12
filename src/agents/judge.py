"""
Judge Agent - Synthesizes debate into final forecast.
"""
import re
from src.agents.base import BaseAgent, load_prompt
from src.models import AgentRole


class JudgeAgent(BaseAgent):
    """Agent that synthesizes the debate and produces final forecast."""

    def __init__(self, ollama, system_prompt: str | None = None):
        """
        Initialize Judge Agent.

        Args:
            ollama: OllamaClient instance
            system_prompt: Optional custom system prompt (defaults to judge_agent.yaml)
        """
        if system_prompt is None:
            system_prompt = load_prompt("judge_agent")

        super().__init__(
            role=AgentRole.JUDGE,
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
        Build prompt for final judgment.

        Judge needs all previous arguments to synthesize.

        Args:
            context: Market analysis context
            instruction: Base instruction
            previous_arguments: List of all arguments from debate

        Returns:
            Formatted prompt string
        """
        prompt_parts = [
            "# Context",
            context,
            "",
        ]

        # Add all debate arguments
        if previous_arguments:
            prompt_parts.extend([
                "# Debate Transcript",
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
                "Synthesize the entire debate to produce your final forecast. "
                "Follow your superforecasting process: evaluate evidence quality, "
                "anchor on base rates, adjust for specifics, identify key uncertainties, "
                "and generate a precise probability with confidence interval. "
                "Your forecast should be well-calibrated and clearly reasoned."
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

    def parse_forecast(self, response: str) -> dict[str, float | None]:
        """
        Parse complete forecast including confidence intervals.

        Args:
            response: Agent's text response

        Returns:
            Dictionary with keys: probability, lower_bound, upper_bound
        """
        result = {
            "probability": self._extract_probability(response),
            "lower_bound": None,
            "upper_bound": None,
        }

        # Try to extract confidence interval
        # Format: "Confidence Interval: 30% to 45% (90% CI)"
        # Or: "30-45%", "30%-45%", "Lower: 30%, Upper: 45%"
        ci_patterns = [
            r"Confidence Interval:\s*(\d+(?:\.\d+)?)%\s*to\s*(\d+(?:\.\d+)?)%",
            r"(\d+(?:\.\d+)?)%\s*[-–]\s*(\d+(?:\.\d+)?)%",
            r"Lower[:\s]+(\d+(?:\.\d+)?)%.*?Upper[:\s]+(\d+(?:\.\d+)?)%",
            r"\[(\d+(?:\.\d+)?)%,\s*(\d+(?:\.\d+)?)%\]",
        ]

        for pattern in ci_patterns:
            match = re.search(pattern, response, re.IGNORECASE)
            if match:
                try:
                    lower = float(match.group(1))
                    upper = float(match.group(2))
                    # Convert percentages to probabilities
                    result["lower_bound"] = lower / 100 if lower > 1 else lower
                    result["upper_bound"] = upper / 100 if upper > 1 else upper
                    break
                except (ValueError, IndexError):
                    continue

        return result
