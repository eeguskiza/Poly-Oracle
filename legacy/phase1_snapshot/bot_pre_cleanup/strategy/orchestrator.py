"""
Debate Orchestrator - Manages multi-round debate between agents.
"""
from datetime import datetime, timezone
from loguru import logger

from bot.strategy.bull import BullAgent
from bot.strategy.bear import BearAgent
from bot.strategy.devil import DevilsAdvocateAgent
from bot.strategy.judge import JudgeAgent
from bot.models import AgentResponse, SimpleForecast


class DebateOrchestrator:
    """Orchestrates multi-round debates between forecasting agents."""

    def __init__(
        self,
        bull: BullAgent,
        bear: BearAgent,
        devil: DevilsAdvocateAgent,
        judge: JudgeAgent,
    ):
        """
        Initialize orchestrator with debate agents.

        Args:
            bull: Bull agent (argues YES)
            bear: Bear agent (argues NO)
            devil: Devil's advocate agent (critiques both)
            judge: Judge agent (synthesizes forecast)
        """
        self.bull = bull
        self.bear = bear
        self.devil = devil
        self.judge = judge

    async def run_debate(
        self,
        market_id: str,
        context: str,
        rounds: int = 2,
        temperature: float = 0.7,
        max_tokens: int = 2000,
        verbose: bool = False,
    ) -> SimpleForecast:
        """
        Run a multi-round debate and produce final forecast.

        Args:
            market_id: ID of the market being forecasted
            context: Market analysis context
            rounds: Number of debate rounds (default: 2)
            temperature: LLM temperature parameter
            max_tokens: Max tokens per response
            verbose: Whether to log detailed progress

        Returns:
            Final Forecast object with probability and metadata
        """
        logger.info(f"Starting {rounds}-round debate for market {market_id}")

        # Track all arguments across rounds
        all_arguments: list[dict[str, str]] = []
        round_probabilities: dict[str, list[float]] = {
            "bull": [],
            "bear": [],
            "judge": [],
        }

        # Run debate rounds
        for round_num in range(1, rounds + 1):
            if verbose:
                logger.info(f"\n{'='*80}\nROUND {round_num}\n{'='*80}")

            # Phase 1: Bull initial argument
            if verbose:
                logger.info("Bull Agent - Initial Argument")

            bull_response = await self.bull.generate(
                context=context,
                instruction="Present your opening argument.",
                round_number=round_num,
                phase="initial_argument",
                temperature=temperature,
                max_tokens=max_tokens,
            )
            all_arguments.append({
                "role": "BULL",
                "content": bull_response.content,
                "round": round_num,
            })
            if bull_response.probability:
                round_probabilities["bull"].append(bull_response.probability)

            if verbose:
                logger.info(f"Bull P(YES) = {bull_response.probability}")

            # Phase 2: Bear initial argument
            if verbose:
                logger.info("\nBear Agent - Initial Argument")

            bear_response = await self.bear.generate(
                context=context,
                instruction="Present your opening argument.",
                round_number=round_num,
                phase="initial_argument",
                temperature=temperature,
                max_tokens=max_tokens,
            )
            all_arguments.append({
                "role": "BEAR",
                "content": bear_response.content,
                "round": round_num,
            })
            if bear_response.probability:
                round_probabilities["bear"].append(bear_response.probability)

            if verbose:
                logger.info(f"Bear P(YES) = {bear_response.probability}")

            # Phase 3: Devil's Advocate critique
            if verbose:
                logger.info("\nDevil's Advocate - Critique")

            devil_response = await self.devil.generate(
                context=context,
                instruction="Critique both arguments.",
                round_number=round_num,
                previous_arguments=[
                    {"role": "BULL", "content": bull_response.content},
                    {"role": "BEAR", "content": bear_response.content},
                ],
                temperature=temperature,
                max_tokens=max_tokens,
            )
            all_arguments.append({
                "role": "DEVIL",
                "content": devil_response.content,
                "round": round_num,
            })

            if verbose:
                logger.info("Critique complete")

            # Phase 4: Bull rebuttal
            if verbose:
                logger.info("\nBull Agent - Rebuttal")

            bull_rebuttal = await self.bull.generate(
                context=context,
                instruction="Respond to critiques and reinforce your position.",
                round_number=round_num,
                phase="rebuttal",
                previous_arguments=[
                    {"role": "BEAR", "content": bear_response.content},
                    {"role": "DEVIL", "content": devil_response.content},
                ],
                temperature=temperature,
                max_tokens=max_tokens,
            )
            all_arguments.append({
                "role": "BULL_REBUTTAL",
                "content": bull_rebuttal.content,
                "round": round_num,
            })
            if bull_rebuttal.probability:
                round_probabilities["bull"].append(bull_rebuttal.probability)

            if verbose:
                logger.info(f"Bull (updated) P(YES) = {bull_rebuttal.probability}")

            # Phase 5: Bear rebuttal
            if verbose:
                logger.info("\nBear Agent - Rebuttal")

            bear_rebuttal = await self.bear.generate(
                context=context,
                instruction="Respond to critiques and reinforce your position.",
                round_number=round_num,
                phase="rebuttal",
                previous_arguments=[
                    {"role": "BULL", "content": bull_response.content},
                    {"role": "DEVIL", "content": devil_response.content},
                ],
                temperature=temperature,
                max_tokens=max_tokens,
            )
            all_arguments.append({
                "role": "BEAR_REBUTTAL",
                "content": bear_rebuttal.content,
                "round": round_num,
            })
            if bear_rebuttal.probability:
                round_probabilities["bear"].append(bear_rebuttal.probability)

            if verbose:
                logger.info(f"Bear (updated) P(YES) = {bear_rebuttal.probability}")

        # Final phase: Judge synthesizes
        if verbose:
            logger.info(f"\n{'='*80}\nFINAL JUDGMENT\n{'='*80}")
            logger.info("Judge Agent - Synthesizing Forecast")

        judge_response = await self.judge.generate(
            context=context,
            instruction="Provide your final forecast based on the complete debate.",
            round_number=rounds + 1,
            previous_arguments=all_arguments,
            temperature=temperature,
            max_tokens=max_tokens,
        )

        # Parse complete forecast with confidence intervals
        forecast_data = self.judge.parse_forecast(judge_response.content)

        if verbose:
            logger.info(f"Judge P(YES) = {forecast_data['probability']}")
            if forecast_data['lower_bound'] and forecast_data['upper_bound']:
                logger.info(
                    f"Confidence Interval: {forecast_data['lower_bound']:.2%} to "
                    f"{forecast_data['upper_bound']:.2%}"
                )

        # Create final forecast
        forecast = SimpleForecast(
            market_id=market_id,
            probability=forecast_data["probability"] or 0.5,
            confidence_lower=forecast_data["lower_bound"],
            confidence_upper=forecast_data["upper_bound"],
            reasoning=judge_response.content,
            created_at=datetime.now(timezone.utc),
            model_name=self.judge.ollama.model,
            debate_rounds=rounds,
            bull_probabilities=round_probabilities["bull"],
            bear_probabilities=round_probabilities["bear"],
        )

        logger.info(f"Debate complete. Final P(YES) = {forecast.probability:.2%}")

        return forecast

    def get_debate_summary(self, forecast: SimpleForecast) -> str:
        """
        Generate a human-readable summary of the debate.

        Args:
            forecast: The forecast object from run_debate

        Returns:
            Formatted summary string
        """
        summary_parts = [
            "=" * 80,
            f"FORECAST SUMMARY - {forecast.market_id}",
            "=" * 80,
            "",
            f"Final Probability: {forecast.probability:.1%}",
        ]

        if forecast.confidence_lower and forecast.confidence_upper:
            summary_parts.append(
                f"Confidence Interval: {forecast.confidence_lower:.1%} to "
                f"{forecast.confidence_upper:.1%}"
            )

        summary_parts.extend([
            "",
            f"Debate Rounds: {forecast.debate_rounds}",
            f"Model: {forecast.model_name}",
            f"Timestamp: {forecast.created_at.strftime('%Y-%m-%d %H:%M:%S UTC')}",
            "",
        ])

        # Show probability evolution
        if forecast.bull_probabilities or forecast.bear_probabilities:
            summary_parts.append("Probability Evolution:")
            for i, (bull_p, bear_p) in enumerate(
                zip(forecast.bull_probabilities, forecast.bear_probabilities), 1
            ):
                summary_parts.append(f"  Round {i}: Bull={bull_p:.1%}, Bear={bear_p:.1%}")
            summary_parts.append("")

        summary_parts.extend([
            "Judge's Reasoning:",
            "-" * 80,
            forecast.reasoning,
            "",
            "=" * 80,
        ])

        return "\n".join(summary_parts)
