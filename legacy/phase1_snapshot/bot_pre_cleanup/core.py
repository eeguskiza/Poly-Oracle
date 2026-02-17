"""Poly-Oracle v1 — Single-market BTC trading bot for Polymarket.

Lifecycle:
  1. Validate config & credentials (fail fast)
  2. Initialize storage, data feed, execution components
  3. Verify Ollama LLM availability
  4. Enter trading loop:
     a. Resolve any closed positions -> update P&L
     b. Fetch current market data
     c. Generate forecast via LLM debate
     d. Compute edge vs market price
     e. If edge sufficient -> Kelly size -> risk check -> execute
     f. Log every decision
     g. Sleep until next cycle
  5. Ctrl+C -> graceful shutdown
"""

from __future__ import annotations

import asyncio
import signal
import time
from datetime import datetime, timezone
from typing import Any, Optional

from loguru import logger

from bot.config.settings import Settings
from bot.strategy import create_debate_system
from bot.strategy.base import OllamaClient
from bot.data.polymarket import PolymarketClient
from bot.state.store import SQLiteClient
from bot.execution.paper import PaperTradingExecutor
from bot.execution.live import LiveTradingExecutor
from bot.risk.manager import RiskManager
from bot.execution.sizer import PositionSizer
from bot.models.forecast import EdgeAnalysis, SimpleForecast
from bot.models.market import Market
from bot.exceptions import ConfigError
from bot.monitoring.logger import setup_logging


class TradingBot:
    """Single-market BTC trading bot for Polymarket."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._shutdown = False

    async def run(self) -> None:
        """Main entry point. Validates, initializes, loops."""
        # --- Logging ---
        setup_logging(
            self.settings.log_level,
            self.settings.database.db_dir / "logs",
        )

        logger.info("=" * 60)
        logger.info("Poly-Oracle v1 starting")
        logger.info("=" * 60)

        # --- Phase 1: Validate (fail fast) ---
        self._validate_config()

        mode = "PAPER" if self.settings.paper_trading else "LIVE"
        logger.info(f"Mode: {mode}")
        logger.info(f"Market: {self.settings.btc_market_id}")
        logger.info(f"Bankroll: ${self.settings.risk.initial_bankroll:.2f}")
        logger.info(
            f"Bet range: ${self.settings.risk.min_bet:.2f}"
            f" - ${self.settings.risk.max_bet:.2f}"
        )
        logger.info(f"Loop interval: {self.settings.loop_interval_minutes}m")

        # --- Phase 2: Storage ---
        self.settings.database.db_dir.mkdir(parents=True, exist_ok=True)
        sqlite = SQLiteClient(self.settings.database.sqlite_path)
        sqlite.initialize_schema()
        sqlite.seed_initial_bankroll(self.settings.risk.initial_bankroll)
        logger.info("Storage initialized")

        # --- Phase 3: Execution ---
        sizer = PositionSizer(risk_settings=self.settings.risk)
        risk_mgr = RiskManager(risk_settings=self.settings.risk)

        if self.settings.paper_trading:
            executor: PaperTradingExecutor | LiveTradingExecutor = (
                PaperTradingExecutor(sqlite=sqlite, sizer=sizer, risk=risk_mgr)
            )
            logger.info("Executor: paper trading")
        else:
            executor = LiveTradingExecutor(
                sqlite=sqlite, sizer=sizer, risk=risk_mgr, settings=self.settings,
            )
            logger.info("Executor: LIVE trading")

        # --- Phase 4: Data feed ---
        async with PolymarketClient() as polymarket:
            market = await polymarket.get_market(self.settings.btc_market_id)
            logger.info(f"Market verified: {market.question}")
            logger.info(f"Current price: {market.current_price:.1%}")

            # --- Phase 5: LLM check ---
            await self._check_ollama()
            logger.info("All systems GO. Entering trading loop.")
            logger.info("=" * 60)

            # Signal handlers for graceful shutdown
            loop = asyncio.get_event_loop()
            for sig in (signal.SIGINT, signal.SIGTERM):
                loop.add_signal_handler(sig, self._request_shutdown)

            # --- Main loop ---
            cycle_num = 0
            while not self._shutdown:
                cycle_num += 1
                try:
                    await self._run_cycle(
                        cycle_num, polymarket, sqlite, executor,
                    )
                except Exception:
                    logger.exception("Cycle failed")

                if not self._shutdown:
                    mins = self.settings.loop_interval_minutes
                    logger.info(f"Next cycle in {mins} minutes")
                    await self._interruptible_sleep(mins * 60)

        sqlite.close()
        logger.info("Poly-Oracle v1 stopped cleanly")

    # ------------------------------------------------------------------
    # Trading cycle
    # ------------------------------------------------------------------

    async def _run_cycle(
        self,
        cycle_num: int,
        polymarket: PolymarketClient,
        sqlite: SQLiteClient,
        executor: PaperTradingExecutor | LiveTradingExecutor,
    ) -> None:
        """One complete trading cycle: resolve -> fetch -> forecast -> edge -> execute."""
        ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        logger.info(f"=== Cycle {cycle_num} [{ts}] START ===")
        t0 = time.monotonic()

        # 1. Resolve closed positions
        resolved_pnl = await self._resolve_positions(polymarket, sqlite)
        if resolved_pnl != 0.0:
            logger.info(f"Resolution P&L: ${resolved_pnl:+.2f}")

        # 2. Fetch market
        market = await polymarket.get_market(self.settings.btc_market_id)
        logger.info(
            f"Market: price={market.current_price:.1%} "
            f"vol24h=${market.volume_24h:,.0f} liq=${market.liquidity:,.0f}"
        )

        # 3. Forecast
        forecast = await self._generate_forecast(market)
        if forecast is None:
            logger.warning("Forecast failed — skipping cycle")
            return

        confidence = forecast.compute_confidence()
        logger.info(
            f"Forecast: P(YES)={forecast.probability:.1%} "
            f"confidence={confidence:.1%}"
        )

        # 4. Edge analysis
        edge = self._compute_edge(forecast, market)
        logger.info(
            f"Edge: raw={edge.raw_edge:+.1%} abs={edge.abs_edge:.1%} "
            f"dir={edge.direction} action={edge.recommended_action} "
            f"| {edge.reasoning}"
        )

        # 5. Execute
        if edge.recommended_action == "TRADE":
            bankroll = sqlite.get_current_bankroll()
            logger.info(f"Attempting trade | bankroll=${bankroll:.2f}")

            result = await executor.execute(
                edge_analysis=edge,
                calibrated_probability=forecast.probability,
                market=market,
                bankroll=bankroll,
            )

            if result is None:
                logger.info("TRADE SKIPPED: position size below minimum")
            elif result.success:
                logger.info(f"TRADE EXECUTED: {result.message}")
            else:
                logger.warning(f"TRADE REJECTED: {result.message}")
        else:
            logger.info("No trade this cycle")

        elapsed = time.monotonic() - t0
        logger.info(f"=== Cycle {cycle_num} END ({elapsed:.1f}s) ===")

    # ------------------------------------------------------------------
    # Forecast generation
    # ------------------------------------------------------------------

    async def _generate_forecast(self, market: Market) -> SimpleForecast | None:
        """Run LLM debate to forecast market outcome."""
        context = self._build_market_context(market)

        orchestrator, ollama = create_debate_system(
            base_url=self.settings.llm.base_url,
            model=self.settings.llm.model,
            timeout=self.settings.llm.timeout,
        )

        try:
            forecast = await orchestrator.run_debate(
                market_id=market.id,
                context=context,
                rounds=self.settings.debate_rounds,
                temperature=self.settings.llm.temperature,
                verbose=False,
            )
            return forecast
        except Exception:
            logger.exception("LLM debate failed")
            return None
        finally:
            await ollama.close()

    @staticmethod
    def _build_market_context(market: Market) -> str:
        """Build plain-text context from market data for the LLM agents."""
        return (
            f"# Market Analysis\n"
            f"Question: {market.question}\n"
            f"Description: {market.description}\n\n"
            f"## Current Data\n"
            f"- Current price (market consensus P(YES)): {market.current_price:.1%}\n"
            f"- 24h trading volume: ${market.volume_24h:,.0f}\n"
            f"- Liquidity: ${market.liquidity:,.0f}\n"
            f"- Days until resolution: {market.days_until_resolution:.1f}\n"
            f"- Resolution date: {market.resolution_date.strftime('%Y-%m-%d')}\n"
            f"- Outcomes: {', '.join(market.outcomes)}\n"
        )

    # ------------------------------------------------------------------
    # Edge computation
    # ------------------------------------------------------------------

    def _compute_edge(
        self, forecast: SimpleForecast, market: Market,
    ) -> EdgeAnalysis:
        """Deterministic edge analysis: forecast vs market price."""
        our_prob = forecast.probability
        market_prob = market.current_price

        raw_edge = our_prob - market_prob
        abs_edge = abs(raw_edge)
        confidence = forecast.compute_confidence()
        weighted_edge = abs_edge * confidence

        direction = "BUY_YES" if raw_edge > 0 else "BUY_NO"

        has_edge = (
            abs_edge >= self.settings.risk.min_edge
            and confidence >= self.settings.risk.min_confidence
            and market.liquidity >= self.settings.risk.min_liquidity
        )
        action = "TRADE" if has_edge else "SKIP"

        reasons: list[str] = []
        if abs_edge < self.settings.risk.min_edge:
            reasons.append(
                f"edge {abs_edge:.1%} < min {self.settings.risk.min_edge:.1%}"
            )
        if confidence < self.settings.risk.min_confidence:
            reasons.append(
                f"confidence {confidence:.1%} < min {self.settings.risk.min_confidence:.1%}"
            )
        if market.liquidity < self.settings.risk.min_liquidity:
            reasons.append(
                f"liquidity ${market.liquidity:,.0f} < min ${self.settings.risk.min_liquidity:,.0f}"
            )

        reasoning = (
            f"{action}. "
            + (" | ".join(reasons) if reasons else "All thresholds met.")
        )

        return EdgeAnalysis(
            our_forecast=our_prob,
            market_price=market_prob,
            raw_edge=raw_edge,
            abs_edge=abs_edge,
            weighted_edge=weighted_edge,
            direction=direction,
            has_actionable_edge=has_edge,
            recommended_action=action,
            reasoning=reasoning,
        )

    # ------------------------------------------------------------------
    # Position resolution
    # ------------------------------------------------------------------

    async def _resolve_positions(
        self,
        polymarket: PolymarketClient,
        sqlite: SQLiteClient,
    ) -> float:
        """Check and resolve any closed positions. Returns realized P&L."""
        positions = sqlite.get_open_positions()
        if not positions:
            return 0.0

        market_ids = [p["market_id"] for p in positions]
        resolved = await polymarket.check_resolutions(market_ids)

        if not resolved:
            return 0.0

        total_pnl = 0.0
        for market_id, outcome in resolved.items():
            pnl = self._settle_position(sqlite, market_id, outcome)
            total_pnl += pnl

        return total_pnl

    @staticmethod
    def _settle_position(
        sqlite: SQLiteClient,
        market_id: str,
        outcome: bool,
    ) -> float:
        """Settle a single resolved position. Returns realized P&L."""
        position = sqlite.get_position(market_id)
        if not position or position["num_shares"] <= 0:
            return 0.0

        direction = position["direction"]
        num_shares = position["num_shares"]
        amount_usd = position["amount_usd"]

        # Binary market: winner gets $1/share, loser gets $0
        if direction == "BUY_YES":
            pnl = (num_shares * 1.0) - amount_usd if outcome else -amount_usd
        else:
            pnl = (num_shares * 1.0) - amount_usd if not outcome else -amount_usd

        outcome_str = "YES" if outcome else "NO"
        logger.info(
            f"RESOLVED: {market_id} -> {outcome_str} | "
            f"{direction} {num_shares:.2f} shares | P&L: ${pnl:+.2f}"
        )

        # Close position
        sqlite.upsert_position({
            **position,
            "num_shares": 0.0,
            "amount_usd": 0.0,
            "unrealized_pnl": 0.0,
            "updated_at": datetime.now(timezone.utc).isoformat(),
        })

        # Update daily stats
        today = datetime.now(timezone.utc).date().isoformat()
        stats = sqlite.get_daily_stats(today)
        if stats:
            stats["ending_bankroll"] = stats["ending_bankroll"] + pnl
            stats["net_pnl"] = stats["net_pnl"] + pnl
            stats["gross_pnl"] = stats["gross_pnl"] + pnl
            if pnl > 0:
                stats["trades_won"] = stats["trades_won"] + 1
        else:
            bankroll = sqlite.get_current_bankroll()
            stats = {
                "date": today,
                "starting_bankroll": bankroll,
                "ending_bankroll": bankroll + pnl,
                "trades_executed": 1,
                "trades_won": 1 if pnl > 0 else 0,
                "gross_pnl": pnl,
                "fees_paid": 0.0,
                "net_pnl": pnl,
            }
        sqlite.upsert_daily_stats(stats)
        return pnl

    # ------------------------------------------------------------------
    # Validation & health checks
    # ------------------------------------------------------------------

    def _validate_config(self) -> None:
        """Fail fast on missing or invalid critical configuration."""
        s = self.settings

        if not s.btc_market_id:
            raise ConfigError(
                "BTC_MARKET_ID is required. "
                "Set it in .env to the Polymarket condition ID you want to trade."
            )

        if not s.paper_trading:
            ps = s.polymarket
            if not all([ps.api_key, ps.api_secret, ps.api_passphrase]):
                raise ConfigError(
                    "Live trading requires POLYMARKET_API_KEY, "
                    "POLYMARKET_API_SECRET, and POLYMARKET_API_PASSPHRASE in .env"
                )

        if s.risk.initial_bankroll <= 0:
            raise ConfigError("INITIAL_BANKROLL must be positive")
        if s.risk.min_bet <= 0:
            raise ConfigError("MIN_BET must be positive")
        if s.risk.max_bet < s.risk.min_bet:
            raise ConfigError("MAX_BET must be >= MIN_BET")

        logger.info("Configuration validated")

    async def _check_ollama(self) -> None:
        """Verify Ollama is running and the configured model is available."""
        async with OllamaClient(
            base_url=self.settings.llm.base_url,
            model=self.settings.llm.model,
            timeout=30,
        ) as client:
            if not await client.is_available():
                raise ConfigError(
                    f"Ollama model '{self.settings.llm.model}' is not available. "
                    f"Run: ollama serve && ollama pull {self.settings.llm.model}"
                )
        logger.info(f"Ollama ready: model={self.settings.llm.model}")

    # ------------------------------------------------------------------
    # Shutdown
    # ------------------------------------------------------------------

    def _request_shutdown(self) -> None:
        logger.info("Shutdown signal received")
        self._shutdown = True

    async def _interruptible_sleep(self, seconds: float) -> None:
        """Sleep that exits early on shutdown signal."""
        deadline = time.monotonic() + seconds
        while time.monotonic() < deadline and not self._shutdown:
            remaining = deadline - time.monotonic()
            await asyncio.sleep(min(1.0, max(0, remaining)))
