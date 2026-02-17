"""Startup orchestration for `python cli.py start`."""

from __future__ import annotations

import logging
import os
import signal
import sys
import threading
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Callable

from bot.config.loader import load_environment, load_json_config
from bot.config.schema import AppSettings, validate_settings
from bot.data.feeds import FallbackPolymarketFeed, PrimaryPolymarketFeed, TargetMarket
from bot.errors import FatalStartupError, StartupError
from bot.monitoring.logger import initialize_logger, log_step
from bot.state.store import SQLiteStateStore
from bot.strategy.coordinator import TradingLoopCoordinator

DEFAULT_CONFIG_PATH = Path("bot/config/startup.json")
DEFAULT_ENV_PATH = Path(".env")


@dataclass
class StartupContext:
    """Validated startup dependency container."""

    settings: AppSettings
    logger: logging.Logger
    state_store: SQLiteStateStore
    target_market: TargetMarket
    primary_feed: PrimaryPolymarketFeed
    fallback_feed: FallbackPolymarketFeed
    coordinator: TradingLoopCoordinator
    stop_event: threading.Event


@dataclass(frozen=True)
class StartupResult:
    exit_code: int
    reason: str


def run_startup(
    *,
    check: bool,
    config_path: Path | None = None,
    env_path: Path | None = None,
    signal_module: ModuleType = signal,
    state_store_factory: Callable[[Path], SQLiteStateStore] = SQLiteStateStore,
    primary_feed_factory: Callable[[str, str], PrimaryPolymarketFeed] = PrimaryPolymarketFeed,
    fallback_feed_factory: Callable[[str, str], FallbackPolymarketFeed] = FallbackPolymarketFeed,
    coordinator_factory: Callable[[logging.Logger, int], TradingLoopCoordinator] = TradingLoopCoordinator,
) -> StartupResult:
    """Execute the deterministic startup pipeline with fail-fast behavior."""
    context: StartupContext | None = None
    loop_thread: threading.Thread | None = None

    try:
        # 1) Load environment and JSON config (chosen for strict stdlib parsing).
        dotenv = env_path or DEFAULT_ENV_PATH
        config_file = config_path or DEFAULT_CONFIG_PATH
        loaded_env = load_environment(dotenv)
        raw_config = load_json_config(config_file)

        # 2) Validate required settings with strict schema.
        settings = validate_settings(raw_config, os.environ)

        # 3) Initialize structured logger.
        logger = initialize_logger(settings.logging.level)
        log_step(
            logger,
            1,
            "Environment and config loaded.",
            env_path=str(dotenv),
            config_path=str(config_file),
            loaded_env_keys=sorted(loaded_env.keys()),
            config_format="json",
        )
        log_step(logger, 2, "Settings validated.", app_name=settings.app_name)
        log_step(logger, 3, "Structured logger initialized.", level=settings.logging.level)

        # 4) Initialize state store (sqlite).
        state_store = state_store_factory(settings.state.sqlite_path)
        state_store.connect()
        state_store.initialize_schema()
        state_store.record_event("startup", "state_store_initialized")
        log_step(logger, 4, "State store initialized.", sqlite_path=str(settings.state.sqlite_path))

        # 5) Resolve target Polymarket market (BTC single-market only).
        target_market = TargetMarket(
            market_id=settings.polymarket.btc_market_id,
            symbol=settings.polymarket.market_symbol,
            source="config+env",
        )
        state_store.record_event("market_resolution", target_market.market_id)
        log_step(
            logger,
            5,
            "Target market resolved.",
            market_symbol=target_market.symbol,
            market_id=target_market.market_id,
        )

        # 6) Initialize data feeds (primary + fallback placeholders).
        primary_feed = primary_feed_factory(settings.polymarket.base_url, target_market.market_id)
        fallback_feed = fallback_feed_factory(settings.polymarket.base_url, target_market.market_id)
        log_step(
            logger,
            6,
            "Data feeds initialized.",
            primary=settings.data_feeds.primary_name,
            fallback=settings.data_feeds.fallback_name,
        )

        # 7) Warmup data buffers.
        primary_snapshot = primary_feed.warmup(settings.data_feeds.warmup_cycles)
        fallback_snapshot = fallback_feed.warmup(settings.data_feeds.warmup_cycles)
        state_store.record_event(
            "warmup",
            f"primary={len(primary_snapshot)},fallback={len(fallback_snapshot)}",
        )
        log_step(
            logger,
            7,
            "Data feed warmup completed.",
            primary_cycles=len(primary_snapshot),
            fallback_cycles=len(fallback_snapshot),
        )

        # 8) Start trading loop coordinator scaffold.
        stop_event = threading.Event()
        coordinator = coordinator_factory(logger, settings.loop.interval_seconds)
        if check:
            log_step(
                logger,
                8,
                "Trading loop coordinator prepared (check mode).",
                interval_seconds=settings.loop.interval_seconds,
                check_mode=True,
            )
        else:
            loop_thread = threading.Thread(
                target=coordinator.run,
                args=(stop_event,),
                name="trading-loop-coordinator",
                daemon=False,
            )
            loop_thread.start()
            log_step(
                logger,
                8,
                "Trading loop coordinator started.",
                interval_seconds=settings.loop.interval_seconds,
                check_mode=False,
            )

        context = StartupContext(
            settings=settings,
            logger=logger,
            state_store=state_store,
            target_market=target_market,
            primary_feed=primary_feed,
            fallback_feed=fallback_feed,
            coordinator=coordinator,
            stop_event=stop_event,
        )

        # 9) Register graceful shutdown handlers.
        register_shutdown_handlers(stop_event, logger, signal_module=signal_module)
        log_step(logger, 9, "Graceful shutdown handlers registered.")

        if check:
            # 10) Exit cleanly for preflight.
            log_step(logger, 10, "Preflight check succeeded; loop not started.")
            return StartupResult(exit_code=0, reason="preflight_ok")

        if loop_thread is None:
            raise FatalStartupError("Trading loop thread failed to start.", reason="loop_not_started")

        loop_thread.join()
        log_step(logger, 10, "Startup pipeline finished after graceful shutdown.")
        return StartupResult(exit_code=0, reason="graceful_shutdown")

    except StartupError as exc:
        if context is not None:
            context.stop_event.set()
        if loop_thread is not None and loop_thread.is_alive():
            loop_thread.join(timeout=1.0)
        _emit_fatal_message(context.logger if context else None, exc)
        return StartupResult(exit_code=exc.exit_code, reason=exc.reason)
    except Exception as exc:  # pragma: no cover - defensive fallback
        if context is not None:
            context.stop_event.set()
        if loop_thread is not None and loop_thread.is_alive():
            loop_thread.join(timeout=1.0)
        fatal = FatalStartupError(f"Unexpected startup exception: {exc}")
        _emit_fatal_message(context.logger if context else None, fatal)
        return StartupResult(exit_code=fatal.exit_code, reason=fatal.reason)
    finally:
        if context is not None:
            context.state_store.close()


def register_shutdown_handlers(
    stop_event: threading.Event,
    logger: logging.Logger,
    *,
    signal_module: ModuleType = signal,
) -> None:
    """Attach SIGINT/SIGTERM handlers that trigger graceful shutdown."""

    def _handler(signum: int, _frame: object) -> None:
        logger.warning(
            "Shutdown signal received.",
            extra={"event": "shutdown_signal", "context": {"signal": signum}},
        )
        stop_event.set()

    signal_module.signal(signal_module.SIGINT, _handler)
    signal_module.signal(signal_module.SIGTERM, _handler)


def _emit_fatal_message(logger: logging.Logger | None, exc: StartupError) -> None:
    payload = f"Startup failed: {exc} | exit_code={exc.exit_code} | reason={exc.reason}"
    if logger is not None:
        logger.error(payload, extra={"event": "startup_failure"})
    else:
        print(payload, file=sys.stderr)
