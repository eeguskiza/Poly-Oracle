"""Tests for startup orchestration branches."""

from __future__ import annotations

import json
import logging
import threading
from pathlib import Path

from bot.startup import register_shutdown_handlers, run_startup


class DummyStore:
    def __init__(self, _db_path: Path) -> None:
        self.connected = False
        self.initialized = False
        self.closed = False
        self.events: list[tuple[str, str]] = []

    def connect(self) -> None:
        self.connected = True

    def initialize_schema(self) -> None:
        self.initialized = True

    def record_event(self, event_name: str, details: str) -> None:
        self.events.append((event_name, details))

    def close(self) -> None:
        self.closed = True


class DummyFeed:
    def __init__(self, _base_url: str, market_id: str) -> None:
        self.market_id = market_id
        self.warmup_calls: list[int] = []

    def warmup(self, cycles: int):
        self.warmup_calls.append(cycles)
        return [{"market_id": self.market_id, "cycle": i + 1} for i in range(cycles)]


class DummyCoordinator:
    def __init__(self, _logger: logging.Logger, _interval_seconds: int) -> None:
        self.run_called = False

    def run(self, stop_event: threading.Event) -> None:
        self.run_called = True
        stop_event.set()


class FakeSignalModule:
    SIGINT = 2
    SIGTERM = 15

    def __init__(self) -> None:
        self.handlers = {}

    def signal(self, signum, handler):
        self.handlers[signum] = handler


def _write_config(path: Path) -> None:
    path.write_text(
        json.dumps(
            {
                "app": {"name": "poly-oracle-test"},
                "logging": {"level": "INFO"},
                "state": {"sqlite_path": str(path.parent / "state.db")},
                "polymarket": {
                    "base_url": "https://clob.polymarket.com",
                    "market_symbol": "BTC",
                },
                "data_feeds": {
                    "primary": "primary_placeholder",
                    "fallback": "fallback_placeholder",
                    "warmup_cycles": 1,
                },
                "loop": {"interval_seconds": 1},
            }
        ),
        encoding="utf-8",
    )


def _write_env(path: Path, *, include_credentials: bool = True) -> None:
    lines = ["BTC_MARKET_ID=btc-market-1"]
    if include_credentials:
        lines.extend(
            [
                "POLYMARKET_API_KEY=test-key",
                "POLYMARKET_API_SECRET=test-secret",
                "POLYMARKET_API_PASSPHRASE=test-passphrase",
            ]
        )
    path.write_text("\n".join(lines), encoding="utf-8")


def test_startup_fails_when_credentials_missing(tmp_path: Path, monkeypatch) -> None:
    config = tmp_path / "startup.json"
    env_file = tmp_path / ".env"
    _write_config(config)
    _write_env(env_file, include_credentials=False)

    monkeypatch.delenv("POLYMARKET_API_KEY", raising=False)
    monkeypatch.delenv("POLYMARKET_API_SECRET", raising=False)
    monkeypatch.delenv("POLYMARKET_API_PASSPHRASE", raising=False)
    monkeypatch.delenv("BTC_MARKET_ID", raising=False)

    result = run_startup(check=True, config_path=config, env_path=env_file)

    assert result.exit_code == 2
    assert result.reason == "settings_validation_failed"


def test_startup_success_path_with_mocks(tmp_path: Path, monkeypatch) -> None:
    config = tmp_path / "startup.json"
    env_file = tmp_path / ".env"
    _write_config(config)
    _write_env(env_file, include_credentials=True)

    monkeypatch.delenv("POLYMARKET_API_KEY", raising=False)
    monkeypatch.delenv("POLYMARKET_API_SECRET", raising=False)
    monkeypatch.delenv("POLYMARKET_API_PASSPHRASE", raising=False)
    monkeypatch.delenv("BTC_MARKET_ID", raising=False)

    result = run_startup(
        check=False,
        config_path=config,
        env_path=env_file,
        signal_module=FakeSignalModule(),
        state_store_factory=DummyStore,
        primary_feed_factory=DummyFeed,
        fallback_feed_factory=DummyFeed,
        coordinator_factory=DummyCoordinator,
    )

    assert result.exit_code == 0
    assert result.reason == "graceful_shutdown"


def test_register_shutdown_handlers_sets_stop_event() -> None:
    stop_event = threading.Event()
    logger = logging.getLogger("test-shutdown")
    signal_mod = FakeSignalModule()

    register_shutdown_handlers(stop_event, logger, signal_module=signal_mod)
    signal_mod.handlers[signal_mod.SIGTERM](signal_mod.SIGTERM, None)

    assert stop_event.is_set()
