"""Strict startup configuration schema and validation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from bot.errors import SettingsValidationError


@dataclass(frozen=True)
class LoggingSettings:
    level: str


@dataclass(frozen=True)
class StateSettings:
    sqlite_path: Path


@dataclass(frozen=True)
class PolymarketCredentials:
    api_key: str
    api_secret: str
    api_passphrase: str


@dataclass(frozen=True)
class PolymarketSettings:
    base_url: str
    market_symbol: str
    btc_market_id: str


@dataclass(frozen=True)
class DataFeedSettings:
    primary_name: str
    fallback_name: str
    warmup_cycles: int


@dataclass(frozen=True)
class LoopSettings:
    interval_seconds: int


@dataclass(frozen=True)
class AppSettings:
    app_name: str
    logging: LoggingSettings
    state: StateSettings
    polymarket: PolymarketSettings
    credentials: PolymarketCredentials
    data_feeds: DataFeedSettings
    loop: LoopSettings


_VALID_LOG_LEVELS = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}


def validate_settings(config: Mapping[str, Any], environ: Mapping[str, str]) -> AppSettings:
    """Validate startup settings with fail-fast, human-readable errors."""
    errors: list[str] = []

    app_name = _required_str(config, ["app", "name"], errors, "app.name")
    log_level = _required_str(config, ["logging", "level"], errors, "logging.level")
    sqlite_path_raw = _required_str(config, ["state", "sqlite_path"], errors, "state.sqlite_path")
    base_url = _required_str(config, ["polymarket", "base_url"], errors, "polymarket.base_url")
    market_symbol = _required_str(
        config,
        ["polymarket", "market_symbol"],
        errors,
        "polymarket.market_symbol",
    )
    primary_name = _required_str(
        config,
        ["data_feeds", "primary"],
        errors,
        "data_feeds.primary",
    )
    fallback_name = _required_str(
        config,
        ["data_feeds", "fallback"],
        errors,
        "data_feeds.fallback",
    )

    warmup_cycles = _required_int(
        config,
        ["data_feeds", "warmup_cycles"],
        errors,
        "data_feeds.warmup_cycles",
    )
    interval_seconds = _required_int(
        config,
        ["loop", "interval_seconds"],
        errors,
        "loop.interval_seconds",
    )

    btc_market_id = _required_env(environ, "BTC_MARKET_ID", errors)
    api_key = _required_env(environ, "POLYMARKET_API_KEY", errors)
    api_secret = _required_env(environ, "POLYMARKET_API_SECRET", errors)
    api_passphrase = _required_env(environ, "POLYMARKET_API_PASSPHRASE", errors)

    if log_level and log_level not in _VALID_LOG_LEVELS:
        errors.append(
            f"logging.level must be one of {sorted(_VALID_LOG_LEVELS)}; got '{log_level}'."
        )

    if market_symbol and market_symbol.upper() != "BTC":
        errors.append(
            "polymarket.market_symbol must be 'BTC' for single-market mode. "
            f"Got '{market_symbol}'."
        )

    if warmup_cycles is not None and warmup_cycles < 1:
        errors.append("data_feeds.warmup_cycles must be >= 1.")

    if interval_seconds is not None and interval_seconds < 1:
        errors.append("loop.interval_seconds must be >= 1.")

    if sqlite_path_raw and Path(sqlite_path_raw).name in {"", "."}:
        errors.append(
            "state.sqlite_path must point to a sqlite file path, e.g. 'state/poly_oracle.db'."
        )

    if errors:
        raise SettingsValidationError(errors)

    return AppSettings(
        app_name=app_name,
        logging=LoggingSettings(level=log_level),
        state=StateSettings(sqlite_path=Path(sqlite_path_raw)),
        polymarket=PolymarketSettings(
            base_url=base_url,
            market_symbol=market_symbol.upper(),
            btc_market_id=btc_market_id,
        ),
        credentials=PolymarketCredentials(
            api_key=api_key,
            api_secret=api_secret,
            api_passphrase=api_passphrase,
        ),
        data_feeds=DataFeedSettings(
            primary_name=primary_name,
            fallback_name=fallback_name,
            warmup_cycles=warmup_cycles,
        ),
        loop=LoopSettings(interval_seconds=interval_seconds),
    )


def _required_env(environ: Mapping[str, str], key: str, errors: list[str]) -> str:
    value = environ.get(key, "").strip()
    if not value:
        errors.append(
            f"Missing required environment variable '{key}'. "
            f"Set it in .env or export it before running startup."
        )
    return value


def _required_str(
    source: Mapping[str, Any],
    path: list[str],
    errors: list[str],
    label: str,
) -> str:
    value = _nested_get(source, path)
    if value is None:
        errors.append(f"Missing required config key '{label}'.")
        return ""
    if not isinstance(value, str):
        errors.append(f"Config key '{label}' must be a string.")
        return ""
    stripped = value.strip()
    if not stripped:
        errors.append(f"Config key '{label}' cannot be empty.")
    return stripped


def _required_int(
    source: Mapping[str, Any],
    path: list[str],
    errors: list[str],
    label: str,
) -> int | None:
    value = _nested_get(source, path)
    if value is None:
        errors.append(f"Missing required config key '{label}'.")
        return None

    if isinstance(value, bool):
        errors.append(f"Config key '{label}' must be an integer, not boolean.")
        return None

    if isinstance(value, int):
        return value

    errors.append(f"Config key '{label}' must be an integer.")
    return None


def _nested_get(source: Mapping[str, Any], path: list[str]) -> Any:
    current: Any = source
    for part in path:
        if not isinstance(current, Mapping) or part not in current:
            return None
        current = current[part]
    return current
