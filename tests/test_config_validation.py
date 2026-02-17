"""Unit tests for strict startup config validation."""

from __future__ import annotations

import pytest

from bot.config.schema import validate_settings
from bot.errors import SettingsValidationError


def _valid_config() -> dict:
    return {
        "app": {"name": "poly-oracle-test"},
        "logging": {"level": "INFO"},
        "state": {"sqlite_path": "state/poly_oracle.db"},
        "polymarket": {
            "base_url": "https://clob.polymarket.com",
            "market_symbol": "BTC",
        },
        "data_feeds": {
            "primary": "primary_placeholder",
            "fallback": "fallback_placeholder",
            "warmup_cycles": 2,
        },
        "loop": {"interval_seconds": 30},
    }


def _valid_env() -> dict[str, str]:
    return {
        "BTC_MARKET_ID": "btc-market-1",
        "POLYMARKET_API_KEY": "key",
        "POLYMARKET_API_SECRET": "secret",
        "POLYMARKET_API_PASSPHRASE": "passphrase",
    }


def test_validate_settings_valid_payload() -> None:
    settings = validate_settings(_valid_config(), _valid_env())

    assert settings.app_name == "poly-oracle-test"
    assert settings.polymarket.market_symbol == "BTC"
    assert settings.polymarket.btc_market_id == "btc-market-1"
    assert settings.loop.interval_seconds == 30


def test_validate_settings_missing_required_config_key() -> None:
    payload = _valid_config()
    del payload["logging"]["level"]

    with pytest.raises(SettingsValidationError) as exc:
        validate_settings(payload, _valid_env())

    assert "logging.level" in str(exc.value)


def test_validate_settings_invalid_values() -> None:
    payload = _valid_config()
    payload["polymarket"]["market_symbol"] = "ETH"
    payload["loop"]["interval_seconds"] = "fast"

    with pytest.raises(SettingsValidationError) as exc:
        validate_settings(payload, _valid_env())

    message = str(exc.value)
    assert "must be 'BTC'" in message
    assert "loop.interval_seconds" in message


def test_validate_settings_missing_credentials() -> None:
    env = _valid_env()
    del env["POLYMARKET_API_SECRET"]

    with pytest.raises(SettingsValidationError) as exc:
        validate_settings(_valid_config(), env)

    assert "POLYMARKET_API_SECRET" in str(exc.value)
