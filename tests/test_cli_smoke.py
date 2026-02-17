"""Smoke tests for CLI startup boot path."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _run_cli(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "cli.py", *args],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )


def _write_config(path: Path) -> None:
    payload = {
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
    path.write_text(json.dumps(payload), encoding="utf-8")


def _write_env(path: Path) -> None:
    path.write_text(
        "\n".join(
            [
                "BTC_MARKET_ID=btc-market-1",
                "POLYMARKET_API_KEY=test-key",
                "POLYMARKET_API_SECRET=test-secret",
                "POLYMARKET_API_PASSPHRASE=test-passphrase",
            ]
        ),
        encoding="utf-8",
    )


def test_start_help_exit_code_zero() -> None:
    result = _run_cli("start", "--help")
    assert result.returncode == 0
    assert "--check" in result.stdout


def test_start_check_exit_code_zero(tmp_path: Path) -> None:
    config_path = tmp_path / "startup.json"
    env_path = tmp_path / ".env"
    _write_config(config_path)
    _write_env(env_path)

    result = _run_cli(
        "start",
        "--check",
        "--config",
        str(config_path),
        "--env-file",
        str(env_path),
    )

    merged_output = result.stdout + result.stderr
    assert result.returncode == 0
    assert '"event": "startup_step"' in merged_output
    assert "Preflight check succeeded" in merged_output
