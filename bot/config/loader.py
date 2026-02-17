"""Load .env values and JSON startup configuration."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from bot.errors import ConfigFileLoadError


def load_environment(dotenv_path: Path) -> dict[str, str]:
    """Load .env file into process environment without overwriting existing vars."""
    if not dotenv_path.exists():
        return {}

    loaded: dict[str, str] = {}

    try:
        for raw_line in dotenv_path.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" not in line:
                raise ConfigFileLoadError(
                    f"Invalid .env line '{raw_line}'. Expected KEY=VALUE format."
                )

            key, value = line.split("=", 1)
            key = key.strip()
            value = _strip_quotes(value.strip())

            if not key:
                raise ConfigFileLoadError("Found empty key in .env file.")

            loaded[key] = value
            os.environ.setdefault(key, value)

    except OSError as exc:
        raise ConfigFileLoadError(f"Failed to read .env file '{dotenv_path}': {exc}") from exc

    return loaded


def load_json_config(config_path: Path) -> dict[str, Any]:
    """Load JSON startup config file."""
    if not config_path.exists():
        raise ConfigFileLoadError(
            f"Startup config file not found: '{config_path}'. "
            "Provide --config or create the default JSON config."
        )

    try:
        raw = config_path.read_text(encoding="utf-8")
        payload = json.loads(raw)
    except OSError as exc:
        raise ConfigFileLoadError(f"Failed to read config file '{config_path}': {exc}") from exc
    except json.JSONDecodeError as exc:
        raise ConfigFileLoadError(
            f"Invalid JSON in config file '{config_path}': {exc.msg} "
            f"(line {exc.lineno}, column {exc.colno})."
        ) from exc

    if not isinstance(payload, dict):
        raise ConfigFileLoadError(
            f"Startup config root must be a JSON object in '{config_path}'."
        )

    return payload


def _strip_quotes(value: str) -> str:
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {'"', "'"}:
        return value[1:-1]
    return value
