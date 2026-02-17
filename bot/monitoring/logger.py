"""Structured startup logging utilities."""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from typing import Any


class JsonFormatter(logging.Formatter):
    """Render logs as compact JSON objects for deterministic startup traces."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }

        event = getattr(record, "event", None)
        if event:
            payload["event"] = event

        context = getattr(record, "context", None)
        if isinstance(context, dict):
            payload.update(context)

        return json.dumps(payload, sort_keys=True)


def initialize_logger(level: str) -> logging.Logger:
    """Initialize singleton structured logger."""
    logger = logging.getLogger("poly_oracle.startup")
    logger.setLevel(level)
    logger.handlers.clear()
    logger.propagate = False

    handler = logging.StreamHandler()
    handler.setFormatter(JsonFormatter())
    logger.addHandler(handler)
    return logger


def log_step(logger: logging.Logger, step: int, description: str, **context: Any) -> None:
    logger.info(
        f"Step {step}: {description}",
        extra={"event": "startup_step", "context": {"step": step, **context}},
    )
