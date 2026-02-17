"""Trading loop coordinator scaffold (no trading logic yet)."""

from __future__ import annotations

import logging
import threading


class TradingLoopCoordinator:
    """Scaffold coordinator for loop lifecycle only."""

    def __init__(self, logger: logging.Logger, interval_seconds: int) -> None:
        self.logger = logger
        self.interval_seconds = interval_seconds

    def run(self, stop_event: threading.Event) -> None:
        self.logger.info(
            "Trading loop coordinator started.",
            extra={"event": "loop_started", "context": {"interval_seconds": self.interval_seconds}},
        )

        while not stop_event.is_set():
            # Phase 2 scope: heartbeat only, no trading execution.
            self.logger.info("Loop scaffold tick.", extra={"event": "loop_tick"})
            stop_event.wait(timeout=self.interval_seconds)

        self.logger.info("Trading loop coordinator stopped.", extra={"event": "loop_stopped"})
