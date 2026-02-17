"""Poly-Oracle v1 CLI skeleton.

Usage:
    python cli.py start
    python cli.py start --help
"""

from __future__ import annotations

import argparse
import logging
import sys

LOGGER_NAME = "poly_oracle"

BANNER_LINES = [
    "============================================================",
    "POLY-ORACLE V1 STARTUP | SKELETON MODE | NO TRADING EXECUTION",
    "STRICT MODE: START COMMAND IS A BOOT PATH STUB ONLY",
    "============================================================",
]

TODO_CHECKPOINTS = [
    "Wire config loader and env validation.",
    "Connect market data adapters and cache layer.",
    "Attach strategy orchestration pipeline.",
    "Attach execution + risk guards.",
    "Attach state persistence and monitoring hooks.",
]


def _configure_logging() -> logging.Logger:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    return logging.getLogger(LOGGER_NAME)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="poly-oracle",
        description="Poly-Oracle v1 CLI skeleton.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    start_parser = subparsers.add_parser(
        "start",
        help="Boot the v1 startup path (stub).",
        description="Boot the v1 startup path and emit migration TODO checkpoints.",
    )
    start_parser.add_argument(
        "--live",
        action="store_true",
        help="Preview live mode startup path (still stubbed).",
    )
    start_parser.set_defaults(handler=_handle_start)

    return parser


def _handle_start(args: argparse.Namespace) -> int:
    logger = _configure_logging()
    mode = "LIVE" if args.live else "PAPER"

    for line in BANNER_LINES:
        logger.info(line)

    logger.info("BOOT MODE: %s", mode)
    for idx, checkpoint in enumerate(TODO_CHECKPOINTS, start=1):
        logger.warning("TODO-CHECKPOINT-%02d: %s", idx, checkpoint)

    logger.info("Startup stub completed successfully.")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    handler = getattr(args, "handler", None)
    if handler is None:
        parser.print_help()
        return 1
    return int(handler(args))


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
