"""Poly-Oracle v1 CLI skeleton.

Usage:
    python cli.py start
    python cli.py start --check
    python cli.py start --help
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from bot.startup import run_startup


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="poly-oracle",
        description="Poly-Oracle v1 CLI.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    start_parser = subparsers.add_parser(
        "start",
        help="Execute deterministic startup orchestration.",
        description="Run the startup pipeline with strict validation and fail-fast behavior.",
    )
    start_parser.add_argument(
        "--check",
        action="store_true",
        help="Run preflight startup checks without entering the loop.",
    )
    start_parser.add_argument(
        "--config",
        type=Path,
        default=Path("bot/config/startup.json"),
        help="Path to JSON startup config file.",
    )
    start_parser.add_argument(
        "--env-file",
        type=Path,
        default=Path(".env"),
        help="Path to .env file to load before validation.",
    )
    start_parser.set_defaults(handler=_handle_start)

    return parser


def _handle_start(args: argparse.Namespace) -> int:
    result = run_startup(
        check=bool(args.check),
        config_path=args.config,
        env_path=args.env_file,
    )
    if result.exit_code != 0:
        print(
            f"STARTUP_EXIT reason={result.reason} code={result.exit_code}",
            file=sys.stderr,
        )
    return result.exit_code


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
