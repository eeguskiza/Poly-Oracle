"""Smoke tests for CLI boot path."""

from __future__ import annotations

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


def test_start_help_exit_code_zero() -> None:
    result = _run_cli("start", "--help")
    assert result.returncode == 0
    assert "Boot the v1 startup path" in result.stdout


def test_start_boot_path_exit_code_zero_and_todos() -> None:
    result = _run_cli("start")
    merged_output = result.stdout + result.stderr

    assert result.returncode == 0
    assert "SKELETON MODE" in merged_output
    assert "TODO-CHECKPOINT-01" in merged_output
