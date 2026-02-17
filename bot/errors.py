"""Startup-specific error types with explicit exit codes and reasons."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class StartupError(Exception):
    """Base startup error with explicit process exit details."""

    message: str
    exit_code: int
    reason: str

    def __str__(self) -> str:
        return self.message


class ConfigFileLoadError(StartupError):
    """Raised when configuration or environment files cannot be loaded."""

    def __init__(self, message: str) -> None:
        super().__init__(message=message, exit_code=2, reason="config_load_failed")


class SettingsValidationError(StartupError):
    """Raised when startup settings fail strict schema validation."""

    def __init__(self, errors: list[str]) -> None:
        rendered = "\n".join(f"- {item}" for item in errors)
        super().__init__(
            message=f"Startup settings validation failed:\n{rendered}",
            exit_code=2,
            reason="settings_validation_failed",
        )
        self.errors = errors


class DependencyInitializationError(StartupError):
    """Raised when a startup dependency cannot be initialized."""

    def __init__(self, message: str) -> None:
        super().__init__(message=message, exit_code=3, reason="dependency_init_failed")


class FatalStartupError(StartupError):
    """Raised for fatal runtime startup conditions."""

    def __init__(self, message: str, reason: str = "fatal_startup_error") -> None:
        super().__init__(message=message, exit_code=4, reason=reason)
