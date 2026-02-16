try:
    from src.dashboard.terminal import TerminalDashboard, create_dashboard
except ModuleNotFoundError:  # Optional terminal dependency (questionary/rich)
    TerminalDashboard = None  # type: ignore[assignment]
    create_dashboard = None  # type: ignore[assignment]

__all__ = [
    "TerminalDashboard",
    "create_dashboard",
]
