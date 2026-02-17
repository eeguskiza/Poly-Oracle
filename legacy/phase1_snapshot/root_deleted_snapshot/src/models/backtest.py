"""BacktestResult model for historical simulation output."""
from __future__ import annotations

from pydantic import BaseModel


class BacktestResult(BaseModel):
    """Aggregated backtest output with equity curve data."""

    total_forecasts: int = 0
    simulated_trades: int = 0
    final_equity: float = 0.0
    max_drawdown: float = 0.0
    sharpe_ratio: float | None = None
    win_rate: float = 0.0
    brier_score: float = 0.0
    equity_curve: list[float] = []
    daily_returns: list[float] = []
    trade_log: list[dict] = []
