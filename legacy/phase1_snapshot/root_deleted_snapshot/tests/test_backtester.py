import pytest
import math
from datetime import datetime, timezone
from unittest.mock import Mock, MagicMock

from src.calibration.backtester import Backtester
from src.models.backtest import BacktestResult


def _make_settings():
    s = Mock()
    s.risk = Mock()
    s.risk.initial_bankroll = 100.0
    s.risk.min_edge = 0.05
    s.risk.max_position_pct = 0.10
    s.risk.min_bet = 1.0
    s.risk.max_bet = 10.0
    s.llm = Mock()
    s.llm.base_url = "http://localhost:11434"
    s.llm.model = "mistral"
    s.llm.timeout = 300
    return s


def _make_forecast(prob=0.70, market_price=0.50, outcome=True):
    return {
        "id": "fc-1",
        "market_id": "mkt-1",
        "question": "test",
        "market_type": "general",
        "timestamp": datetime(2026, 1, 15, tzinfo=timezone.utc).isoformat(),
        "raw_probability": prob,
        "calibrated_probability": prob,
        "confidence": 0.8,
        "market_price_at_forecast": market_price,
        "edge": prob - market_price,
        "recommended_action": "TRADE",
        "debate_log": "[]",
        "judge_reasoning": "test",
        "resolved": True,
        "outcome": int(outcome),
        "brier_score": (prob - (1.0 if outcome else 0.0)) ** 2,
    }


def _mock_duckdb(forecasts):
    """Create a mock DuckDB that returns given forecasts."""
    mock = Mock()
    columns = list(forecasts[0].keys()) if forecasts else []
    rows = [tuple(f[c] for c in columns) for f in forecasts]

    mock_result = Mock()
    mock_result.fetchall.return_value = rows
    mock.conn = Mock()

    desc = [(c,) for c in columns]

    def execute_fn(query, params=None):
        m = Mock()
        m.fetchall.return_value = rows
        mock.conn.description = desc
        return m

    mock.conn.execute = execute_fn
    return mock


class TestBacktestResult:
    def test_empty_result(self):
        r = BacktestResult()
        assert r.total_forecasts == 0
        assert r.equity_curve == []

    def test_populated_result(self):
        r = BacktestResult(
            total_forecasts=10,
            simulated_trades=5,
            final_equity=110.0,
            equity_curve=[100, 105, 110],
        )
        assert r.final_equity == 110.0
        assert len(r.equity_curve) == 3


class TestMaxDrawdown:
    def test_no_drawdown(self):
        dd = Backtester._max_drawdown([100, 105, 110, 115])
        assert dd == 0.0

    def test_simple_drawdown(self):
        dd = Backtester._max_drawdown([100, 110, 88, 95])
        # Peak 110, trough 88, dd = (110-88)/110 = 0.2
        assert dd == pytest.approx(0.2, abs=0.01)

    def test_empty_curve(self):
        dd = Backtester._max_drawdown([100])
        assert dd == 0.0


class TestSharpe:
    def test_positive_returns(self):
        returns = [0.01, 0.02, 0.01, 0.015, 0.01]
        sharpe = Backtester._sharpe(returns)
        assert sharpe > 0

    def test_zero_returns(self):
        returns = [0.0, 0.0, 0.0]
        sharpe = Backtester._sharpe(returns)
        assert sharpe == 0.0

    def test_single_return(self):
        sharpe = Backtester._sharpe([0.05])
        assert sharpe == 0.0


class TestRunReplay:
    def test_replay_with_forecasts(self):
        forecasts = [
            _make_forecast(prob=0.70, market_price=0.50, outcome=True),
            _make_forecast(prob=0.30, market_price=0.50, outcome=False),
        ]
        mock_db = _mock_duckdb(forecasts)
        settings = _make_settings()

        bt = Backtester(duckdb=mock_db, settings=settings)
        result = bt.run_replay()

        assert result.total_forecasts == 2
        assert result.simulated_trades >= 0
        assert len(result.equity_curve) >= 1

    def test_replay_empty(self):
        mock_db = Mock()
        mock_db.conn = Mock()
        mock_result = Mock()
        mock_result.fetchall.return_value = []
        mock_db.conn.execute = Mock(return_value=mock_result)

        settings = _make_settings()
        bt = Backtester(duckdb=mock_db, settings=settings)
        result = bt.run_replay()

        assert result.total_forecasts == 0
        assert result.simulated_trades == 0

    def test_replay_skips_low_edge(self):
        forecasts = [
            _make_forecast(prob=0.51, market_price=0.50, outcome=True),
        ]
        mock_db = _mock_duckdb(forecasts)
        settings = _make_settings()
        settings.risk.min_edge = 0.05

        bt = Backtester(duckdb=mock_db, settings=settings)
        result = bt.run_replay()

        assert result.simulated_trades == 0  # edge 0.01 < min_edge 0.05
