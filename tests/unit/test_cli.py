"""Tests for CLI commands."""
import pytest
from unittest.mock import patch, Mock, AsyncMock, MagicMock
from typer.testing import CliRunner

from cli import app


runner = CliRunner()


class TestStartCommand:
    @patch("src.dashboard.terminal.create_dashboard")
    @patch("cli.asyncio.run")
    def test_start_default_launches_terminal(self, mock_run, mock_create):
        mock_dashboard = Mock()
        mock_dashboard.run = AsyncMock()
        mock_create.return_value = mock_dashboard
        result = runner.invoke(app, ["start"])
        assert result.exit_code == 0

    @patch("src.dashboard.terminal.create_dashboard")
    @patch("cli.asyncio.run")
    def test_start_paper_mode(self, mock_run, mock_create):
        mock_create.return_value = Mock()
        result = runner.invoke(app, ["start", "--paper"])
        assert result.exit_code == 0

    @patch("src.dashboard.terminal.create_dashboard")
    @patch("cli.asyncio.run")
    def test_start_live_mode(self, mock_run, mock_create):
        mock_create.return_value = Mock()
        result = runner.invoke(app, ["start", "--live"])
        assert result.exit_code == 0

    def test_start_paper_and_live_error(self):
        result = runner.invoke(app, ["start", "--paper", "--live"])
        assert result.exit_code == 1
        assert "mutually exclusive" in result.output


class TestTradeCommand:
    @patch("cli.paper")
    def test_trade_mode_auto(self, mock_paper):
        result = runner.invoke(app, ["trade", "--mode", "auto", "--once"])
        assert result.exit_code == 0

    @patch("cli.paper")
    def test_trade_mode_crypto(self, mock_paper):
        result = runner.invoke(app, ["trade", "--mode", "crypto", "--once"])
        assert result.exit_code == 0

    @patch("cli.forecast")
    def test_trade_mode_chosen_needs_market_id(self, mock_forecast):
        result = runner.invoke(app, ["trade", "--mode", "chosen"])
        assert result.exit_code == 1
        assert "market-id" in result.output.lower()


class TestBacktestCommand:
    @patch("src.calibration.backtester.Backtester.run_replay")
    @patch("src.data.storage.duckdb_client.DuckDBClient")
    @patch("cli.setup_logging")
    @patch("cli.get_settings")
    def test_backtest_replay_no_forecasts(
        self, mock_get_settings, mock_logging, mock_duckdb_cls, mock_replay
    ):
        from pathlib import Path
        from src.models.backtest import BacktestResult

        s = Mock()
        s.log_level = "DEBUG"
        s.database = Mock()
        s.database.db_dir = Path("/tmp/test-db")
        s.database.duckdb_path = Path("/tmp/test-db/analytics.duckdb")
        s.risk = Mock()
        s.risk.initial_bankroll = 100
        s.risk.min_edge = 0.05
        s.risk.max_position_pct = 0.10
        s.risk.min_bet = 1.0
        s.risk.max_bet = 10.0
        mock_get_settings.return_value = s

        # Mock the DuckDB context manager
        mock_db = MagicMock()
        mock_duckdb_cls.return_value = mock_db
        mock_db.__enter__ = Mock(return_value=mock_db)
        mock_db.__exit__ = Mock(return_value=False)

        # Mock replay returning empty result
        mock_replay.return_value = BacktestResult()

        result = runner.invoke(app, ["backtest"])
        assert result.exit_code == 0
        assert "No resolved forecasts" in result.output


class TestLiveCommand:
    @patch("cli.setup_logging")
    @patch("cli.get_settings")
    def test_live_missing_credentials(self, mock_get_settings, mock_logging):
        from pathlib import Path

        s = Mock()
        s.log_level = "DEBUG"
        s.database = Mock()
        s.database.db_dir = Path("/tmp/test-db")
        s.polymarket = Mock()
        s.polymarket.api_key = None
        s.polymarket.api_secret = None
        s.polymarket.api_passphrase = None
        mock_get_settings.return_value = s

        result = runner.invoke(app, ["live", "--once"])
        assert result.exit_code == 1


class TestHelpOutput:
    def test_no_web_references_in_help(self):
        """Verify web dashboard references are removed from CLI help."""
        result = runner.invoke(app, ["start", "--help"])
        assert result.exit_code == 0
        assert "--host" not in result.output
        assert "--port" not in result.output
        assert "--terminal" not in result.output

    def test_start_has_paper_and_live(self):
        result = runner.invoke(app, ["start", "--help"])
        assert "--paper" in result.output
        assert "--live" in result.output

    def test_backtest_has_full_flag(self):
        result = runner.invoke(app, ["backtest", "--help"])
        assert "--full" in result.output

    def test_trade_has_modes(self):
        result = runner.invoke(app, ["trade", "--help"])
        assert "crypto" in result.output
        assert "auto" in result.output
        assert "chosen" in result.output
