import pytest
from datetime import datetime, timezone, timedelta
from unittest.mock import Mock, AsyncMock, patch, MagicMock

from bot.execution.live import LiveTradingExecutor, HAS_CLOB
from bot.models import Market, EdgeAnalysis, ExecutionResult, PositionSize, RiskCheck


def _make_settings(has_creds=True):
    s = Mock()
    s.polymarket = Mock()
    s.polymarket.clob_url = "https://clob.polymarket.com"
    if has_creds:
        s.polymarket.api_key = "test-key"
        s.polymarket.api_secret = "test-secret"
        s.polymarket.api_passphrase = "test-pass"
    else:
        s.polymarket.api_key = None
        s.polymarket.api_secret = None
        s.polymarket.api_passphrase = None
    return s


def _make_market():
    now = datetime.now(timezone.utc)
    return Market(
        id="test-market",
        question="Will BTC reach $100k?",
        description="Bitcoin price",
        market_type="crypto",
        current_price=0.55,
        volume_24h=10_000,
        liquidity=50_000,
        resolution_date=now + timedelta(days=30),
        created_at=now - timedelta(days=5),
        outcomes=["Yes", "No"],
        token_ids={"Yes": "tok_yes", "No": "tok_no"},
    )


def _make_edge_analysis(action="TRADE", direction="BUY_YES"):
    return EdgeAnalysis(
        our_forecast=0.60,
        market_price=0.50,
        raw_edge=0.10,
        abs_edge=0.10,
        weighted_edge=0.10,
        direction=direction,
        has_actionable_edge=(action == "TRADE"),
        recommended_action=action,
        reasoning="Edge above threshold",
    )


@pytest.fixture
def mock_sqlite():
    m = Mock()
    m.get_open_positions = Mock(return_value=[])
    m.get_daily_stats = Mock(return_value=None)
    m.insert_trade = Mock(return_value="trade-123")
    return m


@pytest.fixture
def mock_sizer():
    m = Mock()
    m.calculate = Mock(return_value=PositionSize(
        amount_usd=5.0,
        num_shares=10.0,
        kelly_fraction=0.05,
        applied_fraction=0.05,
        constraints_applied={},
    ))
    return m


@pytest.fixture
def mock_risk():
    m = Mock()
    m.check = Mock(return_value=RiskCheck(
        passed=True,
        violations=[],
        daily_loss_pct=0.0,
        num_open_positions=0,
        proposed_market_exposure=0.1,
    ))
    return m


class TestLiveExecutorSkipCases:
    @pytest.mark.asyncio
    async def test_skip_when_action_is_skip(self, mock_sqlite, mock_sizer, mock_risk):
        settings = _make_settings()
        executor = LiveTradingExecutor(mock_sqlite, mock_sizer, mock_risk, settings)
        edge = _make_edge_analysis(action="SKIP")

        result = await executor.execute(edge, 0.60, _make_market(), 100.0)
        assert result is None

    @pytest.mark.asyncio
    async def test_skip_when_position_size_zero(self, mock_sqlite, mock_sizer, mock_risk):
        settings = _make_settings()
        mock_sizer.calculate.return_value = PositionSize(
            amount_usd=0, num_shares=0, kelly_fraction=0, applied_fraction=0,
            constraints_applied={},
        )
        executor = LiveTradingExecutor(mock_sqlite, mock_sizer, mock_risk, settings)
        edge = _make_edge_analysis()

        result = await executor.execute(edge, 0.60, _make_market(), 100.0)
        assert result is None


class TestRiskChecks:
    @pytest.mark.asyncio
    async def test_risk_check_failure(self, mock_sqlite, mock_sizer, mock_risk):
        settings = _make_settings()
        mock_risk.check.return_value = RiskCheck(
            passed=False,
            violations=["max_daily_loss"],
            daily_loss_pct=0.15,
            num_open_positions=5,
            proposed_market_exposure=0.2,
        )
        executor = LiveTradingExecutor(mock_sqlite, mock_sizer, mock_risk, settings)
        edge = _make_edge_analysis()

        result = await executor.execute(edge, 0.60, _make_market(), 100.0)
        assert result is not None
        assert result.success is False
        assert "Risk check failed" in result.message


class TestCredentialValidation:
    def test_missing_clob_or_credentials_raises(self, mock_sqlite, mock_sizer, mock_risk):
        """_ensure_clob raises RuntimeError when clob not installed or creds missing."""
        settings = _make_settings(has_creds=False)
        executor = LiveTradingExecutor(mock_sqlite, mock_sizer, mock_risk, settings)

        with pytest.raises(RuntimeError):
            executor._ensure_clob()


class TestOrderSubmission:
    @pytest.mark.asyncio
    async def test_successful_order(self, mock_sqlite, mock_sizer, mock_risk):
        settings = _make_settings()
        executor = LiveTradingExecutor(mock_sqlite, mock_sizer, mock_risk, settings)

        # Mock the CLOB client directly
        mock_clob = MagicMock()
        mock_clob.create_and_post_order.return_value = {"orderID": "order-abc"}
        executor._clob = mock_clob

        edge = _make_edge_analysis()
        result = await executor.execute(edge, 0.60, _make_market(), 100.0)

        assert result.success is True
        assert "order-abc" in result.message
        mock_clob.create_and_post_order.assert_called_once()

    @pytest.mark.asyncio
    async def test_order_failure_returns_error(self, mock_sqlite, mock_sizer, mock_risk):
        settings = _make_settings()
        executor = LiveTradingExecutor(mock_sqlite, mock_sizer, mock_risk, settings)

        mock_clob = MagicMock()
        mock_clob.create_and_post_order.side_effect = Exception("API timeout")
        executor._clob = mock_clob

        edge = _make_edge_analysis()
        result = await executor.execute(edge, 0.60, _make_market(), 100.0)

        assert result.success is False
        assert "API timeout" in result.message
