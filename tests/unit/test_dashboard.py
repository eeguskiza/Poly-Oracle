import pytest
from datetime import datetime, timezone
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from pathlib import Path

from src.dashboard.terminal import TerminalDashboard, create_dashboard


@pytest.fixture
def mock_settings():
    settings = Mock()
    settings.paper_trading = True
    settings.log_level = "DEBUG"

    settings.llm = Mock()
    settings.llm.model = "mistral"
    settings.llm.base_url = "http://localhost:11434"
    settings.llm.embedding_model = "nomic-embed-text"
    settings.llm.temperature = 0.7
    settings.llm.max_tokens = 2000
    settings.llm.timeout = 300

    settings.risk = Mock()
    settings.risk.initial_bankroll = 50.0
    settings.risk.max_position_pct = 0.10
    settings.risk.min_bet = 1.0
    settings.risk.max_bet = 10.0
    settings.risk.max_daily_loss_pct = 0.10
    settings.risk.max_daily_loss = 0.20
    settings.risk.max_open_positions = 8
    settings.risk.max_single_market_exposure = 0.15
    settings.risk.min_edge = 0.08
    settings.risk.min_confidence = 0.65
    settings.risk.min_liquidity = 1000.0

    settings.data = Mock()
    settings.data.newsapi_key = None
    settings.data.cache_ttl_news = 3600

    settings.polymarket = Mock()
    settings.polymarket.api_key = None
    settings.polymarket.api_secret = None
    settings.polymarket.api_passphrase = None
    settings.polymarket.gamma_url = "https://gamma-api.polymarket.com"

    settings.database = Mock()
    settings.database.db_dir = Path("db")
    settings.database.duckdb_path = Path("db/analytics.duckdb")
    settings.database.sqlite_path = Path("db/poly_oracle.db")
    settings.database.chroma_path = Path("db/chroma")

    return settings


@pytest.fixture
def mock_sqlite():
    mock = Mock()
    mock.initialize_schema = Mock()
    mock.get_open_positions = Mock(return_value=[])
    mock.get_current_bankroll = Mock(return_value=50.0)
    mock.get_daily_stats = Mock(return_value=None)
    mock.conn = Mock()
    mock.conn.cursor = Mock(return_value=Mock(
        execute=Mock(return_value=Mock(fetchone=Mock(return_value=(0,)), fetchall=Mock(return_value=[])))
    ))
    return mock


@pytest.fixture
def mock_duckdb():
    mock = Mock()
    mock.initialize_schema = Mock()
    mock.get_calibration_stats = Mock(return_value={
        "overall": {"count": 10, "avg_brier_score": 0.15},
        "by_type": {},
    })
    mock.conn = Mock()
    mock.conn.execute = Mock(return_value=Mock(fetchone=Mock(return_value=(5,))))
    return mock


@pytest.fixture
def mock_chroma():
    mock = Mock()
    mock.get_collection_stats = Mock(return_value={"news": 100, "market_context": 50})
    return mock


@pytest.fixture
def mock_polymarket():
    return AsyncMock()


@pytest.fixture
def mock_news():
    return AsyncMock()


@pytest.fixture
def mock_resolver():
    mock = AsyncMock()
    mock.run_resolution_cycle = AsyncMock(return_value={"checked": 0, "resolved": 0, "pnl": 0.0})
    return mock


@pytest.fixture
def mock_calibrator():
    return Mock()


@pytest.fixture
def mock_analyzer():
    return Mock()


@pytest.fixture
def mock_feedback():
    mock = Mock()
    mock.get_performance_summary = Mock(return_value={
        "total_forecasts": 10,
        "resolved_forecasts": 5,
        "pending_forecasts": 5,
        "overall_brier_raw": 0.20,
        "overall_brier_calibrated": 0.18,
        "calibration_improvement": 0.02,
        "brier_by_type": {},
        "win_rate": 0.6,
        "avg_edge": 0.05,
        "market_brier": 0.22,
        "value_added_vs_market": 0.04,
    })
    return mock


@pytest.fixture
def mock_sizer():
    return Mock()


@pytest.fixture
def mock_risk_manager():
    return Mock()


@pytest.fixture
def mock_executor():
    return AsyncMock()


@pytest.fixture
def dashboard(
    mock_settings,
    mock_polymarket,
    mock_news,
    mock_sqlite,
    mock_duckdb,
    mock_chroma,
    mock_resolver,
    mock_calibrator,
    mock_analyzer,
    mock_feedback,
    mock_sizer,
    mock_risk_manager,
    mock_executor,
):
    return TerminalDashboard(
        settings=mock_settings,
        polymarket=mock_polymarket,
        news=mock_news,
        sqlite=mock_sqlite,
        duckdb=mock_duckdb,
        chroma=mock_chroma,
        resolver=mock_resolver,
        calibrator=mock_calibrator,
        analyzer=mock_analyzer,
        feedback=mock_feedback,
        sizer=mock_sizer,
        risk_manager=mock_risk_manager,
        executor=mock_executor,
    )


class TestTerminalDashboardInstantiation:
    def test_instantiation(self, dashboard):
        """TerminalDashboard can be instantiated with all dependencies."""
        assert dashboard is not None
        assert dashboard.settings.paper_trading is True
        assert dashboard.polymarket is not None
        assert dashboard.sqlite is not None
        assert dashboard.duckdb is not None

    def test_instantiation_stores_all_deps(self, dashboard, mock_settings, mock_sqlite):
        """All injected dependencies are stored correctly."""
        assert dashboard.settings is mock_settings
        assert dashboard.sqlite is mock_sqlite


class TestGetSystemStatus:
    @pytest.mark.asyncio
    async def test_system_checks_returns_list(self, dashboard):
        """_get_system_checks returns a list of (name, status, detail) tuples."""
        with patch("src.dashboard.terminal.OllamaClient") as MockOllama:
            mock_client = AsyncMock()
            mock_client.is_available = AsyncMock(return_value=True)
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            MockOllama.return_value = mock_client

            with patch("src.dashboard.terminal.httpx.get") as mock_get:
                mock_get.return_value = Mock(status_code=200)
                checks = await dashboard._get_system_checks()

        assert isinstance(checks, list)
        assert len(checks) > 0
        for name, status, detail in checks:
            assert isinstance(name, str)
            assert isinstance(status, str)
            assert isinstance(detail, str)

    @pytest.mark.asyncio
    async def test_system_checks_includes_all_components(self, dashboard):
        """System checks include all expected components."""
        with patch("src.dashboard.terminal.OllamaClient") as MockOllama:
            mock_client = AsyncMock()
            mock_client.is_available = AsyncMock(return_value=True)
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            MockOllama.return_value = mock_client

            with patch("src.dashboard.terminal.httpx.get") as mock_get:
                mock_get.return_value = Mock(status_code=200)
                checks = await dashboard._get_system_checks()

        component_names = [name for name, _, _ in checks]
        assert "Ollama" in component_names
        assert "DuckDB" in component_names
        assert "SQLite" in component_names
        assert "ChromaDB" in component_names
        assert "NewsAPI" in component_names
        assert "Polymarket" in component_names
        assert "Mode" in component_names
        assert "Bankroll" in component_names

    @pytest.mark.asyncio
    async def test_system_checks_newsapi_warn_when_no_key(self, dashboard):
        """NewsAPI shows WARN when no API key is configured."""
        dashboard.settings.data.newsapi_key = None

        with patch("src.dashboard.terminal.OllamaClient") as MockOllama:
            mock_client = AsyncMock()
            mock_client.is_available = AsyncMock(return_value=True)
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            MockOllama.return_value = mock_client

            with patch("src.dashboard.terminal.httpx.get") as mock_get:
                mock_get.return_value = Mock(status_code=200)
                checks = await dashboard._get_system_checks()

        news_check = [c for c in checks if c[0] == "NewsAPI"][0]
        assert news_check[1] == "[WARN]"
        assert "RSS only" in news_check[2]


class TestPortfolioSummary:
    async def _get_portfolio_data(self, dashboard):
        """Helper to get portfolio-relevant data the same way the screen does."""
        positions = dashboard.sqlite.get_open_positions()
        bankroll = dashboard.sqlite.get_current_bankroll()
        return {
            "positions": positions,
            "bankroll": bankroll,
            "num_positions": len(positions),
            "total_invested": sum(p["amount_usd"] for p in positions),
            "total_unrealized": sum(p["unrealized_pnl"] for p in positions),
        }

    @pytest.mark.asyncio
    async def test_portfolio_summary_empty(self, dashboard):
        """Portfolio summary works with no positions."""
        data = await self._get_portfolio_data(dashboard)
        assert data["num_positions"] == 0
        assert data["bankroll"] == 50.0
        assert data["total_invested"] == 0.0
        assert data["total_unrealized"] == 0.0

    @pytest.mark.asyncio
    async def test_portfolio_summary_with_positions(self, dashboard):
        """Portfolio summary correctly aggregates positions."""
        dashboard.sqlite.get_open_positions.return_value = [
            {
                "market_id": "market_1",
                "direction": "BUY_YES",
                "num_shares": 10.0,
                "amount_usd": 5.0,
                "avg_entry_price": 0.50,
                "current_price": 0.60,
                "unrealized_pnl": 1.0,
                "updated_at": "2026-01-01T00:00:00Z",
            },
            {
                "market_id": "market_2",
                "direction": "BUY_NO",
                "num_shares": 8.0,
                "amount_usd": 4.0,
                "avg_entry_price": 0.40,
                "current_price": 0.35,
                "unrealized_pnl": 0.40,
                "updated_at": "2026-01-01T00:00:00Z",
            },
        ]
        dashboard.sqlite.get_current_bankroll.return_value = 41.0

        data = await self._get_portfolio_data(dashboard)
        assert data["num_positions"] == 2
        assert data["bankroll"] == 41.0
        assert data["total_invested"] == 9.0
        assert data["total_unrealized"] == pytest.approx(1.4)


class TestCreateDashboard:
    @staticmethod
    def _make_fake_settings():
        s = Mock()
        s.paper_trading = True
        s.log_level = "DEBUG"
        s.llm = Mock()
        s.llm.model = "mistral"
        s.llm.base_url = "http://localhost:11434"
        s.llm.embedding_model = "nomic-embed-text"
        s.llm.timeout = 300
        s.risk = Mock()
        s.risk.min_edge = 0.08
        s.risk.min_confidence = 0.65
        s.risk.min_liquidity = 1000.0
        s.database = Mock()
        s.database.db_dir = Mock()
        s.database.db_dir.mkdir = Mock()
        s.database.sqlite_path = Mock()
        s.database.duckdb_path = Mock()
        s.database.chroma_path = Mock()
        s.data = Mock()
        s.polymarket = Mock()
        return s

    @patch("src.dashboard.terminal.SQLiteClient")
    @patch("src.dashboard.terminal.DuckDBClient")
    @patch("src.dashboard.terminal.ChromaClient")
    @patch("src.dashboard.terminal.PolymarketClient")
    @patch("src.dashboard.terminal.NewsClient")
    @patch("src.dashboard.terminal.get_settings")
    def test_create_dashboard_returns_dashboard(
        self,
        mock_get_settings,
        mock_news_cls,
        mock_poly_cls,
        mock_chroma_cls,
        mock_duckdb_cls,
        mock_sqlite_cls,
    ):
        """create_dashboard() returns a TerminalDashboard instance."""
        fake_settings = self._make_fake_settings()
        mock_get_settings.return_value = fake_settings

        mock_sqlite_cls.return_value = Mock()
        mock_sqlite_cls.return_value.initialize_schema = Mock()
        mock_duckdb_cls.return_value = Mock()
        mock_duckdb_cls.return_value.initialize_schema = Mock()
        mock_chroma_cls.return_value = Mock()
        mock_poly_cls.return_value = Mock()
        mock_news_cls.return_value = Mock()

        dashboard = create_dashboard()
        assert isinstance(dashboard, TerminalDashboard)

    @patch("src.dashboard.terminal.SQLiteClient")
    @patch("src.dashboard.terminal.DuckDBClient")
    @patch("src.dashboard.terminal.ChromaClient")
    @patch("src.dashboard.terminal.PolymarketClient")
    @patch("src.dashboard.terminal.NewsClient")
    @patch("src.dashboard.terminal.get_settings")
    def test_create_dashboard_initializes_schemas(
        self,
        mock_get_settings,
        mock_news_cls,
        mock_poly_cls,
        mock_chroma_cls,
        mock_duckdb_cls,
        mock_sqlite_cls,
    ):
        """create_dashboard() initializes database schemas."""
        fake_settings = self._make_fake_settings()
        mock_get_settings.return_value = fake_settings

        mock_sqlite_inst = Mock()
        mock_sqlite_cls.return_value = mock_sqlite_inst
        mock_duckdb_inst = Mock()
        mock_duckdb_cls.return_value = mock_duckdb_inst
        mock_chroma_cls.return_value = Mock()
        mock_poly_cls.return_value = Mock()
        mock_news_cls.return_value = Mock()

        create_dashboard()
        mock_sqlite_inst.initialize_schema.assert_called_once()
        mock_duckdb_inst.initialize_schema.assert_called_once()
