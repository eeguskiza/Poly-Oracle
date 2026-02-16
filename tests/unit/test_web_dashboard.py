from datetime import datetime, timezone

from config.settings import Settings
from src.dashboard import web
from src.dashboard.web import build_dashboard_summary
from src.data.storage.sqlite_client import SQLiteClient


def _settings_for_tmp_db(tmp_path, paper_trading=True, polymarket=None):
    polymarket = polymarket or {}
    settings = Settings()
    settings.database.db_dir = tmp_path
    settings.paper_trading = paper_trading
    settings.polymarket.username = polymarket.get("username")
    settings.polymarket.wallet_address = polymarket.get("wallet_address")
    return settings


def test_build_dashboard_summary_empty_db(tmp_path):
    settings = _settings_for_tmp_db(tmp_path)

    with SQLiteClient(settings.database.sqlite_path) as sqlite_client:
        sqlite_client.initialize_schema()

    summary = build_dashboard_summary(settings)

    assert summary["mode"] == "paper"
    assert summary["bankroll"] == 0.0
    assert summary["open_positions_count"] == 0
    assert summary["total_unrealized_pnl"] == 0.0
    assert summary["trades_today"] == 0
    assert summary["positions"] == []
    assert summary["account"]["status"] == "paper mode"


def test_build_dashboard_summary_with_data(tmp_path):
    settings = _settings_for_tmp_db(tmp_path)

    with SQLiteClient(settings.database.sqlite_path) as sqlite_client:
        sqlite_client.initialize_schema()
        sqlite_client.seed_initial_bankroll(100.0)

        sqlite_client.upsert_position(
            {
                "market_id": "mkt-1",
                "direction": "BUY_YES",
                "num_shares": 10.0,
                "amount_usd": 5.0,
                "avg_entry_price": 0.5,
                "current_price": 0.6,
                "unrealized_pnl": 1.0,
                "updated_at": datetime.now(timezone.utc).isoformat(),
            }
        )

        sqlite_client.insert_trade(
            {
                "market_id": "mkt-1",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "direction": "BUY_YES",
                "amount_usd": 5.0,
                "num_shares": 10.0,
                "entry_price": 0.5,
                "status": "FILLED",
            }
        )

        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        sqlite_client.upsert_daily_stats(
            {
                "date": today,
                "starting_bankroll": 100.0,
                "ending_bankroll": 101.0,
                "trades_executed": 2,
                "trades_won": 1,
                "gross_pnl": 1.0,
                "fees_paid": 0.0,
                "net_pnl": 1.0,
            }
        )

    summary = build_dashboard_summary(settings)

    assert summary["bankroll"] == 101.0
    assert summary["open_positions_count"] == 1
    assert summary["total_unrealized_pnl"] == 1.0
    assert summary["trades_today"] == 2
    assert summary["win_rate"] == 0.5
    assert len(summary["recent_trades"]) == 1
    assert summary["positions"][0]["market_id"] == "mkt-1"


def test_build_dashboard_summary_live_mode_uses_account_snapshot(tmp_path, monkeypatch):
    settings = _settings_for_tmp_db(
        tmp_path,
        paper_trading=False,
        polymarket={
            "username": "test-user",
            "wallet_address": "0xabc",
        },
    )

    monkeypatch.setattr(
        web,
        "_fetch_live_account_snapshot",
        lambda _settings: {
            "username": "test-user",
            "wallet_address": "0xabc",
            "live_balance_usdc": 123.45,
            "status": "live profile loaded from gamma-api",
        },
    )

    with SQLiteClient(settings.database.sqlite_path) as sqlite_client:
        sqlite_client.initialize_schema()

    summary = build_dashboard_summary(settings)

    assert summary["mode"] == "live"
    assert summary["account"]["username"] == "test-user"
    assert summary["account"]["live_balance_usdc"] == 123.45
