from pathlib import Path

import pytest

from src.data.storage.sqlite_client import SQLiteClient


@pytest.fixture
def sqlite_client(tmp_path: Path) -> SQLiteClient:
    db_path = tmp_path / "test.db"
    client = SQLiteClient(db_path)
    client.initialize_schema()
    yield client
    client.close()


def test_initialize_schema(tmp_path: Path) -> None:
    db_path = tmp_path / "test.db"
    with SQLiteClient(db_path) as client:
        client.initialize_schema()
        cursor = client.conn.cursor()
        cursor.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        )
        table_names = [row[0] for row in cursor.fetchall()]
        assert "trades" in table_names
        assert "positions" in table_names
        assert "daily_stats" in table_names


def test_insert_trade(sqlite_client: SQLiteClient) -> None:
    trade = {
        "market_id": "market_123",
        "timestamp": "2026-02-12T10:00:00",
        "direction": "BUY_YES",
        "amount_usd": 5.0,
        "num_shares": 8.33,
        "entry_price": 0.6,
        "status": "PENDING",
    }

    trade_id = sqlite_client.insert_trade(trade)
    assert trade_id is not None

    cursor = sqlite_client.conn.cursor()
    cursor.execute("SELECT * FROM trades WHERE id = ?", [trade_id])
    row = cursor.fetchone()
    assert row is not None
    assert dict(row)["market_id"] == "market_123"
    assert dict(row)["status"] == "PENDING"


def test_get_open_positions(sqlite_client: SQLiteClient) -> None:
    from datetime import datetime, timezone

    position1 = {
        "market_id": "market_1",
        "direction": "BUY_YES",
        "num_shares": 10.0,
        "amount_usd": 5.0,
        "avg_entry_price": 0.5,
        "current_price": 0.5,
        "unrealized_pnl": 0.0,
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }

    position2 = {
        "market_id": "market_2",
        "direction": "BUY_NO",
        "num_shares": 0.0,
        "amount_usd": 0.0,
        "avg_entry_price": 0.4,
        "current_price": 0.4,
        "unrealized_pnl": 0.0,
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }

    sqlite_client.upsert_position(position1)
    sqlite_client.upsert_position(position2)

    open_positions = sqlite_client.get_open_positions()
    assert len(open_positions) == 1
    assert open_positions[0]["market_id"] == "market_1"


def test_get_position(sqlite_client: SQLiteClient) -> None:
    from datetime import datetime, timezone

    position = {
        "market_id": "market_123",
        "direction": "BUY_YES",
        "num_shares": 10.0,
        "amount_usd": 5.0,
        "avg_entry_price": 0.5,
        "current_price": 0.5,
        "unrealized_pnl": 0.0,
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }

    sqlite_client.upsert_position(position)

    retrieved = sqlite_client.get_position("market_123")
    assert retrieved is not None
    assert retrieved["num_shares"] == 10.0
    assert retrieved["direction"] == "BUY_YES"


def test_upsert_position(sqlite_client: SQLiteClient) -> None:
    from datetime import datetime, timezone

    position = {
        "market_id": "market_123",
        "direction": "BUY_YES",
        "num_shares": 10.0,
        "amount_usd": 5.0,
        "avg_entry_price": 0.5,
        "current_price": 0.5,
        "unrealized_pnl": 0.0,
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }

    sqlite_client.upsert_position(position)

    updated_position = {
        "market_id": "market_123",
        "direction": "BUY_YES",
        "num_shares": 15.0,
        "amount_usd": 7.8,
        "avg_entry_price": 0.52,
        "current_price": 0.6,
        "unrealized_pnl": 1.2,
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }

    sqlite_client.upsert_position(updated_position)

    retrieved = sqlite_client.get_position("market_123")
    assert retrieved is not None
    assert retrieved["num_shares"] == 15.0
    assert retrieved["unrealized_pnl"] == 1.2


def test_get_daily_stats(sqlite_client: SQLiteClient) -> None:
    stats = {
        "date": "2026-02-12",
        "starting_bankroll": 50.0,
        "ending_bankroll": 52.5,
        "trades_executed": 3,
        "trades_won": 2,
        "gross_pnl": 3.0,
        "fees_paid": 0.5,
        "net_pnl": 2.5,
    }

    sqlite_client.upsert_daily_stats(stats)

    retrieved = sqlite_client.get_daily_stats("2026-02-12")
    assert retrieved is not None
    assert retrieved["ending_bankroll"] == 52.5
    assert retrieved["trades_executed"] == 3


def test_upsert_daily_stats(sqlite_client: SQLiteClient) -> None:
    stats = {
        "date": "2026-02-12",
        "starting_bankroll": 50.0,
        "ending_bankroll": 52.5,
        "trades_executed": 3,
        "trades_won": 2,
        "gross_pnl": 3.0,
        "fees_paid": 0.5,
        "net_pnl": 2.5,
    }

    sqlite_client.upsert_daily_stats(stats)

    updated_stats = {
        "date": "2026-02-12",
        "starting_bankroll": 50.0,
        "ending_bankroll": 55.0,
        "trades_executed": 5,
        "trades_won": 4,
        "gross_pnl": 6.0,
        "fees_paid": 1.0,
        "net_pnl": 5.0,
    }

    sqlite_client.upsert_daily_stats(updated_stats)

    retrieved = sqlite_client.get_daily_stats("2026-02-12")
    assert retrieved is not None
    assert retrieved["ending_bankroll"] == 55.0
    assert retrieved["trades_executed"] == 5


def test_seed_initial_bankroll_empty_db(sqlite_client: SQLiteClient) -> None:
    """seed_initial_bankroll inserts a row when daily_stats is empty."""
    assert sqlite_client.get_current_bankroll() == 0.0
    sqlite_client.seed_initial_bankroll(50.0)
    assert sqlite_client.get_current_bankroll() == 50.0


def test_seed_initial_bankroll_idempotent(sqlite_client: SQLiteClient) -> None:
    """seed_initial_bankroll does nothing when daily_stats already has rows."""
    sqlite_client.seed_initial_bankroll(50.0)
    sqlite_client.upsert_daily_stats({
        "date": "2099-12-31",
        "starting_bankroll": 50.0,
        "ending_bankroll": 55.0,
        "trades_executed": 1,
        "trades_won": 1,
        "gross_pnl": 5.0,
        "fees_paid": 0.0,
        "net_pnl": 5.0,
    })
    # Calling again should NOT overwrite
    sqlite_client.seed_initial_bankroll(50.0)
    assert sqlite_client.get_current_bankroll() == 55.0


def test_get_current_bankroll(sqlite_client: SQLiteClient) -> None:
    stats1 = {
        "date": "2026-02-11",
        "starting_bankroll": 50.0,
        "ending_bankroll": 52.0,
        "trades_executed": 2,
        "trades_won": 1,
        "gross_pnl": 2.5,
        "fees_paid": 0.5,
        "net_pnl": 2.0,
    }

    stats2 = {
        "date": "2026-02-12",
        "starting_bankroll": 52.0,
        "ending_bankroll": 55.0,
        "trades_executed": 3,
        "trades_won": 2,
        "gross_pnl": 3.5,
        "fees_paid": 0.5,
        "net_pnl": 3.0,
    }

    sqlite_client.upsert_daily_stats(stats1)
    sqlite_client.upsert_daily_stats(stats2)

    bankroll = sqlite_client.get_current_bankroll()
    assert bankroll == 55.0
