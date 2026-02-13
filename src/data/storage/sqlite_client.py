import sqlite3
from pathlib import Path
from typing import Any
from uuid import uuid4

from loguru import logger


class SQLiteClient:
    def __init__(self, db_path: Path) -> None:
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(db_path))
        self.conn.row_factory = sqlite3.Row
        logger.info(f"Connected to SQLite at {db_path}")

    def initialize_schema(self) -> None:
        cursor = self.conn.cursor()

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS trades (
                id TEXT PRIMARY KEY,
                market_id TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                direction TEXT NOT NULL,
                amount_usd REAL NOT NULL,
                num_shares REAL NOT NULL,
                entry_price REAL NOT NULL,
                status TEXT NOT NULL
            )
        """)

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS positions (
                market_id TEXT PRIMARY KEY,
                direction TEXT NOT NULL,
                num_shares REAL NOT NULL,
                amount_usd REAL NOT NULL,
                avg_entry_price REAL NOT NULL,
                current_price REAL NOT NULL,
                unrealized_pnl REAL NOT NULL,
                updated_at TEXT NOT NULL
            )
        """)

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS daily_stats (
                date TEXT PRIMARY KEY,
                starting_bankroll REAL NOT NULL,
                ending_bankroll REAL NOT NULL,
                trades_executed INTEGER NOT NULL,
                trades_won INTEGER NOT NULL,
                gross_pnl REAL NOT NULL,
                fees_paid REAL NOT NULL,
                net_pnl REAL NOT NULL
            )
        """)

        self.conn.commit()
        logger.info("SQLite schema initialized")

    def insert_trade(self, trade: dict[str, Any]) -> str:
        trade_id = str(uuid4())
        cursor = self.conn.cursor()

        cursor.execute("""
            INSERT INTO trades (
                id, market_id, timestamp, direction,
                amount_usd, num_shares, entry_price, status
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, [
            trade_id,
            trade["market_id"],
            trade["timestamp"],
            trade["direction"],
            trade["amount_usd"],
            trade["num_shares"],
            trade["entry_price"],
            trade["status"],
        ])

        self.conn.commit()
        logger.debug(f"Inserted trade {trade_id}")
        return trade_id

    def get_open_positions(self) -> list[dict[str, Any]]:
        cursor = self.conn.cursor()
        cursor.execute("SELECT * FROM positions WHERE num_shares > 0")
        rows = cursor.fetchall()
        return [dict(row) for row in rows]

    def get_position(self, market_id: str) -> dict[str, Any] | None:
        cursor = self.conn.cursor()
        cursor.execute("SELECT * FROM positions WHERE market_id = ?", [market_id])
        row = cursor.fetchone()
        return dict(row) if row else None

    def upsert_position(self, position: dict[str, Any]) -> None:
        cursor = self.conn.cursor()

        cursor.execute("""
            INSERT INTO positions (
                market_id, direction, num_shares, amount_usd, avg_entry_price,
                current_price, unrealized_pnl, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(market_id) DO UPDATE SET
                direction = excluded.direction,
                num_shares = excluded.num_shares,
                amount_usd = excluded.amount_usd,
                avg_entry_price = excluded.avg_entry_price,
                current_price = excluded.current_price,
                unrealized_pnl = excluded.unrealized_pnl,
                updated_at = excluded.updated_at
        """, [
            position["market_id"],
            position["direction"],
            position["num_shares"],
            position["amount_usd"],
            position["avg_entry_price"],
            position["current_price"],
            position["unrealized_pnl"],
            position["updated_at"],
        ])

        self.conn.commit()
        logger.debug(f"Upserted position for market {position['market_id']}")

    def get_daily_stats(self, date: str) -> dict[str, Any] | None:
        cursor = self.conn.cursor()
        cursor.execute("SELECT * FROM daily_stats WHERE date = ?", [date])
        row = cursor.fetchone()
        return dict(row) if row else None

    def upsert_daily_stats(self, stats: dict[str, Any]) -> None:
        cursor = self.conn.cursor()

        cursor.execute("""
            INSERT INTO daily_stats (
                date, starting_bankroll, ending_bankroll, trades_executed,
                trades_won, gross_pnl, fees_paid, net_pnl
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(date) DO UPDATE SET
                starting_bankroll = excluded.starting_bankroll,
                ending_bankroll = excluded.ending_bankroll,
                trades_executed = excluded.trades_executed,
                trades_won = excluded.trades_won,
                gross_pnl = excluded.gross_pnl,
                fees_paid = excluded.fees_paid,
                net_pnl = excluded.net_pnl
        """, [
            stats["date"],
            stats["starting_bankroll"],
            stats["ending_bankroll"],
            stats["trades_executed"],
            stats["trades_won"],
            stats["gross_pnl"],
            stats["fees_paid"],
            stats["net_pnl"],
        ])

        self.conn.commit()
        logger.debug(f"Upserted daily stats for {stats['date']}")

    def get_current_bankroll(self) -> float:
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT ending_bankroll
            FROM daily_stats
            ORDER BY date DESC
            LIMIT 1
        """)
        row = cursor.fetchone()
        return row[0] if row else 0.0

    def close(self) -> None:
        self.conn.close()
        logger.info("Closed SQLite connection")

    def __enter__(self) -> "SQLiteClient":
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.close()
