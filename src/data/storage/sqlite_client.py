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
                forecast_id TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                direction TEXT NOT NULL,
                amount_usd REAL NOT NULL,
                price REAL NOT NULL,
                shares REAL NOT NULL,
                order_type TEXT NOT NULL,
                status TEXT NOT NULL,
                fill_price REAL,
                fill_amount REAL,
                fees REAL,
                tx_hash TEXT
            )
        """)

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS positions (
                market_id TEXT PRIMARY KEY,
                direction TEXT NOT NULL,
                shares REAL NOT NULL,
                avg_entry_price REAL NOT NULL,
                total_cost REAL NOT NULL,
                realized_pnl REAL NOT NULL
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
                id, market_id, forecast_id, timestamp, direction,
                amount_usd, price, shares, order_type, status
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, [
            trade_id,
            trade["market_id"],
            trade["forecast_id"],
            trade["timestamp"],
            trade["direction"],
            trade["amount_usd"],
            trade["price"],
            trade["shares"],
            trade["order_type"],
            trade["status"],
        ])

        self.conn.commit()
        logger.debug(f"Inserted trade {trade_id}")
        return trade_id

    def update_trade_status(
        self,
        trade_id: str,
        status: str,
        fill_price: float,
        fill_amount: float,
        fees: float
    ) -> None:
        cursor = self.conn.cursor()

        cursor.execute("""
            UPDATE trades
            SET status = ?, fill_price = ?, fill_amount = ?, fees = ?
            WHERE id = ?
        """, [status, fill_price, fill_amount, fees, trade_id])

        self.conn.commit()
        logger.info(f"Updated trade {trade_id} to status {status}")

    def get_open_positions(self) -> list[dict[str, Any]]:
        cursor = self.conn.cursor()
        cursor.execute("SELECT * FROM positions WHERE shares > 0")
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
                market_id, direction, shares, avg_entry_price, total_cost, realized_pnl
            ) VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(market_id) DO UPDATE SET
                direction = excluded.direction,
                shares = excluded.shares,
                avg_entry_price = excluded.avg_entry_price,
                total_cost = excluded.total_cost,
                realized_pnl = excluded.realized_pnl
        """, [
            position["market_id"],
            position["direction"],
            position["shares"],
            position["avg_entry_price"],
            position["total_cost"],
            position["realized_pnl"],
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
