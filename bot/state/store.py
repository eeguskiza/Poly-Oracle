"""SQLite state store bootstrap for startup lifecycle."""

from __future__ import annotations

import sqlite3
from pathlib import Path

from bot.errors import DependencyInitializationError


class SQLiteStateStore:
    """Minimal sqlite state store for startup events and loop metadata."""

    def __init__(self, db_path: Path) -> None:
        self.db_path = db_path
        self._conn: sqlite3.Connection | None = None

    def connect(self) -> None:
        try:
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
            self._conn = sqlite3.connect(str(self.db_path))
        except sqlite3.Error as exc:
            raise DependencyInitializationError(
                f"Unable to connect sqlite state store at '{self.db_path}': {exc}"
            ) from exc

    def initialize_schema(self) -> None:
        conn = self._require_conn()
        try:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS startup_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    event_name TEXT NOT NULL,
                    details TEXT,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
                """
            )
            conn.commit()
        except sqlite3.Error as exc:
            raise DependencyInitializationError(
                f"Unable to initialize sqlite schema at '{self.db_path}': {exc}"
            ) from exc

    def record_event(self, event_name: str, details: str) -> None:
        conn = self._require_conn()
        conn.execute(
            "INSERT INTO startup_events (event_name, details) VALUES (?, ?)",
            (event_name, details),
        )
        conn.commit()

    def close(self) -> None:
        if self._conn is not None:
            self._conn.close()
            self._conn = None

    def _require_conn(self) -> sqlite3.Connection:
        if self._conn is None:
            raise DependencyInitializationError(
                "SQLite state store used before connection was initialized."
            )
        return self._conn
