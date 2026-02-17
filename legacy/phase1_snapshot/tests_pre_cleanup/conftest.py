"""Shared test fixtures for Poly-Oracle v1."""

import os
import pytest
from pathlib import Path
from unittest.mock import patch

# Ensure tests never hit real APIs or real .env
os.environ.setdefault("BTC_MARKET_ID", "test-market-id-000")
os.environ.setdefault("PAPER_TRADING", "true")
os.environ.setdefault("INITIAL_BANKROLL", "10.0")
os.environ.setdefault("MIN_BET", "0.10")
os.environ.setdefault("MAX_BET", "1.0")
os.environ.setdefault("OLLAMA_MODEL", "mistral")
os.environ.setdefault("OLLAMA_BASE_URL", "http://localhost:11434")
os.environ.setdefault("LOG_LEVEL", "WARNING")
os.environ.setdefault("DB_DIR", "/tmp/poly-oracle-test-db")
os.environ.setdefault("LOOP_INTERVAL_MINUTES", "1")
os.environ.setdefault("DEBATE_ROUNDS", "1")


@pytest.fixture
def settings():
    """Fresh Settings instance for each test (bypasses lru_cache)."""
    from bot.config.settings import Settings
    return Settings()


@pytest.fixture
def tmp_db(tmp_path):
    """Temporary SQLite database for testing."""
    from bot.state.store import SQLiteClient

    db_path = tmp_path / "test.db"
    client = SQLiteClient(db_path)
    client.initialize_schema()
    client.seed_initial_bankroll(10.0)
    yield client
    client.close()
