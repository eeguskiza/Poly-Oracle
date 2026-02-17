from datetime import datetime
from pathlib import Path

import pytest

from src.data.storage.duckdb_client import DuckDBClient


@pytest.fixture
def duckdb_client(tmp_path: Path) -> DuckDBClient:
    db_path = tmp_path / "test.duckdb"
    client = DuckDBClient(db_path)
    client.initialize_schema()
    yield client
    client.close()


def test_initialize_schema(tmp_path: Path) -> None:
    db_path = tmp_path / "test.duckdb"
    with DuckDBClient(db_path) as client:
        client.initialize_schema()
        result = client.conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()
        table_names = [row[0] for row in result]
        assert "forecasts" in table_names
        assert "calibration_data" in table_names


def test_insert_and_get_forecast(duckdb_client: DuckDBClient) -> None:
    forecast = {
        "market_id": "market_123",
        "question": "Will it rain tomorrow?",
        "market_type": "binary",
        "timestamp": datetime.now(),
        "raw_probability": 0.65,
        "calibrated_probability": 0.62,
        "confidence": 0.75,
        "market_price_at_forecast": 0.58,
        "edge": 0.04,
        "recommended_action": "BUY",
        "debate_log": [{"agent": "bull", "statement": "bullish"}],
        "judge_reasoning": "Market underpriced",
    }

    forecast_id = duckdb_client.insert_forecast(forecast)
    assert forecast_id is not None

    retrieved = duckdb_client.get_forecast(forecast_id)
    assert retrieved is not None
    assert retrieved["market_id"] == "market_123"
    assert retrieved["question"] == "Will it rain tomorrow?"
    assert retrieved["raw_probability"] == 0.65
    assert retrieved["resolved"] is False


def test_get_forecasts_by_type(duckdb_client: DuckDBClient) -> None:
    forecast1 = {
        "market_id": "market_1",
        "question": "Question 1",
        "market_type": "binary",
        "timestamp": datetime.now(),
        "raw_probability": 0.6,
        "calibrated_probability": 0.58,
        "confidence": 0.7,
        "market_price_at_forecast": 0.5,
        "edge": 0.08,
        "recommended_action": "BUY",
        "debate_log": [],
        "judge_reasoning": "Good edge",
    }

    forecast2 = {
        "market_id": "market_2",
        "question": "Question 2",
        "market_type": "categorical",
        "timestamp": datetime.now(),
        "raw_probability": 0.4,
        "calibrated_probability": 0.38,
        "confidence": 0.6,
        "market_price_at_forecast": 0.3,
        "edge": 0.08,
        "recommended_action": "BUY",
        "debate_log": [],
        "judge_reasoning": "Good edge",
    }

    duckdb_client.insert_forecast(forecast1)
    duckdb_client.insert_forecast(forecast2)

    binary_forecasts = duckdb_client.get_forecasts_by_type("binary")
    assert len(binary_forecasts) == 1
    assert binary_forecasts[0]["market_type"] == "binary"


def test_get_unresolved_forecasts(duckdb_client: DuckDBClient) -> None:
    forecast = {
        "market_id": "market_123",
        "question": "Test question",
        "market_type": "binary",
        "timestamp": datetime.now(),
        "raw_probability": 0.6,
        "calibrated_probability": 0.58,
        "confidence": 0.7,
        "market_price_at_forecast": 0.5,
        "edge": 0.08,
        "recommended_action": "BUY",
        "debate_log": [],
        "judge_reasoning": "Test",
    }

    forecast_id = duckdb_client.insert_forecast(forecast)

    unresolved = duckdb_client.get_unresolved_forecasts()
    assert len(unresolved) == 1
    assert unresolved[0]["id"] == forecast_id


def test_resolve_forecast(duckdb_client: DuckDBClient) -> None:
    forecast = {
        "market_id": "market_123",
        "question": "Test question",
        "market_type": "binary",
        "timestamp": datetime.now(),
        "raw_probability": 0.6,
        "calibrated_probability": 0.58,
        "confidence": 0.7,
        "market_price_at_forecast": 0.5,
        "edge": 0.08,
        "recommended_action": "BUY",
        "debate_log": [],
        "judge_reasoning": "Test",
    }

    forecast_id = duckdb_client.insert_forecast(forecast)
    duckdb_client.resolve_forecast(forecast_id, outcome=True, brier_score=0.15)

    resolved = duckdb_client.get_forecast(forecast_id)
    assert resolved is not None
    assert resolved["resolved"] is True
    assert resolved["outcome"] == 1
    assert resolved["brier_score"] == 0.15


def test_get_calibration_stats(duckdb_client: DuckDBClient) -> None:
    forecast1 = {
        "market_id": "market_1",
        "question": "Question 1",
        "market_type": "binary",
        "timestamp": datetime.now(),
        "raw_probability": 0.6,
        "calibrated_probability": 0.58,
        "confidence": 0.7,
        "market_price_at_forecast": 0.5,
        "edge": 0.08,
        "recommended_action": "BUY",
        "debate_log": [],
        "judge_reasoning": "Test",
    }

    forecast2 = {
        "market_id": "market_2",
        "question": "Question 2",
        "market_type": "categorical",
        "timestamp": datetime.now(),
        "raw_probability": 0.4,
        "calibrated_probability": 0.38,
        "confidence": 0.6,
        "market_price_at_forecast": 0.3,
        "edge": 0.08,
        "recommended_action": "BUY",
        "debate_log": [],
        "judge_reasoning": "Test",
    }

    id1 = duckdb_client.insert_forecast(forecast1)
    id2 = duckdb_client.insert_forecast(forecast2)

    duckdb_client.resolve_forecast(id1, outcome=True, brier_score=0.1)
    duckdb_client.resolve_forecast(id2, outcome=False, brier_score=0.2)

    stats = duckdb_client.get_calibration_stats()
    assert stats["overall"]["count"] == 2
    assert pytest.approx(stats["overall"]["avg_brier_score"], abs=1e-9) == 0.15
    assert "binary" in stats["by_type"]
    assert "categorical" in stats["by_type"]
