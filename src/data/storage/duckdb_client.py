import json
from pathlib import Path
from typing import Any
from uuid import uuid4

import duckdb
from loguru import logger


class DuckDBClient:
    def __init__(self, db_path: Path) -> None:
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = duckdb.connect(str(db_path))
        logger.info(f"Connected to DuckDB at {db_path}")

    def initialize_schema(self) -> None:
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS forecasts (
                id VARCHAR PRIMARY KEY,
                market_id VARCHAR NOT NULL,
                question VARCHAR NOT NULL,
                market_type VARCHAR NOT NULL,
                timestamp TIMESTAMP NOT NULL,
                raw_probability DOUBLE NOT NULL,
                calibrated_probability DOUBLE NOT NULL,
                confidence DOUBLE NOT NULL,
                market_price_at_forecast DOUBLE NOT NULL,
                edge DOUBLE NOT NULL,
                recommended_action VARCHAR NOT NULL,
                debate_log VARCHAR NOT NULL,
                judge_reasoning VARCHAR NOT NULL,
                resolved BOOLEAN DEFAULT false,
                outcome INTEGER,
                brier_score DOUBLE,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS calibration_data (
                id VARCHAR PRIMARY KEY,
                market_type VARCHAR NOT NULL,
                prediction_bucket DOUBLE NOT NULL,
                count INTEGER NOT NULL,
                positive_outcomes INTEGER NOT NULL,
                empirical_probability DOUBLE NOT NULL,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        logger.info("DuckDB schema initialized")

    def insert_forecast(self, forecast: dict[str, Any]) -> str:
        forecast_id = str(uuid4())

        debate_log_json = json.dumps(forecast.get("debate_log", []))

        self.conn.execute("""
            INSERT INTO forecasts (
                id, market_id, question, market_type, timestamp,
                raw_probability, calibrated_probability, confidence,
                market_price_at_forecast, edge, recommended_action,
                debate_log, judge_reasoning
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, [
            forecast_id,
            forecast["market_id"],
            forecast["question"],
            forecast["market_type"],
            forecast["timestamp"],
            forecast["raw_probability"],
            forecast["calibrated_probability"],
            forecast["confidence"],
            forecast["market_price_at_forecast"],
            forecast["edge"],
            forecast["recommended_action"],
            debate_log_json,
            forecast["judge_reasoning"],
        ])

        logger.debug(f"Inserted forecast {forecast_id}")
        return forecast_id

    def get_forecast(self, forecast_id: str) -> dict[str, Any] | None:
        result = self.conn.execute(
            "SELECT * FROM forecasts WHERE id = ?", [forecast_id]
        ).fetchone()

        if not result:
            return None

        columns = [desc[0] for desc in self.conn.description]
        forecast = dict(zip(columns, result))

        if forecast.get("debate_log"):
            forecast["debate_log"] = json.loads(forecast["debate_log"])

        return forecast

    def get_forecasts_by_type(self, market_type: str) -> list[dict[str, Any]]:
        results = self.conn.execute(
            "SELECT * FROM forecasts WHERE market_type = ? ORDER BY created_at DESC",
            [market_type]
        ).fetchall()

        columns = [desc[0] for desc in self.conn.description]
        forecasts = []

        for row in results:
            forecast = dict(zip(columns, row))
            if forecast.get("debate_log"):
                forecast["debate_log"] = json.loads(forecast["debate_log"])
            forecasts.append(forecast)

        return forecasts

    def get_unresolved_forecasts(self) -> list[dict[str, Any]]:
        results = self.conn.execute(
            "SELECT * FROM forecasts WHERE resolved = false ORDER BY created_at DESC"
        ).fetchall()

        columns = [desc[0] for desc in self.conn.description]
        forecasts = []

        for row in results:
            forecast = dict(zip(columns, row))
            if forecast.get("debate_log"):
                forecast["debate_log"] = json.loads(forecast["debate_log"])
            forecasts.append(forecast)

        return forecasts

    def resolve_forecast(
        self, forecast_id: str, outcome: bool, brier_score: float
    ) -> None:
        self.conn.execute("""
            UPDATE forecasts
            SET resolved = true, outcome = ?, brier_score = ?
            WHERE id = ?
        """, [int(outcome), brier_score, forecast_id])

        logger.info(f"Resolved forecast {forecast_id}: outcome={outcome}, brier={brier_score:.4f}")

    def get_calibration_stats(self) -> dict[str, Any]:
        overall = self.conn.execute("""
            SELECT
                COUNT(*) as count,
                AVG(brier_score) as avg_brier_score
            FROM forecasts
            WHERE resolved = true AND brier_score IS NOT NULL
        """).fetchone()

        by_type = self.conn.execute("""
            SELECT
                market_type,
                COUNT(*) as count,
                AVG(brier_score) as avg_brier_score
            FROM forecasts
            WHERE resolved = true AND brier_score IS NOT NULL
            GROUP BY market_type
        """).fetchall()

        stats = {
            "overall": {
                "count": overall[0] if overall else 0,
                "avg_brier_score": overall[1] if overall and overall[1] else 0.0,
            },
            "by_type": {}
        }

        for row in by_type:
            stats["by_type"][row[0]] = {
                "count": row[1],
                "avg_brier_score": row[2],
            }

        return stats

    def close(self) -> None:
        self.conn.close()
        logger.info("Closed DuckDB connection")

    def __enter__(self) -> "DuckDBClient":
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.close()
