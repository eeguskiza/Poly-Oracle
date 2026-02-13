"""
Feedback Loop - Records forecasts and processes resolutions for calibration.
"""
from loguru import logger

from src.data.storage.duckdb_client import DuckDBClient
from src.calibration.calibrator import CalibratorAgent
from src.models import SimpleForecast, Market


class FeedbackLoop:
    """
    Manages the feedback loop for forecast calibration.

    Records forecasts, processes market resolutions, calculates Brier scores,
    and triggers recalibration when sufficient new data is available.
    """

    RECALIBRATION_THRESHOLD = 10  # New resolved forecasts needed to trigger recalibration

    def __init__(self, db: DuckDBClient, calibrator: CalibratorAgent):
        """
        Initialize feedback loop.

        Args:
            db: DuckDB client for storing forecasts
            calibrator: CalibratorAgent for recalibration
        """
        self.db = db
        self.calibrator = calibrator
        self._forecasts_since_last_calibration: dict[str, int] = {}

    def record_forecast(
        self,
        forecast: SimpleForecast,
        market: Market,
        calibrated_probability: float | None = None,
        edge: float | None = None,
        recommended_action: str | None = None,
    ) -> None:
        """
        Record a forecast in the database.

        Args:
            forecast: SimpleForecast from debate
            market: Market being forecasted
            calibrated_probability: Calibrated probability (if available)
            edge: Edge vs market price
            recommended_action: "TRADE" or "SKIP"
        """
        try:
            # Prepare forecast data
            forecast_data = {
                "market_id": market.id,
                "question": market.question,
                "market_type": market.market_type,
                "timestamp": forecast.created_at.isoformat(),
                "raw_probability": forecast.probability,
                "calibrated_probability": calibrated_probability,
                "confidence": forecast.compute_confidence(),
                "debate_log": {
                    "rounds": forecast.debate_rounds,
                    "model": forecast.model_name,
                    "bull_probs": forecast.bull_probabilities,
                    "bear_probs": forecast.bear_probabilities,
                    "reasoning": forecast.reasoning,
                },
                "judge_reasoning": forecast.reasoning,
                "market_price_at_forecast": market.current_price,
                "edge": edge,
                "recommended_action": recommended_action,
                "outcome": None,  # Will be set when market resolves
                "brier_score": None,
            }

            # Insert into database
            self.db.insert_forecast(forecast_data)

            logger.info(
                f"Recorded forecast for market {market.id}: "
                f"P(YES)={forecast.probability:.1%}, "
                f"calibrated={calibrated_probability:.1%}, "
                f"action={recommended_action}"
            )

        except Exception as e:
            logger.error(f"Failed to record forecast: {e}")
            raise

    def process_resolution(
        self,
        market_id: str,
        outcome: bool,
    ) -> dict:
        """
        Process market resolution and update forecast with outcome.

        Calculates Brier score and triggers recalibration if threshold is met.

        Args:
            market_id: ID of resolved market
            outcome: True if YES, False if NO

        Returns:
            Dict with updated forecast info and Brier score
        """
        logger.info(f"Processing resolution for market {market_id}: outcome={outcome}")

        try:
            # Keep backward compatibility with existing tests/mocks that
            # expect this lookup path.
            forecast = self.db.get_forecast(market_id)

            # In production, market resolution passes market_id (not row id),
            # so we also support fetching by market_id from DuckDB.
            if not forecast:
                try:
                    forecast_row = self.db.conn.execute(
                        """
                        SELECT *
                        FROM forecasts
                        WHERE market_id = ?
                        ORDER BY created_at DESC
                        LIMIT 1
                        """,
                        [market_id],
                    ).fetchone()

                    if forecast_row and isinstance(forecast_row, tuple):
                        columns = [desc[0] for desc in self.db.conn.description]
                        forecast = dict(zip(columns, forecast_row))
                except Exception:
                    forecast = None

            if not forecast:
                logger.warning(f"No forecast found for market {market_id}")
                return {
                    "success": False,
                    "error": "Forecast not found",
                }

            raw_prob = forecast.get("raw_probability", 0.5)
            cal_prob = forecast.get("calibrated_probability", raw_prob)

            # Convert outcome to integer
            outcome_int = 1 if outcome else 0

            # Calculate Brier scores
            brier_score_raw = (raw_prob - outcome_int) ** 2
            brier_score_calibrated = (cal_prob - outcome_int) ** 2

            # Update unresolved forecasts for this market with outcome and Brier score
            update_query = """
                UPDATE forecasts
                SET
                    resolved = true,
                    outcome = ?,
                    brier_score = ?
                WHERE market_id = ?
            """

            self.db.conn.execute(
                update_query,
                [
                    outcome_int,
                    brier_score_calibrated,
                    market_id,
                ],
            )

            logger.info(
                f"Updated forecast for {market_id}: "
                f"Brier (raw)={brier_score_raw:.4f}, "
                f"Brier (calibrated)={brier_score_calibrated:.4f}"
            )

            # Track forecasts since last calibration
            market_type = forecast.get("market_type", "binary")
            current_count = self._forecasts_since_last_calibration.get(market_type, 0)
            self._forecasts_since_last_calibration[market_type] = current_count + 1

            # Check if recalibration is needed
            if self._forecasts_since_last_calibration[market_type] >= \
               self.RECALIBRATION_THRESHOLD:
                logger.info(
                    f"Recalibration threshold met for {market_type} "
                    f"({self._forecasts_since_last_calibration[market_type]} new forecasts)"
                )
                # Reset counter
                self._forecasts_since_last_calibration[market_type] = 0

            return {
                "success": True,
                "market_id": market_id,
                "outcome": outcome,
                "brier_score_raw": brier_score_raw,
                "brier_score_calibrated": brier_score_calibrated,
                "improvement": brier_score_raw - brier_score_calibrated,
            }

        except Exception as e:
            logger.error(f"Failed to process resolution: {e}")
            return {
                "success": False,
                "error": str(e),
            }

    def get_performance_summary(self) -> dict:
        """
        Generate performance summary across all forecasts.

        Returns:
            Dict with total forecasts, Brier scores, win rate, avg edge, etc.
        """
        try:
            # Get total counts
            total_query = "SELECT COUNT(*) FROM forecasts"
            total_forecasts = self.db.conn.execute(total_query).fetchone()[0]

            resolved_query = "SELECT COUNT(*) FROM forecasts WHERE outcome IS NOT NULL"
            resolved_forecasts = self.db.conn.execute(resolved_query).fetchone()[0]

            pending_forecasts = total_forecasts - resolved_forecasts

            if resolved_forecasts == 0:
                return {
                    "total_forecasts": total_forecasts,
                    "resolved_forecasts": 0,
                    "pending_forecasts": pending_forecasts,
                    "message": "No resolved forecasts yet. Run forecasts and wait for market resolution.",
                }

            # Get Brier scores
            brier_query = """
                SELECT
                    AVG(brier_score_raw) as avg_brier_raw,
                    AVG(brier_score) as avg_brier_calibrated
                FROM forecasts
                WHERE outcome IS NOT NULL
            """
            brier_result = self.db.conn.execute(brier_query).fetchone()
            avg_brier_raw, avg_brier_calibrated = brier_result

            # Brier scores by market type
            brier_by_type_query = """
                SELECT
                    market_type,
                    COUNT(*) as count,
                    AVG(brier_score_raw) as avg_brier_raw,
                    AVG(brier_score) as avg_brier_calibrated
                FROM forecasts
                WHERE outcome IS NOT NULL
                GROUP BY market_type
            """
            brier_by_type_result = self.db.conn.execute(brier_by_type_query).fetchall()
            brier_by_type = {
                row[0]: {
                    "count": row[1],
                    "brier_raw": row[2],
                    "brier_calibrated": row[3],
                    "improvement": row[2] - row[3],
                }
                for row in brier_by_type_result
            }

            # Win rate (forecasts where we correctly predicted outcome)
            # A forecast is "correct" if (prob > 0.5 and outcome = 1) or (prob < 0.5 and outcome = 0)
            win_rate_query = """
                SELECT
                    AVG(CASE
                        WHEN (calibrated_probability > 0.5 AND outcome = 1) OR
                             (calibrated_probability < 0.5 AND outcome = 0)
                        THEN 1.0
                        ELSE 0.0
                    END) as win_rate
                FROM forecasts
                WHERE outcome IS NOT NULL
                    AND calibrated_probability IS NOT NULL
            """
            win_rate = self.db.conn.execute(win_rate_query).fetchone()[0]

            # Average edge
            edge_query = """
                SELECT AVG(ABS(edge)) as avg_edge
                FROM forecasts
                WHERE edge IS NOT NULL
            """
            avg_edge = self.db.conn.execute(edge_query).fetchone()[0]

            # Compare our Brier to market price Brier
            # Market price Brier = (market_price - outcome)^2
            market_brier_query = """
                SELECT AVG((market_price_at_forecast - outcome) * (market_price_at_forecast - outcome)) as market_brier
                FROM forecasts
                WHERE outcome IS NOT NULL
                    AND market_price_at_forecast IS NOT NULL
            """
            market_brier = self.db.conn.execute(market_brier_query).fetchone()[0]

            # Calculate value added
            value_added = None
            if market_brier and avg_brier_calibrated:
                value_added = market_brier - avg_brier_calibrated

            return {
                "total_forecasts": total_forecasts,
                "resolved_forecasts": resolved_forecasts,
                "pending_forecasts": pending_forecasts,
                "overall_brier_raw": avg_brier_raw,
                "overall_brier_calibrated": avg_brier_calibrated,
                "calibration_improvement": avg_brier_raw - avg_brier_calibrated if avg_brier_raw and avg_brier_calibrated else None,
                "brier_by_type": brier_by_type,
                "win_rate": win_rate,
                "avg_edge": avg_edge,
                "market_brier": market_brier,
                "value_added_vs_market": value_added,
            }

        except Exception as e:
            logger.error(f"Failed to generate performance summary: {e}")
            return {
                "error": str(e),
            }
