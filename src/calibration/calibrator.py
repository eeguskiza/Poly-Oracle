"""
Calibrator Agent - Calibrates raw forecasts based on historical performance.
"""
from typing import Callable
import numpy as np
from sklearn.isotonic import IsotonicRegression
from loguru import logger

from src.data.storage.duckdb_client import DuckDBClient
from src.models import CalibratedForecast


class CalibratorAgent:
    """
    Calibrates raw probability forecasts using historical performance data.

    Uses isotonic regression to build calibration curves when sufficient
    historical data is available. For small sample sizes, returns identity
    calibration (raw = calibrated).
    """

    MIN_SAMPLES_FOR_CALIBRATION = 50

    def __init__(self, history_db: DuckDBClient):
        """
        Initialize calibrator with historical forecast database.

        Args:
            history_db: DuckDB client with forecast history
        """
        self.history_db = history_db
        self._calibration_curves: dict[str, Callable] = {}
        self._last_calibration_sample_count: dict[str, int] = {}

    def calibrate(
        self,
        raw_forecast: float,
        market_type: str,
        confidence: float,
    ) -> CalibratedForecast:
        """
        Calibrate a raw probability forecast.

        Args:
            raw_forecast: Raw probability from debate (0-1)
            market_type: Type of market (e.g., "binary", "categorical")
            confidence: Confidence score (0-1)

        Returns:
            CalibratedForecast with raw, calibrated, method, and sample count
        """
        # Validate inputs
        if not 0.0 <= raw_forecast <= 1.0:
            raise ValueError(f"raw_forecast must be in [0, 1], got {raw_forecast}")
        if not 0.0 <= confidence <= 1.0:
            raise ValueError(f"confidence must be in [0, 1], got {confidence}")

        # Get historical forecasts for this market type
        historical = self._get_historical_forecasts(market_type)
        num_samples = len(historical)

        logger.debug(
            f"Calibrating forecast for {market_type}: "
            f"{num_samples} historical samples available"
        )

        # If insufficient data, use identity calibration
        if num_samples < self.MIN_SAMPLES_FOR_CALIBRATION:
            logger.info(
                f"Insufficient data for calibration ({num_samples} < "
                f"{self.MIN_SAMPLES_FOR_CALIBRATION}). Using identity calibration."
            )
            calibrated = raw_forecast
            method = "identity"
        else:
            # Build or use cached calibration curve
            if market_type not in self._calibration_curves or \
               self._last_calibration_sample_count.get(market_type, 0) != num_samples:
                logger.info(f"Building calibration curve for {market_type}")
                self._calibration_curves[market_type] = \
                    self._build_calibration_curve(historical)
                self._last_calibration_sample_count[market_type] = num_samples

            # Apply calibration curve
            calibration_fn = self._calibration_curves[market_type]
            calibrated = float(calibration_fn(np.array([raw_forecast]))[0])
            method = "isotonic_regression"

            logger.debug(f"Applied calibration: {raw_forecast:.3f} -> {calibrated:.3f}")

        # Apply shrinkage for extreme predictions
        calibrated = self._shrink_extremes(calibrated, confidence)

        return CalibratedForecast(
            raw=raw_forecast,
            calibrated=calibrated,
            confidence=confidence,
            calibration_method=method,
            historical_samples=num_samples,
        )

    def _get_historical_forecasts(self, market_type: str) -> list[dict]:
        """
        Get resolved forecasts from history for a specific market type.

        Args:
            market_type: Type of market to filter by

        Returns:
            List of dicts with 'prediction' and 'outcome' keys
        """
        query = """
            SELECT
                raw_probability as prediction,
                CAST(outcome AS INTEGER) as outcome
            FROM forecasts
            WHERE market_type = ?
                AND outcome IS NOT NULL
                AND raw_probability IS NOT NULL
            ORDER BY timestamp DESC
        """

        try:
            result = self.history_db.conn.execute(query, [market_type]).fetchall()
            return [
                {"prediction": row[0], "outcome": row[1]}
                for row in result
            ]
        except Exception as e:
            logger.warning(f"Error fetching historical forecasts: {e}")
            return []

    def _build_calibration_curve(self, historical: list[dict]) -> Callable:
        """
        Build a calibration curve using isotonic regression.

        Groups predictions into buckets, calculates empirical frequencies,
        and fits an isotonic regression model for monotonicity.

        Args:
            historical: List of dicts with 'prediction' and 'outcome' keys

        Returns:
            Callable that takes array of predictions and returns calibrated values
        """
        if len(historical) < self.MIN_SAMPLES_FOR_CALIBRATION:
            # Return identity function
            return lambda x: x

        # Extract predictions and outcomes
        predictions = np.array([h["prediction"] for h in historical])
        outcomes = np.array([h["outcome"] for h in historical])

        # Fit isotonic regression
        # y_min/y_max bounds to avoid extreme calibrations
        iso_reg = IsotonicRegression(
            y_min=0.01,
            y_max=0.99,
            out_of_bounds="clip",
        )
        iso_reg.fit(predictions, outcomes)

        return lambda x: iso_reg.predict(x)

    def _shrink_extremes(self, probability: float, confidence: float) -> float:
        """
        Shrink extreme predictions towards 0.5 based on confidence.

        Args:
            probability: Calibrated probability
            confidence: Confidence score (0-1)

        Returns:
            Shrunk probability
        """
        # Shrink factor increases as confidence decreases
        shrink_factor = 0.1 * (1 - confidence)

        # Only shrink if prediction is extreme (>0.9 or <0.1)
        if probability > 0.9:
            shrinkage = (probability - 0.5) * shrink_factor
            return probability - shrinkage
        elif probability < 0.1:
            shrinkage = (0.5 - probability) * shrink_factor
            return probability + shrinkage
        else:
            return probability

    def get_calibration_report(self) -> dict:
        """
        Generate calibration performance report.

        Returns:
            Dict with Brier scores, calibration curve data, and sample counts
        """
        # Get all resolved forecasts
        query = """
            SELECT
                market_type,
                raw_probability,
                calibrated_probability,
                CAST(outcome AS INTEGER) as outcome
            FROM forecasts
            WHERE outcome IS NOT NULL
        """

        try:
            result = self.history_db.conn.execute(query).fetchall()
        except Exception as e:
            logger.error(f"Error generating calibration report: {e}")
            return {
                "error": str(e),
                "total_forecasts": 0,
                "resolved_forecasts": 0,
            }

        if not result:
            return {
                "total_forecasts": 0,
                "resolved_forecasts": 0,
                "brier_score_raw": None,
                "brier_score_calibrated": None,
                "brier_by_type": {},
                "calibration_curve": [],
            }

        # Calculate overall Brier scores
        raw_preds = []
        cal_preds = []
        outcomes = []
        by_type = {}

        for row in result:
            market_type, raw_prob, cal_prob, outcome = row
            raw_preds.append(raw_prob)
            cal_preds.append(cal_prob if cal_prob is not None else raw_prob)
            outcomes.append(outcome)

            if market_type not in by_type:
                by_type[market_type] = {"raw": [], "cal": [], "outcomes": []}
            by_type[market_type]["raw"].append(raw_prob)
            by_type[market_type]["cal"].append(
                cal_prob if cal_prob is not None else raw_prob
            )
            by_type[market_type]["outcomes"].append(outcome)

        # Calculate Brier scores
        brier_raw = self._calculate_brier_score(raw_preds, outcomes)
        brier_cal = self._calculate_brier_score(cal_preds, outcomes)

        # Brier by type
        brier_by_type = {}
        for mtype, data in by_type.items():
            brier_by_type[mtype] = {
                "raw": self._calculate_brier_score(data["raw"], data["outcomes"]),
                "calibrated": self._calculate_brier_score(data["cal"], data["outcomes"]),
                "count": len(data["outcomes"]),
            }

        # Build calibration curve data (bucketed)
        calibration_curve = self._build_calibration_curve_data(cal_preds, outcomes)

        # Get total forecast count
        total_count_query = "SELECT COUNT(*) FROM forecasts"
        total_count = self.history_db.conn.execute(total_count_query).fetchone()[0]

        return {
            "total_forecasts": total_count,
            "resolved_forecasts": len(result),
            "brier_score_raw": brier_raw,
            "brier_score_calibrated": brier_cal,
            "improvement": brier_raw - brier_cal if brier_raw and brier_cal else None,
            "brier_by_type": brier_by_type,
            "calibration_curve": calibration_curve,
        }

    def _calculate_brier_score(
        self,
        predictions: list[float],
        outcomes: list[int],
    ) -> float:
        """
        Calculate Brier score.

        Args:
            predictions: List of probability predictions
            outcomes: List of binary outcomes (0 or 1)

        Returns:
            Brier score (lower is better, 0 = perfect)
        """
        if not predictions or len(predictions) != len(outcomes):
            return None

        predictions = np.array(predictions)
        outcomes = np.array(outcomes)

        return float(np.mean((predictions - outcomes) ** 2))

    def _build_calibration_curve_data(
        self,
        predictions: list[float],
        outcomes: list[int],
        num_buckets: int = 10,
    ) -> list[dict]:
        """
        Build calibration curve data by bucketing predictions.

        Args:
            predictions: List of predictions
            outcomes: List of outcomes
            num_buckets: Number of buckets to group predictions into

        Returns:
            List of dicts with predicted_prob, actual_freq, count
        """
        predictions = np.array(predictions)
        outcomes = np.array(outcomes)

        # Create buckets
        buckets = np.linspace(0, 1, num_buckets + 1)
        curve_data = []

        for i in range(num_buckets):
            lower = buckets[i]
            upper = buckets[i + 1]

            # Find predictions in this bucket
            mask = (predictions >= lower) & (predictions < upper)
            if i == num_buckets - 1:  # Include upper bound in last bucket
                mask = (predictions >= lower) & (predictions <= upper)

            bucket_preds = predictions[mask]
            bucket_outcomes = outcomes[mask]

            if len(bucket_outcomes) > 0:
                curve_data.append({
                    "predicted_prob": float(bucket_preds.mean()),
                    "actual_freq": float(bucket_outcomes.mean()),
                    "count": len(bucket_outcomes),
                    "bucket_range": f"{lower:.1f}-{upper:.1f}",
                })

        return curve_data
