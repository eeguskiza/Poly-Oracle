from datetime import datetime, timezone
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator


class DebateRound(BaseModel):
    model_config = {"from_attributes": True}

    round_number: int
    bull_argument: str
    bear_argument: str
    devil_critique: str


class RawForecast(BaseModel):
    model_config = {"from_attributes": True}

    probability: float = Field(ge=0.0, le=1.0)
    confidence: float = Field(ge=0.0, le=1.0)
    reasoning: str

    @field_validator("probability", "confidence")
    @classmethod
    def validate_probability_range(cls, v: float) -> float:
        if not 0.0 <= v <= 1.0:
            raise ValueError("Probability and confidence must be between 0 and 1")
        return v


class CalibratedForecast(BaseModel):
    model_config = {"from_attributes": True}

    raw: float = Field(ge=0.0, le=1.0)
    calibrated: float = Field(ge=0.0, le=1.0)
    confidence: float = Field(ge=0.0, le=1.0)
    calibration_method: str
    historical_samples: int

    @field_validator("raw", "calibrated", "confidence")
    @classmethod
    def validate_probability_range(cls, v: float) -> float:
        if not 0.0 <= v <= 1.0:
            raise ValueError("Probabilities must be between 0 and 1")
        return v


class SimpleForecast(BaseModel):
    """
    Simple forecast output from debate orchestrator.

    This is the immediate output from a debate session,
    before calibration and edge analysis.
    """

    model_config = {"from_attributes": True}

    market_id: str
    probability: float = Field(ge=0.0, le=1.0)
    confidence_lower: float | None = Field(default=None, ge=0.0, le=1.0)
    confidence_upper: float | None = Field(default=None, ge=0.0, le=1.0)
    reasoning: str
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    model_name: str
    debate_rounds: int
    bull_probabilities: list[float] = Field(default_factory=list)
    bear_probabilities: list[float] = Field(default_factory=list)

    @field_validator("probability")
    @classmethod
    def validate_probability(cls, v: float) -> float:
        if not 0.0 <= v <= 1.0:
            raise ValueError("Probability must be between 0 and 1")
        return v

    @field_validator("confidence_lower", "confidence_upper")
    @classmethod
    def validate_confidence_bounds(cls, v: float | None) -> float | None:
        if v is not None and not 0.0 <= v <= 1.0:
            raise ValueError("Confidence bounds must be between 0 and 1")
        return v

    def compute_confidence(self) -> float:
        """Compute confidence from CI bounds or bull/bear convergence.

        Priority:
        1. CI bounds from judge (1 - interval width)
        2. Bull/bear spread convergence (1 - last round spread)
        3. Default 0.5
        """
        if self.confidence_lower is not None and self.confidence_upper is not None:
            return max(0.0, min(1.0, 1.0 - (self.confidence_upper - self.confidence_lower)))
        if self.bull_probabilities and self.bear_probabilities:
            last_bull = self.bull_probabilities[-1]
            last_bear = self.bear_probabilities[-1]
            spread = abs(last_bull - last_bear)
            return max(0.5, 1.0 - spread)
        return 0.5


class Forecast(BaseModel):
    model_config = {"from_attributes": True}

    market_id: str
    timestamp: datetime
    raw_probability: float = Field(ge=0.0, le=1.0)
    calibrated_probability: float = Field(ge=0.0, le=1.0)
    confidence: float = Field(ge=0.0, le=1.0)
    debate_rounds: list[DebateRound]
    judge_reasoning: str
    market_price_at_forecast: float = Field(ge=0.0, le=1.0)
    edge: float
    recommended_action: Literal["TRADE", "SKIP"]

    def to_db_dict(self) -> dict[str, Any]:
        return {
            "market_id": self.market_id,
            "timestamp": self.timestamp.isoformat(),
            "raw_probability": self.raw_probability,
            "calibrated_probability": self.calibrated_probability,
            "confidence": self.confidence,
            "debate_log": [
                {
                    "round_number": r.round_number,
                    "bull_argument": r.bull_argument,
                    "bear_argument": r.bear_argument,
                    "devil_critique": r.devil_critique,
                }
                for r in self.debate_rounds
            ],
            "judge_reasoning": self.judge_reasoning,
            "market_price_at_forecast": self.market_price_at_forecast,
            "edge": self.edge,
            "recommended_action": self.recommended_action,
        }


class EdgeAnalysis(BaseModel):
    model_config = {"from_attributes": True}

    our_forecast: float = Field(ge=0.0, le=1.0)
    market_price: float = Field(ge=0.0, le=1.0)
    raw_edge: float
    abs_edge: float
    weighted_edge: float
    direction: Literal["BUY_YES", "BUY_NO"]
    has_actionable_edge: bool
    recommended_action: Literal["TRADE", "SKIP"]
    reasoning: str
