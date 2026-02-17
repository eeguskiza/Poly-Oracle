from datetime import datetime
from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel

from bot.models.forecast import DebateRound, RawForecast


class AgentRole(str, Enum):
    BULL = "BULL"
    BEAR = "BEAR"
    DEVIL = "DEVIL"
    JUDGE = "JUDGE"


class DebateConfig(BaseModel):
    model_config = {"from_attributes": True}

    num_rounds: int = 3
    temperature: float = 0.7
    max_tokens_per_turn: int = 500


class AgentResponse(BaseModel):
    model_config = {"from_attributes": True}

    role: AgentRole
    round_number: int
    content: str
    probability: Optional[float] = None
    timestamp: datetime


class DebateResult(BaseModel):
    model_config = {"from_attributes": True}

    market_id: str
    debate_rounds: list[DebateRound]
    forecast: RawForecast
    config: DebateConfig
    total_tokens: int
    duration_seconds: float
    timestamp: datetime

    def to_db_dict(self) -> dict[str, Any]:
        return {
            "market_id": self.market_id,
            "debate_rounds": [
                {
                    "round_number": r.round_number,
                    "bull_argument": r.bull_argument,
                    "bear_argument": r.bear_argument,
                    "devil_critique": r.devil_critique,
                }
                for r in self.debate_rounds
            ],
            "forecast": {
                "probability": self.forecast.probability,
                "confidence": self.forecast.confidence,
                "reasoning": self.forecast.reasoning,
            },
            "config": {
                "num_rounds": self.config.num_rounds,
                "temperature": self.config.temperature,
                "max_tokens_per_turn": self.config.max_tokens_per_turn,
            },
            "total_tokens": self.total_tokens,
            "duration_seconds": self.duration_seconds,
            "timestamp": self.timestamp.isoformat(),
        }
