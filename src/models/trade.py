from datetime import datetime
from typing import Any, Literal, Optional

from pydantic import BaseModel, Field, field_validator, computed_field


class PositionSize(BaseModel):
    model_config = {"from_attributes": True}

    amount_usd: float = Field(ge=0)
    num_shares: float = Field(ge=0)
    kelly_fraction: float
    applied_fraction: float
    constraints_applied: dict[str, Any]


class Trade(BaseModel):
    model_config = {"from_attributes": True}

    id: str
    market_id: str
    timestamp: datetime
    direction: Literal["BUY_YES", "BUY_NO"]
    amount_usd: float = Field(gt=0)
    num_shares: float
    entry_price: float = Field(ge=0.0, le=1.0)
    status: Literal["PENDING", "FILLED", "CANCELLED", "FAILED"]

    @field_validator("amount_usd")
    @classmethod
    def validate_positive_amount(cls, v: float) -> float:
        if v <= 0:
            raise ValueError("amount_usd must be positive")
        return v

    def to_db_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "market_id": self.market_id,
            "timestamp": self.timestamp.isoformat(),
            "direction": self.direction,
            "amount_usd": self.amount_usd,
            "num_shares": self.num_shares,
            "entry_price": self.entry_price,
            "status": self.status,
        }


class Position(BaseModel):
    model_config = {"from_attributes": True}

    market_id: str
    direction: str
    num_shares: float
    amount_usd: float
    avg_entry_price: float = Field(ge=0.0, le=1.0)
    current_price: float = Field(ge=0.0, le=1.0)
    unrealized_pnl: float
    updated_at: datetime

    def to_db_dict(self) -> dict[str, Any]:
        return {
            "market_id": self.market_id,
            "direction": self.direction,
            "num_shares": self.num_shares,
            "amount_usd": self.amount_usd,
            "avg_entry_price": self.avg_entry_price,
            "current_price": self.current_price,
            "unrealized_pnl": self.unrealized_pnl,
            "updated_at": self.updated_at.isoformat(),
        }


class RiskCheck(BaseModel):
    model_config = {"from_attributes": True}

    passed: bool
    violations: list[str]
    daily_loss_pct: float
    num_open_positions: int
    proposed_market_exposure: float


class ExecutionResult(BaseModel):
    model_config = {"from_attributes": True}

    success: bool
    trade_id: Optional[str] = None
    message: Optional[str] = None
    risk_check: Optional[RiskCheck] = None
