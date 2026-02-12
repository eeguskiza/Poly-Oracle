from datetime import datetime
from typing import Any, Literal, Optional

from pydantic import BaseModel, Field, field_validator, computed_field


class PositionSize(BaseModel):
    model_config = {"from_attributes": True}

    full_kelly: float
    fractional_kelly: float
    constrained_pct: float
    amount_usd: float = Field(gt=0)
    direction: str

    @field_validator("amount_usd")
    @classmethod
    def validate_positive_amount(cls, v: float) -> float:
        if v <= 0:
            raise ValueError("amount_usd must be positive")
        return v


class Trade(BaseModel):
    model_config = {"from_attributes": True}

    id: str
    market_id: str
    timestamp: datetime
    direction: Literal["BUY_YES", "BUY_NO"]
    amount_usd: float = Field(gt=0)
    price: float = Field(ge=0.0, le=1.0)
    shares: float
    order_type: Literal["LIMIT", "MARKET"]
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
            "price": self.price,
            "shares": self.shares,
            "order_type": self.order_type,
            "status": self.status,
        }


class Position(BaseModel):
    model_config = {"from_attributes": True}

    market_id: str
    direction: str
    shares: float
    avg_entry_price: float = Field(ge=0.0, le=1.0)
    current_price: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    realized_pnl: float

    @computed_field
    @property
    def unrealized_pnl(self) -> float:
        if self.current_price is None or self.shares == 0:
            return 0.0
        return self.shares * (self.current_price - self.avg_entry_price)

    def to_db_dict(self) -> dict[str, Any]:
        return {
            "market_id": self.market_id,
            "direction": self.direction,
            "shares": self.shares,
            "avg_entry_price": self.avg_entry_price,
            "total_cost": self.shares * self.avg_entry_price,
            "realized_pnl": self.realized_pnl,
        }


class RiskCheck(BaseModel):
    model_config = {"from_attributes": True}

    passed: bool
    violations: list[str]
    current_daily_pnl: float
    open_positions: int
    proposed_exposure: float


class ExecutionResult(BaseModel):
    model_config = {"from_attributes": True}

    success: bool
    order_id: Optional[str] = None
    filled_price: Optional[float] = None
    filled_amount: Optional[float] = None
    fees: Optional[float] = None
    error: Optional[str] = None
    timestamp: datetime

    def to_db_dict(self) -> dict[str, Any]:
        return {
            "success": self.success,
            "order_id": self.order_id,
            "filled_price": self.filled_price,
            "filled_amount": self.filled_amount,
            "fees": self.fees,
            "error": self.error,
            "timestamp": self.timestamp.isoformat(),
        }
