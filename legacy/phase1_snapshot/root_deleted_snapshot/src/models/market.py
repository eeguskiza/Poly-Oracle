from datetime import datetime, timezone
from typing import Any, Optional

from pydantic import BaseModel, Field, field_validator, computed_field


class Market(BaseModel):
    model_config = {"from_attributes": True}

    id: str
    question: str
    description: str
    market_type: str
    current_price: float = Field(ge=0.0, le=1.0)
    volume_24h: float
    liquidity: float
    resolution_date: datetime
    created_at: datetime
    outcomes: list[str]
    token_ids: dict[str, str]

    @field_validator("current_price")
    @classmethod
    def validate_price_range(cls, v: float) -> float:
        if not 0.0 <= v <= 1.0:
            raise ValueError("current_price must be between 0 and 1")
        return v

    @computed_field
    @property
    def days_until_resolution(self) -> float:
        now = datetime.now(timezone.utc)
        if self.resolution_date.tzinfo is None:
            resolution = self.resolution_date.replace(tzinfo=timezone.utc)
        else:
            resolution = self.resolution_date
        delta = resolution - now
        return delta.total_seconds() / 86400

    def to_db_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "question": self.question,
            "description": self.description,
            "market_type": self.market_type,
            "current_price": self.current_price,
            "volume_24h": self.volume_24h,
            "liquidity": self.liquidity,
            "resolution_date": self.resolution_date.isoformat(),
            "created_at": self.created_at.isoformat(),
            "outcomes": self.outcomes,
            "token_ids": self.token_ids,
        }


class MarketSnapshot(BaseModel):
    model_config = {"from_attributes": True}

    market_id: str
    timestamp: datetime
    price: float = Field(ge=0.0, le=1.0)
    volume: float
    orderbook: dict[str, Any]

    def to_db_dict(self) -> dict[str, Any]:
        return {
            "market_id": self.market_id,
            "timestamp": self.timestamp.isoformat(),
            "price": self.price,
            "volume": self.volume,
            "orderbook": self.orderbook,
        }


class MarketFilter(BaseModel):
    min_liquidity: Optional[float] = None
    max_days_to_resolution: Optional[float] = None
    min_volume: Optional[float] = None
    market_types: Optional[list[str]] = None
