from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field


class NewsItem(BaseModel):
    model_config = {"from_attributes": True}

    title: str
    summary: str
    source: str
    published_at: datetime
    url: str
    relevance_score: float = Field(default=0.0, ge=0.0, le=1.0)
    sentiment: float = Field(default=0.0, ge=-1.0, le=1.0)
    entities: list[str] = Field(default_factory=list)

    def to_db_dict(self) -> dict[str, Any]:
        return {
            "title": self.title,
            "summary": self.summary,
            "source": self.source,
            "published_at": self.published_at.isoformat(),
            "url": self.url,
            "relevance_score": self.relevance_score,
            "sentiment": self.sentiment,
            "entities": self.entities,
        }
