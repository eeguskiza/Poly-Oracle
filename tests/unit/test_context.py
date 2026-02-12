from datetime import datetime, timezone
from unittest.mock import AsyncMock, Mock, patch

import pytest

from src.data.context import ContextBuilder
from src.models import Market, NewsItem


@pytest.fixture
def mock_market() -> Market:
    return Market(
        id="test_market_1",
        question="Will SpaceX launch Starship to Mars in 2026?",
        description="This market resolves YES if SpaceX successfully launches Starship to Mars in 2026.",
        market_type="binary",
        current_price=0.35,
        volume_24h=50000,
        liquidity=150000,
        resolution_date=datetime(2026, 12, 31, tzinfo=timezone.utc),
        created_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
        outcomes=["YES", "NO"],
        token_ids={"YES": "token_yes", "NO": "token_no"},
    )


@pytest.fixture
def mock_news_items() -> list[NewsItem]:
    return [
        NewsItem(
            title="SpaceX completes successful Starship test",
            summary="SpaceX achieved a milestone with Starship testing",
            source="Space News",
            published_at=datetime(2026, 2, 10, tzinfo=timezone.utc),
            url="https://example.com/news1",
            relevance_score=0.9,
            sentiment=0.8,
            entities=["SpaceX", "Starship"],
        ),
        NewsItem(
            title="Mars mission faces technical challenges",
            summary="Engineers identify issues with propulsion system",
            source="Tech Times",
            published_at=datetime(2026, 2, 9, tzinfo=timezone.utc),
            url="https://example.com/news2",
            relevance_score=0.7,
            sentiment=-0.4,
            entities=["Mars", "SpaceX"],
        ),
        NewsItem(
            title="NASA collaborates with SpaceX on Mars plans",
            summary="Partnership aims to accelerate Mars mission timeline",
            source="NASA Today",
            published_at=datetime(2026, 2, 8, tzinfo=timezone.utc),
            url="https://example.com/news3",
            relevance_score=0.6,
            sentiment=0.5,
            entities=["NASA", "SpaceX", "Mars"],
        ),
    ]


@pytest.mark.asyncio
async def test_build_context_basic(mock_market: Market, mock_news_items: list[NewsItem]) -> None:
    mock_poly_client = Mock()
    mock_news_client = AsyncMock()
    mock_news_client.get_market_news = AsyncMock(return_value=mock_news_items)

    mock_chroma_client = Mock()
    mock_chroma_client.query = Mock(return_value=[])
    mock_chroma_client.add_documents = Mock()

    builder = ContextBuilder(
        polymarket_client=mock_poly_client,
        news_client=mock_news_client,
        chroma_client=mock_chroma_client,
    )

    context = await builder.build_context(mock_market)

    assert "Market Analysis Context" in context
    assert mock_market.question in context
    assert "Current Market State" in context
    assert "35.0%" in context
    assert "$50,000" in context
    assert "$150,000" in context
    assert "Recent News" in context
    assert "SpaceX completes successful Starship test" in context


@pytest.mark.asyncio
async def test_build_context_with_news(mock_market: Market, mock_news_items: list[NewsItem]) -> None:
    mock_poly_client = Mock()
    mock_news_client = AsyncMock()
    mock_news_client.get_market_news = AsyncMock(return_value=mock_news_items)

    mock_chroma_client = Mock()
    mock_chroma_client.query = Mock(return_value=[])
    mock_chroma_client.add_documents = Mock()

    builder = ContextBuilder(
        polymarket_client=mock_poly_client,
        news_client=mock_news_client,
        chroma_client=mock_chroma_client,
    )

    context = await builder.build_context(mock_market)

    assert "Recent News" in context
    assert "Space News" in context
    assert "Positive" in context
    assert "Negative" in context


@pytest.mark.asyncio
async def test_build_context_stores_in_chromadb(
    mock_market: Market,
    mock_news_items: list[NewsItem]
) -> None:
    mock_poly_client = Mock()
    mock_news_client = AsyncMock()
    mock_news_client.get_market_news = AsyncMock(return_value=mock_news_items)

    mock_chroma_client = Mock()
    mock_chroma_client.query = Mock(return_value=[])
    mock_chroma_client.add_documents = Mock()

    builder = ContextBuilder(
        polymarket_client=mock_poly_client,
        news_client=mock_news_client,
        chroma_client=mock_chroma_client,
    )

    await builder.build_context(mock_market)

    assert mock_chroma_client.add_documents.call_count == 2

    news_call = mock_chroma_client.add_documents.call_args_list[0]
    assert news_call[1]["collection_name"] == "news"
    assert len(news_call[1]["documents"]) == 3

    context_call = mock_chroma_client.add_documents.call_args_list[1]
    assert context_call[1]["collection_name"] == "market_context"
    assert len(context_call[1]["documents"]) == 1


@pytest.mark.asyncio
async def test_build_context_with_similar_markets(
    mock_market: Market,
    mock_news_items: list[NewsItem]
) -> None:
    mock_poly_client = Mock()
    mock_news_client = AsyncMock()
    mock_news_client.get_market_news = AsyncMock(return_value=mock_news_items)

    similar_markets = [
        {
            "metadata": {
                "question": "Will SpaceX land on Mars in 2025?",
                "outcome": "NO",
                "final_price": 0.15,
            }
        },
    ]

    mock_chroma_client = Mock()
    mock_chroma_client.query = Mock(return_value=similar_markets)
    mock_chroma_client.add_documents = Mock()

    builder = ContextBuilder(
        polymarket_client=mock_poly_client,
        news_client=mock_news_client,
        chroma_client=mock_chroma_client,
    )

    context = await builder.build_context(mock_market)

    assert "Historical Similar Markets" in context
    assert "Will SpaceX land on Mars in 2025?" in context
    assert "NO" in context


def test_extract_entities(mock_news_items: list[NewsItem]) -> None:
    mock_poly_client = Mock()
    mock_news_client = Mock()
    mock_chroma_client = Mock()

    builder = ContextBuilder(
        polymarket_client=mock_poly_client,
        news_client=mock_news_client,
        chroma_client=mock_chroma_client,
    )

    entities = builder._extract_entities(mock_news_items)

    assert "SpaceX" in entities
    assert "Mars" in entities
    assert "NASA" in entities


def test_format_context_template(mock_market: Market, mock_news_items: list[NewsItem]) -> None:
    mock_poly_client = Mock()
    mock_news_client = Mock()
    mock_chroma_client = Mock()

    builder = ContextBuilder(
        polymarket_client=mock_poly_client,
        news_client=mock_news_client,
        chroma_client=mock_chroma_client,
    )

    context = builder._format_context(
        market=mock_market,
        news_items=mock_news_items,
        similar_markets=[],
        entities=["SpaceX", "Mars", "NASA"],
    )

    assert "# Market Analysis Context" in context
    assert "## Question" in context
    assert "## Current Market State" in context
    assert "## Recent News" in context
    assert "## Key Entities" in context
    assert "SpaceX, Mars, NASA" in context


def test_format_context_with_description(mock_market: Market) -> None:
    mock_poly_client = Mock()
    mock_news_client = Mock()
    mock_chroma_client = Mock()

    builder = ContextBuilder(
        polymarket_client=mock_poly_client,
        news_client=mock_news_client,
        chroma_client=mock_chroma_client,
    )

    context = builder._format_context(
        market=mock_market,
        news_items=[],
        similar_markets=[],
        entities=[],
    )

    assert "## Description" in context
    assert mock_market.description in context


@pytest.mark.asyncio
async def test_build_context_no_news(mock_market: Market) -> None:
    mock_poly_client = Mock()
    mock_news_client = AsyncMock()
    mock_news_client.get_market_news = AsyncMock(return_value=[])

    mock_chroma_client = Mock()
    mock_chroma_client.query = Mock(return_value=[])
    mock_chroma_client.add_documents = Mock()

    builder = ContextBuilder(
        polymarket_client=mock_poly_client,
        news_client=mock_news_client,
        chroma_client=mock_chroma_client,
    )

    context = await builder.build_context(mock_market)

    assert "Market Analysis Context" in context
    assert mock_market.question in context
    assert "Recent News" not in context or context.count("Recent News") == 1


@pytest.mark.asyncio
async def test_build_context_chromadb_error_handling(
    mock_market: Market,
    mock_news_items: list[NewsItem]
) -> None:
    mock_poly_client = Mock()
    mock_news_client = AsyncMock()
    mock_news_client.get_market_news = AsyncMock(return_value=mock_news_items)

    mock_chroma_client = Mock()
    mock_chroma_client.query = Mock(return_value=[])
    mock_chroma_client.add_documents = Mock(side_effect=Exception("ChromaDB error"))

    builder = ContextBuilder(
        polymarket_client=mock_poly_client,
        news_client=mock_news_client,
        chroma_client=mock_chroma_client,
    )

    context = await builder.build_context(mock_market)

    assert "Market Analysis Context" in context
    assert mock_market.question in context
