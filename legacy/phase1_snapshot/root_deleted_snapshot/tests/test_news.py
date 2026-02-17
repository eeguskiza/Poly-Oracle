from datetime import datetime, timezone
from unittest.mock import AsyncMock, Mock, patch

import pytest

from src.data.sources.news import NewsClient, SimpleSentiment
from src.models import Market, NewsItem


@pytest.fixture
def mock_google_rss() -> str:
    return """<?xml version="1.0" encoding="UTF-8"?>
<rss version="2.0">
    <channel>
        <title>Google News</title>
        <item>
            <title>Trump wins election with record turnout</title>
            <link>https://example.com/article1</link>
            <pubDate>Wed, 12 Feb 2026 12:00:00 GMT</pubDate>
            <description>Breaking news about the election</description>
            <source url="https://example.com">Example News</source>
        </item>
        <item>
            <title>Market crashes amid uncertainty</title>
            <link>https://example.com/article2</link>
            <pubDate>Wed, 12 Feb 2026 11:00:00 GMT</pubDate>
            <description>Stock market sees major decline</description>
            <source url="https://example.com">Financial Times</source>
        </item>
    </channel>
</rss>"""


@pytest.fixture
def mock_newsapi_response() -> dict:
    return {
        "status": "ok",
        "totalResults": 2,
        "articles": [
            {
                "source": {"id": "cnn", "name": "CNN"},
                "title": "Trump wins presidential election",
                "description": "Donald Trump has won the election",
                "url": "https://cnn.com/article1",
                "publishedAt": "2026-02-12T12:00:00Z",
                "content": "Full article content here...",
            },
            {
                "source": {"id": "bbc", "name": "BBC"},
                "title": "Economic growth slows down",
                "description": "GDP growth shows signs of weakness",
                "url": "https://bbc.com/article2",
                "publishedAt": "2026-02-12T11:00:00Z",
                "content": "Economic analysis content...",
            },
        ],
    }


def test_simple_sentiment_positive() -> None:
    analyzer = SimpleSentiment()
    text = "The team won with great success and strong performance"
    sentiment = analyzer.analyze(text)
    assert sentiment > 0.5


def test_simple_sentiment_negative() -> None:
    analyzer = SimpleSentiment()
    text = "The market crashed and failed with terrible losses"
    sentiment = analyzer.analyze(text)
    assert sentiment < -0.5


def test_simple_sentiment_neutral() -> None:
    analyzer = SimpleSentiment()
    text = "The meeting happened at noon with several people"
    sentiment = analyzer.analyze(text)
    assert -0.2 <= sentiment <= 0.2


def test_simple_sentiment_mixed() -> None:
    analyzer = SimpleSentiment()
    text = "The company won awards but lost market share"
    sentiment = analyzer.analyze(text)
    assert -0.3 <= sentiment <= 0.3


@pytest.mark.asyncio
async def test_fetch_google_news(mock_google_rss: str) -> None:
    with patch("httpx.AsyncClient") as mock_client_class:
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.text = mock_google_rss
        mock_response.raise_for_status = Mock()

        mock_client = AsyncMock()
        mock_client.get = AsyncMock(return_value=mock_response)
        mock_client.aclose = AsyncMock()
        mock_client_class.return_value = mock_client

        async with NewsClient() as client:
            news_items = await client.fetch_google_news("Trump election", max_results=10)

        assert len(news_items) == 2
        assert all(isinstance(item, NewsItem) for item in news_items)
        assert news_items[0].title == "Trump wins election with record turnout"
        assert news_items[0].sentiment > 0
        assert news_items[1].title == "Market crashes amid uncertainty"
        assert news_items[1].sentiment < 0


@pytest.mark.asyncio
async def test_fetch_newsapi(mock_newsapi_response: dict) -> None:
    with patch("httpx.AsyncClient") as mock_client_class:
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = mock_newsapi_response
        mock_response.raise_for_status = Mock()

        mock_client = AsyncMock()
        mock_client.get = AsyncMock(return_value=mock_response)
        mock_client.aclose = AsyncMock()
        mock_client_class.return_value = mock_client

        async with NewsClient() as client:
            news_items = await client.fetch_newsapi("Trump election", max_results=5)

        assert len(news_items) == 2
        assert all(isinstance(item, NewsItem) for item in news_items)
        assert news_items[0].source == "CNN"
        assert news_items[1].source == "BBC"


@pytest.mark.asyncio
async def test_fetch_newsapi_no_key() -> None:
    with patch("src.data.sources.news.get_settings") as mock_settings:
        mock_settings.return_value.data.newsapi_key = None

        async with NewsClient() as client:
            news_items = await client.fetch_newsapi("test", max_results=5)

        assert len(news_items) == 0


@pytest.mark.asyncio
async def test_search_news_deduplication(mock_google_rss: str, mock_newsapi_response: dict) -> None:
    mock_newsapi_response["articles"][0]["title"] = "Trump wins election with record turnout"

    with patch("httpx.AsyncClient") as mock_client_class:
        mock_response_rss = Mock()
        mock_response_rss.status_code = 200
        mock_response_rss.text = mock_google_rss
        mock_response_rss.raise_for_status = Mock()

        mock_response_api = Mock()
        mock_response_api.status_code = 200
        mock_response_api.json.return_value = mock_newsapi_response
        mock_response_api.raise_for_status = Mock()

        mock_client = AsyncMock()
        mock_client.get = AsyncMock(
            side_effect=[mock_response_rss, mock_response_api]
        )
        mock_client.aclose = AsyncMock()
        mock_client_class.return_value = mock_client

        async with NewsClient() as client:
            news_items = await client.search_news("Trump election", max_results=15)

        assert len(news_items) == 3


def test_title_similarity() -> None:
    client = NewsClient()

    similarity = client._calculate_title_similarity(
        "Trump wins election",
        "Trump wins presidential election"
    )
    assert similarity > 0.7

    similarity = client._calculate_title_similarity(
        "Trump wins election",
        "Market crashes today"
    )
    assert similarity < 0.3


def test_extract_keywords() -> None:
    client = NewsClient()

    question = "Will Trump win the 2028 presidential election?"
    keywords = client._extract_keywords(question)

    assert "trump" in keywords.lower()
    assert "2028" in keywords
    assert "presidential" in keywords
    assert "election" in keywords
    assert "will" not in keywords
    assert "the" not in keywords


@pytest.mark.asyncio
async def test_get_market_news(mock_google_rss: str) -> None:
    market = Market(
        id="test_market",
        question="Will Trump win the 2028 election?",
        description="Test market",
        market_type="binary",
        current_price=0.5,
        volume_24h=1000,
        liquidity=5000,
        resolution_date=datetime.now(timezone.utc),
        created_at=datetime.now(timezone.utc),
        outcomes=["YES", "NO"],
        token_ids={},
    )

    with patch("httpx.AsyncClient") as mock_client_class:
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.text = mock_google_rss
        mock_response.raise_for_status = Mock()

        mock_client = AsyncMock()
        mock_client.get = AsyncMock(return_value=mock_response)
        mock_client.aclose = AsyncMock()
        mock_client_class.return_value = mock_client

        async with NewsClient() as client:
            news_items = await client.get_market_news(market, max_results=10)

        assert len(news_items) > 0
        assert all(isinstance(item, NewsItem) for item in news_items)
        assert all(item.relevance_score > 0 for item in news_items)


@pytest.mark.asyncio
async def test_newsapi_limit_warning() -> None:
    with patch("httpx.AsyncClient") as mock_client_class:
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"articles": []}
        mock_response.raise_for_status = Mock()

        mock_client = AsyncMock()
        mock_client.get = AsyncMock(return_value=mock_response)
        mock_client.aclose = AsyncMock()
        mock_client_class.return_value = mock_client

        async with NewsClient() as client:
            client.newsapi_requests_today = 95

            news_items = await client.fetch_newsapi("test", max_results=5)

            assert client.newsapi_requests_today == 96
