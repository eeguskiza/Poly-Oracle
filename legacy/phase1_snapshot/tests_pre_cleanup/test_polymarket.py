from datetime import datetime, timezone, timedelta
from unittest.mock import AsyncMock, Mock, patch

import httpx
import pytest

from bot.data.polymarket import PolymarketClient
from bot.models import Market, MarketFilter
from bot.exceptions import DataFetchError, MarketNotFoundError


@pytest.fixture
def mock_market_data() -> dict:
    return {
        "id": "test_market_1",
        "question": "Will test pass?",
        "description": "Test market description",
        "lastTradePrice": 0.65,
        "outcomePrices": "[\"0.65\", \"0.35\"]",
        "volume24hr": 5000,
        "liquidityNum": 10000,
        "endDate": "2026-03-15T00:00:00Z",
        "createdAt": "2026-02-01T00:00:00Z",
        "outcomes": "[\"YES\", \"NO\"]",
        "clobTokenIds": "[\"token_yes_1\", \"token_no_1\"]",
    }


@pytest.fixture
def mock_market_list_data(mock_market_data: dict) -> list[dict]:
    market_2 = mock_market_data.copy()
    market_2["id"] = "test_market_2"
    market_2["question"] = "Will second test pass?"
    market_2["liquidityNum"] = 500

    market_3 = mock_market_data.copy()
    market_3["id"] = "test_market_3"
    market_3["question"] = "Will third test pass?"
    market_3["liquidityNum"] = 2000

    return [mock_market_data, market_2, market_3]


@pytest.mark.asyncio
async def test_get_market_success(mock_market_data: dict) -> None:
    with patch("httpx.AsyncClient") as mock_client_class:
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = mock_market_data
        mock_response.raise_for_status = Mock()

        mock_client = AsyncMock()
        mock_client.request = AsyncMock(return_value=mock_response)
        mock_client.aclose = AsyncMock()
        mock_client_class.return_value = mock_client

        async with PolymarketClient() as client:
            market = await client.get_market("test_market_1")

        assert isinstance(market, Market)
        assert market.id == "test_market_1"
        assert market.question == "Will test pass?"
        assert market.current_price == 0.65
        assert market.liquidity == 10000.0
        assert market.volume_24h == 5000.0


@pytest.mark.asyncio
async def test_get_market_not_found() -> None:
    with patch("httpx.AsyncClient") as mock_client_class:
        mock_response = Mock()
        mock_response.status_code = 404

        mock_client = AsyncMock()
        mock_client.request = AsyncMock(return_value=mock_response)
        mock_client.aclose = AsyncMock()
        mock_client_class.return_value = mock_client

        async with PolymarketClient() as client:
            with pytest.raises(MarketNotFoundError):
                await client.get_market("nonexistent_market")


@pytest.mark.asyncio
async def test_get_active_markets_success(mock_market_list_data: list[dict]) -> None:
    with patch("httpx.AsyncClient") as mock_client_class:
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = mock_market_list_data
        mock_response.raise_for_status = Mock()

        mock_client = AsyncMock()
        mock_client.request = AsyncMock(return_value=mock_response)
        mock_client.aclose = AsyncMock()
        mock_client_class.return_value = mock_client

        async with PolymarketClient() as client:
            markets = await client.get_active_markets(limit=10)

        assert len(markets) == 3
        assert all(isinstance(m, Market) for m in markets)
        assert markets[0].id == "test_market_1"
        assert markets[1].id == "test_market_2"
        assert markets[2].id == "test_market_3"


@pytest.mark.asyncio
async def test_filter_markets_by_liquidity(mock_market_list_data: list[dict]) -> None:
    with patch("httpx.AsyncClient") as mock_client_class:
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = mock_market_list_data
        mock_response.raise_for_status = Mock()

        mock_client = AsyncMock()
        mock_client.request = AsyncMock(return_value=mock_response)
        mock_client.aclose = AsyncMock()
        mock_client_class.return_value = mock_client

        async with PolymarketClient() as client:
            filter_obj = MarketFilter(min_liquidity=1000)
            markets = await client.filter_markets(filter_obj)

        assert len(markets) == 2
        assert all(m.liquidity >= 1000 for m in markets)
        assert markets[0].id == "test_market_1"
        assert markets[1].id == "test_market_3"


@pytest.mark.asyncio
async def test_filter_markets_by_volume(mock_market_list_data: list[dict]) -> None:
    mock_market_list_data[1]["volume24hr"] = 100

    with patch("httpx.AsyncClient") as mock_client_class:
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = mock_market_list_data
        mock_response.raise_for_status = Mock()

        mock_client = AsyncMock()
        mock_client.request = AsyncMock(return_value=mock_response)
        mock_client.aclose = AsyncMock()
        mock_client_class.return_value = mock_client

        async with PolymarketClient() as client:
            filter_obj = MarketFilter(min_volume=1000)
            markets = await client.filter_markets(filter_obj)

        assert len(markets) == 2
        assert all(m.volume_24h >= 1000 for m in markets)


@pytest.mark.asyncio
async def test_filter_markets_by_type(mock_market_list_data: list[dict]) -> None:
    with patch("httpx.AsyncClient") as mock_client_class:
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = mock_market_list_data
        mock_response.raise_for_status = Mock()

        mock_client = AsyncMock()
        mock_client.request = AsyncMock(return_value=mock_response)
        mock_client.aclose = AsyncMock()
        mock_client_class.return_value = mock_client

        async with PolymarketClient() as client:
            filter_obj = MarketFilter(market_types=["binary"])
            markets = await client.filter_markets(filter_obj)

        assert len(markets) == 3
        assert all(m.market_type == "binary" for m in markets)


@pytest.mark.asyncio
async def test_filter_markets_by_days_to_resolution(mock_market_list_data: list[dict]) -> None:
    far_future = (datetime.now(timezone.utc) + timedelta(days=60)).isoformat().replace("+00:00", "Z")
    mock_market_list_data[1]["endDate"] = far_future

    with patch("httpx.AsyncClient") as mock_client_class:
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = mock_market_list_data
        mock_response.raise_for_status = Mock()

        mock_client = AsyncMock()
        mock_client.request = AsyncMock(return_value=mock_response)
        mock_client.aclose = AsyncMock()
        mock_client_class.return_value = mock_client

        async with PolymarketClient() as client:
            filter_obj = MarketFilter(max_days_to_resolution=45)
            markets = await client.filter_markets(filter_obj)

        assert len(markets) == 2
        assert all(m.days_until_resolution <= 45 for m in markets)


@pytest.mark.asyncio
async def test_search_markets(mock_market_list_data: list[dict]) -> None:
    with patch("httpx.AsyncClient") as mock_client_class:
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = mock_market_list_data
        mock_response.raise_for_status = Mock()

        mock_client = AsyncMock()
        mock_client.request = AsyncMock(return_value=mock_response)
        mock_client.aclose = AsyncMock()
        mock_client_class.return_value = mock_client

        async with PolymarketClient() as client:
            markets = await client.search_markets("second")

        assert len(markets) == 1
        assert markets[0].id == "test_market_2"
        assert "second" in markets[0].question.lower()


@pytest.mark.asyncio
async def test_caching_behavior(mock_market_data: dict) -> None:
    with patch("httpx.AsyncClient") as mock_client_class:
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = mock_market_data
        mock_response.raise_for_status = Mock()

        mock_client = AsyncMock()
        mock_client.request = AsyncMock(return_value=mock_response)
        mock_client.aclose = AsyncMock()
        mock_client_class.return_value = mock_client

        async with PolymarketClient() as client:
            market1 = await client.get_market("test_market_1")
            market2 = await client.get_market("test_market_1")

        assert mock_client.request.call_count == 1
        assert market1.id == market2.id


@pytest.mark.asyncio
async def test_http_error_retry(mock_market_data: dict) -> None:
    with patch("httpx.AsyncClient") as mock_client_class:
        mock_response_fail = Mock()
        mock_response_fail.status_code = 500
        mock_response_fail.raise_for_status = Mock(
            side_effect=httpx.HTTPStatusError(
                "Server error",
                request=Mock(),
                response=Mock(status_code=500)
            )
        )

        mock_response_success = Mock()
        mock_response_success.status_code = 200
        mock_response_success.json.return_value = mock_market_data
        mock_response_success.raise_for_status = Mock()

        mock_client = AsyncMock()
        mock_client.request = AsyncMock(
            side_effect=[mock_response_fail, mock_response_success]
        )
        mock_client.aclose = AsyncMock()
        mock_client_class.return_value = mock_client

        with patch("asyncio.sleep", new_callable=AsyncMock):
            async with PolymarketClient() as client:
                market = await client.get_market("test_market_1")

        assert mock_client.request.call_count == 2
        assert market.id == "test_market_1"


@pytest.mark.asyncio
async def test_http_error_max_retries() -> None:
    with patch("httpx.AsyncClient") as mock_client_class:
        mock_response_fail = Mock()
        mock_response_fail.status_code = 500
        mock_response_fail.raise_for_status = Mock(
            side_effect=httpx.HTTPStatusError(
                "Server error",
                request=Mock(),
                response=Mock(status_code=500)
            )
        )

        mock_client = AsyncMock()
        mock_client.request = AsyncMock(return_value=mock_response_fail)
        mock_client.aclose = AsyncMock()
        mock_client_class.return_value = mock_client

        with patch("asyncio.sleep", new_callable=AsyncMock):
            async with PolymarketClient() as client:
                with pytest.raises(DataFetchError):
                    await client.get_market("test_market_1")

        assert mock_client.request.call_count == 3


@pytest.mark.asyncio
async def test_timeout_error_retry() -> None:
    with patch("httpx.AsyncClient") as mock_client_class:
        mock_client = AsyncMock()
        mock_client.request = AsyncMock(
            side_effect=httpx.TimeoutException("Request timeout")
        )
        mock_client.aclose = AsyncMock()
        mock_client_class.return_value = mock_client

        with patch("asyncio.sleep", new_callable=AsyncMock):
            async with PolymarketClient() as client:
                with pytest.raises(DataFetchError):
                    await client.get_market("test_market_1")

        assert mock_client.request.call_count == 3


@pytest.mark.asyncio
async def test_rate_limiting_gamma() -> None:
    with patch("httpx.AsyncClient") as mock_client_class, \
         patch("asyncio.sleep", new_callable=AsyncMock) as mock_sleep:

        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"id": "test"}
        mock_response.raise_for_status = Mock()

        mock_client = AsyncMock()
        mock_client.request = AsyncMock(return_value=mock_response)
        mock_client.aclose = AsyncMock()
        mock_client_class.return_value = mock_client

        async with PolymarketClient() as client:
            client.gamma_rate_limit = 2

            for _ in range(3):
                try:
                    await client._request_with_retry(
                        "GET",
                        f"{client.gamma_url}/test",
                        "gamma"
                    )
                except Exception:
                    pass

        assert mock_sleep.called


@pytest.mark.asyncio
async def test_get_market_price_success() -> None:
    with patch("httpx.AsyncClient") as mock_client_class:
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"mid": "0.72"}
        mock_response.raise_for_status = Mock()

        mock_client = AsyncMock()
        mock_client.request = AsyncMock(return_value=mock_response)
        mock_client.aclose = AsyncMock()
        mock_client_class.return_value = mock_client

        async with PolymarketClient() as client:
            price = await client.get_market_price("token_yes_1")

        assert price == 0.72


@pytest.mark.asyncio
async def test_get_market_price_fallback() -> None:
    with patch("httpx.AsyncClient") as mock_client_class:
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {}
        mock_response.raise_for_status = Mock()

        mock_client = AsyncMock()
        mock_client.request = AsyncMock(return_value=mock_response)
        mock_client.aclose = AsyncMock()
        mock_client_class.return_value = mock_client

        async with PolymarketClient() as client:
            price = await client.get_market_price("token_yes_1")

        assert price == 0.5


@pytest.mark.asyncio
async def test_get_orderbook_success() -> None:
    orderbook_data = {
        "bids": [{"price": "0.71", "size": "100"}],
        "asks": [{"price": "0.73", "size": "150"}]
    }

    with patch("httpx.AsyncClient") as mock_client_class:
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = orderbook_data
        mock_response.raise_for_status = Mock()

        mock_client = AsyncMock()
        mock_client.request = AsyncMock(return_value=mock_response)
        mock_client.aclose = AsyncMock()
        mock_client_class.return_value = mock_client

        async with PolymarketClient() as client:
            orderbook = await client.get_orderbook("token_yes_1")

        assert "bids" in orderbook
        assert "asks" in orderbook
        assert len(orderbook["bids"]) == 1
        assert len(orderbook["asks"]) == 1


@pytest.mark.asyncio
async def test_parse_market_invalid_data() -> None:
    invalid_data = {"id": "test"}

    with patch("httpx.AsyncClient") as mock_client_class:
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = invalid_data
        mock_response.raise_for_status = Mock()

        mock_client = AsyncMock()
        mock_client.request = AsyncMock(return_value=mock_response)
        mock_client.aclose = AsyncMock()
        mock_client_class.return_value = mock_client

        async with PolymarketClient() as client:
            with pytest.raises(DataFetchError):
                await client.get_market("test_market_1")


@pytest.mark.asyncio
async def test_get_active_markets_skips_invalid() -> None:
    valid_market = {
        "id": "test_market_1",
        "question": "Will test pass?",
        "description": "Test",
        "lastTradePrice": 0.65,
        "outcomePrices": "[\"0.65\", \"0.35\"]",
        "volume24hr": 5000,
        "liquidityNum": 10000,
        "endDate": "2026-03-15T00:00:00Z",
        "createdAt": "2026-02-01T00:00:00Z",
        "outcomes": "[\"YES\", \"NO\"]",
        "clobTokenIds": "[]",
    }
    invalid_market = {"id": "invalid"}

    with patch("httpx.AsyncClient") as mock_client_class:
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = [valid_market, invalid_market, valid_market]
        mock_response.raise_for_status = Mock()

        mock_client = AsyncMock()
        mock_client.request = AsyncMock(return_value=mock_response)
        mock_client.aclose = AsyncMock()
        mock_client_class.return_value = mock_client

        async with PolymarketClient() as client:
            markets = await client.get_active_markets()

        assert len(markets) == 2
        assert all(m.id == "test_market_1" for m in markets)
