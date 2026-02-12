import asyncio
import json
from datetime import datetime, timezone, timedelta
from typing import Any

import httpx
from loguru import logger

from config.settings import get_settings
from src.models import Market, MarketSnapshot, MarketFilter
from src.utils.exceptions import DataFetchError, MarketNotFoundError


class PolymarketClient:
    def __init__(self) -> None:
        self.settings = get_settings()
        self.gamma_url = self.settings.polymarket.gamma_url
        self.clob_url = self.settings.polymarket.clob_url

        self.gamma_rate_limit = 100
        self.clob_rate_limit = 60

        self._cache: dict[str, tuple[Any, datetime]] = {}
        self._gamma_request_times: list[datetime] = []
        self._clob_request_times: list[datetime] = []

        self.client = httpx.AsyncClient(timeout=30.0)

        logger.info("PolymarketClient initialized")

    async def _check_rate_limit(self, api_type: str) -> None:
        now = datetime.now(timezone.utc)

        if api_type == "gamma":
            self._gamma_request_times = [
                t for t in self._gamma_request_times
                if now - t < timedelta(minutes=1)
            ]
            if len(self._gamma_request_times) >= self.gamma_rate_limit:
                sleep_time = 60 - (now - self._gamma_request_times[0]).total_seconds()
                if sleep_time > 0:
                    logger.warning(f"Gamma rate limit reached, sleeping {sleep_time:.1f}s")
                    await asyncio.sleep(sleep_time)
            self._gamma_request_times.append(now)
        else:
            self._clob_request_times = [
                t for t in self._clob_request_times
                if now - t < timedelta(minutes=1)
            ]
            if len(self._clob_request_times) >= self.clob_rate_limit:
                sleep_time = 60 - (now - self._clob_request_times[0]).total_seconds()
                if sleep_time > 0:
                    logger.warning(f"CLOB rate limit reached, sleeping {sleep_time:.1f}s")
                    await asyncio.sleep(sleep_time)
            self._clob_request_times.append(now)

    def _get_from_cache(self, key: str, ttl: int) -> Any | None:
        if key in self._cache:
            value, timestamp = self._cache[key]
            age = (datetime.now(timezone.utc) - timestamp).total_seconds()
            if age < ttl:
                logger.debug(f"Cache hit for {key}, age: {age:.1f}s")
                return value
        return None

    def _set_cache(self, key: str, value: Any) -> None:
        self._cache[key] = (value, datetime.now(timezone.utc))

    async def _request_with_retry(
        self,
        method: str,
        url: str,
        api_type: str,
        **kwargs: Any
    ) -> dict[str, Any]:
        await self._check_rate_limit(api_type)

        for attempt in range(3):
            try:
                logger.debug(f"{method} {url} (attempt {attempt + 1})")
                response = await self.client.request(method, url, **kwargs)

                if response.status_code == 404:
                    raise MarketNotFoundError(url)

                response.raise_for_status()
                return response.json()

            except httpx.HTTPStatusError as e:
                if e.response.status_code == 404:
                    raise MarketNotFoundError(url)
                if attempt == 2:
                    raise DataFetchError(
                        f"HTTP {e.response.status_code}",
                        source="Polymarket",
                        url=url
                    )
                sleep_time = 2 ** attempt
                logger.warning(f"Request failed, retrying in {sleep_time}s")
                await asyncio.sleep(sleep_time)

            except (httpx.RequestError, httpx.TimeoutException) as e:
                if attempt == 2:
                    raise DataFetchError(
                        str(e),
                        source="Polymarket",
                        url=url
                    )
                sleep_time = 2 ** attempt
                logger.warning(f"Request error, retrying in {sleep_time}s: {e}")
                await asyncio.sleep(sleep_time)

        raise DataFetchError("Max retries exceeded", source="Polymarket", url=url)

    def _parse_market(self, data: dict[str, Any]) -> Market:
        try:
            outcomes = data.get("outcomes", ["YES", "NO"])
            if isinstance(outcomes, str):
                outcomes = json.loads(outcomes)

            outcome_prices_str = data.get("outcomePrices", "[\"0.5\", \"0.5\"]")
            outcome_prices = json.loads(outcome_prices_str) if isinstance(outcome_prices_str, str) else outcome_prices_str
            current_price = float(outcome_prices[0]) if outcome_prices else 0.5

            if "lastTradePrice" in data and data["lastTradePrice"]:
                current_price = float(data["lastTradePrice"])

            token_ids_str = data.get("clobTokenIds", "[]")
            token_ids_list = json.loads(token_ids_str) if isinstance(token_ids_str, str) else token_ids_str
            token_ids = {}
            if token_ids_list and len(token_ids_list) >= 2:
                token_ids = {
                    outcomes[0]: token_ids_list[0],
                    outcomes[1]: token_ids_list[1] if len(token_ids_list) > 1 else ""
                }

            if "endDate" in data and data["endDate"]:
                resolution_date = datetime.fromisoformat(
                    data["endDate"].replace("Z", "+00:00")
                )
            else:
                resolution_date = datetime.now(timezone.utc) + timedelta(days=365)
                logger.debug(f"Market {data.get('id', 'unknown')} missing endDate, using default")

            created_at = datetime.now(timezone.utc)
            if "createdAt" in data and data["createdAt"]:
                created_at = datetime.fromisoformat(
                    data["createdAt"].replace("Z", "+00:00")
                )

            return Market(
                id=data["id"],
                question=data["question"],
                description=data.get("description", ""),
                market_type="binary",
                current_price=current_price,
                volume_24h=float(data.get("volume24hr", 0)),
                liquidity=float(data.get("liquidityNum", 0)),
                resolution_date=resolution_date,
                created_at=created_at,
                outcomes=outcomes,
                token_ids=token_ids,
            )
        except (KeyError, ValueError, TypeError) as e:
            market_id = data.get("id", "unknown")
            logger.error(f"Failed to parse market {market_id}: {e}")
            raise DataFetchError(
                f"Invalid market data format: {e}",
                source="Polymarket",
                url=""
            )

    async def get_active_markets(
        self,
        limit: int = 100,
        offset: int = 0
    ) -> list[Market]:
        cache_key = f"active_markets_{limit}_{offset}"
        cached = self._get_from_cache(
            cache_key,
            self.settings.data.cache_ttl_market_list
        )
        if cached:
            return cached

        try:
            url = f"{self.gamma_url}/markets"
            params = {
                "active": "true",
                "closed": "false",
                "limit": limit,
                "offset": offset,
            }

            data = await self._request_with_retry("GET", url, "gamma", params=params)

            markets = []
            market_list = data if isinstance(data, list) else data.get("data", [])

            for market_data in market_list:
                try:
                    market = self._parse_market(market_data)
                    markets.append(market)
                except DataFetchError as e:
                    logger.warning(f"Skipping invalid market: {e}")
                    continue

            logger.info(f"Fetched {len(markets)} active markets")
            self._set_cache(cache_key, markets)
            return markets

        except Exception as e:
            if isinstance(e, (DataFetchError, MarketNotFoundError)):
                raise
            raise DataFetchError(
                str(e),
                source="Polymarket",
                url=f"{self.gamma_url}/markets"
            )

    async def get_market(self, market_id: str) -> Market:
        cache_key = f"market_{market_id}"
        cached = self._get_from_cache(
            cache_key,
            self.settings.data.cache_ttl_market_detail
        )
        if cached:
            return cached

        try:
            url = f"{self.gamma_url}/markets/{market_id}"
            data = await self._request_with_retry("GET", url, "gamma")

            market = self._parse_market(data)

            logger.info(f"Fetched market: {market.question}")
            self._set_cache(cache_key, market)
            return market

        except MarketNotFoundError:
            raise
        except Exception as e:
            if isinstance(e, DataFetchError):
                raise
            raise DataFetchError(
                str(e),
                source="Polymarket",
                url=f"{self.gamma_url}/markets/{market_id}"
            )

    async def get_market_price(self, token_id: str) -> float:
        try:
            url = f"{self.clob_url}/price"
            params = {"token_id": token_id}

            data = await self._request_with_retry("GET", url, "clob", params=params)

            mid_price = float(data.get("mid", 0.5))
            logger.debug(f"Fetched price for {token_id}: {mid_price}")
            return mid_price

        except Exception as e:
            if isinstance(e, DataFetchError):
                raise
            logger.warning(f"Failed to get price for {token_id}: {e}")
            return 0.5

    async def get_orderbook(self, token_id: str) -> dict[str, Any]:
        try:
            url = f"{self.clob_url}/book"
            params = {"token_id": token_id}

            data = await self._request_with_retry("GET", url, "clob", params=params)

            logger.debug(f"Fetched orderbook for {token_id}")
            return data

        except Exception as e:
            if isinstance(e, DataFetchError):
                raise
            logger.warning(f"Failed to get orderbook for {token_id}: {e}")
            return {"bids": [], "asks": []}

    async def get_price_history(
        self,
        market_id: str,
        interval: str = "1d"
    ) -> list[MarketSnapshot]:
        try:
            market = await self.get_market(market_id)

            snapshots = [
                MarketSnapshot(
                    market_id=market_id,
                    timestamp=datetime.now(timezone.utc),
                    price=market.current_price,
                    volume=market.volume_24h,
                    orderbook={},
                )
            ]

            logger.debug(f"Created {len(snapshots)} price history snapshots")
            return snapshots

        except Exception as e:
            if isinstance(e, (DataFetchError, MarketNotFoundError)):
                raise
            raise DataFetchError(
                str(e),
                source="Polymarket",
                url=f"price_history/{market_id}"
            )

    async def filter_markets(self, filter: MarketFilter) -> list[Market]:
        markets = await self.get_active_markets(limit=200)

        filtered = []
        for market in markets:
            if filter.min_liquidity and market.liquidity < filter.min_liquidity:
                continue

            if filter.max_days_to_resolution:
                if market.days_until_resolution > filter.max_days_to_resolution:
                    continue

            if filter.min_volume and market.volume_24h < filter.min_volume:
                continue

            if filter.market_types:
                if market.market_type not in filter.market_types:
                    continue

            filtered.append(market)

        filtered.sort(key=lambda m: m.liquidity, reverse=True)

        logger.info(f"Filtered to {len(filtered)} markets from {len(markets)}")
        return filtered

    async def search_markets(self, query: str) -> list[Market]:
        markets = await self.get_active_markets(limit=200)

        query_lower = query.lower()
        matching = [
            m for m in markets
            if query_lower in m.question.lower() or query_lower in m.description.lower()
        ]

        logger.info(f"Found {len(matching)} markets matching '{query}'")
        return matching

    async def close(self) -> None:
        await self.client.aclose()
        logger.info("PolymarketClient closed")

    async def __aenter__(self) -> "PolymarketClient":
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        await self.close()
