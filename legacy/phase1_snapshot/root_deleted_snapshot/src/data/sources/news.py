import asyncio
from datetime import datetime, timezone
from typing import Any

import feedparser
import httpx
from loguru import logger

from config.settings import get_settings
from src.models import Market, NewsItem
from src.utils.exceptions import DataFetchError


class SimpleSentiment:
    def __init__(self) -> None:
        self.positive_words = [
            "win", "wins", "won", "winning", "success", "successful", "gain", "gains",
            "rise", "rising", "increase", "increased", "positive", "good", "great",
            "best", "better", "strong", "stronger", "up", "breakthrough", "achieve",
            "achieved", "growth", "boom", "surge", "soar", "record", "high"
        ]

        self.negative_words = [
            "lose", "loss", "lost", "losing", "fail", "failed", "failure", "decline",
            "drop", "fall", "falling", "decrease", "decreased", "negative", "bad",
            "worse", "worst", "weak", "weaker", "down", "crash", "crisis", "plunge",
            "collapse", "risk", "threat", "low", "concern", "worry"
        ]

    def analyze(self, text: str) -> float:
        text_lower = text.lower()
        words = text_lower.split()

        positive_count = sum(1 for word in words if any(pos in word for pos in self.positive_words))
        negative_count = sum(1 for word in words if any(neg in word for neg in self.negative_words))

        total_count = positive_count + negative_count

        if total_count == 0:
            return 0.0

        sentiment = (positive_count - negative_count) / total_count
        return max(-1.0, min(1.0, sentiment))


class NewsClient:
    def __init__(self) -> None:
        self.settings = get_settings()
        self.client = httpx.AsyncClient(timeout=30.0, follow_redirects=True)
        self.sentiment_analyzer = SimpleSentiment()
        self.newsapi_requests_today = 0
        self.newsapi_limit = 100

        logger.info("NewsClient initialized")

    async def fetch_google_news(self, query: str, max_results: int = 10) -> list[NewsItem]:
        try:
            url = f"https://news.google.com/rss/search?q={query}&hl=en"

            logger.debug(f"Fetching Google News for query: {query}")
            response = await self.client.get(url)
            response.raise_for_status()

            feed = feedparser.parse(response.text)

            news_items = []
            for entry in feed.entries[:max_results]:
                try:
                    published_dt = datetime.now(timezone.utc)
                    if hasattr(entry, "published_parsed") and entry.published_parsed:
                        published_dt = datetime(*entry.published_parsed[:6], tzinfo=timezone.utc)

                    source = entry.get("source", {}).get("title", "Google News")
                    if isinstance(source, dict):
                        source = source.get("title", "Google News")

                    title = entry.get("title", "")
                    summary = entry.get("summary", entry.get("description", ""))

                    sentiment = self.sentiment_analyzer.analyze(f"{title} {summary}")

                    news_item = NewsItem(
                        title=title,
                        summary=summary,
                        source=source,
                        published_at=published_dt,
                        url=entry.get("link", ""),
                        sentiment=sentiment,
                    )
                    news_items.append(news_item)
                except Exception as e:
                    logger.warning(f"Failed to parse Google News entry: {e}")
                    continue

            logger.info(f"Fetched {len(news_items)} items from Google News")
            return news_items

        except Exception as e:
            logger.error(f"Failed to fetch Google News: {e}")
            raise DataFetchError(
                str(e),
                source="Google News",
                url=url if 'url' in locals() else ""
            )

    async def fetch_newsapi(self, query: str, max_results: int = 5) -> list[NewsItem]:
        if not self.settings.data.newsapi_key:
            logger.debug("NewsAPI key not configured, skipping")
            return []

        if self.newsapi_requests_today >= self.newsapi_limit:
            logger.warning("NewsAPI daily limit reached, skipping")
            return []

        try:
            url = "https://newsapi.org/v2/everything"
            params = {
                "q": query,
                "sortBy": "relevancy",
                "pageSize": max_results,
                "apiKey": self.settings.data.newsapi_key,
            }

            logger.debug(f"Fetching NewsAPI for query: {query}")
            response = await self.client.get(url, params=params)
            response.raise_for_status()

            data = response.json()

            self.newsapi_requests_today += 1

            if self.newsapi_requests_today >= self.newsapi_limit * 0.9:
                logger.warning(
                    f"NewsAPI approaching daily limit: {self.newsapi_requests_today}/{self.newsapi_limit}"
                )

            news_items = []
            for article in data.get("articles", []):
                try:
                    published_str = article.get("publishedAt", "")
                    published_dt = datetime.now(timezone.utc)
                    if published_str:
                        published_dt = datetime.fromisoformat(
                            published_str.replace("Z", "+00:00")
                        )

                    title = article.get("title", "")
                    description = article.get("description", "")
                    content = article.get("content", "")
                    summary = description or content[:500]

                    sentiment = self.sentiment_analyzer.analyze(f"{title} {summary}")

                    news_item = NewsItem(
                        title=title,
                        summary=summary,
                        source=article.get("source", {}).get("name", "NewsAPI"),
                        published_at=published_dt,
                        url=article.get("url", ""),
                        sentiment=sentiment,
                    )
                    news_items.append(news_item)
                except Exception as e:
                    logger.warning(f"Failed to parse NewsAPI article: {e}")
                    continue

            logger.info(f"Fetched {len(news_items)} items from NewsAPI")
            return news_items

        except Exception as e:
            logger.error(f"Failed to fetch NewsAPI: {e}")
            return []

    def _calculate_title_similarity(self, title1: str, title2: str) -> float:
        words1 = set(title1.lower().split())
        words2 = set(title2.lower().split())

        if not words1 or not words2:
            return 0.0

        intersection = words1.intersection(words2)
        union = words1.union(words2)

        return len(intersection) / len(union)

    async def search_news(self, query: str, max_results: int = 15) -> list[NewsItem]:
        google_task = asyncio.create_task(
            self.fetch_google_news(query, max_results=max_results)
        )
        newsapi_task = asyncio.create_task(
            self.fetch_newsapi(query, max_results=5)
        )

        google_news, newsapi_news = await asyncio.gather(
            google_task, newsapi_task, return_exceptions=True
        )

        if isinstance(google_news, Exception):
            logger.error(f"Google News failed: {google_news}")
            google_news = []

        if isinstance(newsapi_news, Exception):
            logger.error(f"NewsAPI failed: {newsapi_news}")
            newsapi_news = []

        all_news = list(google_news) + list(newsapi_news)

        deduplicated = []
        seen_titles = []

        for item in all_news:
            is_duplicate = False
            for seen_title in seen_titles:
                similarity = self._calculate_title_similarity(item.title, seen_title)
                if similarity > 0.7:
                    is_duplicate = True
                    break

            if not is_duplicate:
                deduplicated.append(item)
                seen_titles.append(item.title)

        deduplicated.sort(key=lambda x: x.published_at, reverse=True)

        logger.info(
            f"Search returned {len(deduplicated)} unique items from {len(all_news)} total"
        )

        return deduplicated[:max_results]

    def _extract_keywords(self, question: str) -> str:
        stopwords = {
            "will", "the", "a", "an", "is", "are", "was", "were", "be", "been",
            "being", "have", "has", "had", "do", "does", "did", "can", "could",
            "would", "should", "may", "might", "must", "shall", "of", "to", "in",
            "on", "at", "by", "for", "with", "from", "as", "or", "and", "but",
            "if", "than", "when", "where", "who", "which", "what", "how", "why",
            "his", "her", "their", "its", "this", "that", "these", "those"
        }

        words = question.lower().split()
        keywords = [w.strip("?.,!") for w in words if w.strip("?.,!") not in stopwords]

        return " ".join(keywords)

    async def get_market_news(self, market: Market, max_results: int = 10) -> list[NewsItem]:
        keywords = self._extract_keywords(market.question)

        logger.info(f"Searching news for market: {market.question[:50]}...")
        logger.debug(f"Extracted keywords: {keywords}")

        news_items = await self.search_news(keywords, max_results=max_results)

        filtered_items = []
        for item in news_items:
            relevance_score = 0.0

            title_lower = item.title.lower()
            question_words = market.question.lower().split()

            matches = sum(1 for word in question_words if word in title_lower and len(word) > 3)
            relevance_score = min(1.0, matches / max(5, len(question_words) * 0.3))

            item.relevance_score = relevance_score

            if relevance_score > 0.1:
                filtered_items.append(item)

        filtered_items.sort(key=lambda x: (x.relevance_score, x.published_at), reverse=True)

        logger.info(f"Found {len(filtered_items)} relevant items for market")

        return filtered_items[:max_results]

    async def close(self) -> None:
        await self.client.aclose()
        logger.info("NewsClient closed")

    async def __aenter__(self) -> "NewsClient":
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        await self.close()
