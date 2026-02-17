from datetime import datetime, timezone
from typing import Any

from loguru import logger

from src.data.sources.polymarket import PolymarketClient
from src.data.sources.news import NewsClient
from src.data.storage.chroma_client import ChromaClient
from src.models import Market


class ContextBuilder:
    def __init__(
        self,
        polymarket_client: PolymarketClient,
        news_client: NewsClient,
        chroma_client: ChromaClient,
    ) -> None:
        self.polymarket_client = polymarket_client
        self.news_client = news_client
        self.chroma_client = chroma_client

        logger.info("ContextBuilder initialized")

    async def build_context(self, market: Market) -> str:
        logger.info(f"Building context for market: {market.id}")

        news_items = await self.news_client.get_market_news(market, max_results=5)
        logger.debug(f"Retrieved {len(news_items)} news items")

        similar_markets = self._query_similar_markets(market)
        logger.debug(f"Found {len(similar_markets)} similar historical markets")

        entities = self._extract_entities(news_items)
        logger.debug(f"Extracted {len(entities)} entities")

        context = self._format_context(
            market=market,
            news_items=news_items,
            similar_markets=similar_markets,
            entities=entities,
        )

        self._store_in_chromadb(market, news_items, context)

        logger.info(f"Context built for market {market.id}")
        return context

    def _query_similar_markets(self, market: Market) -> list[dict[str, Any]]:
        try:
            results = self.chroma_client.query(
                collection_name="market_context",
                query_text=market.question,
                n_results=3,
                where={"resolved": True} if hasattr(self.chroma_client, "query_with_metadata") else None,
            )

            similar = []
            for i, doc in enumerate(results):
                metadata = doc.get("metadata", {})
                similar.append({
                    "question": metadata.get("question", "Unknown"),
                    "outcome": metadata.get("outcome", "Unknown"),
                    "final_price": metadata.get("final_price", 0.5),
                })

            return similar
        except Exception as e:
            logger.warning(f"Failed to query similar markets: {e}")
            return []

    def _extract_entities(self, news_items: list[Any]) -> list[str]:
        entities = set()

        for item in news_items:
            if hasattr(item, "entities") and item.entities:
                entities.update(item.entities)

        for item in news_items:
            title_words = item.title.split()
            for word in title_words:
                if len(word) > 3 and word[0].isupper():
                    entities.add(word.strip(".,!?"))

        return sorted(list(entities))[:10]

    def _format_context(
        self,
        market: Market,
        news_items: list[Any],
        similar_markets: list[dict[str, Any]],
        entities: list[str],
    ) -> str:
        context_parts = []

        context_parts.append("# Market Analysis Context")
        context_parts.append("")

        context_parts.append("## Question")
        context_parts.append(market.question)
        context_parts.append("")

        if market.description:
            context_parts.append("## Description")
            context_parts.append(market.description[:500])
            context_parts.append("")

        context_parts.append("## Current Market State")
        context_parts.append(f"- Current Price (Market P(YES)): {market.current_price:.1%}")
        context_parts.append(f"- Volume 24h: ${market.volume_24h:,.0f}")
        context_parts.append(f"- Liquidity: ${market.liquidity:,.0f}")
        context_parts.append(f"- Resolution Date: {market.resolution_date.strftime('%Y-%m-%d')}")
        context_parts.append(f"- Days Remaining: {market.days_until_resolution:.0f}")
        context_parts.append("")

        if news_items:
            context_parts.append("## Recent News")
            for item in news_items[:5]:
                date_str = item.published_at.strftime("%Y-%m-%d")
                sentiment_str = "Positive" if item.sentiment > 0.2 else "Negative" if item.sentiment < -0.2 else "Neutral"
                context_parts.append(
                    f"- [{item.source}] {item.title} ({date_str}) - Sentiment: {sentiment_str}"
                )
            context_parts.append("")

        if similar_markets:
            context_parts.append("## Historical Similar Markets")
            for market_data in similar_markets:
                context_parts.append(
                    f"- Question: {market_data['question']}"
                )
                context_parts.append(
                    f"  Outcome: {market_data['outcome']}, Final Price: {market_data['final_price']:.1%}"
                )
            context_parts.append("")

        if entities:
            context_parts.append("## Key Entities")
            context_parts.append(", ".join(entities))
            context_parts.append("")

        return "\n".join(context_parts)

    def _store_in_chromadb(
        self,
        market: Market,
        news_items: list[Any],
        context: str,
    ) -> None:
        try:
            timestamp = datetime.now(timezone.utc).isoformat()

            if news_items:
                news_docs = []
                news_metadatas = []
                news_ids = []

                for i, item in enumerate(news_items):
                    news_docs.append(f"{item.title}\n\n{item.summary}")
                    news_metadatas.append({
                        "market_id": market.id,
                        "source": item.source,
                        "url": item.url,
                        "published_at": item.published_at.isoformat(),
                        "sentiment": item.sentiment,
                        "relevance_score": item.relevance_score,
                    })
                    news_ids.append(f"{market.id}_news_{i}_{timestamp}")

                self.chroma_client.add_documents(
                    collection_name="news",
                    documents=news_docs,
                    metadatas=news_metadatas,
                    ids=news_ids,
                )
                logger.debug(f"Stored {len(news_items)} news items in ChromaDB")

            context_doc = [context]
            context_metadata = [{
                "market_id": market.id,
                "question": market.question,
                "timestamp": timestamp,
                "price": market.current_price,
                "resolved": False,
            }]
            context_id = [f"{market.id}_context_{timestamp}"]

            self.chroma_client.add_documents(
                collection_name="market_context",
                documents=context_doc,
                metadatas=context_metadata,
                ids=context_id,
            )
            logger.debug("Stored market context in ChromaDB")

        except Exception as e:
            logger.error(f"Failed to store in ChromaDB: {e}")
