from pathlib import Path
from typing import Any

import chromadb
from chromadb.utils.embedding_functions import DefaultEmbeddingFunction
from loguru import logger


class ChromaClient:
    def __init__(self, persist_dir: Path, embedding_model: str) -> None:
        self.persist_dir = persist_dir
        self.persist_dir.mkdir(parents=True, exist_ok=True)

        self.client = chromadb.PersistentClient(path=str(persist_dir))
        self.embedding_function = DefaultEmbeddingFunction()

        logger.info(f"Connected to ChromaDB at {persist_dir}")

    def initialize_collections(self) -> None:
        self.client.get_or_create_collection(
            name="news",
            embedding_function=self.embedding_function,
            metadata={"description": "News articles and press releases"}
        )

        self.client.get_or_create_collection(
            name="social",
            embedding_function=self.embedding_function,
            metadata={"description": "Social media discussions"}
        )

        self.client.get_or_create_collection(
            name="market_context",
            embedding_function=self.embedding_function,
            metadata={"description": "Market-specific contextual data"}
        )

        self.client.get_or_create_collection(
            name="historical_forecasts",
            embedding_function=self.embedding_function,
            metadata={"description": "Historical forecast records"}
        )

        logger.info("ChromaDB collections initialized")

    def add_documents(
        self,
        collection_name: str,
        documents: list[str],
        metadatas: list[dict[str, Any]],
        ids: list[str]
    ) -> None:
        collection = self.client.get_collection(
            name=collection_name,
            embedding_function=self.embedding_function
        )

        collection.add(
            documents=documents,
            metadatas=metadatas,
            ids=ids
        )

        logger.debug(f"Added {len(documents)} documents to {collection_name}")

    def query(
        self,
        collection_name: str,
        query_text: str,
        n_results: int = 5,
        where: dict[str, Any] | None = None
    ) -> list[dict[str, Any]]:
        collection = self.client.get_collection(
            name=collection_name,
            embedding_function=self.embedding_function
        )

        results = collection.query(
            query_texts=[query_text],
            n_results=n_results,
            where=where
        )

        documents = []
        if results["documents"] and results["documents"][0]:
            for i, doc in enumerate(results["documents"][0]):
                documents.append({
                    "document": doc,
                    "metadata": results["metadatas"][0][i] if results["metadatas"] else {},
                    "id": results["ids"][0][i] if results["ids"] else "",
                    "distance": results["distances"][0][i] if results.get("distances") else 0.0,
                })

        return documents

    def delete_by_market(self, collection_name: str, market_id: str) -> None:
        collection = self.client.get_collection(
            name=collection_name,
            embedding_function=self.embedding_function
        )

        collection.delete(where={"market_id": market_id})
        logger.debug(f"Deleted documents for market {market_id} from {collection_name}")

    def get_collection_stats(self) -> dict[str, int]:
        collections = self.client.list_collections()
        stats = {}

        for collection_info in collections:
            collection = self.client.get_collection(
                name=collection_info.name,
                embedding_function=self.embedding_function
            )
            stats[collection_info.name] = collection.count()

        return stats

    def close(self) -> None:
        logger.info("ChromaDB client closed")

    def __enter__(self) -> "ChromaClient":
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.close()
