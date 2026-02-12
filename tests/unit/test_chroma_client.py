from pathlib import Path

import pytest

from src.data.storage.chroma_client import ChromaClient


@pytest.fixture
def chroma_client(tmp_path: Path) -> ChromaClient:
    persist_dir = tmp_path / "chroma"
    client = ChromaClient(persist_dir, "nomic-embed-text")
    client.initialize_collections()
    yield client
    client.close()


def test_initialize_collections(tmp_path: Path) -> None:
    persist_dir = tmp_path / "chroma"
    with ChromaClient(persist_dir, "nomic-embed-text") as client:
        client.initialize_collections()
        stats = client.get_collection_stats()
        assert "news" in stats
        assert "social" in stats
        assert "market_context" in stats
        assert "historical_forecasts" in stats


def test_add_documents(chroma_client: ChromaClient) -> None:
    documents = [
        "Breaking news about the election",
        "Market analysis suggests bullish trend",
    ]
    metadatas = [
        {"source": "NewsAPI", "date": "2026-02-12", "market_id": "market_123"},
        {"source": "Analysis", "date": "2026-02-12", "market_id": "market_123"},
    ]
    ids = ["news_1", "news_2"]

    chroma_client.add_documents("news", documents, metadatas, ids)

    stats = chroma_client.get_collection_stats()
    assert stats["news"] == 2


def test_query_documents(chroma_client: ChromaClient) -> None:
    documents = [
        "Bitcoin price reached new all-time high",
        "Election polls show tight race",
        "Stock market sees major gains",
    ]
    metadatas = [
        {"source": "NewsAPI", "date": "2026-02-12", "market_id": "crypto_1"},
        {"source": "NewsAPI", "date": "2026-02-12", "market_id": "politics_1"},
        {"source": "NewsAPI", "date": "2026-02-12", "market_id": "finance_1"},
    ]
    ids = ["news_1", "news_2", "news_3"]

    chroma_client.add_documents("news", documents, metadatas, ids)

    results = chroma_client.query("news", "cryptocurrency prices", n_results=2)
    assert len(results) <= 2
    assert len(results) > 0
    assert "document" in results[0]
    assert "metadata" in results[0]


def test_query_with_filter(chroma_client: ChromaClient) -> None:
    documents = [
        "Market A shows bullish signals",
        "Market B shows bearish signals",
        "Market A continues upward trend",
    ]
    metadatas = [
        {"market_id": "market_A", "date": "2026-02-12"},
        {"market_id": "market_B", "date": "2026-02-12"},
        {"market_id": "market_A", "date": "2026-02-13"},
    ]
    ids = ["ctx_1", "ctx_2", "ctx_3"]

    chroma_client.add_documents("market_context", documents, metadatas, ids)

    results = chroma_client.query(
        "market_context",
        "market trends",
        n_results=5,
        where={"market_id": "market_A"}
    )

    assert len(results) == 2
    for result in results:
        assert result["metadata"]["market_id"] == "market_A"


def test_delete_by_market(chroma_client: ChromaClient) -> None:
    documents = [
        "Document for market A",
        "Document for market B",
        "Another document for market A",
    ]
    metadatas = [
        {"market_id": "market_A"},
        {"market_id": "market_B"},
        {"market_id": "market_A"},
    ]
    ids = ["doc_1", "doc_2", "doc_3"]

    chroma_client.add_documents("market_context", documents, metadatas, ids)

    stats_before = chroma_client.get_collection_stats()
    assert stats_before["market_context"] == 3

    chroma_client.delete_by_market("market_context", "market_A")

    stats_after = chroma_client.get_collection_stats()
    assert stats_after["market_context"] == 1


def test_get_collection_stats(chroma_client: ChromaClient) -> None:
    chroma_client.add_documents(
        "news",
        ["News 1", "News 2"],
        [{"source": "test"}, {"source": "test"}],
        ["n1", "n2"]
    )

    chroma_client.add_documents(
        "social",
        ["Post 1"],
        [{"platform": "hn"}],
        ["s1"]
    )

    stats = chroma_client.get_collection_stats()
    assert stats["news"] == 2
    assert stats["social"] == 1
    assert stats["market_context"] == 0
    assert stats["historical_forecasts"] == 0
