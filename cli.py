import asyncio
import httpx
import typer
from loguru import logger

from config.settings import get_settings
from src.utils.logging import setup_logging
from src.utils.exceptions import ConfigError, DataFetchError, MarketNotFoundError
from src.data.storage.duckdb_client import DuckDBClient
from src.data.storage.sqlite_client import SQLiteClient
from src.data.storage.chroma_client import ChromaClient
from src.data.sources.polymarket import PolymarketClient
from src.data.sources.news import NewsClient
from src.models import MarketFilter

app = typer.Typer(no_args_is_help=True)


@app.command()
def init() -> None:
    """Initialize Poly-Oracle directories and verify setup."""
    try:
        settings = get_settings()
        setup_logging(settings.log_level, settings.database.db_dir / "logs")

        settings.database.db_dir.mkdir(parents=True, exist_ok=True)
        (settings.database.db_dir / "logs").mkdir(parents=True, exist_ok=True)
        settings.database.chroma_path.mkdir(parents=True, exist_ok=True)

        typer.echo(f"Created directory: {settings.database.db_dir}")
        typer.echo(f"Created directory: {settings.database.db_dir / 'logs'}")
        typer.echo(f"Created directory: {settings.database.chroma_path}")

        with DuckDBClient(settings.database.duckdb_path) as duckdb_client:
            duckdb_client.initialize_schema()
            typer.echo("Initialized DuckDB schema")

        with SQLiteClient(settings.database.sqlite_path) as sqlite_client:
            sqlite_client.initialize_schema()
            typer.echo("Initialized SQLite schema")

        with ChromaClient(settings.database.chroma_path, settings.llm.embedding_model) as chroma_client:
            chroma_client.initialize_collections()
            typer.echo("Initialized ChromaDB collections")

        typer.echo("Poly-Oracle initialized successfully")
        logger.info("Poly-Oracle directories and databases initialized")

    except Exception as e:
        typer.echo(f"Initialization failed: {e}", err=True)
        raise typer.Exit(code=1)


@app.command()
def status() -> None:
    """Check system status and configuration."""
    try:
        settings = get_settings()
        setup_logging(settings.log_level, settings.database.db_dir / "logs")

        typer.echo("Poly-Oracle System Status")
        typer.echo("=" * 50)

        if not settings.database.db_dir.exists():
            typer.echo(f"Database directory: MISSING (run 'init' first)")
            raise ConfigError(f"Database directory {settings.database.db_dir} does not exist")
        else:
            typer.echo(f"Database directory: {settings.database.db_dir}")

        typer.echo(f"DuckDB path: {settings.database.duckdb_path}")
        typer.echo(f"SQLite path: {settings.database.sqlite_path}")
        typer.echo(f"ChromaDB path: {settings.database.chroma_path}")
        typer.echo("")

        ollama_status = "OFFLINE"
        try:
            response = httpx.get(
                f"{settings.llm.base_url}/api/tags",
                timeout=5.0,
            )
            if response.status_code == 200:
                ollama_status = "ONLINE"
        except Exception:
            pass

        typer.echo(f"Ollama status: {ollama_status}")
        typer.echo(f"Ollama URL: {settings.llm.base_url}")
        typer.echo(f"Model: {settings.llm.model}")
        typer.echo(f"Embedding model: {settings.llm.embedding_model}")
        typer.echo("")

        mode = "PAPER TRADING" if settings.paper_trading else "LIVE TRADING"
        typer.echo(f"Trading mode: {mode}")
        typer.echo(f"Initial bankroll: ${settings.risk.initial_bankroll:.2f}")
        typer.echo(f"Max position: {settings.risk.max_position_pct * 100:.0f}%")
        typer.echo(f"Bet range: ${settings.risk.min_bet:.2f} - ${settings.risk.max_bet:.2f}")
        typer.echo(f"Min edge: {settings.risk.min_edge * 100:.0f}%")
        typer.echo(f"Min confidence: {settings.risk.min_confidence * 100:.0f}%")
        typer.echo("")

        has_newsapi = "YES" if settings.data.newsapi_key else "NO"
        typer.echo(f"NewsAPI configured: {has_newsapi}")

        has_polymarket = "YES" if all([
            settings.polymarket.api_key,
            settings.polymarket.api_secret,
            settings.polymarket.api_passphrase
        ]) else "NO"
        typer.echo(f"Polymarket API configured: {has_polymarket}")
        typer.echo("")

        typer.echo("Database Status:")
        duckdb_exists = settings.database.duckdb_path.exists()
        typer.echo(f"DuckDB: {'EXISTS' if duckdb_exists else 'NOT FOUND'}")

        if duckdb_exists:
            try:
                with DuckDBClient(settings.database.duckdb_path) as duckdb_client:
                    stats = duckdb_client.get_calibration_stats()
                    typer.echo(f"  Forecasts: {stats['overall']['count']}")
            except Exception:
                typer.echo(f"  Forecasts: ERROR")

        sqlite_exists = settings.database.sqlite_path.exists()
        typer.echo(f"SQLite: {'EXISTS' if sqlite_exists else 'NOT FOUND'}")

        if sqlite_exists:
            try:
                with SQLiteClient(settings.database.sqlite_path) as sqlite_client:
                    positions = sqlite_client.get_open_positions()
                    typer.echo(f"  Open positions: {len(positions)}")
            except Exception:
                typer.echo(f"  Open positions: ERROR")

        chroma_exists = settings.database.chroma_path.exists()
        typer.echo(f"ChromaDB: {'EXISTS' if chroma_exists else 'NOT FOUND'}")

        if chroma_exists:
            try:
                with ChromaClient(settings.database.chroma_path, settings.llm.embedding_model) as chroma_client:
                    stats = chroma_client.get_collection_stats()
                    total_docs = sum(stats.values())
                    typer.echo(f"  Total documents: {total_docs}")
                    for collection, count in stats.items():
                        typer.echo(f"    {collection}: {count}")
            except Exception:
                typer.echo(f"  Total documents: ERROR")

        logger.info("Status check completed")

    except ConfigError as e:
        typer.echo(f"Configuration error: {e}", err=True)
        raise typer.Exit(code=1)
    except Exception as e:
        typer.echo(f"Status check failed: {e}", err=True)
        raise typer.Exit(code=1)


@app.command()
def markets(
    limit: int = typer.Option(20, help="Number of markets to fetch"),
    min_liquidity: float = typer.Option(1000, help="Minimum liquidity filter"),
    market_type: str = typer.Option(None, help="Filter by market type"),
) -> None:
    """List active Polymarket markets."""
    try:
        settings = get_settings()
        setup_logging(settings.log_level, settings.database.db_dir / "logs")

        async def fetch_markets() -> None:
            async with PolymarketClient() as client:
                filter_obj = MarketFilter(
                    min_liquidity=min_liquidity,
                    market_types=[market_type] if market_type else None,
                )

                market_list = await client.filter_markets(filter_obj)
                market_list = market_list[:limit]

                if not market_list:
                    typer.echo("No markets found matching criteria")
                    return

                typer.echo(f"\nActive Markets (showing {len(market_list)}):")
                typer.echo("=" * 120)
                typer.echo(
                    f"{'Question':<50} | {'Price':<8} | {'Volume':<12} | {'Liquidity':<12} | {'Days Left':<10}"
                )
                typer.echo("=" * 120)

                for market in market_list:
                    question = market.question[:47] + "..." if len(market.question) > 50 else market.question
                    price_str = f"{market.current_price:.3f}"
                    volume_str = f"${market.volume_24h:,.0f}"
                    liquidity_str = f"${market.liquidity:,.0f}"
                    days_str = f"{market.days_until_resolution:.1f}"

                    typer.echo(
                        f"{question:<50} | {price_str:<8} | {volume_str:<12} | {liquidity_str:<12} | {days_str:<10}"
                    )

        asyncio.run(fetch_markets())

    except DataFetchError as e:
        typer.echo(f"Failed to fetch markets: {e}", err=True)
        raise typer.Exit(code=1)
    except Exception as e:
        typer.echo(f"Error: {e}", err=True)
        logger.exception("Markets command failed")
        raise typer.Exit(code=1)


@app.command()
def market(market_id: str) -> None:
    """Show details for a specific market."""
    try:
        settings = get_settings()
        setup_logging(settings.log_level, settings.database.db_dir / "logs")

        async def fetch_market() -> None:
            async with PolymarketClient() as client:
                try:
                    m = await client.get_market(market_id)

                    typer.echo("\nMarket Details:")
                    typer.echo("=" * 80)
                    typer.echo(f"ID: {m.id}")
                    typer.echo(f"Question: {m.question}")
                    typer.echo(f"Description: {m.description}")
                    typer.echo(f"Type: {m.market_type}")
                    typer.echo("")
                    typer.echo(f"Current Price: {m.current_price:.3f}")
                    typer.echo(f"24h Volume: ${m.volume_24h:,.2f}")
                    typer.echo(f"Liquidity: ${m.liquidity:,.2f}")
                    typer.echo("")
                    typer.echo(f"Created: {m.created_at.strftime('%Y-%m-%d %H:%M UTC')}")
                    typer.echo(f"Resolves: {m.resolution_date.strftime('%Y-%m-%d %H:%M UTC')}")
                    typer.echo(f"Days until resolution: {m.days_until_resolution:.1f}")
                    typer.echo("")
                    typer.echo(f"Outcomes: {', '.join(m.outcomes)}")

                except MarketNotFoundError:
                    typer.echo(f"Market {market_id} not found", err=True)
                    raise typer.Exit(code=1)

        asyncio.run(fetch_market())

    except DataFetchError as e:
        typer.echo(f"Failed to fetch market: {e}", err=True)
        raise typer.Exit(code=1)
    except typer.Exit:
        raise
    except Exception as e:
        typer.echo(f"Error: {e}", err=True)
        logger.exception("Market command failed")
        raise typer.Exit(code=1)


@app.command()
def news(
    query: str = typer.Argument(..., help="Search query for news"),
    limit: int = typer.Option(10, help="Number of news items to fetch"),
) -> None:
    """Search for news articles."""
    try:
        settings = get_settings()
        setup_logging(settings.log_level, settings.database.db_dir / "logs")

        async def fetch_news() -> None:
            async with NewsClient() as client:
                news_items = await client.search_news(query, max_results=limit)

                if not news_items:
                    typer.echo("No news found matching query")
                    return

                typer.echo(f"\nNews Results (showing {len(news_items)}):")
                typer.echo("=" * 120)
                typer.echo(
                    f"{'Date':<20} | {'Source':<25} | {'Sentiment':<10} | {'Title':<50}"
                )
                typer.echo("=" * 120)

                for item in news_items:
                    date_str = item.published_at.strftime("%Y-%m-%d %H:%M")
                    source = item.source[:22] + "..." if len(item.source) > 25 else item.source
                    sentiment_str = f"{item.sentiment:+.2f}"

                    title = item.title[:47] + "..." if len(item.title) > 50 else item.title

                    typer.echo(
                        f"{date_str:<20} | {source:<25} | {sentiment_str:<10} | {title:<50}"
                    )

        asyncio.run(fetch_news())

    except DataFetchError as e:
        typer.echo(f"Failed to fetch news: {e}", err=True)
        raise typer.Exit(code=1)
    except Exception as e:
        typer.echo(f"Error: {e}", err=True)
        logger.exception("News command failed")
        raise typer.Exit(code=1)


@app.command()
def market_news(market_id: str) -> None:
    """Get news relevant to a specific market."""
    try:
        settings = get_settings()
        setup_logging(settings.log_level, settings.database.db_dir / "logs")

        async def fetch_market_news() -> None:
            async with PolymarketClient() as poly_client:
                market = await poly_client.get_market(market_id)

                typer.echo(f"\nMarket: {market.question}")
                typer.echo("=" * 120)

                async with NewsClient() as news_client:
                    news_items = await news_client.get_market_news(market, max_results=10)

                    if not news_items:
                        typer.echo("No relevant news found for this market")
                        return

                    typer.echo(f"\nRelevant News (showing {len(news_items)}):")
                    typer.echo("=" * 120)
                    typer.echo(
                        f"{'Date':<20} | {'Source':<25} | {'Relevance':<10} | {'Sentiment':<10} | {'Title':<40}"
                    )
                    typer.echo("=" * 120)

                    for item in news_items:
                        date_str = item.published_at.strftime("%Y-%m-%d %H:%M")
                        source = item.source[:22] + "..." if len(item.source) > 25 else item.source
                        relevance_str = f"{item.relevance_score:.2f}"
                        sentiment_str = f"{item.sentiment:+.2f}"
                        title = item.title[:37] + "..." if len(item.title) > 40 else item.title

                        typer.echo(
                            f"{date_str:<20} | {source:<25} | {relevance_str:<10} | {sentiment_str:<10} | {title:<40}"
                        )

        asyncio.run(fetch_market_news())

    except MarketNotFoundError:
        typer.echo(f"Market {market_id} not found", err=True)
        raise typer.Exit(code=1)
    except DataFetchError as e:
        typer.echo(f"Failed to fetch data: {e}", err=True)
        raise typer.Exit(code=1)
    except Exception as e:
        typer.echo(f"Error: {e}", err=True)
        logger.exception("Market news command failed")
        raise typer.Exit(code=1)


@app.command()
def context() -> None:
    """Build context for a market."""
    typer.echo("Command not yet implemented")


@app.command()
def forecast() -> None:
    """Generate forecast for a market."""
    typer.echo("Command not yet implemented")


@app.command()
def backtest() -> None:
    """Run historical calibration."""
    typer.echo("Command not yet implemented")


@app.command()
def paper() -> None:
    """Run in paper trading mode."""
    typer.echo("Command not yet implemented")


@app.command()
def live() -> None:
    """Run in live trading mode."""
    typer.echo("Command not yet implemented")


@app.command()
def positions() -> None:
    """View open positions."""
    typer.echo("Command not yet implemented")


if __name__ == "__main__":
    app()
