import asyncio
import httpx
import typer
from datetime import datetime
from loguru import logger

from config.settings import get_settings
from src.utils.logging import setup_logging
from src.utils.exceptions import ConfigError, DataFetchError, MarketNotFoundError
from src.data.storage.duckdb_client import DuckDBClient
from src.data.storage.sqlite_client import SQLiteClient
from src.data.storage.chroma_client import ChromaClient
from src.data.sources.polymarket import PolymarketClient
from src.data.sources.news import NewsClient
from src.data.context import ContextBuilder
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
        model_available = False
        available_models = []

        try:
            response = httpx.get(
                f"{settings.llm.base_url}/api/tags",
                timeout=5.0,
            )
            if response.status_code == 200:
                ollama_status = "ONLINE"
                data = response.json()
                models = data.get("models", [])
                available_models = [m.get("name", "") for m in models]

                for model_name in available_models:
                    if settings.llm.model in model_name or model_name.startswith(settings.llm.model):
                        model_available = True
                        break
        except Exception:
            pass

        typer.echo(f"Ollama status: {ollama_status}")
        typer.echo(f"Ollama URL: {settings.llm.base_url}")
        typer.echo(f"Model: {settings.llm.model} - {'AVAILABLE' if model_available else 'NOT FOUND'}")
        if available_models and not model_available:
            typer.echo(f"  Available models: {', '.join(available_models[:3])}")
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
def context(market_id: str = typer.Argument(..., help="Market ID to build context for")) -> None:
    """Build context for a market."""
    try:
        settings = get_settings()
        setup_logging(settings.log_level, settings.database.db_dir / "logs")

        async def build_and_display_context() -> None:
            async with PolymarketClient() as poly_client, \
                       NewsClient() as news_client:

                market = await poly_client.get_market(market_id)

                typer.echo(f"\nBuilding context for market: {market.id}")
                typer.echo("=" * 80)

                with ChromaClient(settings.database.chroma_path, settings.llm.embedding_model) as chroma_client:
                    builder = ContextBuilder(
                        polymarket_client=poly_client,
                        news_client=news_client,
                        chroma_client=chroma_client,
                    )

                    context_text = await builder.build_context(market)

                typer.echo(context_text)
                typer.echo("=" * 80)
                typer.echo(f"\nContext built and stored in ChromaDB")

        asyncio.run(build_and_display_context())

    except MarketNotFoundError:
        typer.echo(f"Market {market_id} not found", err=True)
        raise typer.Exit(code=1)
    except DataFetchError as e:
        typer.echo(f"Failed to fetch data: {e}", err=True)
        raise typer.Exit(code=1)
    except Exception as e:
        typer.echo(f"Error: {e}", err=True)
        logger.exception("Context command failed")
        raise typer.Exit(code=1)


@app.command()
def forecast(
    market_id: str = typer.Argument(..., help="Market ID to forecast"),
    rounds: int = typer.Option(2, "--rounds", "-r", help="Number of debate rounds"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Show detailed debate output"),
    temperature: float = typer.Option(0.7, "--temperature", "-t", help="LLM temperature (0.0-1.0)"),
) -> None:
    """
    Generate forecast for a market using multi-agent debate.

    Example:
        poly-oracle forecast 0x123abc --rounds 2 --verbose
    """
    from src.agents import create_debate_system
    from src.data.sources.polymarket import PolymarketClient
    from src.data.sources.news import NewsClient
    from src.data.context import ContextBuilder
    from src.data.storage.chroma_client import ChromaClient
    from src.data.storage.duckdb_client import DuckDBClient
    from src.calibration import CalibratorAgent, MetaAnalyzer, FeedbackLoop
    from src.utils.exceptions import MarketNotFoundError, DataFetchError

    try:
        settings = get_settings()

        async def run_forecast() -> None:
            # Check Ollama availability first
            typer.echo("Checking Ollama availability...")
            from src.agents.base import OllamaClient

            async with OllamaClient(
                base_url=settings.llm.base_url,
                model=settings.llm.model,
                timeout=120
            ) as test_ollama:
                is_available = await test_ollama.is_available()

                if not is_available:
                    typer.echo(f"\nError: Ollama model '{settings.llm.model}' is not available.", err=True)
                    typer.echo("\nPlease ensure:")
                    typer.echo(f"1. Ollama is running (ollama serve)")
                    typer.echo(f"2. Model is pulled (ollama pull {settings.llm.model})")
                    raise typer.Exit(code=1)

            typer.echo(f"✓ Ollama model '{settings.llm.model}' is ready\n")

            # Fetch market and build context
            typer.echo(f"Fetching market {market_id}...")
            async with PolymarketClient() as poly_client, \
                       NewsClient() as news_client:

                market = await poly_client.get_market(market_id)

                typer.echo(f"Market: {market.question}")
                typer.echo(f"Current Price: {market.current_price:.1%}")
                typer.echo("\nBuilding context...")

                with ChromaClient(settings.database.chroma_path, settings.llm.embedding_model) as chroma_client:
                    builder = ContextBuilder(
                        polymarket_client=poly_client,
                        news_client=news_client,
                        chroma_client=chroma_client,
                    )

                    context_text = await builder.build_context(market)

                typer.echo("✓ Context ready\n")

                # Create debate system and run forecast
                typer.echo(f"Starting {rounds}-round debate...\n")
                typer.echo("=" * 80)

                orchestrator, ollama = create_debate_system(
                    base_url=settings.llm.base_url,
                    model=settings.llm.model,
                    timeout=120,
                )

                try:
                    forecast_result = await orchestrator.run_debate(
                        market_id=market_id,
                        context=context_text,
                        rounds=rounds,
                        temperature=temperature,
                        verbose=verbose,
                    )

                    # Initialize calibration system
                    with DuckDBClient(settings.database.duckdb_path) as duckdb_client:
                        calibrator = CalibratorAgent(history_db=duckdb_client)
                        analyzer = MetaAnalyzer(
                            min_edge=settings.risk.min_edge,
                            min_confidence=0.6,  # Could be configurable
                            min_liquidity=settings.risk.min_liquidity,
                        )
                        feedback = FeedbackLoop(db=duckdb_client, calibrator=calibrator)

                        # Calculate confidence from interval if available
                        confidence = 0.5
                        if forecast_result.confidence_lower and forecast_result.confidence_upper:
                            confidence = 1.0 - (forecast_result.confidence_upper - forecast_result.confidence_lower)

                        # Calibrate forecast
                        calibrated = calibrator.calibrate(
                            raw_forecast=forecast_result.probability,
                            market_type=market.market_type,
                            confidence=confidence,
                        )

                        # Analyze edge
                        edge_analysis = analyzer.analyze(
                            our_forecast=calibrated,
                            market_price=market.current_price,
                            liquidity=market.liquidity,
                        )

                        # Record forecast in database
                        feedback.record_forecast(
                            forecast=forecast_result,
                            market=market,
                            calibrated_probability=calibrated.calibrated,
                            edge=edge_analysis.raw_edge,
                            recommended_action=edge_analysis.recommended_action,
                        )

                    # Display results
                    typer.echo("\n" + "=" * 80)
                    typer.echo("FORECAST COMPLETE")
                    typer.echo("=" * 80)
                    typer.echo(f"\nMarket: {market.question}")
                    typer.echo(f"Market Price: {market.current_price:.1%}")
                    typer.echo(f"Raw P(YES): {forecast_result.probability:.1%}")
                    typer.echo(f"Calibrated P(YES): {calibrated.calibrated:.1%}")

                    if calibrated.calibration_method != "identity":
                        adjustment = calibrated.calibrated - calibrated.raw
                        typer.echo(f"Calibration adjustment: {adjustment:+.1%} (based on {calibrated.historical_samples} samples)")

                    if forecast_result.confidence_lower and forecast_result.confidence_upper:
                        typer.echo(
                            f"Confidence Interval: {forecast_result.confidence_lower:.1%} to "
                            f"{forecast_result.confidence_upper:.1%}"
                        )

                    typer.echo(f"\nEdge Analysis:")
                    typer.echo(f"  Raw Edge: {edge_analysis.raw_edge:+.1%}")
                    typer.echo(f"  Absolute Edge: {edge_analysis.abs_edge:.1%}")
                    typer.echo(f"  Direction: {edge_analysis.direction}")
                    typer.echo(f"  Recommendation: {edge_analysis.recommended_action}")

                    typer.echo(f"\nDebate Rounds: {forecast_result.debate_rounds}")
                    typer.echo(f"Model: {forecast_result.model_name}")
                    typer.echo(f"Timestamp: {forecast_result.created_at.strftime('%Y-%m-%d %H:%M:%S UTC')}")

                    # Show probability evolution
                    if forecast_result.bull_probabilities or forecast_result.bear_probabilities:
                        typer.echo("\nProbability Evolution:")
                        for i, (bull_p, bear_p) in enumerate(
                            zip(forecast_result.bull_probabilities, forecast_result.bear_probabilities), 1
                        ):
                            typer.echo(f"  Round {i}: Bull={bull_p:.1%}, Bear={bear_p:.1%}")

                    typer.echo("\n" + "=" * 80)
                    typer.echo("EDGE ANALYSIS REASONING")
                    typer.echo("=" * 80)
                    typer.echo(edge_analysis.reasoning)

                    typer.echo("\n" + "=" * 80)
                    typer.echo("JUDGE'S REASONING")
                    typer.echo("=" * 80)
                    typer.echo(forecast_result.reasoning)
                    typer.echo("=" * 80)

                finally:
                    await ollama.close()

        asyncio.run(run_forecast())

    except MarketNotFoundError:
        typer.echo(f"Market {market_id} not found", err=True)
        raise typer.Exit(code=1)
    except DataFetchError as e:
        typer.echo(f"Failed to fetch data: {e}", err=True)
        raise typer.Exit(code=1)
    except Exception as e:
        typer.echo(f"Error: {e}", err=True)
        logger.exception("Forecast command failed")
        raise typer.Exit(code=1)


@app.command()
def calibration() -> None:
    """
    Show calibration performance report.

    Displays Brier scores, calibration curves, and performance metrics
    based on resolved forecasts.
    """
    from src.data.storage.duckdb_client import DuckDBClient
    from src.calibration import CalibratorAgent

    try:
        settings = get_settings()

        with DuckDBClient(settings.database.duckdb_path) as duckdb_client:
            calibrator = CalibratorAgent(history_db=duckdb_client)
            report = calibrator.get_calibration_report()

        if report.get("resolved_forecasts", 0) == 0:
            typer.echo("No resolved forecasts yet.")
            typer.echo("\nTo build calibration data:")
            typer.echo("1. Run forecasts with: poly-oracle forecast <market_id>")
            typer.echo("2. Wait for markets to resolve or use: poly-oracle resolve <market_id> <yes|no>")
            return

        typer.echo("=" * 80)
        typer.echo("CALIBRATION REPORT")
        typer.echo("=" * 80)
        typer.echo(f"\nTotal Forecasts: {report['total_forecasts']}")
        typer.echo(f"Resolved Forecasts: {report['resolved_forecasts']}")
        typer.echo(f"Pending Forecasts: {report['total_forecasts'] - report['resolved_forecasts']}")

        typer.echo(f"\nOverall Performance:")
        if report.get("brier_score_raw"):
            typer.echo(f"  Brier Score (Raw): {report['brier_score_raw']:.4f}")
        if report.get("brier_score_calibrated"):
            typer.echo(f"  Brier Score (Calibrated): {report['brier_score_calibrated']:.4f}")
        if report.get("improvement"):
            typer.echo(f"  Calibration Improvement: {report['improvement']:.4f}")

        if report.get("brier_by_type"):
            typer.echo(f"\nPerformance by Market Type:")
            for mtype, metrics in report["brier_by_type"].items():
                typer.echo(f"\n  {mtype}:")
                typer.echo(f"    Count: {metrics['count']}")
                typer.echo(f"    Brier (Raw): {metrics['brier_raw']:.4f}")
                typer.echo(f"    Brier (Calibrated): {metrics['brier_calibrated']:.4f}")
                typer.echo(f"    Improvement: {metrics['improvement']:.4f}")

        if report.get("calibration_curve"):
            typer.echo(f"\nCalibration Curve:")
            typer.echo(f"  {'Predicted':<12} {'Actual':<12} {'Count':<8}")
            typer.echo("  " + "-" * 32)
            for point in report["calibration_curve"]:
                typer.echo(
                    f"  {point['predicted_prob']:.1%}          "
                    f"{point['actual_freq']:.1%}          "
                    f"{point['count']}"
                )

        typer.echo("\n" + "=" * 80)

    except Exception as e:
        typer.echo(f"Error: {e}", err=True)
        logger.exception("Calibration command failed")
        raise typer.Exit(code=1)


# Resolution is now fully automatic via the paper trading loop
# Keeping this for manual debugging if needed
# @app.command()
def _manual_resolve(
    market_id: str = typer.Argument(..., help="Market ID to resolve"),
    outcome: str = typer.Argument(..., help="Outcome: 'yes' or 'no'"),
) -> None:
    """
    Manually resolve a market outcome (for debugging only).

    Resolution is now automatic via the paper trading loop.
    Use this only for manual testing/debugging.

    Example:
        python cli.py _manual_resolve 0x123abc yes
    """
    from src.data.storage.duckdb_client import DuckDBClient
    from src.calibration import CalibratorAgent, FeedbackLoop

    try:
        settings = get_settings()

        # Parse outcome
        outcome_lower = outcome.lower()
        if outcome_lower not in ["yes", "no"]:
            typer.echo("Outcome must be 'yes' or 'no'", err=True)
            raise typer.Exit(code=1)

        outcome_bool = outcome_lower == "yes"

        # Process resolution
        with DuckDBClient(settings.database.duckdb_path) as duckdb_client:
            calibrator = CalibratorAgent(history_db=duckdb_client)
            feedback = FeedbackLoop(db=duckdb_client, calibrator=calibrator)

            result = feedback.process_resolution(
                market_id=market_id,
                outcome=outcome_bool,
            )

        if not result.get("success"):
            typer.echo(f"Error: {result.get('error', 'Unknown error')}", err=True)
            raise typer.Exit(code=1)

        typer.echo("=" * 80)
        typer.echo("MARKET RESOLVED")
        typer.echo("=" * 80)
        typer.echo(f"\nMarket ID: {market_id}")
        typer.echo(f"Outcome: {'YES' if outcome_bool else 'NO'}")
        typer.echo(f"\nBrier Score (Raw): {result['brier_score_raw']:.4f}")
        typer.echo(f"Brier Score (Calibrated): {result['brier_score_calibrated']:.4f}")
        typer.echo(f"Improvement: {result['improvement']:+.4f}")

        if result['improvement'] > 0:
            typer.echo("\n✓ Calibration improved forecast accuracy")
        elif result['improvement'] < 0:
            typer.echo("\n✗ Calibration made forecast worse")
        else:
            typer.echo("\n= No change from calibration")

        typer.echo("=" * 80)

    except Exception as e:
        typer.echo(f"Error: {e}", err=True)
        logger.exception("Resolve command failed")
        raise typer.Exit(code=1)


@app.command()
def backtest() -> None:
    """Run historical calibration."""
    typer.echo("Command not yet implemented")


@app.command()
def paper(
    once: bool = typer.Option(False, "--once", help="Run one cycle and exit"),
    interval: int = typer.Option(60, "--interval", "-i", help="Minutes between cycles (if not --once)"),
    top_markets: int = typer.Option(5, "--top", "-n", help="Number of top markets to analyze per cycle"),
) -> None:
    """
    Run in paper trading mode.

    Continuously monitors markets, generates forecasts, and executes paper trades.

    Example:
        poly-oracle paper --once              # Run one cycle
        poly-oracle paper --interval 30       # Run every 30 minutes
        poly-oracle paper --once --top 3      # Analyze top 3 markets by liquidity
    """
    import asyncio
    import time
    from src.agents import create_debate_system
    from src.data.sources.polymarket import PolymarketClient
    from src.data.sources.news import NewsClient
    from src.data.context import ContextBuilder
    from src.data.storage.chroma_client import ChromaClient
    from src.data.storage.duckdb_client import DuckDBClient
    from src.data.storage.sqlite_client import SQLiteClient
    from src.calibration import CalibratorAgent, MetaAnalyzer, FeedbackLoop
    from src.calibration.resolver import MarketResolver
    from src.execution import PositionSizer, RiskManager, PaperTradingExecutor

    settings = get_settings()

    async def run_cycle() -> None:
        """Run one complete trading cycle."""
        cycle_start = time.time()

        typer.echo("\n" + "=" * 80)
        typer.echo(f"PAPER TRADING CYCLE - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        typer.echo("=" * 80)

        # Initialize clients
        with SQLiteClient(settings.database.sqlite_path) as sqlite_client, \
             DuckDBClient(settings.database.duckdb_path) as duckdb_client:

            # Initialize calibration
            calibrator = CalibratorAgent(history_db=duckdb_client)
            analyzer = MetaAnalyzer(
                min_edge=settings.risk.min_edge,
                min_confidence=0.6,
                min_liquidity=settings.risk.min_liquidity,
            )
            feedback = FeedbackLoop(db=duckdb_client, calibrator=calibrator)

            # Initialize Polymarket client for resolution checks
            async with PolymarketClient() as poly_client, \
                       NewsClient() as news_client:

                # PHASE 1: Resolve closed markets (ALWAYS FIRST)
                typer.echo("\nPHASE 1: Checking for resolved markets...")
                resolver = MarketResolver(
                    polymarket=poly_client,
                    feedback=feedback,
                    sqlite=sqlite_client,
                    duckdb=duckdb_client,
                )

                resolution = await resolver.run_resolution_cycle()
                if resolution["resolved"] > 0:
                    typer.echo(
                        f"Resolved {resolution['resolved']} markets. "
                        f"P&L: {resolution['pnl']:+.2f} EUR"
                    )
                else:
                    typer.echo("No markets resolved")

                # Get current bankroll (updated with P&L from resolutions)
                bankroll = sqlite_client.get_current_bankroll()
                typer.echo(f"Current Bankroll: ${bankroll:.2f}")

                # Initialize execution components
                sizer = PositionSizer(risk_settings=settings.risk)
                risk_manager = RiskManager(risk_settings=settings.risk)
                executor = PaperTradingExecutor(
                    sqlite=sqlite_client,
                    sizer=sizer,
                    risk=risk_manager,
                )

                # PHASE 2: Find new trading opportunities
                typer.echo(f"\nPHASE 2: Fetching active markets...")

                markets = await poly_client.get_active_markets()

                # Filter by liquidity
                filtered_markets = [
                    m for m in markets
                    if m.liquidity >= settings.risk.min_liquidity
                ]

                # Sort by liquidity and take top N
                filtered_markets.sort(key=lambda m: m.liquidity, reverse=True)
                target_markets = filtered_markets[:top_markets]

                typer.echo(
                    f"Found {len(markets)} active markets, "
                    f"{len(filtered_markets)} with sufficient liquidity, "
                    f"analyzing top {len(target_markets)}"
                )

                trades_attempted = 0
                trades_executed = 0
                trades_skipped = 0

                # Process each market
                for idx, market in enumerate(target_markets, 1):
                    typer.echo(f"\n[{idx}/{len(target_markets)}] Analyzing {market.question[:60]}...")
                    typer.echo(f"  Liquidity: ${market.liquidity:,.0f} | Price: {market.current_price:.1%}")

                    try:
                        # Build context
                        with ChromaClient(settings.database.chroma_path, settings.llm.embedding_model) as chroma_client:
                            builder = ContextBuilder(
                                polymarket_client=poly_client,
                                news_client=news_client,
                                chroma_client=chroma_client,
                            )
                            context_text = await builder.build_context(market)

                        # Create debate system and run forecast
                        orchestrator, ollama = create_debate_system(
                            base_url=settings.llm.base_url,
                            model=settings.llm.model,
                            timeout=120,
                        )

                        try:
                            forecast_result = await orchestrator.run_debate(
                                market_id=market.id,
                                context=context_text,
                                rounds=1,  # Use 1 round for faster cycles
                                temperature=0.7,
                                verbose=False,
                            )

                            # Calibrate forecast
                            confidence = 0.5
                            if forecast_result.confidence_lower and forecast_result.confidence_upper:
                                confidence = 1.0 - (forecast_result.confidence_upper - forecast_result.confidence_lower)

                            calibrated = calibrator.calibrate(
                                raw_forecast=forecast_result.probability,
                                market_type=market.market_type,
                                confidence=confidence,
                            )

                            # Analyze edge
                            edge_analysis = analyzer.analyze(
                                our_forecast=calibrated,
                                market_price=market.current_price,
                                liquidity=market.liquidity,
                            )

                            typer.echo(f"  Forecast: {calibrated.calibrated:.1%} (raw: {forecast_result.probability:.1%})")
                            typer.echo(f"  Edge: {edge_analysis.raw_edge:+.1%} | Recommendation: {edge_analysis.recommended_action}")

                            # Record forecast
                            feedback.record_forecast(
                                forecast=forecast_result,
                                market=market,
                                calibrated_probability=calibrated.calibrated,
                                edge=edge_analysis.raw_edge,
                                recommended_action=edge_analysis.recommended_action,
                            )

                            # Attempt execution
                            if edge_analysis.recommended_action == "TRADE":
                                trades_attempted += 1

                                execution_result = await executor.execute(
                                    edge_analysis=edge_analysis,
                                    calibrated_probability=calibrated.calibrated,
                                    market=market,
                                    bankroll=bankroll,
                                )

                                if execution_result and execution_result.success:
                                    trades_executed += 1
                                    typer.echo(f"  ✓ Trade executed: {execution_result.message}")
                                elif execution_result:
                                    typer.echo(f"  ✗ Trade rejected: {execution_result.message}")
                                else:
                                    typer.echo(f"  ○ Trade skipped (insufficient size)")
                            else:
                                trades_skipped += 1

                        finally:
                            await ollama.close()

                    except Exception as e:
                        typer.echo(f"  ✗ Error processing market: {e}")
                        logger.exception(f"Error processing market {market.id}")

            # Cycle summary
            cycle_duration = time.time() - cycle_start
            typer.echo("\n" + "=" * 80)
            typer.echo("CYCLE SUMMARY")
            typer.echo("=" * 80)
            typer.echo(f"Markets Analyzed: {len(target_markets)}")
            typer.echo(f"Trades Attempted: {trades_attempted}")
            typer.echo(f"Trades Executed: {trades_executed}")
            typer.echo(f"Trades Skipped: {trades_skipped}")
            typer.echo(f"Cycle Duration: {cycle_duration:.1f}s")
            typer.echo("=" * 80)

    # Main loop
    try:
        if once:
            typer.echo("Running single paper trading cycle...")
            asyncio.run(run_cycle())
            typer.echo("\nCycle complete. Use 'poly-oracle positions' to view portfolio.")
        else:
            typer.echo(f"Starting continuous paper trading (interval: {interval} minutes)")
            typer.echo("Press Ctrl+C to stop")

            while True:
                asyncio.run(run_cycle())

                if interval > 0:
                    typer.echo(f"\nSleeping for {interval} minutes...")
                    time.sleep(interval * 60)
                else:
                    typer.echo("\nInterval is 0, exiting after one cycle")
                    break

    except KeyboardInterrupt:
        typer.echo("\n\nPaper trading stopped by user")
    except Exception as e:
        typer.echo(f"\nError in paper trading: {e}", err=True)
        logger.exception("Paper trading failed")
        raise typer.Exit(code=1)


@app.command()
def live() -> None:
    """Run in live trading mode."""
    typer.echo("Command not yet implemented")


@app.command()
def positions() -> None:
    """
    Display current open positions.

    Shows all open positions with current P&L and portfolio summary.
    """
    from src.data.storage.sqlite_client import SQLiteClient

    try:
        settings = get_settings()

        with SQLiteClient(settings.database.sqlite_path) as sqlite_client:
            positions = sqlite_client.get_open_positions()
            bankroll = sqlite_client.get_current_bankroll()

            if not positions:
                typer.echo("No open positions")
                typer.echo(f"\nCurrent Bankroll: ${bankroll:.2f}")
                return

            typer.echo("=" * 120)
            typer.echo("OPEN POSITIONS")
            typer.echo("=" * 120)

            # Header
            typer.echo(
                f"{'Market ID':<20} {'Direction':<10} {'Shares':<12} {'Avg Entry':<12} "
                f"{'Current':<12} {'P&L':<15} {'P&L %':<10}"
            )
            typer.echo("-" * 120)

            total_pnl = 0.0

            for pos in positions:
                # Calculate unrealized P&L
                if pos.direction == "BUY_YES":
                    unrealized_pnl = (pos.current_price - pos.avg_entry_price) * pos.num_shares
                else:  # BUY_NO
                    unrealized_pnl = (pos.avg_entry_price - pos.current_price) * pos.num_shares

                pnl_pct = (unrealized_pnl / pos.amount_usd * 100) if pos.amount_usd > 0 else 0
                total_pnl += unrealized_pnl

                # Color code P&L
                pnl_str = f"${unrealized_pnl:+.2f}"
                pnl_pct_str = f"{pnl_pct:+.1f}%"

                typer.echo(
                    f"{pos.market_id[:20]:<20} {pos.direction:<10} {pos.num_shares:<12.2f} "
                    f"${pos.avg_entry_price:<11.2f} ${pos.current_price:<11.2f} "
                    f"{pnl_str:<15} {pnl_pct_str:<10}"
                )

            typer.echo("=" * 120)
            typer.echo(f"Total Positions: {len(positions)}")
            typer.echo(f"Total Unrealized P&L: ${total_pnl:+.2f}")
            typer.echo(f"Current Bankroll: ${bankroll:.2f}")
            typer.echo(f"Total Portfolio Value: ${bankroll + total_pnl:.2f}")
            typer.echo("=" * 120)

    except Exception as e:
        typer.echo(f"Error: {e}", err=True)
        logger.exception("Positions command failed")
        raise typer.Exit(code=1)


@app.command()
def trades(
    limit: int = typer.Option(20, "--limit", "-n", help="Number of recent trades to show"),
) -> None:
    """
    Display trade history.

    Shows recent executed trades.

    Example:
        poly-oracle trades --limit 10
    """
    from src.data.storage.sqlite_client import SQLiteClient

    try:
        settings = get_settings()

        with SQLiteClient(settings.database.sqlite_path) as sqlite_client:
            # Get all trades (SQLiteClient doesn't have get_trades yet, so we query directly)
            result = sqlite_client.conn.execute(
                """
                SELECT
                    id, market_id, direction, amount_usd, num_shares,
                    entry_price, timestamp, status
                FROM trades
                ORDER BY timestamp DESC
                LIMIT ?
                """,
                [limit],
            ).fetchall()

            if not result:
                typer.echo("No trades found")
                return

            typer.echo("=" * 120)
            typer.echo("TRADE HISTORY")
            typer.echo("=" * 120)

            # Header
            typer.echo(
                f"{'Timestamp':<20} {'Market ID':<20} {'Direction':<10} "
                f"{'Amount':<12} {'Shares':<12} {'Entry':<10} {'Status':<10}"
            )
            typer.echo("-" * 120)

            for row in result:
                trade_id, market_id, direction, amount, shares, entry, timestamp, status = row

                # Parse timestamp
                if isinstance(timestamp, str):
                    dt = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
                    timestamp_str = dt.strftime("%Y-%m-%d %H:%M:%S")
                else:
                    timestamp_str = str(timestamp)[:19]

                typer.echo(
                    f"{timestamp_str:<20} {market_id[:20]:<20} {direction:<10} "
                    f"${amount:<11.2f} {shares:<12.2f} ${entry:<9.2f} {status:<10}"
                )

            typer.echo("=" * 120)
            typer.echo(f"Showing {len(result)} most recent trades")
            typer.echo("=" * 120)

    except Exception as e:
        typer.echo(f"Error: {e}", err=True)
        logger.exception("Trades command failed")
        raise typer.Exit(code=1)


@app.command()
def positions() -> None:
    """
    View open positions.

    Shows all currently open positions with unrealized P&L.

    Example:
        poly-oracle positions
    """
    from src.data.storage.sqlite_client import SQLiteClient

    try:
        settings = get_settings()

        with SQLiteClient(settings.database.sqlite_path) as sqlite_client:
            open_positions = sqlite_client.get_open_positions()

            if not open_positions:
                typer.echo("No open positions")
                return

            typer.echo("=" * 130)
            typer.echo("OPEN POSITIONS")
            typer.echo("=" * 130)

            # Header
            typer.echo(
                f"{'Market ID':<25} {'Direction':<10} {'Shares':<12} "
                f"{'Amount':<12} {'Entry':<10} {'Current':<10} {'Unrealized P&L':<15}"
            )
            typer.echo("-" * 130)

            total_pnl = 0.0
            for pos in open_positions:
                market_id = pos["market_id"]
                direction = pos["direction"]
                shares = pos["num_shares"]
                amount = pos["amount_usd"]
                entry = pos["avg_entry_price"]
                current = pos["current_price"]
                pnl = pos["unrealized_pnl"]

                total_pnl += pnl

                # Format P&L with color
                pnl_str = f"${pnl:+.2f}"

                typer.echo(
                    f"{market_id[:25]:<25} {direction:<10} {shares:<12.2f} "
                    f"${amount:<11.2f} ${entry:<9.3f} ${current:<9.3f} {pnl_str:<15}"
                )

            typer.echo("=" * 130)
            typer.echo(f"Total Unrealized P&L: ${total_pnl:+.2f}")
            typer.echo("=" * 130)

    except Exception as e:
        typer.echo(f"Error: {e}", err=True)
        logger.exception("Positions command failed")
        raise typer.Exit(code=1)


if __name__ == "__main__":
    app()
