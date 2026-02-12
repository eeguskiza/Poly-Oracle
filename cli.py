import httpx
import typer
from loguru import logger

from config.settings import get_settings
from src.utils.logging import setup_logging
from src.utils.exceptions import ConfigError
from src.data.storage.duckdb_client import DuckDBClient
from src.data.storage.sqlite_client import SQLiteClient
from src.data.storage.chroma_client import ChromaClient

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
def markets() -> None:
    """List active markets."""
    typer.echo("Command not yet implemented")


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
