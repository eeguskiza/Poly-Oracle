# Legacy Code

This directory contains the pre-v1 codebase preserved for reference during migration.

## What's here

- `src/` — Original flat package layout (agents, execution, models, etc.)
- `config/` — Original root-level configuration (settings.py, prompts/)

## What was already removed (from git history)

These modules were deleted during the initial v1 cleanup because they are not
needed for a single-market BTC trading bot:

| Module | Reason |
|--------|--------|
| `src/dashboard/` | Interactive TUI — v1 is a headless loop |
| `src/calibration/` | Isotonic regression, Brier scoring, backtesting — premature for v1 |
| `src/data/storage/duckdb_client.py` | Analytics DB — SQLite is sufficient |
| `src/data/storage/chroma_client.py` | Vector embeddings — no RAG needed |
| `src/data/sources/news.py` | News aggregation — unreliable external dependency |
| `src/data/context.py` | Context builder — depended on ChromaDB + news |
| `src/data/selectors/` | Multi-market selectors — single-market bot |
| `src/models/news.py` | News data model |
| `src/models/backtest.py` | Backtest result model |

## Why preserved

The code here is structurally sound and can be reused if v2 needs:
- Multi-market support (selectors)
- Forecast calibration (calibration/)
- News-augmented context (news, context)
- Interactive dashboard (dashboard/)

## Do not import from this directory

All active code lives under `bot/`. This directory exists only for reference.
