# CLAUDE.md

## Project: Poly-Oracle

Multi-agent forecasting system for Polymarket prediction markets.

## Core Constraints

1. Zero budget for APIs - Only free tiers
2. Local LLM inference - Ollama + 7B-8B models on MacBook M1 Pro (16GB)
3. Initial capital: 50 EUR - Validation phase
4. Must run on M1 Pro without dedicated GPU

## Code Rules

- No emojis anywhere (code, logs, CLI)
- Type hints on all functions
- Minimal comments, self-documenting code
- English only in code
- Async for all I/O
- Custom exceptions with context
- Pydantic for all config
- No hardcoded values

## Stack

Python 3.11+, Ollama, ChromaDB, DuckDB, SQLite, Typer, Loguru, Pydantic, httpx, ruff

## Architecture

Layer 1 (Data): NewsAgent + MarketAgent + SocialAgent + DataAgent -> ChromaDB
Layer 2 (Reasoning): BullAgent vs BearAgent vs Devil's Advocate -> JudgeAgent -> P(YES)
Layer 3 (Calibration): Isotonic regression + Edge detection
Layer 4 (Execution): Kelly sizing -> Risk checks -> Polymarket CLOB API

## Approved Free APIs

Polymarket CLOB/Gamma (unlimited), NewsAPI (100/day), Google News RSS (unlimited), Reddit (100/min)

## Forbidden

OpenAI API, Anthropic API, Twitter/X API, any paid service, models >14B params

## Risk Limits (50 EUR)

MAX_POSITION_PCT=0.10, MIN_BET=1.0, MAX_BET=10.0, MAX_DAILY_LOSS_PCT=0.10
MAX_OPEN_POSITIONS=8, MIN_EDGE=0.08, MIN_CONFIDENCE=0.65, MIN_LIQUIDITY=1000

## CLI Commands

status, markets, context, forecast, backtest, paper, live, positions

## Development Order

1. Foundation (CLI + DB + Settings)
2. Data Layer (Polymarket + News + ChromaDB)
3. Agents (Bull/Bear/Devil/Judge + Orchestrator)
4. Calibration (Backtest + Adjustment)
5. Execution (Paper trading -> Live)
