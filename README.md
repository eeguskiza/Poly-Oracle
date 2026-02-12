# Poly-Oracle

Multi-agent forecasting system for Polymarket prediction markets.
Uses local LLM inference (Ollama) for zero-cost analysis and decision making.

## Setup

```bash
# Install Ollama
brew install ollama

# Pull required models
ollama pull mistral
ollama pull nomic-embed-text

# Create virtual environment
python3 -m venv venv

# Activate virtual environment
source venv/bin/activate

# Install dependencies
pip install --upgrade pip setuptools wheel
pip install typer loguru pydantic pydantic-settings httpx duckdb chromadb pytest pytest-asyncio pytest-cov ruff

# Configure environment
cp .env.example .env
# Edit .env with your API keys

# Test installation
python cli.py status
```

## CLI Commands

- `poly-oracle status` - Check system status
- `poly-oracle markets` - List active markets
- `poly-oracle context` - Build context for a market
- `poly-oracle forecast` - Generate forecast for a market
- `poly-oracle backtest` - Run historical calibration
- `poly-oracle paper` - Paper trading mode
- `poly-oracle live` - Live trading mode
- `poly-oracle positions` - View open positions

## Project Structure

```
poly-oracle/
├── cli.py                  # CLI entrypoint
├── config/                 # Settings and prompts
├── src/                    # Source code
│   ├── agents/            # Multi-agent system
│   ├── data/              # Data collection and storage
│   ├── calibration/       # Forecast calibration
│   ├── execution/         # Trade execution
│   └── utils/             # Utilities
├── db/                     # Local databases (gitignored)
└── tests/                  # Test suite
```

## Running Tests

```bash
# Ensure virtual environment is activated
source venv/bin/activate

# Run tests
pytest
pytest --cov=src tests/
```
