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
pip install --upgrade pip
pip install -r requirements.txt

# Configure environment
cp .env.example .env
# Edit .env with your API keys

# Test installation
python cli.py status
```

## CLI Commands

### Setup & Status
- `python cli.py init` - Initialize Poly-Oracle directories and verify setup
- `python cli.py status` - Check system status and configuration

### Market Discovery
- `python cli.py markets [--limit N] [--sort-by volume|liquidity|trending]` - List active markets
  - `--sort-by trending` (default) - Shows markets with highest volume/liquidity ratio (most active)
  - `--sort-by volume` - Shows markets with highest 24h trading volume
  - `--sort-by liquidity` - Shows markets with highest total liquidity
- `python cli.py market <market_id>` - Show details for a specific market

### News & Context
- `python cli.py news <query>` - Search for news articles
- `python cli.py market-news <market_id>` - Get news relevant to a specific market
- `python cli.py context <market_id>` - Build context for a market (news + historical data)

### Forecasting
- `python cli.py forecast <market_id>` - Generate forecast using multi-agent debate
- `python cli.py calibration` - Show calibration performance report
- `python cli.py backtest` - Run historical calibration

### Trading
- `python cli.py paper [--once]` - Run in paper trading mode (--once runs one cycle)
- `python cli.py live` - Run in live trading mode (not yet implemented)
- `python cli.py positions` - View open positions with unrealized P&L
- `python cli.py trades [--limit N]` - Display trade history (default: 20)

Note: Market resolution is fully automatic and runs as part of the paper trading loop.

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
# Run all tests (197 tests covering all layers)
./venv/bin/pytest tests/ -v

# Run with coverage report
./venv/bin/pytest --cov=src tests/

# Run specific test modules
./venv/bin/pytest tests/unit/test_resolver.py -v
./venv/bin/pytest tests/integration/test_full_pipeline.py -v
```
