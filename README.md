# Poly-Oracle

Multi-agent forecasting system for Polymarket prediction markets.
Uses local LLM inference (Ollama + GPU) for zero-cost analysis.

## Setup

```bash
# Install Ollama (official, with GPU support)
curl -fsSL https://ollama.com/install.sh | sh

# Pull required models
ollama pull mistral
ollama pull nomic-embed-text

# Clone and setup
git clone <repo-url> && cd Poly-Oracle
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

# Configure
cp .env.example .env
nano .env  # Add your API keys

# Initialize databases
python cli.py init
```

## Usage

### Interactive Dashboard (recommended)

```bash
python cli.py start
```

Opens an interactive menu with:
- **Start Trading** -- Autonomous paper/live trading loop
- **Market Scanner** -- Browse and filter active markets
- **Single Forecast** -- Run debate on one market
- **Portfolio** -- Open positions and P&L
- **Performance** -- Brier scores and accuracy
- **System Status** -- Component health checks
- **Settings** -- Current configuration

### CLI Commands

```bash
# Status
python cli.py status

# Markets
python cli.py markets --sort-by trending
python cli.py market <market_id>

# Forecast
python cli.py forecast <market_id> --rounds 2 --verbose

# Paper trading (one cycle)
python cli.py paper --once --top 5

# Portfolio
python cli.py positions
python cli.py trades --limit 10
```

## Tests

```bash
source venv/bin/activate
python -m pytest tests/ -q
```
