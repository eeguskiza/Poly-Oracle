# Poly-Oracle

Sistema de forecasting multi-agente para Polymarket. LLM local (Ollama + GPU), coste cero.

## Setup rapido

```bash
curl -fsSL https://ollama.com/install.sh | sh
ollama pull mistral && ollama pull nomic-embed-text

python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env    # meter API keys
python cli.py init      # crear BBDDs
```

## Como funciona

```
Seleccion mercados → Contexto (news+data) → Debate 4 agentes → Calibracion → Sizing Kelly → Trade
```

4 agentes debaten: Bull (alcista), Bear (bajista), Devil's Advocate (desafia), Judge (decide probabilidad final). Despues se calibra con Brier scores y se ejecuta con Kelly criterion.

## Dashboard terminal

```bash
python cli.py start              # lanzar dashboard
python cli.py start --paper      # forzar paper
python cli.py start --live       # forzar live
```

### Que hace cada opcion del menu

| Opcion | Que hace |
|--------|----------|
| **Start Trading** | Loop autonomo: escanea mercados, debate, ejecuta trades. Corre hasta Ctrl+C |
| **Market Scanner** | Lista mercados activos de Polymarket con precio, volumen, liquidez. Puedes lanzar forecast desde aqui |
| **Single Forecast** | Metes un market ID, corre el debate completo y te dice si hay edge para tradear |
| **Portfolio** | Posiciones abiertas, P&L por posicion, resumen de bankroll |
| **Trade History** | Ultimos 30 trades ejecutados con timestamps y status |
| **Performance** | Brier scores (raw vs calibrado), win rate, value vs market |
| **Equity Curve** | Grafico ASCII de la evolucion del bankroll + drawdown |
| **Backtest** | Simula trades sobre forecasts historicos resueltos |
| **System Status** | Health check: Ollama, DuckDB, SQLite, ChromaDB, Polymarket API |
| **Settings** | Muestra config actual (LLM, risk, data, polymarket) |

## Comandos CLI

```bash
# Trading
python cli.py paper --once              # 1 ciclo paper trading
python cli.py trade --mode crypto       # solo mercados BTC/ETH/SOL
python cli.py trade --mode auto         # selector inteligente por viabilidad
python cli.py trade --mode chosen --market-id <id>  # mercado especifico
python cli.py live --once               # trading real (necesita API keys)

# Analisis
python cli.py forecast <id> -r 2 -v     # debate sobre un mercado
python cli.py backtest                   # replay rapido sin LLM
python cli.py backtest --full            # simulacion completa con LLM

# Info
python cli.py markets --sort-by trending
python cli.py positions
python cli.py trades
python cli.py calibration
python cli.py status
```

## Paper → Live

No pasar a live hasta cumplir:

| KPI | Minimo |
|-----|--------|
| Brier Score | < 0.20 |
| Win Rate | > 50% |
| Sharpe | > 0 |
| Forecasts resueltos | >= 50 |
| Max Drawdown | < 30% |

Luego: meter `POLYMARKET_API_KEY`, `POLYMARKET_API_SECRET`, `POLYMARKET_API_PASSPHRASE` en `.env` y arrancar con `python cli.py live --once`.

## Tests

```bash
python -m pytest tests/ -q    # 287 tests
```
