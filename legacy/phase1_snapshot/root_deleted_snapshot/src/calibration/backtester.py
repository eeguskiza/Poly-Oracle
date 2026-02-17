"""
Backtester — replay historical forecasts to compute simulated P&L.

Supports two modes:
- ``run_replay()``: fast mode using existing DuckDB forecasts (no LLM)
- ``run_full()``: re-runs the debate pipeline on resolved markets
"""
from __future__ import annotations

import math
from datetime import datetime
from typing import Any

from loguru import logger

from config.settings import Settings
from src.data.storage.duckdb_client import DuckDBClient
from src.models.backtest import BacktestResult


class Backtester:
    """Backtesting over historical forecasts stored in DuckDB."""

    def __init__(self, duckdb: DuckDBClient, settings: Settings) -> None:
        self.duckdb = duckdb
        self.settings = settings

    # ------------------------------------------------------------------
    # Replay mode (fast — no LLM calls)
    # ------------------------------------------------------------------
    def run_replay(
        self,
        date_from: str | None = None,
        date_to: str | None = None,
    ) -> BacktestResult:
        """Replay over existing resolved forecasts.

        Re-calculates Kelly sizing, Brier score, and simulated P&L
        without calling the LLM.
        """
        forecasts = self._get_resolved_forecasts(date_from, date_to)

        if not forecasts:
            logger.info("No resolved forecasts for backtesting")
            return BacktestResult()

        bankroll = float(self.settings.risk.initial_bankroll)
        equity_curve = [bankroll]
        daily_returns: list[float] = []
        trade_log: list[dict] = []
        wins = 0
        brier_sum = 0.0

        for fc in forecasts:
            prob = float(fc["calibrated_probability"])
            market_price = float(fc["market_price_at_forecast"])
            outcome = bool(fc["outcome"])
            edge = prob - market_price

            # Kelly fraction (simplified half-Kelly)
            if abs(edge) < self.settings.risk.min_edge:
                continue  # Would have skipped

            if edge > 0:
                kelly = edge / (1.0 - market_price) if market_price < 1.0 else 0
                direction = "BUY_YES"
            else:
                kelly = -edge / market_price if market_price > 0 else 0
                direction = "BUY_NO"

            fraction = min(kelly * 0.5, self.settings.risk.max_position_pct)
            bet = bankroll * fraction
            bet = max(min(bet, self.settings.risk.max_bet), self.settings.risk.min_bet)
            bet = min(bet, bankroll * 0.99)  # never go all-in

            if bet < self.settings.risk.min_bet:
                continue

            # Simulate outcome
            if direction == "BUY_YES":
                entry_price = market_price
                payout = bet / entry_price if entry_price > 0 else 0
                pnl = (payout - bet) if outcome else -bet
            else:
                entry_price = 1.0 - market_price
                payout = bet / entry_price if entry_price > 0 else 0
                pnl = (payout - bet) if not outcome else -bet

            bankroll += pnl
            equity_curve.append(bankroll)

            prev = equity_curve[-2] if len(equity_curve) >= 2 else bankroll
            ret = pnl / prev if prev > 0 else 0
            daily_returns.append(ret)

            if pnl > 0:
                wins += 1

            # Brier
            brier = (prob - (1.0 if outcome else 0.0)) ** 2
            brier_sum += brier

            trade_log.append({
                "market_id": fc["market_id"],
                "direction": direction,
                "bet": round(bet, 2),
                "pnl": round(pnl, 2),
                "outcome": outcome,
                "prob": round(prob, 3),
                "market_price": round(market_price, 3),
            })

        total = len(trade_log)
        max_dd = self._max_drawdown(equity_curve)
        sharpe = self._sharpe(daily_returns) if daily_returns else None

        return BacktestResult(
            total_forecasts=len(forecasts),
            simulated_trades=total,
            final_equity=round(bankroll, 2),
            max_drawdown=round(max_dd, 4),
            sharpe_ratio=round(sharpe, 4) if sharpe is not None else None,
            win_rate=round(wins / total, 4) if total > 0 else 0.0,
            brier_score=round(brier_sum / len(forecasts), 4) if forecasts else 0.0,
            equity_curve=[round(e, 2) for e in equity_curve],
            daily_returns=[round(r, 6) for r in daily_returns],
            trade_log=trade_log,
        )

    # ------------------------------------------------------------------
    # Full mode (re-runs debate — async, needs LLM)
    # ------------------------------------------------------------------
    async def run_full(
        self,
        polymarket: Any,
        news_client: Any,
        chroma: Any,
        date_from: str | None = None,
        date_to: str | None = None,
    ) -> BacktestResult:
        """Full simulation: rebuild context + re-run debate for each resolved forecast.

        This is the slow, thorough mode that calls the LLM.
        """
        from src.agents import create_debate_system
        from src.data.context import ContextBuilder
        from src.calibration import CalibratorAgent, MetaAnalyzer

        forecasts = self._get_resolved_forecasts(date_from, date_to)

        if not forecasts:
            return BacktestResult()

        calibrator = CalibratorAgent(history_db=self.duckdb)
        analyzer = MetaAnalyzer(
            min_edge=self.settings.risk.min_edge,
            min_confidence=0.6,
            min_liquidity=self.settings.risk.min_liquidity,
        )

        bankroll = float(self.settings.risk.initial_bankroll)
        equity_curve = [bankroll]
        daily_returns: list[float] = []
        trade_log: list[dict] = []
        wins = 0
        brier_sum = 0.0

        for fc in forecasts:
            market_id = fc["market_id"]
            outcome = bool(fc["outcome"])
            market_price = float(fc["market_price_at_forecast"])

            try:
                market = await polymarket.get_market(market_id)
            except Exception:
                logger.warning(f"Backtest: cannot fetch market {market_id}, skipping")
                continue

            try:
                builder = ContextBuilder(
                    polymarket_client=polymarket,
                    news_client=news_client,
                    chroma_client=chroma,
                )
                context_text = await builder.build_context(market)

                orchestrator, ollama = create_debate_system(
                    base_url=self.settings.llm.base_url,
                    model=self.settings.llm.model,
                    timeout=self.settings.llm.timeout,
                )
                try:
                    result = await orchestrator.run_debate(
                        market_id=market_id,
                        context=context_text,
                        rounds=1,
                        temperature=0.7,
                        verbose=False,
                    )
                    confidence = result.compute_confidence()
                    calibrated = calibrator.calibrate(
                        raw_forecast=result.probability,
                        market_type=market.market_type,
                        confidence=confidence,
                    )
                    prob = calibrated.calibrated
                finally:
                    await ollama.close()
            except Exception as exc:
                logger.warning(f"Backtest full: error on {market_id}: {exc}")
                continue

            edge = prob - market_price
            if abs(edge) < self.settings.risk.min_edge:
                continue

            if edge > 0:
                kelly = edge / (1.0 - market_price) if market_price < 1.0 else 0
                direction = "BUY_YES"
            else:
                kelly = -edge / market_price if market_price > 0 else 0
                direction = "BUY_NO"

            fraction = min(kelly * 0.5, self.settings.risk.max_position_pct)
            bet = bankroll * fraction
            bet = max(min(bet, self.settings.risk.max_bet), self.settings.risk.min_bet)
            bet = min(bet, bankroll * 0.99)

            if bet < self.settings.risk.min_bet:
                continue

            if direction == "BUY_YES":
                entry_price = market_price
                payout = bet / entry_price if entry_price > 0 else 0
                pnl = (payout - bet) if outcome else -bet
            else:
                entry_price = 1.0 - market_price
                payout = bet / entry_price if entry_price > 0 else 0
                pnl = (payout - bet) if not outcome else -bet

            bankroll += pnl
            equity_curve.append(bankroll)

            prev = equity_curve[-2] if len(equity_curve) >= 2 else bankroll
            daily_returns.append(pnl / prev if prev > 0 else 0)

            if pnl > 0:
                wins += 1

            brier = (prob - (1.0 if outcome else 0.0)) ** 2
            brier_sum += brier

            trade_log.append({
                "market_id": market_id,
                "direction": direction,
                "bet": round(bet, 2),
                "pnl": round(pnl, 2),
                "outcome": outcome,
                "prob": round(prob, 3),
                "market_price": round(market_price, 3),
            })

        total = len(trade_log)
        return BacktestResult(
            total_forecasts=len(forecasts),
            simulated_trades=total,
            final_equity=round(bankroll, 2),
            max_drawdown=round(self._max_drawdown(equity_curve), 4),
            sharpe_ratio=round(self._sharpe(daily_returns), 4) if daily_returns else None,
            win_rate=round(wins / total, 4) if total > 0 else 0.0,
            brier_score=round(brier_sum / len(forecasts), 4) if forecasts else 0.0,
            equity_curve=[round(e, 2) for e in equity_curve],
            daily_returns=[round(r, 6) for r in daily_returns],
            trade_log=trade_log,
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _get_resolved_forecasts(
        self, date_from: str | None = None, date_to: str | None = None
    ) -> list[dict]:
        query = "SELECT * FROM forecasts WHERE resolved = true AND outcome IS NOT NULL"
        params: list[Any] = []
        if date_from:
            query += " AND timestamp >= ?"
            params.append(date_from)
        if date_to:
            query += " AND timestamp <= ?"
            params.append(date_to)
        query += " ORDER BY timestamp ASC"

        rows = self.duckdb.conn.execute(query, params).fetchall()
        if not rows:
            return []
        columns = [desc[0] for desc in self.duckdb.conn.description]
        return [dict(zip(columns, row)) for row in rows]

    @staticmethod
    def _max_drawdown(equity_curve: list[float]) -> float:
        if len(equity_curve) < 2:
            return 0.0
        peak = equity_curve[0]
        max_dd = 0.0
        for val in equity_curve[1:]:
            if val > peak:
                peak = val
            dd = (peak - val) / peak if peak > 0 else 0
            max_dd = max(max_dd, dd)
        return max_dd

    @staticmethod
    def _sharpe(returns: list[float], risk_free: float = 0.0) -> float:
        if len(returns) < 2:
            return 0.0
        mean = sum(returns) / len(returns) - risk_free
        variance = sum((r - mean) ** 2 for r in returns) / (len(returns) - 1)
        std = math.sqrt(variance) if variance > 0 else 0.0
        if std == 0:
            return 0.0
        return (mean / std) * math.sqrt(252)  # Annualized
