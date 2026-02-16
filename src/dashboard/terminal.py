"""
Terminal Dashboard - Interactive Rich + questionary dashboard for Poly-Oracle.
"""
import asyncio
import io
import time
from collections import deque
from datetime import datetime, timezone
from typing import Any

import httpx
import questionary
from loguru import logger
from rich.console import Console
from rich.live import Live
from rich.panel import Panel
from rich.table import Table, box
from rich.text import Text

try:
    import plotext as plt

    HAS_PLOTEXT = True
except ImportError:
    HAS_PLOTEXT = False

from config.settings import VERSION, Settings, get_settings
from src.agents import create_debate_system
from src.agents.base import OllamaClient
from src.calibration import CalibratorAgent, FeedbackLoop, MetaAnalyzer
from src.calibration.resolver import MarketResolver
from src.data.context import ContextBuilder
from src.data.sources.news import NewsClient
from src.data.sources.polymarket import PolymarketClient
from src.data.storage.chroma_client import ChromaClient
from src.data.storage.duckdb_client import DuckDBClient
from src.data.storage.sqlite_client import SQLiteClient
from src.execution import PaperTradingExecutor, PositionSizer, RiskManager
from src.utils.exceptions import LLMError


console = Console()

MAIN_MENU_CHOICES = [
    "Auto Trading (Crypto) - Autonomous BTC/ETH/SOL every 15 min",
    "Auto Trading (All)    - Autonomous all viable markets",
    "Portfolio             - Positions and P&L",
    "Advanced              - Scanner, backtest, settings...",
    "Exit",
]

ADVANCED_MENU_CHOICES = [
    "Market Scanner      - Browse active markets",
    "Single Forecast     - Run debate on one market",
    "Trade History       - Recent executed trades",
    "Performance         - Brier score and accuracy",
    "Equity Curve        - Portfolio value over time",
    "Backtest            - Historical simulation",
    "System Status       - Component health",
    "Settings            - Current configuration",
    "Back",
]


def _truncate(text: str, max_len: int) -> str:
    if len(text) <= max_len:
        return text
    return text[: max_len - 3] + "..."


class TerminalDashboard:
    """Interactive terminal dashboard for monitoring and controlling Poly-Oracle."""

    def __init__(
        self,
        settings: Settings,
        polymarket: PolymarketClient,
        news: NewsClient,
        sqlite: SQLiteClient,
        duckdb: DuckDBClient,
        chroma: ChromaClient,
        resolver: MarketResolver,
        calibrator: CalibratorAgent,
        analyzer: MetaAnalyzer,
        feedback: FeedbackLoop,
        sizer: PositionSizer,
        risk_manager: RiskManager,
        executor: PaperTradingExecutor,
    ) -> None:
        self.settings = settings
        self.polymarket = polymarket
        self.news = news
        self.sqlite = sqlite
        self.duckdb = duckdb
        self.chroma = chroma
        self.resolver = resolver
        self.calibrator = calibrator
        self.analyzer = analyzer
        self.feedback = feedback
        self.sizer = sizer
        self.risk_manager = risk_manager
        self.executor = executor

    async def _show_splash(self) -> None:
        """Display splash screen with ASCII logo, version, mode, and health summary."""
        mode = "PAPER" if self.settings.paper_trading else "LIVE"
        mode_style = "yellow" if mode == "PAPER" else "red bold"

        logo = r"""
  ____       _            ___                  _
 |  _ \ ___ | |_   _     / _ \ _ __ __ _  ___| | ___
 | |_) / _ \| | | | |___| | | | '__/ _` |/ __| |/ _ \
 |  __/ (_) | | |_| |___| |_| | | | (_| | (__| |  __/
 |_|   \___/|_|\__, |    \___/|_|  \__,_|\___|_|\___|
               |___/
"""
        console.print(f"[bold cyan]{logo}[/bold cyan]")

        # Banner info
        bankroll = self.sqlite.get_current_bankroll()
        banner = (
            f"  [bold]v{VERSION}[/bold]  |  "
            f"Mode: [{mode_style}]{mode}[/{mode_style}]  |  "
            f"LLM: [bold]{self.settings.llm.model}[/bold]  |  "
            f"Bankroll: [bold]{bankroll:.2f} EUR[/bold]"
        )
        console.print(banner)
        console.print()

        # Quick health checks
        checks = await self._get_system_checks()
        health_table = Table(box=box.SIMPLE, show_header=False, padding=(0, 2))
        health_table.add_column("Component", style="bold", width=14)
        health_table.add_column("Status", width=8)
        health_table.add_column("Detail", ratio=1)

        for name, status, detail in checks:
            if status == "[OK]":
                style = "green"
            elif status == "[WARN]":
                style = "yellow"
            else:
                style = "red"
            health_table.add_row(name, f"[{style}]{status}[/{style}]", detail)

        console.print(Panel(health_table, title="System Health", box=box.ROUNDED))
        console.print()

        # Config summary
        config_summary = (
            f"[bold]LLM:[/bold] {self.settings.llm.model} @ {self.settings.llm.base_url} "
            f"(temp={self.settings.llm.temperature}, timeout={self.settings.llm.timeout}s)\n"
            f"[bold]Risk:[/bold] edge>={self.settings.risk.min_edge:.0%}, "
            f"bet ${self.settings.risk.min_bet:.0f}-${self.settings.risk.max_bet:.0f}, "
            f"max {self.settings.risk.max_open_positions} positions\n"
            f"[bold]Data:[/bold] NewsAPI {'configured' if self.settings.data.newsapi_key else 'not set (RSS only)'}, "
            f"Polymarket API "
            f"{'configured' if all([self.settings.polymarket.api_key, self.settings.polymarket.api_secret, self.settings.polymarket.api_passphrase]) else 'not set'}"
        )
        console.print(Panel(config_summary, title="Configuration", box=box.ROUNDED))
        console.print()

        # Risk alerts
        alerts = self._get_risk_alerts()
        if alerts:
            alert_text = "\n".join(f"[red bold]WARNING:[/red bold] {a}" for a in alerts)
            console.print(Panel(alert_text, title="Risk Alerts", box=box.HEAVY, border_style="red"))
            console.print()

    async def run(self) -> None:
        """Main dashboard loop: menu -> action -> menu. Ctrl+C exits cleanly."""
        await self._show_splash()
        mode = "PAPER MODE" if self.settings.paper_trading else "LIVE MODE"
        while True:
            try:
                console.print()
                choice = await questionary.select(
                    f"Poly-Oracle [{mode}] - Main Menu",
                    choices=MAIN_MENU_CHOICES,
                    qmark=">>",
                    pointer=">",
                ).ask_async()

                if choice is None or choice.startswith("Exit"):
                    console.print("[bold]Exiting Poly-Oracle.[/bold]")
                    break

                label = choice.split(" - ")[0].strip()

                if label == "Auto Trading (Crypto)":
                    await self._screen_auto_trading("crypto")
                elif label == "Auto Trading (All)":
                    await self._screen_auto_trading("all")
                elif label == "Portfolio":
                    await self._screen_portfolio()
                elif label == "Advanced":
                    await self._run_advanced_menu(mode)

            except KeyboardInterrupt:
                console.print("\n[bold]Exiting Poly-Oracle.[/bold]")
                break

    async def _run_advanced_menu(self, mode: str) -> None:
        """Advanced submenu loop. 'Back' returns to main menu."""
        while True:
            try:
                console.print()
                choice = await questionary.select(
                    f"Poly-Oracle [{mode}] - Advanced",
                    choices=ADVANCED_MENU_CHOICES,
                    qmark=">>",
                    pointer=">",
                ).ask_async()

                if choice is None or choice.startswith("Back"):
                    return

                action = choice.split(" - ")[0].strip().lower().replace(" ", "_")
                handler = getattr(self, f"_screen_{action}", None)
                if handler:
                    await handler()
                else:
                    console.print(f"[yellow]Unknown action: {action}[/yellow]")

            except KeyboardInterrupt:
                return

    # ------------------------------------------------------------------
    # Auto Trading (main entry point)
    # ------------------------------------------------------------------
    async def _screen_auto_trading(self, mode: str = "crypto") -> None:
        """Fully autonomous trading loop. No user prompts.

        Args:
            mode: "crypto" (BTC/ETH/SOL, 15 min) or "all" (viability selector, 60 min).
        """
        if mode == "crypto":
            interval = 15
            top_markets = 5
            mode_label = "Crypto"
        else:
            interval = 60
            top_markets = 5
            mode_label = "All Markets"

        # Pre-flight LLM check
        try:
            async with OllamaClient(
                base_url=self.settings.llm.base_url,
                model=self.settings.llm.model,
                timeout=120,
            ) as test_ollama:
                if not await test_ollama.is_available():
                    console.print(
                        f"[red]Ollama model '{self.settings.llm.model}' is not available.[/red]"
                    )
                    console.print("Run: ollama serve && ollama pull " + self.settings.llm.model)
                    return
                if not await test_ollama.can_generate():
                    console.print(
                        f"[red]Ollama model '{self.settings.llm.model}' failed generation check.[/red]"
                    )
                    return
        except Exception as exc:
            console.print(f"[red]Ollama error: {exc}[/red]")
            return

        activity_log: deque[str] = deque(maxlen=15)
        stats: dict[str, Any] = {
            "cycle": 0,
            "start_time": time.time(),
            "resolved_checked": 0,
            "resolved_count": 0,
            "session_pnl": 0.0,
            "scanned": 0,
            "analyzed": 0,
            "traded": 0,
            "skipped": 0,
        }

        trading_mode = "PAPER MODE" if self.settings.paper_trading else "LIVE MODE"

        def _build_display(next_cycle_at: float) -> Table:
            now = time.time()
            uptime_s = int(now - stats["start_time"])
            h, remainder = divmod(uptime_s, 3600)
            m, _ = divmod(remainder, 60)
            uptime_str = f"{h}h {m:02d}m"
            remaining = max(0, int(next_cycle_at - now))
            rm, rs = divmod(remaining, 60)

            bankroll = self.sqlite.get_current_bankroll()
            positions = self.sqlite.get_open_positions()
            invested = sum(p["amount_usd"] for p in positions)
            unrealized = sum(p["unrealized_pnl"] for p in positions)

            perf = self.feedback.get_performance_summary()
            total_forecasts = perf.get("resolved_forecasts", 0)
            brier = perf.get("overall_brier_calibrated")
            win_rate = perf.get("win_rate")
            total_pnl = stats["session_pnl"] + unrealized

            # Build grid
            grid = Table(
                show_header=False, box=None, padding=(0, 1), expand=True
            )
            grid.add_column(ratio=1)

            # Title
            status_text = Text()
            status_text.append(
                f"Poly-Oracle - Auto Trading [{mode_label}] [{trading_mode}]\n",
                style="bold cyan",
            )
            status_text.append(
                f"Status: Running | Cycle: {stats['cycle']} | "
                f"Interval: {interval}m | Uptime: {uptime_str}"
            )
            grid.add_row(status_text)
            grid.add_row("")

            # Sections table
            info = Table(show_header=False, box=box.SIMPLE, expand=True, padding=(0, 2))
            info.add_column("Section", style="bold", min_width=14)
            info.add_column("Details", ratio=1)

            info.add_row(
                "Resolution",
                f"Checked: {stats['resolved_checked']} | "
                f"Resolved: {stats['resolved_count']} | "
                f"Session P&L: {stats['session_pnl']:+.2f} EUR",
            )
            info.add_row(
                "Forecasting",
                f"Scanned: {stats['scanned']} | "
                f"Analyzed: {stats['analyzed']} | "
                f"Traded: {stats['traded']} | "
                f"Skipped: {stats['skipped']}",
            )
            info.add_row(
                "Portfolio",
                f"Open: {len(positions)} | "
                f"Invested: {invested:.2f} EUR | "
                f"Unrealized: {unrealized:+.2f} EUR",
            )

            stats_parts = [f"Bankroll: {bankroll:.2f} EUR", f"Total P&L: {total_pnl:+.2f} EUR"]
            if win_rate is not None:
                stats_parts.append(f"Win rate: {win_rate:.0%}")
            stats_parts.append(f"Resolved: {total_forecasts} forecasts")
            if brier is not None:
                stats_parts.append(f"Brier: {brier:.3f}")
            info.add_row("Stats", " | ".join(stats_parts))

            grid.add_row(info)
            grid.add_row("")

            # Activity log
            if activity_log:
                log_text = Text()
                log_text.append("Recent Activity\n", style="bold")
                for entry in reversed(activity_log):
                    log_text.append(entry + "\n")
                grid.add_row(log_text)

            # Countdown
            grid.add_row(Text(f"Next cycle in: {rm}m {rs:02d}s", style="dim"))
            grid.add_row(Text("Ctrl+C to stop (returns to menu)", style="dim"))

            return grid

        def _log(tag: str, msg: str) -> None:
            ts = datetime.now().strftime("%H:%M")
            style_map = {
                "TRADE": "green",
                "SKIP": "yellow",
                "RESOLVED": "cyan",
                "ERROR": "red",
                "INFO": "dim",
            }
            style = style_map.get(tag, "")
            entry = f"[{ts}] [{style}]{tag}[/{style}] {msg}" if style else f"[{ts}] {tag} {msg}"
            activity_log.append(entry)

        # ---- autonomous loop ----
        console.print(
            f"\n[bold cyan]Starting Auto Trading [{mode_label}] "
            f"— {interval} min cycles, top {top_markets} markets[/bold cyan]\n"
        )
        try:
            while True:
                stats["cycle"] += 1
                next_cycle_at = time.time() + interval * 60

                # -- run one cycle --
                _log("INFO", f"Cycle {stats['cycle']} starting...")

                # Phase 1: Resolution
                try:
                    resolution = await self.resolver.run_resolution_cycle()
                    stats["resolved_checked"] += resolution["checked"]
                    stats["resolved_count"] += resolution["resolved"]
                    stats["session_pnl"] += resolution["pnl"]
                    if resolution["resolved"] > 0:
                        _log(
                            "RESOLVED",
                            f"{resolution['resolved']} markets | P&L: {resolution['pnl']:+.2f} EUR",
                        )
                except Exception as exc:
                    _log("ERROR", f"Resolution failed: {exc}")

                # Phase 2: Select markets using the appropriate selector
                try:
                    if mode == "crypto":
                        from src.data.selectors import CryptoSelector

                        selector = CryptoSelector()
                        targets = await selector.select_markets(
                            self.polymarket, top_n=top_markets
                        )
                        stats["scanned"] += len(targets)
                    else:
                        from src.data.selectors import ViabilitySelector

                        selector = ViabilitySelector(settings=self.settings)
                        scored = await selector.select_markets(
                            self.polymarket, top_n=top_markets
                        )
                        targets = [m for m, score, reasons in scored if not reasons]
                        stats["scanned"] += len(scored)

                    bankroll = self.sqlite.get_current_bankroll()

                    for mkt in targets:
                        stats["analyzed"] += 1
                        try:
                            context_text = await self._build_context(mkt)
                            orchestrator, ollama = create_debate_system(
                                base_url=self.settings.llm.base_url,
                                model=self.settings.llm.model,
                                timeout=self.settings.llm.timeout,
                            )
                            try:
                                forecast_result = await orchestrator.run_debate(
                                    market_id=mkt.id,
                                    context=context_text,
                                    rounds=1,
                                    temperature=0.7,
                                    verbose=False,
                                )

                                confidence = forecast_result.compute_confidence()

                                calibrated = self.calibrator.calibrate(
                                    raw_forecast=forecast_result.probability,
                                    market_type=mkt.market_type,
                                    confidence=confidence,
                                )
                                edge_analysis = self.analyzer.analyze(
                                    our_forecast=calibrated,
                                    market_price=mkt.current_price,
                                    liquidity=mkt.liquidity,
                                )

                                self.feedback.record_forecast(
                                    forecast=forecast_result,
                                    market=mkt,
                                    calibrated_probability=calibrated.calibrated,
                                    edge=edge_analysis.raw_edge,
                                    recommended_action=edge_analysis.recommended_action,
                                )

                                q = _truncate(mkt.question, 50)
                                if edge_analysis.recommended_action == "TRADE":
                                    exec_result = await self.executor.execute(
                                        edge_analysis=edge_analysis,
                                        calibrated_probability=calibrated.calibrated,
                                        market=mkt,
                                        bankroll=bankroll,
                                    )
                                    if exec_result and exec_result.success:
                                        stats["traded"] += 1
                                        _log(
                                            "TRADE",
                                            f'{edge_analysis.direction} "{q}" '
                                            f"@ {mkt.current_price:.2f}",
                                        )
                                    else:
                                        stats["skipped"] += 1
                                        reason = exec_result.message if exec_result else "insufficient size"
                                        _log("SKIP", f'"{q}" | {reason}')
                                else:
                                    stats["skipped"] += 1
                                    _log(
                                        "SKIP",
                                        f'"{q}" | Edge {edge_analysis.abs_edge:.0%} < '
                                        f"{self.settings.risk.min_edge:.0%} min",
                                    )
                            finally:
                                await ollama.close()
                        except LLMError as exc:
                            _log("ERROR", f"LLM error: {exc}")
                        except Exception as exc:
                            _log("ERROR", f"Market {mkt.id[:8]}: {exc}")

                except Exception as exc:
                    _log("ERROR", f"Market scan failed: {exc}")

                _log("INFO", f"Cycle {stats['cycle']} complete.")

                # -- countdown with Live display until next cycle --
                with Live(
                    _build_display(next_cycle_at),
                    console=console,
                    refresh_per_second=1,
                    transient=True,
                ) as live:
                    while time.time() < next_cycle_at:
                        live.update(_build_display(next_cycle_at))
                        await asyncio.sleep(1)

        except KeyboardInterrupt:
            console.print("\n[bold]Trading loop stopped. Returning to menu.[/bold]")

    async def _screen_start_trading(self) -> None:
        """Legacy entry point — redirects to auto trading (all markets)."""
        await self._screen_auto_trading("all")

    # ------------------------------------------------------------------
    # Market Scanner
    # ------------------------------------------------------------------
    async def _screen_market_scanner(self) -> None:
        search = await questionary.text("Search markets (empty for all):").ask_async()
        if search is None:
            return

        with console.status("Fetching markets..."):
            if search.strip():
                all_markets = await self.polymarket.search_markets(search.strip())
            else:
                all_markets = await self.polymarket.get_active_markets()

        filtered = [
            m for m in all_markets if m.liquidity >= self.settings.risk.min_liquidity
        ]
        filtered.sort(key=lambda m: m.liquidity, reverse=True)

        if not filtered:
            console.print("[yellow]No markets found matching criteria.[/yellow]")
            return

        table = Table(title="Active Markets", box=box.ROUNDED, show_lines=False)
        table.add_column("#", style="dim", width=4)
        table.add_column("Market", min_width=40, max_width=52)
        table.add_column("Price", justify="right", width=8)
        table.add_column("Volume 24h", justify="right", width=14)
        table.add_column("Liquidity", justify="right", width=14)
        table.add_column("Days Left", justify="right", width=10)

        display_markets = filtered[:30]
        for idx, m in enumerate(display_markets, 1):
            table.add_row(
                str(idx),
                _truncate(m.question, 50),
                f"{m.current_price:.0%}",
                f"${m.volume_24h:,.0f}",
                f"${m.liquidity:,.0f}",
                f"{m.days_until_resolution:.0f}",
            )

        console.print(table)
        console.print(f"  Showing {len(display_markets)} of {len(filtered)} markets\n")

        action = await questionary.select(
            "Action:",
            choices=[
                "Run forecast on a market",
                "Back to menu",
            ],
        ).ask_async()

        if action and action.startswith("Run forecast"):
            num = await questionary.text("Market # from table:").ask_async()
            if num is None:
                return
            try:
                idx = int(num) - 1
                if 0 <= idx < len(display_markets):
                    await self._run_single_forecast(display_markets[idx])
                else:
                    console.print("[red]Invalid market number.[/red]")
            except ValueError:
                console.print("[red]Invalid input.[/red]")

    # ------------------------------------------------------------------
    # Single Forecast
    # ------------------------------------------------------------------
    async def _screen_single_forecast(self) -> None:
        market_id = await questionary.text("Market ID:").ask_async()
        if not market_id or not market_id.strip():
            return
        try:
            with console.status("Fetching market..."):
                market = await self.polymarket.get_market(market_id.strip())
            await self._run_single_forecast(market)
        except Exception as exc:
            console.print(f"[red]Error fetching market: {exc}[/red]")

    async def _run_single_forecast(self, market: Any) -> None:
        console.print(f"\n[bold]Market:[/bold] {market.question}")
        console.print(f"[bold]Price:[/bold] {market.current_price:.0%}  |  "
                       f"[bold]Liquidity:[/bold] ${market.liquidity:,.0f}\n")

        rounds = 2

        # Build context
        with console.status("[cyan]Building context...[/cyan]"):
            context_text = await self._build_context(market)

        # Create debate system
        orchestrator, ollama = create_debate_system(
            base_url=self.settings.llm.base_url,
            model=self.settings.llm.model,
            timeout=self.settings.llm.timeout,
        )

        try:
            with console.status("") as status:
                # We run the debate and update status along the way
                # Since orchestrator.run_debate is a single call, we show a spinner
                for r in range(1, rounds + 1):
                    status.update(f"[cyan]Debate round {r}/{rounds}: Bull arguing...[/cyan]")
                    await asyncio.sleep(0)  # yield

                status.update(f"[cyan]Running {rounds}-round debate...[/cyan]")
                forecast_result = await orchestrator.run_debate(
                    market_id=market.id,
                    context=context_text,
                    rounds=rounds,
                    temperature=0.7,
                    verbose=False,
                )

                status.update("[cyan]Calibrating...[/cyan]")
                confidence = forecast_result.compute_confidence()

                calibrated = self.calibrator.calibrate(
                    raw_forecast=forecast_result.probability,
                    market_type=market.market_type,
                    confidence=confidence,
                )
                edge_analysis = self.analyzer.analyze(
                    our_forecast=calibrated,
                    market_price=market.current_price,
                    liquidity=market.liquidity,
                )

                self.feedback.record_forecast(
                    forecast=forecast_result,
                    market=market,
                    calibrated_probability=calibrated.calibrated,
                    edge=edge_analysis.raw_edge,
                    recommended_action=edge_analysis.recommended_action,
                )

        finally:
            await ollama.close()

        # Display result panel
        action_style = "green" if edge_analysis.recommended_action == "TRADE" else "yellow"
        action_label = edge_analysis.recommended_action
        if action_label == "SKIP":
            reason = f"edge {edge_analysis.abs_edge:.0%} < {self.settings.risk.min_edge:.0%} threshold"
            action_label = f"SKIP ({reason})"

        result_text = (
            f"[bold]Market:[/bold]       {_truncate(market.question, 50)}\n"
            f"[bold]Market Price:[/bold] {market.current_price:.0%}\n"
            f"[bold]Our Forecast:[/bold] {forecast_result.probability:.0%} (raw) -> "
            f"{calibrated.calibrated:.0%} (calibrated)\n"
            f"[bold]Edge:[/bold]         {edge_analysis.raw_edge:+.0%} vs market\n"
            f"[bold]Direction:[/bold]    {edge_analysis.direction}\n"
            f"[bold]Confidence:[/bold]   {confidence:.0%}\n"
            f"[bold]Action:[/bold]       [{action_style}]{action_label}[/{action_style}]"
        )

        console.print(Panel(result_text, title="Forecast Result", box=box.ROUNDED))

        # Offer to execute if TRADE
        if edge_analysis.recommended_action == "TRADE":
            execute = await questionary.confirm("Execute paper trade?", default=False).ask_async()
            if execute:
                bankroll = self.sqlite.get_current_bankroll()
                exec_result = await self.executor.execute(
                    edge_analysis=edge_analysis,
                    calibrated_probability=calibrated.calibrated,
                    market=market,
                    bankroll=bankroll,
                )
                if exec_result and exec_result.success:
                    console.print(f"[green][TRADE] {exec_result.message}[/green]")
                elif exec_result:
                    console.print(f"[red][FAIL] {exec_result.message}[/red]")
                else:
                    console.print("[yellow][SKIP] Position size too small.[/yellow]")

    # ------------------------------------------------------------------
    # Portfolio
    # ------------------------------------------------------------------
    async def _screen_portfolio(self) -> None:
        positions = self.sqlite.get_open_positions()
        bankroll = self.sqlite.get_current_bankroll()

        # Open positions
        if positions:
            tbl = Table(title="Open Positions", box=box.ROUNDED)
            tbl.add_column("Market", min_width=30, max_width=42)
            tbl.add_column("Dir", width=8)
            tbl.add_column("Entry", justify="right", width=8)
            tbl.add_column("Current", justify="right", width=8)
            tbl.add_column("P&L", justify="right", width=10)
            tbl.add_column("Days Left", justify="right", width=10)

            total_invested = 0.0
            total_unrealized = 0.0

            for pos in positions:
                pnl = pos["unrealized_pnl"]
                total_invested += pos["amount_usd"]
                total_unrealized += pnl
                pnl_style = "green" if pnl >= 0 else "red"
                tbl.add_row(
                    _truncate(pos["market_id"], 40),
                    pos["direction"],
                    f"{pos['avg_entry_price']:.0%}",
                    f"{pos['current_price']:.0%}",
                    f"[{pnl_style}]{pnl:+.2f} EUR[/{pnl_style}]",
                    "-",
                )

            # Total row
            total_style = "green" if total_unrealized >= 0 else "red"
            tbl.add_row(
                f"[bold]TOTAL ({len(positions)} positions)[/bold]",
                "",
                "",
                "",
                f"[bold {total_style}]{total_unrealized:+.2f} EUR[/bold {total_style}]",
                "",
                style="bold",
            )
            console.print(tbl)
        else:
            console.print("[dim]No open positions.[/dim]")

        # Recently closed
        try:
            cursor = self.sqlite.conn.cursor()
            cursor.execute(
                "SELECT * FROM positions WHERE num_shares = 0 ORDER BY updated_at DESC LIMIT 10"
            )
            closed = [dict(r) for r in cursor.fetchall()]
        except Exception:
            closed = []

        if closed:
            ctbl = Table(title="Recently Closed", box=box.ROUNDED)
            ctbl.add_column("Market", min_width=30, max_width=42)
            ctbl.add_column("Dir", width=8)
            ctbl.add_column("Entry", justify="right", width=8)
            ctbl.add_column("P&L", justify="right", width=10)

            for pos in closed:
                pnl = pos.get("unrealized_pnl", 0.0)
                pnl_style = "green" if pnl >= 0 else "red"
                ctbl.add_row(
                    _truncate(pos["market_id"], 40),
                    pos["direction"],
                    f"{pos['avg_entry_price']:.0%}",
                    f"[{pnl_style}]{pnl:+.2f} EUR[/{pnl_style}]",
                )
            console.print(ctbl)

        # Summary
        invested = sum(p["amount_usd"] for p in positions)
        unrealized = sum(p["unrealized_pnl"] for p in positions)
        available = bankroll - invested

        # Count all trades
        try:
            cursor = self.sqlite.conn.cursor()
            total_trades = cursor.execute("SELECT COUNT(*) FROM trades").fetchone()[0]
            won_trades = cursor.execute(
                "SELECT SUM(trades_won) FROM daily_stats"
            ).fetchone()[0] or 0
            win_rate = won_trades / total_trades if total_trades > 0 else 0.0
        except Exception:
            total_trades = 0
            win_rate = 0.0

        realized = bankroll - self.settings.risk.initial_bankroll

        summary = (
            f"[bold]Bankroll:[/bold] {bankroll:.2f} EUR  |  "
            f"[bold]Invested:[/bold] {invested:.2f} EUR  |  "
            f"[bold]Available:[/bold] {available:.2f} EUR\n"
            f"[bold]Realized P&L:[/bold] {realized:+.2f} EUR  |  "
            f"[bold]Unrealized P&L:[/bold] {unrealized:+.2f} EUR\n"
            f"[bold]Trades:[/bold] {total_trades} total  |  "
            f"[bold]Win rate:[/bold] {win_rate:.0%}"
        )
        console.print(Panel(summary, title="Summary", box=box.ROUNDED))

    # ------------------------------------------------------------------
    # Performance
    # ------------------------------------------------------------------
    async def _screen_performance(self) -> None:
        perf = self.feedback.get_performance_summary()

        if perf.get("error"):
            console.print(f"[red]Error loading performance: {perf['error']}[/red]")
            return

        if perf.get("resolved_forecasts", 0) == 0:
            console.print(
                "[dim]No resolved forecasts yet. Start trading to collect data.[/dim]"
            )
            return

        # Overall panel
        brier_raw = perf.get("overall_brier_raw")
        brier_cal = perf.get("overall_brier_calibrated")
        market_brier = perf.get("market_brier")
        improvement = perf.get("calibration_improvement")
        value_added = perf.get("value_added_vs_market")

        overall_parts = [
            f"[bold]Total Forecasts:[/bold] {perf['total_forecasts']}",
            f"[bold]Resolved:[/bold] {perf['resolved_forecasts']}",
            f"[bold]Pending:[/bold] {perf['pending_forecasts']}",
        ]
        if brier_raw is not None:
            overall_parts.append(f"[bold]Brier (raw):[/bold] {brier_raw:.4f}")
        if brier_cal is not None:
            overall_parts.append(f"[bold]Brier (calibrated):[/bold] {brier_cal:.4f}")
        if market_brier is not None:
            overall_parts.append(f"[bold]Market Brier:[/bold] {market_brier:.4f}")
        if improvement is not None:
            style = "green" if improvement > 0 else "red"
            overall_parts.append(
                f"[bold]Calibration Improvement:[/bold] [{style}]{improvement:+.4f}[/{style}]"
            )
        if value_added is not None:
            style = "green" if value_added > 0 else "red"
            overall_parts.append(
                f"[bold]Value vs Market:[/bold] [{style}]{value_added:+.4f}[/{style}]"
            )

        console.print(Panel("\n".join(overall_parts), title="Overall", box=box.ROUNDED))

        # By type table
        brier_by_type = perf.get("brier_by_type", {})
        if brier_by_type:
            tbl = Table(title="Performance by Market Type", box=box.ROUNDED)
            tbl.add_column("Type", min_width=15)
            tbl.add_column("Count", justify="right", width=8)
            tbl.add_column("Brier", justify="right", width=10)
            tbl.add_column("Win Rate", justify="right", width=10)

            for mtype, data in brier_by_type.items():
                tbl.add_row(
                    mtype,
                    str(data["count"]),
                    f"{data['brier_calibrated']:.4f}" if data.get("brier_calibrated") else "-",
                    "-",  # win rate per type not available from current query
                )
            console.print(tbl)

        # Edge accuracy
        avg_edge = perf.get("avg_edge")
        win_rate = perf.get("win_rate")
        if avg_edge is not None or win_rate is not None:
            edge_parts = []
            if avg_edge is not None:
                edge_parts.append(f"[bold]Avg Edge (all):[/bold] {avg_edge:.1%}")
            if win_rate is not None:
                edge_parts.append(f"[bold]Win Rate:[/bold] {win_rate:.0%}")
            console.print(Panel("\n".join(edge_parts), title="Edge Accuracy", box=box.ROUNDED))

        console.print("[dim]Brier < 0.14 = crowd level. Lower is better.[/dim]")

    # ------------------------------------------------------------------
    # System Status
    # ------------------------------------------------------------------
    async def _screen_system_status(self) -> None:
        tbl = Table(title="System Status", box=box.ROUNDED)
        tbl.add_column("Component", min_width=15, style="bold")
        tbl.add_column("Status", width=8)
        tbl.add_column("Detail", ratio=1)

        checks = await self._get_system_checks()
        for name, status, detail in checks:
            if status == "[OK]":
                style = "green"
            elif status == "[WARN]":
                style = "yellow"
            else:
                style = "red"
            tbl.add_row(name, f"[{style}]{status}[/{style}]", detail)

        console.print(tbl)

    async def _get_system_checks(self) -> list[tuple[str, str, str]]:
        checks: list[tuple[str, str, str]] = []

        # Ollama
        try:
            async with OllamaClient(
                base_url=self.settings.llm.base_url,
                model=self.settings.llm.model,
                timeout=30,
            ) as client:
                if await client.is_available():
                    checks.append(("Ollama", "[OK]", f"{self.settings.llm.model} available"))
                else:
                    checks.append(("Ollama", "[FAIL]", f"Model {self.settings.llm.model} not found"))
        except Exception as exc:
            checks.append(("Ollama", "[FAIL]", str(exc)))

        # DuckDB
        try:
            stats = self.duckdb.get_calibration_stats()
            total = stats["overall"]["count"]
            resolved_count = 0
            try:
                row = self.duckdb.conn.execute(
                    "SELECT COUNT(*) FROM forecasts WHERE outcome IS NOT NULL"
                ).fetchone()
                resolved_count = row[0] if row else 0
            except Exception:
                pass
            checks.append(
                ("DuckDB", "[OK]", f"{total} forecasts ({resolved_count} resolved)")
            )
        except Exception as exc:
            checks.append(("DuckDB", "[FAIL]", str(exc)))

        # SQLite
        try:
            positions = self.sqlite.get_open_positions()
            total_trades = 0
            try:
                cursor = self.sqlite.conn.cursor()
                total_trades = cursor.execute("SELECT COUNT(*) FROM trades").fetchone()[0]
            except Exception:
                pass
            checks.append(
                ("SQLite", "[OK]", f"{total_trades} trades, {len(positions)} open positions")
            )
        except Exception as exc:
            checks.append(("SQLite", "[FAIL]", str(exc)))

        # ChromaDB
        try:
            cstats = self.chroma.get_collection_stats()
            total_docs = sum(cstats.values())
            checks.append(("ChromaDB", "[OK]", f"{total_docs:,} documents"))
        except Exception as exc:
            checks.append(("ChromaDB", "[FAIL]", str(exc)))

        # NewsAPI
        if self.settings.data.newsapi_key:
            checks.append(("NewsAPI", "[OK]", "API key configured"))
        else:
            checks.append(("NewsAPI", "[WARN]", "No key configured (RSS only)"))

        # Polymarket
        try:
            resp = httpx.get(
                f"{self.settings.polymarket.gamma_url}/markets?limit=1", timeout=5.0
            )
            if resp.status_code == 200:
                checks.append(("Polymarket", "[OK]", "API reachable"))
            else:
                checks.append(("Polymarket", "[FAIL]", f"HTTP {resp.status_code}"))
        except Exception as exc:
            checks.append(("Polymarket", "[FAIL]", str(exc)))

        # Mode & bankroll
        mode = "PAPER" if self.settings.paper_trading else "LIVE"
        checks.append(("Mode", mode, f"{'Paper' if self.settings.paper_trading else 'Live'} trading active"))
        bankroll = self.sqlite.get_current_bankroll()
        initial = self.settings.risk.initial_bankroll
        if bankroll == 0.0 and initial > 0:
            bankroll_display = f"{initial:.2f} EUR (initial)"
        else:
            bankroll_display = f"{bankroll:.2f} EUR"
        checks.append(("Bankroll", "", bankroll_display))

        return checks

    # ------------------------------------------------------------------
    # Settings
    # ------------------------------------------------------------------
    async def _screen_settings(self) -> None:
        tbl = Table(title="Current Configuration", box=box.ROUNDED)
        tbl.add_column("Section", style="bold", width=12)
        tbl.add_column("Parameter", width=28)
        tbl.add_column("Value", ratio=1)

        s = self.settings

        # LLM
        tbl.add_row("LLM", "model", s.llm.model)
        tbl.add_row("LLM", "base_url", s.llm.base_url)
        tbl.add_row("LLM", "embedding_model", s.llm.embedding_model)
        tbl.add_row("LLM", "temperature", str(s.llm.temperature))
        tbl.add_row("LLM", "max_tokens", str(s.llm.max_tokens))
        tbl.add_row("LLM", "timeout", f"{s.llm.timeout}s")

        # Risk
        tbl.add_row("Risk", "initial_bankroll", f"{s.risk.initial_bankroll:.2f} EUR")
        tbl.add_row("Risk", "max_position_pct", f"{s.risk.max_position_pct:.0%}")
        tbl.add_row("Risk", "min_bet", f"{s.risk.min_bet:.2f} EUR")
        tbl.add_row("Risk", "max_bet", f"{s.risk.max_bet:.2f} EUR")
        tbl.add_row("Risk", "max_daily_loss_pct", f"{s.risk.max_daily_loss_pct:.0%}")
        tbl.add_row("Risk", "max_open_positions", str(s.risk.max_open_positions))
        tbl.add_row("Risk", "min_edge", f"{s.risk.min_edge:.0%}")
        tbl.add_row("Risk", "min_confidence", f"{s.risk.min_confidence:.0%}")
        tbl.add_row("Risk", "min_liquidity", f"${s.risk.min_liquidity:,.0f}")

        # Data
        tbl.add_row("Data", "newsapi_key", "***" if s.data.newsapi_key else "Not set")
        tbl.add_row("Data", "cache_ttl_news", f"{s.data.cache_ttl_news}s")

        # Polymarket
        has_api = all([s.polymarket.api_key, s.polymarket.api_secret, s.polymarket.api_passphrase])
        tbl.add_row("Polymarket", "api_credentials", "Configured" if has_api else "Not set")
        tbl.add_row("Polymarket", "gamma_url", s.polymarket.gamma_url)

        # Database
        tbl.add_row("Database", "db_dir", str(s.database.db_dir))
        tbl.add_row("Database", "duckdb_path", str(s.database.duckdb_path))
        tbl.add_row("Database", "sqlite_path", str(s.database.sqlite_path))

        # General
        tbl.add_row("General", "log_level", s.log_level)
        tbl.add_row("General", "paper_trading", str(s.paper_trading))

        console.print(tbl)
        console.print("[dim]Edit .env file and restart to change settings.[/dim]")

    # ------------------------------------------------------------------
    # Trade History
    # ------------------------------------------------------------------
    async def _screen_trade_history(self) -> None:
        try:
            cursor = self.sqlite.conn.cursor()
            rows = cursor.execute(
                """
                SELECT id, market_id, direction, amount_usd, num_shares,
                       entry_price, timestamp, status
                FROM trades
                ORDER BY timestamp DESC
                LIMIT 30
                """
            ).fetchall()
        except Exception:
            rows = []

        if not rows:
            console.print("[dim]No trades recorded yet. Start trading to see history.[/dim]")
            return

        tbl = Table(title="Trade History (last 30)", box=box.ROUNDED)
        tbl.add_column("Time", width=18)
        tbl.add_column("Market", min_width=25, max_width=35)
        tbl.add_column("Direction", width=10)
        tbl.add_column("Amount", justify="right", width=10)
        tbl.add_column("Shares", justify="right", width=10)
        tbl.add_column("Entry", justify="right", width=8)
        tbl.add_column("Status", width=10)

        total_volume = 0.0
        for row in rows:
            trade_id, market_id, direction, amount, shares, entry, timestamp, status = row
            total_volume += float(amount) if amount else 0.0

            if isinstance(timestamp, str):
                ts_str = timestamp[:19].replace("T", " ")
            else:
                ts_str = str(timestamp)[:19]

            tbl.add_row(
                ts_str,
                _truncate(str(market_id), 33),
                str(direction),
                f"${float(amount):.2f}" if amount else "-",
                f"{float(shares):.2f}" if shares else "-",
                f"${float(entry):.2f}" if entry else "-",
                str(status),
            )

        console.print(tbl)
        console.print(
            f"  [bold]Total:[/bold] {len(rows)} trades  |  "
            f"[bold]Volume:[/bold] ${total_volume:,.2f}\n"
        )

    # ------------------------------------------------------------------
    # Help
    # ------------------------------------------------------------------
    async def _screen_help(self) -> None:
        help_text = (
            f"[bold cyan]Poly-Oracle v{VERSION}[/bold cyan]\n"
            "Multi-agent forecasting system for Polymarket prediction markets.\n"
            "\n"
            "[bold]Architecture[/bold]\n"
            "  Data Layer    : Polymarket API, NewsAPI/RSS, ChromaDB embeddings\n"
            "  Agent Layer   : Bull/Bear/Devil's Advocate debate via local LLM (Ollama)\n"
            "  Calibration   : Brier scoring, isotonic regression, feedback loops\n"
            "  Execution     : Kelly criterion sizing, risk management, paper/live trading\n"
            "\n"
            "[bold]Main Menu[/bold]\n"
            "  Auto Trading (Crypto) - BTC/ETH/SOL markets every 15 min\n"
            "  Auto Trading (All)    - All viable markets every 60 min\n"
            "  Portfolio             - Open positions and P&L summary\n"
            "  Advanced              - All other tools (scanner, backtest, etc.)\n"
            "\n"
            "[bold]Advanced Menu[/bold]\n"
            "  Market Scanner  - Browse and filter active Polymarket markets\n"
            "  Single Forecast - Run a multi-agent debate on a specific market\n"
            "  Trade History   - See recently executed trades\n"
            "  Performance     - Brier scores, calibration curves, win rate\n"
            "  Equity Curve    - Portfolio value and drawdown charts\n"
            "  Backtest        - Historical replay or full LLM simulation\n"
            "  System Status   - Health check of all components\n"
            "  Settings        - View current configuration\n"
            "\n"
            "[bold]CLI Commands[/bold]\n"
            "  python cli.py start                      Launch interactive terminal\n"
            "  python cli.py start --paper              Force paper trading mode\n"
            "  python cli.py start --live               Force live trading mode\n"
            "  python cli.py paper --once               Run one paper trading cycle\n"
            "  python cli.py live --once                Run one live trading cycle\n"
            "  python cli.py trade --mode crypto        Crypto-focused trading\n"
            "  python cli.py trade --mode auto          Auto viability selector\n"
            "  python cli.py backtest                   Fast replay backtest\n"
            "  python cli.py backtest --full            Full LLM backtest\n"
            "  python cli.py forecast <market_id>       Forecast a specific market\n"
            "  python cli.py markets                    List active markets\n"
            "  python cli.py positions                  Show open positions\n"
            "  python cli.py trades                     Show trade history\n"
            "  python cli.py status                     System status check\n"
            "  python cli.py calibration                Calibration report\n"
        )
        console.print(Panel(help_text, title="Help", box=box.ROUNDED))

    # ------------------------------------------------------------------
    # Equity Curve
    # ------------------------------------------------------------------
    async def _screen_equity_curve(self) -> None:
        """Display equity curve using plotext ASCII chart."""
        try:
            cursor = self.sqlite.conn.cursor()
            rows = cursor.execute(
                "SELECT date, ending_bankroll FROM daily_stats ORDER BY date ASC"
            ).fetchall()
        except Exception:
            rows = []

        if not rows:
            console.print("[dim]No daily stats yet. Start trading to build equity data.[/dim]")
            return

        dates = [str(r[0]) for r in rows]
        values = [float(r[1]) for r in rows]

        if HAS_PLOTEXT:
            plt.clear_figure()
            plt.plot(values, label="Equity")
            plt.title("Equity Curve")
            plt.xlabel("Day")
            plt.ylabel("EUR")
            plt.theme("dark")
            chart_str = plt.build()
            console.print(Panel(chart_str, title="Equity Curve", box=box.ROUNDED))
        else:
            tbl = Table(title="Equity Curve (install plotext for chart)", box=box.ROUNDED)
            tbl.add_column("Date", width=12)
            tbl.add_column("Equity", justify="right", width=12)
            for d, v in zip(dates[-20:], values[-20:]):
                tbl.add_row(d, f"{v:.2f}")
            console.print(tbl)

        # Drawdown
        if len(values) >= 2:
            peak = values[0]
            max_dd = 0.0
            dd_values = []
            for v in values:
                if v > peak:
                    peak = v
                dd = (peak - v) / peak if peak > 0 else 0
                dd_values.append(dd)
                max_dd = max(max_dd, dd)

            if HAS_PLOTEXT:
                plt.clear_figure()
                plt.plot([-d * 100 for d in dd_values], label="Drawdown %")
                plt.title("Drawdown")
                plt.xlabel("Day")
                plt.ylabel("% from peak")
                plt.theme("dark")
                dd_chart = plt.build()
                console.print(Panel(dd_chart, title="Drawdown", box=box.ROUNDED))

            console.print(f"[bold]Max Drawdown:[/bold] {max_dd:.1%}")

    # ------------------------------------------------------------------
    # Backtest
    # ------------------------------------------------------------------
    async def _screen_backtest(self) -> None:
        """Run backtest from terminal menu."""
        from src.calibration.backtester import Backtester

        mode = await questionary.select(
            "Backtest mode:",
            choices=["Replay (fast, no LLM)", "Full (slow, with LLM)", "Cancel"],
        ).ask_async()

        if mode is None or mode.startswith("Cancel"):
            return

        bt = Backtester(duckdb=self.duckdb, settings=self.settings)

        if mode.startswith("Replay"):
            with console.status("[cyan]Running replay backtest...[/cyan]"):
                result = bt.run_replay()
        else:
            with console.status("[cyan]Running full backtest (this may take a while)...[/cyan]"):
                result = await bt.run_full(
                    polymarket=self.polymarket,
                    news_client=self.news,
                    chroma=self.chroma,
                )

        if result.total_forecasts == 0:
            console.print("[dim]No resolved forecasts for backtesting.[/dim]")
            return

        # Results table
        tbl = Table(title="Backtest Results", box=box.ROUNDED)
        tbl.add_column("Metric", style="bold", width=20)
        tbl.add_column("Value", justify="right", width=15)

        tbl.add_row("Total Forecasts", str(result.total_forecasts))
        tbl.add_row("Simulated Trades", str(result.simulated_trades))
        tbl.add_row("Final Equity", f"${result.final_equity:.2f}")
        tbl.add_row("Max Drawdown", f"{result.max_drawdown:.1%}")
        if result.sharpe_ratio is not None:
            tbl.add_row("Sharpe Ratio", f"{result.sharpe_ratio:.2f}")
        tbl.add_row("Win Rate", f"{result.win_rate:.1%}")
        tbl.add_row("Brier Score", f"{result.brier_score:.4f}")

        console.print(tbl)

        # Equity curve chart
        if result.equity_curve and HAS_PLOTEXT:
            plt.clear_figure()
            plt.plot(result.equity_curve, label="Equity")
            plt.title("Backtest Equity Curve")
            plt.theme("dark")
            console.print(Panel(plt.build(), title="Equity", box=box.ROUNDED))

    # ------------------------------------------------------------------
    # Risk alerts helper
    # ------------------------------------------------------------------
    def _get_risk_alerts(self) -> list[str]:
        """Return list of active risk warnings."""
        alerts: list[str] = []
        try:
            bankroll = self.sqlite.get_current_bankroll()
            initial = self.settings.risk.initial_bankroll

            # Bankroll critically low
            if initial > 0 and bankroll < initial * 0.20:
                alerts.append(
                    f"Bankroll is {bankroll:.2f} EUR — below 20% of initial ({initial:.2f} EUR)"
                )

            # Max open positions
            positions = self.sqlite.get_open_positions()
            if len(positions) >= self.settings.risk.max_open_positions:
                alerts.append(
                    f"Open positions ({len(positions)}) at maximum ({self.settings.risk.max_open_positions})"
                )

            # Daily loss approaching limit
            try:
                daily_stats = self.sqlite.get_daily_stats(datetime.now(timezone.utc).date())
                if daily_stats:
                    daily_loss = abs(float(daily_stats.get("realized_pnl", 0.0)))
                    limit = initial * self.settings.risk.max_daily_loss_pct
                    if limit > 0 and daily_loss >= limit * 0.80:
                        alerts.append(
                            f"Daily loss ({daily_loss:.2f}) is >=80% of limit ({limit:.2f})"
                        )
            except Exception:
                pass

        except Exception:
            pass
        return alerts

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    async def _build_context(self, market: Any) -> str:
        builder = ContextBuilder(
            polymarket_client=self.polymarket,
            news_client=self.news,
            chroma_client=self.chroma,
        )
        return await builder.build_context(market)


# ------------------------------------------------------------------
# Factory
# ------------------------------------------------------------------
def create_dashboard() -> TerminalDashboard:
    """Create a fully-wired TerminalDashboard ready for .run()."""
    settings = get_settings()

    # Ensure DB dirs exist
    settings.database.db_dir.mkdir(parents=True, exist_ok=True)

    # Storage clients
    sqlite = SQLiteClient(settings.database.sqlite_path)
    sqlite.initialize_schema()
    sqlite.seed_initial_bankroll(settings.risk.initial_bankroll)
    duckdb = DuckDBClient(settings.database.duckdb_path)
    duckdb.initialize_schema()
    chroma = ChromaClient(settings.database.chroma_path, settings.llm.embedding_model)

    # Data sources
    polymarket = PolymarketClient()
    news_client = NewsClient()

    # Calibration
    calibrator = CalibratorAgent(history_db=duckdb)
    analyzer = MetaAnalyzer(
        min_edge=settings.risk.min_edge,
        min_confidence=settings.risk.min_confidence,
        min_liquidity=settings.risk.min_liquidity,
    )
    feedback = FeedbackLoop(db=duckdb, calibrator=calibrator)

    # Resolution
    resolver = MarketResolver(
        polymarket=polymarket,
        feedback=feedback,
        sqlite=sqlite,
        duckdb=duckdb,
    )

    # Execution — use LiveTradingExecutor when not in paper mode
    sizer = PositionSizer(risk_settings=settings.risk)
    risk_manager = RiskManager(risk_settings=settings.risk)

    if settings.paper_trading:
        executor = PaperTradingExecutor(
            sqlite=sqlite,
            sizer=sizer,
            risk=risk_manager,
        )
    else:
        from src.execution import LiveTradingExecutor

        executor = LiveTradingExecutor(
            sqlite=sqlite,
            sizer=sizer,
            risk=risk_manager,
            settings=settings,
        )

    return TerminalDashboard(
        settings=settings,
        polymarket=polymarket,
        news=news_client,
        sqlite=sqlite,
        duckdb=duckdb,
        chroma=chroma,
        resolver=resolver,
        calibrator=calibrator,
        analyzer=analyzer,
        feedback=feedback,
        sizer=sizer,
        risk_manager=risk_manager,
        executor=executor,
    )
