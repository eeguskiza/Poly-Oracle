"""
Simple web dashboard for Poly-Oracle.

This dashboard is intentionally dependency-light (stdlib HTTP server)
so `python cli.py start` can open a local UI without extra setup.
"""
from __future__ import annotations

import json
import threading
import webbrowser
from datetime import datetime, timezone
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

import httpx
from loguru import logger

from config.settings import Settings
from src.data.storage.sqlite_client import SQLiteClient


DASHBOARD_HTML = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Poly-Oracle Dashboard</title>
  <style>
    :root {
      color-scheme: dark;
      --bg: #0b1020;
      --card: #131a2a;
      --muted: #8892b0;
      --text: #e6edf3;
      --good: #22c55e;
      --bad: #ef4444;
      --accent: #60a5fa;
      --warn: #f59e0b;
    }
    body {
      margin: 0;
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Arial, sans-serif;
      background: radial-gradient(circle at top, #15203c, var(--bg));
      color: var(--text);
    }
    .layout {
      display: grid;
      grid-template-columns: 290px minmax(0, 1fr);
      gap: 16px;
      max-width: 1250px;
      margin: 20px auto;
      padding: 0 16px;
    }
    .container { min-width: 0; }
    h1 { margin: 0 0 6px; font-size: 1.8rem; }
    .subtitle { color: var(--muted); margin-bottom: 20px; }
    .grid {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
      gap: 12px;
      margin-bottom: 16px;
    }
    .card {
      background: rgba(19, 26, 42, 0.92);
      border: 1px solid rgba(96, 165, 250, 0.18);
      border-radius: 12px;
      padding: 14px;
      box-shadow: 0 8px 24px rgba(0, 0, 0, 0.25);
    }
    .label { color: var(--muted); font-size: 0.85rem; margin-bottom: 8px; }
    .value { font-size: 1.4rem; font-weight: 700; }
    .good { color: var(--good); }
    .bad { color: var(--bad); }
    .warn { color: var(--warn); }
    .section-title { margin: 20px 0 10px; font-size: 1.05rem; color: var(--accent); }
    .bars { display: flex; flex-direction: column; gap: 8px; }
    .bar-row { display: grid; grid-template-columns: 140px 1fr 90px; gap: 8px; align-items: center; }
    .bar-track { background: #1f2940; border-radius: 999px; height: 12px; overflow: hidden; }
    .bar-fill { height: 100%; background: linear-gradient(90deg, #3b82f6, #22d3ee); }
    table {
      width: 100%;
      border-collapse: collapse;
      font-size: 0.9rem;
      overflow: hidden;
      border-radius: 12px;
    }
    th, td { padding: 10px; border-bottom: 1px solid #26324f; text-align: left; }
    th { color: var(--muted); font-weight: 600; }
    tr:hover { background: rgba(96, 165, 250, 0.08); }
    .badge {
      font-size: 0.75rem;
      color: #cbd5e1;
      border: 1px solid #334155;
      border-radius: 999px;
      padding: 2px 8px;
      margin-left: 6px;
    }
        .sidebar {
      position: sticky;
      top: 16px;
      height: fit-content;
      background: rgba(19, 26, 42, 0.92);
      border: 1px solid rgba(96, 165, 250, 0.18);
      border-radius: 14px;
      padding: 14px;
      box-shadow: 0 8px 24px rgba(0, 0, 0, 0.25);
    }
    .sidebar h2 { margin: 0 0 10px; font-size: 1.05rem; }
    .user-row { margin: 8px 0; color: #cbd5e1; font-size: 0.92rem; }
    .settings-btn {
      background: #1f2940;
      border: 1px solid #334155;
      color: var(--text);
      border-radius: 10px;
      padding: 8px 10px;
      width: 100%;
      text-align: left;
      cursor: pointer;
      margin-top: 10px;
      font-weight: 600;
    }
    .settings-btn:hover { border-color: var(--accent); }
    .settings-panel {
      margin-top: 10px;
      border-top: 1px solid #26324f;
      padding-top: 10px;
      color: #cbd5e1;
      font-size: 0.88rem;
      display: none;
    }
    .settings-row { margin: 6px 0; }
    .settings-row code { color: #93c5fd; }
    .chip { display: inline-block; font-size: 0.75rem; padding: 2px 8px; border-radius: 999px; border: 1px solid #334155; }
    .chart-wrap { height: 240px; }
    #tradeChart { width: 100%; height: 100%; display: block; }
    .toggle-sidebar-btn {
      background: #1f2940;
      border: 1px solid #334155;
      color: var(--text);
      border-radius: 8px;
      padding: 6px 10px;
      cursor: pointer;
      margin-bottom: 12px;
      font-weight: 600;
    }
    body.sidebar-collapsed .layout { grid-template-columns: 1fr; }
    body.sidebar-collapsed .sidebar { display: none; }
    @media (max-width: 980px) {
      .layout { grid-template-columns: 1fr; }
      .sidebar { position: static; }
    }
  </style>
</head>
<body>
  <div class="layout">
    <aside class="sidebar">
      <h2>Operator Panel</h2>
      <div class="user-row"><strong>Mode:</strong> <span id="sidebarMode" class="chip">paper</span></div>
      <div class="user-row"><strong>Username:</strong> <span id="sidebarUsername">-</span></div>
      <div class="user-row"><strong>Wallet:</strong> <span id="sidebarWallet">-</span></div>
      <div class="user-row"><strong>Balance:</strong> <span id="sidebarBalance">-</span></div>
      <div class="user-row"><strong>Status:</strong> <span id="sidebarStatus">-</span></div>
      <button id="settingsButton" class="settings-btn">⚙️ Ajustes</button>
      <div id="settingsPanel" class="settings-panel">
        <div class="settings-row">Refresh dashboard: <code>5s</code></div>
        <div class="settings-row">Comando de control: <code>python cli.py paper ...</code></div>
        <div class="settings-row">Credenciales live: usa claves <code>POLYMARKET_API_*</code> del <code>.env</code>.</div>
      </div>
    </aside>

    <div class="container">
      <h1>Poly-Oracle Dashboard <span class="badge" id="modeBadge">paper mode</span></h1>
      <div class="subtitle" id="updatedAt">Loading...</div>
      <button id="sidebarToggle" class="toggle-sidebar-btn">☰ Ocultar/mostrar barra lateral</button>

      <div class="grid" id="kpis"></div>

      <div class="section-title">Portfolio Allocation (by position amount)</div>
      <div class="card">
        <div class="bars" id="allocationBars"></div>
      </div>

      <div class="section-title">Trade Amount Trend (last 20 trades)</div>
      <div class="card chart-wrap">
        <canvas id="tradeChart" width="1000" height="240"></canvas>
      </div>

      <div class="section-title">Recent Trades</div>
      <div class="card">
        <table>
          <thead>
            <tr>
              <th>Time</th>
              <th>Market</th>
              <th>Direction</th>
              <th>Amount</th>
              <th>Status</th>
            </tr>
          </thead>
          <tbody id="tradesBody"></tbody>
        </table>
      </div>
    </div>
  </div>

  <script>
    function fmtUsd(v) { return `$${Number(v || 0).toFixed(2)}`; }
    function pct(v) { return `${(Number(v || 0) * 100).toFixed(1)}%`; }

    function kpiCard(label, value, cls="") {
      return `<div class="card"><div class="label">${label}</div><div class="value ${cls}">${value}</div></div>`;
    }

    function render(summary) {
      const pnlClass = summary.total_unrealized_pnl >= 0 ? "good" : "bad";
      const wrClass = summary.win_rate >= 0.5 ? "good" : "bad";
      document.getElementById("updatedAt").textContent = `Updated: ${summary.generated_at}`;

      const modeLabel = summary.mode === "live" ? "live mode" : "paper mode";
      document.getElementById("modeBadge").textContent = modeLabel;

      document.getElementById("kpis").innerHTML = [
        kpiCard("Bankroll", fmtUsd(summary.bankroll)),
        kpiCard("Open Positions", String(summary.open_positions_count)),
        kpiCard("Unrealized P&L", fmtUsd(summary.total_unrealized_pnl), pnlClass),
        kpiCard("Trades Today", String(summary.trades_today)),
        kpiCard("Win Rate Today", pct(summary.win_rate), wrClass),
        kpiCard("Net P&L Today", fmtUsd(summary.net_pnl_today), summary.net_pnl_today >= 0 ? "good" : "bad"),
      ].join("");

      const account = summary.account || {};

      document.getElementById("sidebarMode").textContent = summary.mode || "paper";
      document.getElementById("sidebarUsername").textContent = account.username || "not configured";
      document.getElementById("sidebarWallet").textContent = account.wallet_address || "not configured";
      document.getElementById("sidebarBalance").textContent =
        account.live_balance_usdc !== null && account.live_balance_usdc !== undefined
          ? fmtUsd(account.live_balance_usdc)
          : "unavailable";
      document.getElementById("sidebarStatus").textContent = account.status || "unknown";

      const positions = summary.positions || [];
      const maxAmount = Math.max(1, ...positions.map(p => Number(p.amount_usd || 0)));
      document.getElementById("allocationBars").innerHTML = positions.length
        ? positions.map((p) => {
            const width = Math.max(2, (Number(p.amount_usd || 0) / maxAmount) * 100);
            return `<div class="bar-row">
              <div>${p.market_id}</div>
              <div class="bar-track"><div class="bar-fill" style="width:${width}%"></div></div>
              <div>${fmtUsd(p.amount_usd)}</div>
            </div>`;
          }).join("")
        : "<div class='label'>No open positions</div>";

      const trades = summary.recent_trades || [];
      document.getElementById("tradesBody").innerHTML = trades.length
        ? trades.map((t) => `<tr>
            <td>${t.timestamp}</td>
            <td>${t.market_id}</td>
            <td>${t.direction}</td>
            <td>${fmtUsd(t.amount_usd)}</td>
            <td>${t.status}</td>
          </tr>`).join("")
        : "<tr><td colspan='5'>No trades yet</td></tr>";

      drawTradeChart(trades);
    }

    function drawTradeChart(trades) {
      const canvas = document.getElementById("tradeChart");
      if (!canvas) return;
      const ctx = canvas.getContext("2d");
      const width = canvas.width;
      const height = canvas.height;
      ctx.clearRect(0, 0, width, height);

      ctx.fillStyle = "#101828";
      ctx.fillRect(0, 0, width, height);

      if (!trades.length) {
        ctx.fillStyle = "#94a3b8";
        ctx.font = "14px sans-serif";
        ctx.fillText("No trade data yet", 20, 28);
        return;
      }

      const values = [...trades].reverse().map((t) => Number(t.amount_usd || 0));
      const maxVal = Math.max(1, ...values);
      const minVal = Math.min(0, ...values);
      const span = Math.max(1, maxVal - minVal);
      const leftPad = 36;
      const rightPad = 16;
      const topPad = 16;
      const bottomPad = 24;
      const innerW = width - leftPad - rightPad;
      const innerH = height - topPad - bottomPad;

      ctx.strokeStyle = "#24324d";
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(leftPad, topPad + innerH);
      ctx.lineTo(width - rightPad, topPad + innerH);
      ctx.stroke();

      ctx.strokeStyle = "#22d3ee";
      ctx.lineWidth = 2;
      ctx.beginPath();
      values.forEach((v, i) => {
        const x = leftPad + (i / Math.max(1, values.length - 1)) * innerW;
        const y = topPad + innerH - ((v - minVal) / span) * innerH;
        if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
      });
      ctx.stroke();

      ctx.fillStyle = "#94a3b8";
      ctx.font = "11px sans-serif";
      ctx.fillText(`$${maxVal.toFixed(2)}`, 4, topPad + 10);
      ctx.fillText(`$${minVal.toFixed(2)}`, 4, topPad + innerH);
    }

    async function refresh() {
      try {
        const res = await fetch('/api/summary');
        const data = await res.json();
        render(data);
      } catch (_) {
        document.getElementById("updatedAt").textContent = "Dashboard error: could not fetch data";
      }
    }

    refresh();
    setInterval(refresh, 5000);

    document.getElementById("settingsButton").addEventListener("click", () => {
      const panel = document.getElementById("settingsPanel");
      panel.style.display = panel.style.display === "block" ? "none" : "block";
    });

    document.getElementById("sidebarToggle").addEventListener("click", () => {
      document.body.classList.toggle("sidebar-collapsed");
    });
  </script>
</body>
</html>
"""


class DashboardHTTPServer(ThreadingHTTPServer):
    """HTTP server carrying app settings for request handlers."""

    def __init__(self, server_address: tuple[str, int], settings: Settings) -> None:
        super().__init__(server_address, DashboardHandler)
        self.settings = settings


class DashboardHandler(BaseHTTPRequestHandler):
    """Serve dashboard assets and JSON API."""

    server: DashboardHTTPServer

    def do_GET(self) -> None:  # noqa: N802 (BaseHTTPRequestHandler signature)
        if self.path == "/":
            self._send_html(DASHBOARD_HTML)
            return

        if self.path == "/api/summary":
            payload = build_dashboard_summary(self.server.settings)
            self._send_json(payload)
            return

        self.send_error(HTTPStatus.NOT_FOUND, "Not Found")

    def log_message(self, fmt: str, *args: Any) -> None:
        logger.debug("web-dashboard: " + fmt, *args)

    def _send_html(self, body: str) -> None:
        raw = body.encode("utf-8")
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def _send_json(self, payload: dict[str, Any]) -> None:
        raw = json.dumps(payload).encode("utf-8")
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "application/json")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)


def _fetch_live_account_snapshot(settings: Settings) -> dict[str, Any]:
    """Best-effort fetch of live Polymarket account identity and balance."""
    account = {
        "username": settings.polymarket.username,
        "wallet_address": settings.polymarket.wallet_address,
        "live_balance_usdc": None,
        "status": "live mode (no API data yet)",
    }

    if not account["wallet_address"] and not account["username"]:
        account["status"] = "no username/wallet configured, trying API key balance"

    try:
        with httpx.Client(timeout=3.0) as client:
            profile_data = None

            if account["wallet_address"]:
                profile_url = f"{settings.polymarket.gamma_url}/users/{account['wallet_address']}"
                profile_resp = client.get(profile_url)
                if profile_resp.status_code == 200:
                    profile_data = profile_resp.json()

            if profile_data:
                account["username"] = (
                    profile_data.get("username")
                    or profile_data.get("handle")
                    or account["username"]
                )
                account["live_balance_usdc"] = (
                    profile_data.get("balance")
                    or profile_data.get("usdcBalance")
                    or profile_data.get("usdc_balance")
                )
                account["status"] = "live profile loaded from gamma-api"
                return account

            if settings.polymarket.api_key:
                headers = {"POLY_API_KEY": settings.polymarket.api_key}
                if settings.polymarket.api_passphrase:
                    headers["POLY_PASSPHRASE"] = settings.polymarket.api_passphrase

                balance_resp = client.get(
                    f"{settings.polymarket.clob_url}/balance",
                    headers=headers,
                )
                if balance_resp.status_code == 200:
                    data = balance_resp.json()
                    account["live_balance_usdc"] = (
                        data.get("balance")
                        or data.get("available")
                        or data.get("usdc")
                    )
                    account["status"] = "live balance loaded from CLOB"
                    return account

            account["status"] = "live mode active, API did not return account balance"
            return account

    except Exception as exc:
        account["status"] = f"live API unavailable: {exc}"
        return account


def build_dashboard_summary(settings: Settings) -> dict[str, Any]:
    """Aggregate portfolio KPIs for the web dashboard."""
    now = datetime.now(timezone.utc)
    today = now.strftime("%Y-%m-%d")

    summary: dict[str, Any] = {
        "generated_at": now.strftime("%Y-%m-%d %H:%M:%S UTC"),
        "mode": "paper" if settings.paper_trading else "live",
        "bankroll": 0.0,
        "open_positions_count": 0,
        "total_unrealized_pnl": 0.0,
        "trades_today": 0,
        "win_rate": 0.0,
        "net_pnl_today": 0.0,
        "positions": [],
        "recent_trades": [],
        "account": {
            "username": settings.polymarket.username,
            "wallet_address": settings.polymarket.wallet_address,
            "live_balance_usdc": None,
            "status": "paper mode",
        },
    }

    try:
        with SQLiteClient(settings.database.sqlite_path) as sqlite_client:
            positions = sqlite_client.get_open_positions()
            bankroll = sqlite_client.get_current_bankroll()
            daily_stats = sqlite_client.get_daily_stats(today)

            cursor = sqlite_client.conn.cursor()
            cursor.execute(
                """
                SELECT timestamp, market_id, direction, amount_usd, status
                FROM trades
                ORDER BY timestamp DESC
                LIMIT 20
                """
            )
            recent_trades = [dict(row) for row in cursor.fetchall()]

            total_unrealized_pnl = sum(float(p.get("unrealized_pnl", 0.0)) for p in positions)
            trades_today = int(daily_stats.get("trades_executed", 0)) if daily_stats else 0
            trades_won = int(daily_stats.get("trades_won", 0)) if daily_stats else 0
            win_rate = (trades_won / trades_today) if trades_today > 0 else 0.0
            net_pnl_today = float(daily_stats.get("net_pnl", 0.0)) if daily_stats else 0.0

            summary.update(
                {
                    "bankroll": bankroll,
                    "open_positions_count": len(positions),
                    "total_unrealized_pnl": total_unrealized_pnl,
                    "trades_today": trades_today,
                    "win_rate": win_rate,
                    "net_pnl_today": net_pnl_today,
                    "positions": [
                        {
                            "market_id": p.get("market_id", ""),
                            "amount_usd": p.get("amount_usd", 0.0),
                            "direction": p.get("direction", ""),
                            "unrealized_pnl": p.get("unrealized_pnl", 0.0),
                        }
                        for p in positions
                    ],
                    "recent_trades": recent_trades,
                }
            )

    except Exception as exc:
        logger.warning(f"Dashboard summary fallback due to error: {exc}")

    if not settings.paper_trading:
        summary["account"] = _fetch_live_account_snapshot(settings)

    return summary


def run_web_dashboard(settings: Settings, host: str = "127.0.0.1", port: int = 8787) -> None:
    """Run the local web dashboard and open the browser."""
    server = DashboardHTTPServer((host, port), settings)
    url = f"http://{host}:{port}"

    logger.info(f"Starting web dashboard at {url}")

    # Open browser in a background thread so server boot is not blocked.
    threading.Thread(target=lambda: webbrowser.open(url), daemon=True).start()

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        logger.info("Web dashboard stopped by user")
    finally:
        server.server_close()
