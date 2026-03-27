"""
Performance metrics — Sharpe ratio, win rate, max drawdown.

Runs a daily job at 16:05 ET and writes to performance_metrics table.
Also exposes summary for Telegram daily P&L message.
"""
from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

import numpy as np
import structlog

import aiosqlite

from src.history.db import get_closed_trades, get_latest_snapshot

log = structlog.get_logger(__name__)

_DAILY_SUMMARY_HOUR_ET = 16
_DAILY_SUMMARY_MINUTE_ET = 5


async def compute_and_store_metrics(
    conn: aiosqlite.Connection, period: str = "daily"
) -> dict:
    """
    Compute performance metrics from closed trades and write to DB.
    Returns a summary dict for Telegram.
    """
    trades = await get_closed_trades(conn, limit=500)
    total = len(trades)
    wins = [t for t in trades if t.pnl and t.pnl > 0]
    win_rate = (len(wins) / total) if total > 0 else 0.0

    pnl_values = [t.pnl for t in trades if t.pnl is not None]
    total_pnl = sum(pnl_values)

    # Sharpe ratio (annualized from daily pnl% returns)
    pnl_pcts = [t.pnl_pct for t in trades if t.pnl_pct is not None]
    sharpe = 0.0
    if len(pnl_pcts) >= 5:
        arr = np.array(pnl_pcts, dtype=float)
        std = arr.std()
        if std > 0:
            sharpe = float(arr.mean() / std * (252 ** 0.5))

    # Max drawdown
    max_drawdown = _compute_max_drawdown(pnl_values)

    # Average hold time
    hold_hours: list[float] = []
    for t in trades:
        if t.opened_at and t.closed_at:
            delta = t.closed_at - t.opened_at
            hold_hours.append(delta.total_seconds() / 3600)
    avg_hold = float(np.mean(hold_hours)) if hold_hours else 0.0

    row = {
        "period": period,
        "total_trades": total,
        "winning_trades": len(wins),
        "win_rate": round(win_rate, 4),
        "total_pnl": round(total_pnl, 2),
        "sharpe_ratio": round(sharpe, 3),
        "max_drawdown": round(max_drawdown, 4),
        "avg_hold_hours": round(avg_hold, 2),
        "calculated_at": datetime.now(tz=timezone.utc).isoformat(),
    }

    await conn.execute(
        """INSERT INTO performance_metrics
           (period, total_trades, winning_trades, win_rate, total_pnl,
            sharpe_ratio, max_drawdown, avg_hold_hours, calculated_at)
           VALUES (:period, :total_trades, :winning_trades, :win_rate, :total_pnl,
                   :sharpe_ratio, :max_drawdown, :avg_hold_hours, :calculated_at)""",
        row,
    )
    await conn.commit()
    log.info("metrics_computed", **{k: v for k, v in row.items() if k != "calculated_at"})
    return row


def _compute_max_drawdown(pnl_values: list[float]) -> float:
    """Compute maximum drawdown from a list of P&L values."""
    if not pnl_values:
        return 0.0
    equity_curve = np.cumsum(pnl_values)
    peak = np.maximum.accumulate(equity_curve)
    drawdowns = (equity_curve - peak) / (peak + 1e-9)
    return float(abs(drawdowns.min()))


async def daily_summary_message(conn: aiosqlite.Connection) -> str:
    """Build the daily Telegram P&L summary message."""
    metrics = await compute_and_store_metrics(conn, period="daily")
    snapshot = await get_latest_snapshot(conn)
    equity = snapshot.equity if snapshot else 0.0
    daily_pnl_pct = snapshot.daily_pnl_pct if snapshot else 0.0

    emoji = "📈" if daily_pnl_pct >= 0 else "📉"
    return (
        f"{emoji} <b>Daily Summary</b>\n"
        f"Portfolio equity: <b>${equity:,.2f}</b>\n"
        f"Today's P&amp;L: <b>{daily_pnl_pct:+.2f}%</b>\n"
        f"Win rate (all-time): {metrics['win_rate']:.1%}\n"
        f"Sharpe (all-time): {metrics['sharpe_ratio']:.2f}\n"
        f"Total trades: {metrics['total_trades']}\n"
        f"Total P&amp;L: ${metrics['total_pnl']:+,.2f}"
    )


class DailyMetricsJob:
    def __init__(self, conn: aiosqlite.Connection, alert_fn) -> None:
        self._conn = conn
        self._alert = alert_fn

    async def run(self) -> None:
        """Send daily P&L summary at 16:05 ET."""
        _ET = ZoneInfo("America/New_York")
        while True:
            now_et = datetime.now(tz=_ET)
            now_et_hour = now_et.hour
            now_et_minute = now_et.minute

            if now_et_hour == _DAILY_SUMMARY_HOUR_ET and now_et_minute == _DAILY_SUMMARY_MINUTE_ET:
                try:
                    msg = await daily_summary_message(self._conn)
                    await self._alert(msg, priority=False)
                except Exception as exc:  # noqa: BLE001
                    log.error("daily_summary_failed", error=str(exc))
                await asyncio.sleep(60)   # skip the rest of this minute
            else:
                await asyncio.sleep(30)
