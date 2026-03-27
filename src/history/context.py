"""
Build the rolling context window injected into each Ollama decision request.

Aggregates portfolio state, recent decisions, and performance metrics
so the AI has the full picture without unbounded token growth.
"""
from __future__ import annotations

from dataclasses import dataclass

import aiosqlite

from src.history.db import get_closed_trades, get_latest_snapshot, get_recent_decisions


@dataclass
class PortfolioContext:
    equity: float
    cash: float
    positions: list[dict]
    daily_pnl_pct: float
    drawdown_pct: float
    win_rate: float
    sharpe: float
    total_trades: int
    recent_decisions: list[dict]


async def build_context(conn: aiosqlite.Connection) -> PortfolioContext:
    """Compile portfolio state for the AI prompt context injection."""
    snapshot = await get_latest_snapshot(conn)
    equity = snapshot.equity if snapshot else 0.0
    cash = snapshot.cash if snapshot else 0.0
    positions = snapshot.positions if snapshot else []
    daily_pnl_pct = snapshot.daily_pnl_pct if snapshot else 0.0

    # Drawdown from peak (approximate from snapshots)
    drawdown_pct = 0.0

    # Performance from closed trades
    closed = await get_closed_trades(conn, limit=50)
    total_trades = len(closed)
    wins = [t for t in closed if t.pnl and t.pnl > 0]
    win_rate = (len(wins) / total_trades * 100) if total_trades > 0 else 0.0

    # Sharpe ratio (simplified: mean/std of pnl_pct * sqrt(252))
    sharpe = 0.0
    if len(closed) >= 5:
        import numpy as np
        pnl_pcts = [t.pnl_pct for t in closed if t.pnl_pct is not None]
        if pnl_pcts:
            arr = np.array(pnl_pcts, dtype=float)
            std = arr.std()
            if std > 0:
                sharpe = round(float(arr.mean() / std * (252 ** 0.5)), 2)

    # Recent decisions (last 10)
    decisions = await get_recent_decisions(conn, limit=10)
    recent_decisions = [
        {
            "action": d.action,
            "ticker": d.ticker,
            "confidence": d.confidence,
            "hold_period": d.hold_period,
            "reasoning_excerpt": (d.reasoning or "")[:120],
            "created_at": d.created_at.isoformat(),
        }
        for d in decisions
    ]

    return PortfolioContext(
        equity=equity,
        cash=cash,
        positions=positions,
        daily_pnl_pct=daily_pnl_pct,
        drawdown_pct=drawdown_pct,
        win_rate=win_rate,
        sharpe=sharpe,
        total_trades=total_trades,
        recent_decisions=recent_decisions,
    )
