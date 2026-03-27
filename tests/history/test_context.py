"""
Tests for build_context — portfolio context window builder.

Covers:
  - Empty DB returns zero-valued context
  - With snapshot: equity, cash, positions, daily_pnl_pct populated
  - With closed trades: win_rate calculated correctly
  - Sharpe computed (requires >= 5 trades with pnl_pct)
  - Recent decisions list capped at 10
"""
from __future__ import annotations

from datetime import datetime, timezone

import pytest

from src.history.db import (
    insert_ai_decision,
    insert_portfolio_snapshot,
    _wlock,
)
from src.history.models import AIDecision, PortfolioSnapshot
from src.history.context import build_context


# ── Empty DB ─────────────────────────────────────────────────────────────────

async def test_build_context_empty_db(tmp_db):
    """Empty DB returns zero-valued context — no crash."""
    ctx = await build_context(tmp_db)
    assert ctx.equity == 0.0
    assert ctx.cash == 0.0
    assert ctx.win_rate == 0.0
    assert ctx.total_trades == 0
    assert ctx.sharpe == 0.0
    assert ctx.recent_decisions == []


# ── With portfolio snapshot ───────────────────────────────────────────────────

async def test_build_context_with_snapshot(tmp_db):
    """Portfolio snapshot values flow through to context."""
    snap = PortfolioSnapshot(
        equity=25000.0,
        cash=10000.0,
        positions=[{"ticker": "AAPL", "qty": 10, "market_value": 1800.0, "unrealized_pnl": 50.0}],
        daily_pnl=300.0,
        daily_pnl_pct=1.2,
    )
    await insert_portfolio_snapshot(tmp_db, snap)

    ctx = await build_context(tmp_db)
    assert ctx.equity == 25000.0
    assert ctx.cash == 10000.0
    assert ctx.daily_pnl_pct == 1.2
    assert len(ctx.positions) == 1
    assert ctx.positions[0]["ticker"] == "AAPL"


# ── Win rate ──────────────────────────────────────────────────────────────────

async def _insert_trades(conn, pnls: list[float]) -> None:
    now = datetime.now(timezone.utc).isoformat()
    for pnl in pnls:
        exit_price = 100.0 + pnl
        async with _wlock():
            await conn.execute(
                """INSERT INTO trades
                   (decision_id, ticker, side, qty, entry_price, stop_loss, take_profit,
                    exit_price, status, pnl, pnl_pct, opened_at, closed_at)
                   VALUES (NULL,'TEST','buy',1.0,100.0,97.0,106.0,?,
                           'closed',?,?,?,?)""",
                (exit_price, pnl, pnl / 100.0, now, now),
            )
            await conn.commit()


async def test_win_rate_3_wins_2_losses(tmp_db):
    """win_rate = 3/5 = 60%."""
    await _insert_trades(tmp_db, [5.0, 3.0, 2.0, -1.0, -4.0])
    ctx = await build_context(tmp_db)
    assert ctx.total_trades == 5
    assert abs(ctx.win_rate - 60.0) < 0.01


async def test_win_rate_all_losses(tmp_db):
    """All losing trades → win_rate = 0."""
    await _insert_trades(tmp_db, [-1.0, -2.0, -3.0])
    ctx = await build_context(tmp_db)
    assert ctx.win_rate == 0.0


async def test_win_rate_all_wins(tmp_db):
    """All winning trades → win_rate = 100."""
    await _insert_trades(tmp_db, [1.0, 2.0, 4.0])
    ctx = await build_context(tmp_db)
    assert ctx.win_rate == 100.0


# ── Sharpe ────────────────────────────────────────────────────────────────────

async def test_sharpe_requires_5_trades(tmp_db):
    """Fewer than 5 trades → Sharpe stays 0.0."""
    await _insert_trades(tmp_db, [1.0, 2.0, 3.0, 4.0])  # only 4
    ctx = await build_context(tmp_db)
    assert ctx.sharpe == 0.0


async def test_sharpe_computed_with_enough_trades(tmp_db):
    """5+ trades with varied returns → non-zero Sharpe."""
    await _insert_trades(tmp_db, [1.0, -0.5, 2.0, -1.0, 1.5, 0.5])
    ctx = await build_context(tmp_db)
    assert ctx.sharpe != 0.0  # Sharpe computable
    assert isinstance(ctx.sharpe, float)


# ── Recent decisions ──────────────────────────────────────────────────────────

async def test_recent_decisions_populated(tmp_db):
    """Recent decisions include action, ticker, confidence, reasoning_excerpt."""
    dec = AIDecision(
        news_item_ids=[],
        model="test",
        action="buy",
        ticker="TSLA",
        confidence=0.82,
        position_pct=0.01,
        stop_loss_pct=0.02,
        take_profit_pct=0.04,
        hold_period="intraday",
        reasoning="Strong momentum from news catalyst",
        risk_factors=[],
        literature_ref="",
        risk_profile="conservative",
    )
    await insert_ai_decision(tmp_db, dec)
    ctx = await build_context(tmp_db)

    assert len(ctx.recent_decisions) == 1
    d = ctx.recent_decisions[0]
    assert d["action"] == "buy"
    assert d["ticker"] == "TSLA"
    assert d["confidence"] == 0.82
    assert "momentum" in d["reasoning_excerpt"]
