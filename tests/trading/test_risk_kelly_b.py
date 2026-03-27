"""
Tests for Kelly b-parameter and guard improvements from the review pass:
  - _get_historical_b uses mean_winner/mean_loser from trade history
  - _get_historical_b falls back to 2.0 when insufficient data
  - Kelly guard runs at most once per calendar day
"""
from __future__ import annotations

from datetime import date, datetime, timezone

import pytest

from src.trading.risk import RiskEngine


async def _insert_closed_trades(conn, winners: list[float], losers: list[float]) -> None:
    """Seed closed trades with specified PNL values."""
    from src.history.db import _wlock
    now = datetime.now(timezone.utc).isoformat()
    for pnl in winners + losers:
        exit_price = 100.0 + pnl
        async with _wlock():
            await conn.execute(
                """INSERT INTO trades
                   (decision_id, ticker, side, qty, entry_price, stop_loss, take_profit,
                    exit_price, status, pnl, pnl_pct, opened_at, closed_at)
                   VALUES (NULL, 'TEST', 'buy', 1.0, 100.0, 97.0, 106.0,
                           ?, 'closed', ?, ?, ?, ?)""",
                (exit_price, pnl, pnl / 100.0, now, now),
            )
            await conn.commit()


# ── _get_historical_b ────────────────────────────────────────────────────────

async def test_historical_b_falls_back_to_2_with_no_trades(tmp_db):
    """Empty trade history → b=2.0 default."""
    engine = RiskEngine("conservative")
    b = await engine._get_historical_b(tmp_db)
    assert b == 2.0


async def test_historical_b_falls_back_with_only_winners(tmp_db):
    """All winners, no losers → insufficient data → b=2.0."""
    await _insert_closed_trades(tmp_db, winners=[5.0, 3.0], losers=[])
    engine = RiskEngine("conservative")
    b = await engine._get_historical_b(tmp_db)
    assert b == 2.0


async def test_historical_b_computed_correctly(tmp_db):
    """mean_winner=6, mean_loser=2 → b=3.0."""
    await _insert_closed_trades(tmp_db, winners=[6.0, 6.0], losers=[-2.0, -2.0])
    engine = RiskEngine("conservative")
    b = await engine._get_historical_b(tmp_db)
    assert abs(b - 3.0) < 0.01


async def test_historical_b_floored_at_0_1(tmp_db):
    """b can't go below 0.1 (guard against degenerate data)."""
    # Tiny winners, huge losers → b < 0.1 without floor
    await _insert_closed_trades(tmp_db, winners=[0.01], losers=[-100.0])
    engine = RiskEngine("conservative")
    b = await engine._get_historical_b(tmp_db)
    assert b >= 0.1


# ── Kelly guard date-gating ──────────────────────────────────────────────────

async def test_kelly_guard_runs_only_once_per_day(tmp_db):
    """_update_kelly_guard must not re-check if already ran today."""
    # Seed 55 losing trades so Sharpe < 1 and guard would activate
    await _insert_closed_trades(tmp_db, winners=[], losers=[-2.0] * 55)
    engine = RiskEngine("conservative")
    engine._kelly_enabled = True  # bypass Kelly transition check

    # First call should activate the guard (Sharpe of all losers < 1.0)
    await engine._update_kelly_guard(tmp_db)
    assert engine._kelly_guard_active is True
    assert engine._kelly_guard_last_checked == date.today()

    # Manually flip guard off to detect if it re-runs
    engine._kelly_guard_active = False

    # Second call same day must be a no-op (date-gated)
    await engine._update_kelly_guard(tmp_db)
    assert engine._kelly_guard_active is False  # unchanged — guard didn't re-run
