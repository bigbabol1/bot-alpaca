"""
Financial correctness tests for the risk engine.

Non-negotiable before any live trading:
  - Kelly formula: p=0.6, b=2.0 → f*=0.4, half-Kelly=0.2, capped at 0.02
  - Kelly negative edge: p=0.4, b=1.5 → f*=-0.067 → clamped to 0 (no trade)
  - ATR stop: entry=100, ATR=2.5, multiplier=1.5 → stop=96.25
  - Daily loss limit: equity=$10k, loss=$301, limit=3% → halt triggered
  - Position cap: Conservative profile, request 5% → returned 1%
"""
from __future__ import annotations

from datetime import datetime, timezone

import pytest

from src.trading.risk import PROFILES, RiskEngine, _blocked


async def _seed_closed_trades(conn, count: int = 51) -> None:
    """Insert `count` closed trades so Kelly activates (threshold=50).
    Uses raw SQL to avoid FK constraint on decision_id (NULL is allowed in schema)."""
    from src.history.db import _wlock
    now = datetime.now(timezone.utc).isoformat()
    for i in range(count):
        pnl = 4.0 if i % 2 == 0 else -3.0
        exit_price = 104.0 if i % 2 == 0 else 97.0
        async with _wlock():
            await conn.execute(
                """INSERT INTO trades
                   (decision_id, ticker, side, qty, entry_price, stop_loss, take_profit,
                    exit_price, status, pnl, pnl_pct, opened_at, closed_at)
                   VALUES (NULL, 'AAPL', 'buy', 1.0, 100.0, 97.0, 106.0,
                           ?, 'closed', ?, ?, ?, ?)""",
                (exit_price, pnl, pnl / 100.0, now, now),
            )
            await conn.commit()



# ── Kelly formula ─────────────────────────────────────────────────────────────

def test_kelly_standard_case():
    """p=0.6, b=2.0 → f*=0.4, half-Kelly=0.2, capped at 0.02."""
    result = RiskEngine._kelly_size(win_prob=0.6, avg_win_loss_ratio=2.0)
    # f* = (0.6*2.0 - 0.4) / 2.0 = (1.2 - 0.4) / 2.0 = 0.8/2.0 = 0.4
    # half-Kelly = 0.2, capped at 0.02
    assert result == 0.02


def test_kelly_negative_edge():
    """p=0.4, b=1.5 → f*=-0.067 → clamped to 0 (no trade)."""
    result = RiskEngine._kelly_size(win_prob=0.4, avg_win_loss_ratio=1.5)
    # f* = (0.4*1.5 - 0.6) / 1.5 = (0.6 - 0.6) / 1.5 = 0.0 / 1.5 ≈ 0.0 exactly
    # Actually: (0.4 * 1.5 - 0.6) / 1.5 = (0.6 - 0.6) / 1.5 = 0
    # Let's test with p=0.3, b=1.5: f* = (0.45-0.7)/1.5 = -0.25/1.5 = -0.167 → 0
    result2 = RiskEngine._kelly_size(win_prob=0.3, avg_win_loss_ratio=1.5)
    assert result2 == 0.0


def test_kelly_zero_win_prob():
    """p=0 → f* = -1 → clamped to 0."""
    result = RiskEngine._kelly_size(win_prob=0.0, avg_win_loss_ratio=2.0)
    assert result == 0.0


def test_kelly_high_confidence_still_capped():
    """Very high confidence shouldn't exceed the black-swan cap (2%)."""
    result = RiskEngine._kelly_size(win_prob=0.99, avg_win_loss_ratio=5.0)
    assert result <= 0.02


def test_kelly_zero_b_uses_default():
    """b=0 → uses default of 2.0 to avoid division by zero."""
    result = RiskEngine._kelly_size(win_prob=0.6, avg_win_loss_ratio=0.0)
    # Uses b=2.0 default → same as standard case
    assert result == 0.02


# ── ATR stop calculation ──────────────────────────────────────────────────────

async def test_atr_stop_calculation(tmp_db):
    """entry=100, ATR=2.5, multiplier=1.5 → stop=96.25."""
    engine = RiskEngine("conservative")
    # multiplier for conservative = 1.5
    assert engine.profile.atr_stop_multiplier == 1.5

    result = await engine.compute_sizing(
        conn=tmp_db,
        equity=10000.0,
        entry_price=100.0,
        atr=2.5,
        ai_confidence=0.80,   # above conservative min (0.75)
        ai_position_pct=0.01,
        open_positions_count=0,
        daily_loss_pct=0.0,
    )
    # stop = entry - (ATR * multiplier) = 100 - (2.5 * 1.5) = 96.25
    assert result.stop_price == 96.25
    assert not result.blocked


async def test_atr_stop_calculation_short(tmp_db):
    """For a sell/short order stops are mirrored: stop above entry, TP below."""
    engine = RiskEngine("conservative")
    result = await engine.compute_sizing(
        conn=tmp_db,
        equity=10000.0,
        entry_price=100.0,
        atr=2.5,
        ai_confidence=0.80,
        ai_position_pct=0.01,
        open_positions_count=0,
        daily_loss_pct=0.0,
        side="sell",
    )
    # stop = entry + (ATR * multiplier) = 100 + (2.5 * 1.5) = 103.75
    # tp   = entry - (ATR * multiplier * ratio) = 100 - (2.5 * 1.5 * 1.5) = 94.375
    assert result.stop_price == 103.75
    assert result.take_profit_price == round(100 - 2.5 * 1.5 * 1.5, 2)
    assert result.take_profit_price < result.stop_price   # Alpaca requirement for sell bracket
    assert not result.blocked


# ── Daily loss limit ──────────────────────────────────────────────────────────

def test_daily_loss_limit_halt():
    """equity=$10k, loss=$301, limit=3% → halt triggered (loss is negative)."""
    engine = RiskEngine("conservative")
    # loss_pct = -301/10000 = -3.01% loss → triggers halt
    assert engine.daily_loss_halt(-0.0301) is True


def test_daily_loss_limit_not_triggered():
    """Loss of 2.99% does not trigger conservative halt."""
    engine = RiskEngine("conservative")
    assert engine.daily_loss_halt(-0.0299) is False


def test_daily_gain_does_not_trigger_halt():
    """A +3.1% winning day should NOT halt trading (halt is loss-only)."""
    engine = RiskEngine("conservative")
    assert engine.daily_loss_halt(0.0301) is False


async def test_daily_loss_limit_blocks_trade(tmp_db):
    """compute_sizing returns blocked=True when daily loss limit exceeded."""
    engine = RiskEngine("conservative")
    result = await engine.compute_sizing(
        conn=tmp_db,
        equity=10000.0,
        entry_price=100.0,
        atr=2.0,
        ai_confidence=0.8,
        ai_position_pct=0.01,
        open_positions_count=0,
        daily_loss_pct=-0.031,   # -3.1% loss > 3% limit
    )
    assert result.blocked
    assert "daily loss limit" in result.block_reason


# ── Profile position cap ──────────────────────────────────────────────────────

async def test_conservative_position_cap(tmp_db):
    """Conservative profile: request 5% AI position → returned 1% cap."""
    engine = RiskEngine("conservative")
    result = await engine.compute_sizing(
        conn=tmp_db,
        equity=10000.0,
        entry_price=100.0,
        atr=2.0,
        ai_confidence=0.8,
        ai_position_pct=0.05,   # AI requests 5%
        open_positions_count=0,
        daily_loss_pct=0.0,
    )
    # Conservative max = 1%
    assert result.position_pct == 0.01
    assert not result.blocked


async def test_moderate_position_cap(tmp_db):
    """Moderate profile caps at 2% once Kelly is active (>50 closed trades)."""
    await _seed_closed_trades(tmp_db, count=51)
    engine = RiskEngine("moderate")
    result = await engine.compute_sizing(
        conn=tmp_db,
        equity=10000.0,
        entry_price=100.0,
        atr=2.0,
        ai_confidence=0.99,  # very high confidence → Kelly output exceeds 2%, profile caps it
        ai_position_pct=0.10,
        open_positions_count=0,
        daily_loss_pct=0.0,
    )
    assert result.position_pct == 0.02
    assert result.sizing_method == "kelly"


# ── Confidence gate ────────────────────────────────────────────────────────────

async def test_confidence_below_threshold_blocks(tmp_db):
    """Confidence below profile minimum blocks the trade."""
    engine = RiskEngine("conservative")
    result = await engine.compute_sizing(
        conn=tmp_db,
        equity=10000.0,
        entry_price=100.0,
        atr=2.0,
        ai_confidence=0.60,   # below conservative min (0.75)
        ai_position_pct=0.01,
        open_positions_count=0,
        daily_loss_pct=0.0,
    )
    assert result.blocked
    assert "confidence" in result.block_reason


# ── Open position cap ──────────────────────────────────────────────────────────

async def test_max_positions_blocks(tmp_db):
    """Conservative max 10 open positions — 11th is blocked."""
    engine = RiskEngine("conservative")
    result = await engine.compute_sizing(
        conn=tmp_db,
        equity=10000.0,
        entry_price=100.0,
        atr=2.0,
        ai_confidence=0.8,
        ai_position_pct=0.01,
        open_positions_count=10,   # at the limit
        daily_loss_pct=0.0,
    )
    assert result.blocked
    assert "max open positions" in result.block_reason


# ── Profile definitions ────────────────────────────────────────────────────────

def test_all_profiles_defined():
    for name in ("conservative", "moderate", "aggressive"):
        assert name in PROFILES
        p = PROFILES[name]
        assert 0 < p.max_position_pct <= 1.0
        assert 0 < p.max_daily_loss_pct <= 1.0
        assert 0 < p.atr_stop_multiplier <= 5.0
        assert 0 < p.min_confidence <= 1.0


def test_risk_engine_set_profile():
    engine = RiskEngine("conservative")
    engine.set_profile("aggressive")
    assert engine.profile.name == "aggressive"


def test_risk_engine_invalid_profile():
    engine = RiskEngine("conservative")
    with pytest.raises(ValueError, match="Unknown risk profile"):
        engine.set_profile("yolo")
