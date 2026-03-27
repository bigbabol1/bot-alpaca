"""
Risk engine — applies profile limits and position sizing.

Position sizing rules:
  - Trades 1–50:  fixed 1% of equity
  - Trade 51+:    Kelly Criterion (half-Kelly, capped at 2%)
  - Kelly guard:  if last-30-trade Sharpe < 1.0, revert to 1% fixed
  - Black-swan:   effective_size = min(kelly_f * 0.5, 0.02)
  - Profile cap:  never exceed profile's max_position_pct

ATR stop calculation:
  stop_price = entry_price - (ATR * atr_multiplier)
  take_profit = entry_price + (risk * take_profit_ratio)

Financial correctness tests (non-negotiable):
  Kelly: p=0.6, b=2.0 → f*=0.4, half-Kelly=0.2, capped at 0.02
  Kelly edge case: p=0.4, b=1.5 → f*=-0.067 → clamped to 0 (no trade)
  ATR stop: entry=100, ATR=2.5, mult=1.5 → stop=96.25
  Daily loss limit: equity=$10k, loss=$301, limit=3% → halt triggered
  Position cap: Conservative, request 5% → returned 1%
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import aiosqlite
import structlog

from src.history.db import count_closed_trades, get_closed_trades

log = structlog.get_logger(__name__)

ProfileName = Literal["conservative", "moderate", "aggressive"]

_KELLY_TRANSITION = 50          # switch to Kelly after this many closed trades
_KELLY_BLACK_SWAN_CAP = 0.02    # never exceed 2% per trade regardless of Kelly


@dataclass(frozen=True)
class RiskProfile:
    name: str
    max_position_pct: float    # max % of portfolio per trade
    max_daily_loss_pct: float  # halt trading for the day above this drawdown
    max_open_positions: int    # 0 = unlimited
    atr_stop_multiplier: float
    min_confidence: float
    take_profit_ratio: float   # multiple of risk (e.g. 2.0 = take-profit at 2x stop distance)


PROFILES: dict[str, RiskProfile] = {
    "conservative": RiskProfile(
        name="conservative",
        max_position_pct=0.01,
        max_daily_loss_pct=0.03,
        max_open_positions=10,
        atr_stop_multiplier=1.5,
        min_confidence=0.75,
        take_profit_ratio=1.5,
    ),
    "moderate": RiskProfile(
        name="moderate",
        max_position_pct=0.02,
        max_daily_loss_pct=0.05,
        max_open_positions=20,
        atr_stop_multiplier=2.0,
        min_confidence=0.65,
        take_profit_ratio=2.0,
    ),
    "aggressive": RiskProfile(
        name="aggressive",
        max_position_pct=0.05,
        max_daily_loss_pct=0.10,
        max_open_positions=0,   # unlimited
        atr_stop_multiplier=3.0,
        min_confidence=0.55,
        take_profit_ratio=3.0,
    ),
}


@dataclass
class SizingResult:
    position_pct: float         # fraction of equity to use (0.0–1.0)
    stop_price: float
    take_profit_price: float
    qty: float
    sizing_method: str          # "fixed" | "kelly"
    blocked: bool = False
    block_reason: str = ""


class RiskEngine:
    def __init__(self, profile_name: str = "conservative") -> None:
        self._profile_name = profile_name
        self._kelly_enabled = False
        self._kelly_guard_active = False  # True when Kelly is temporarily disabled by Sharpe guard
        self._sharpe_ok_days = 0          # consecutive days Sharpe >= 1.0

    @property
    def profile(self) -> RiskProfile:
        return PROFILES[self._profile_name]

    def set_profile(self, name: str) -> None:
        if name not in PROFILES:
            raise ValueError(f"Unknown risk profile: {name!r}")
        old = self._profile_name
        self._profile_name = name
        log.info("risk_profile_changed", old=old, new=name)

    # ── Main sizing call ───────────────────────────────────────────────────────

    async def compute_sizing(
        self,
        conn: aiosqlite.Connection,
        equity: float,
        entry_price: float,
        atr: float | None,
        ai_confidence: float,
        ai_position_pct: float,
        open_positions_count: int,
        daily_loss_pct: float,
    ) -> SizingResult:
        """
        Compute position size, stop price, and take-profit price.

        Returns SizingResult with blocked=True if the trade should be skipped.
        """
        p = self.profile

        # 1. Confidence gate
        if ai_confidence < p.min_confidence:
            return _blocked(f"confidence {ai_confidence:.2f} < min {p.min_confidence}")

        # 2. Daily loss limit
        if abs(daily_loss_pct) >= p.max_daily_loss_pct:
            return _blocked(
                f"daily loss limit hit: {daily_loss_pct:.2%} >= {p.max_daily_loss_pct:.2%}"
            )

        # 3. Open position cap
        if p.max_open_positions > 0 and open_positions_count >= p.max_open_positions:
            return _blocked(
                f"max open positions reached: {open_positions_count}/{p.max_open_positions}"
            )

        # 4. Position sizing
        await self._update_kelly_state(conn)
        if self._kelly_enabled and not self._kelly_guard_active:
            raw_pct = self._kelly_size(ai_confidence, ai_position_pct)
            method = "kelly"
        else:
            raw_pct = 0.01   # fixed 1% fallback
            method = "fixed"

        # Apply profile cap (never exceed profile max)
        position_pct = min(raw_pct, p.max_position_pct)

        # 5. ATR stop calculation
        if atr and entry_price:
            stop_distance = atr * p.atr_stop_multiplier
            stop_price = entry_price - stop_distance
            take_profit_price = entry_price + (stop_distance * p.take_profit_ratio)
        else:
            # Fallback: percentage-based stops (from AI decision)
            stop_price = entry_price * (1.0 - 0.02)         # 2% default
            take_profit_price = entry_price * (1.0 + 0.04)  # 4% default
            log.warning("risk_no_atr_using_pct_fallback", entry_price=entry_price)

        # 6. Qty calculation
        trade_value = equity * position_pct
        qty = trade_value / entry_price if entry_price > 0 else 0.0
        qty = round(qty, 4)

        return SizingResult(
            position_pct=position_pct,
            stop_price=round(stop_price, 2),
            take_profit_price=round(take_profit_price, 2),
            qty=qty,
            sizing_method=method,
        )

    # ── Kelly logic ────────────────────────────────────────────────────────────

    async def _update_kelly_state(self, conn: aiosqlite.Connection) -> None:
        closed_count = await count_closed_trades(conn)
        if closed_count >= _KELLY_TRANSITION and not self._kelly_enabled:
            self._kelly_enabled = True
            log.info("kelly_criterion_activated", closed_trades=closed_count)

        # Kelly guard: check last-30-trade Sharpe
        if self._kelly_enabled:
            await self._update_kelly_guard(conn)

    async def _update_kelly_guard(self, conn: aiosqlite.Connection) -> None:
        trades = await get_closed_trades(conn, limit=30)
        if len(trades) < 5:
            return
        pnl_pcts = [t.pnl_pct for t in trades if t.pnl_pct is not None]
        if not pnl_pcts:
            return

        import numpy as np
        arr = np.array(pnl_pcts, dtype=float)
        std = arr.std()
        sharpe = (arr.mean() / std * (252 ** 0.5)) if std > 0 else 0.0

        was_guarded = self._kelly_guard_active
        if sharpe < 1.0:
            if not self._kelly_guard_active:
                self._kelly_guard_active = True
                self._sharpe_ok_days = 0
                log.warning("kelly_guard_activated", sharpe=round(sharpe, 2))
        else:
            self._sharpe_ok_days += 1
            if self._sharpe_ok_days >= 5 and self._kelly_guard_active:
                self._kelly_guard_active = False
                log.info("kelly_guard_lifted", sharpe=round(sharpe, 2), ok_days=self._sharpe_ok_days)
            elif not self._kelly_guard_active:
                self._sharpe_ok_days = 0

    @staticmethod
    def _kelly_size(win_prob: float, avg_win_loss_ratio: float) -> float:
        """
        Kelly Criterion: f* = (p*b - q) / b
        Returns half-Kelly, capped at _KELLY_BLACK_SWAN_CAP.

        Args:
            win_prob:           AI confidence (treated as win probability).
            avg_win_loss_ratio: AI position_size_pct hint (used as b proxy;
                                defaults to 2.0 if zero).
        """
        p = win_prob
        q = 1.0 - p
        b = avg_win_loss_ratio if avg_win_loss_ratio > 0 else 2.0

        f_star = (p * b - q) / b

        if f_star <= 0:
            log.debug("kelly_negative_edge", f_star=round(f_star, 4))
            return 0.0   # negative edge — no trade

        half_kelly = f_star * 0.5
        capped = min(half_kelly, _KELLY_BLACK_SWAN_CAP)
        return capped

    # ── Daily loss halt check (public helper) ──────────────────────────────────

    def daily_loss_halt(self, daily_loss_pct: float) -> bool:
        """Return True if daily loss limit is exceeded (trading should halt)."""
        return abs(daily_loss_pct) >= self.profile.max_daily_loss_pct


def _blocked(reason: str) -> SizingResult:
    log.info("risk_trade_blocked", reason=reason)
    return SizingResult(
        position_pct=0.0,
        stop_price=0.0,
        take_profit_price=0.0,
        qty=0.0,
        sizing_method="blocked",
        blocked=True,
        block_reason=reason,
    )
