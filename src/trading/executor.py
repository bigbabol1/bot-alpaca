"""
Trade executor — places orders, handles partial fills, manages crypto stop-loss.

Equity orders: bracket (entry + stop-loss + take-profit server-side).
Crypto orders: market entry + manual stop-limit + position monitor loop.

Partial fill handling:
  - Poll every 10s for up to 60s
  - partially_filled → update DB, submit stop/TP for filled qty
  - If bracket legs fail → submit stop-market immediately (fail-safe)
  - Still pending after 60s → cancel, log timeout_cancelled

Paper→live gate: enforced at startup in main.py; executor trusts the gate.
"""
from __future__ import annotations

import asyncio
import math
import time
from datetime import datetime, timezone

import structlog
from alpaca.common.exceptions import APIError

from src.history import db as db_ops
from src.history.models import Trade
from src.trading.alpaca_client import AlpacaWrapper
from src.trading.risk import SizingResult

log = structlog.get_logger(__name__)

_FILL_POLL_INTERVAL = 10       # seconds between fill status polls (equity)
_FILL_POLL_INTERVAL_CRYPTO = 1 # shorter first-poll for crypto (fills in <1s)
_FILL_POLL_TIMEOUT = 120       # max seconds to wait for fill (paper trading can be slow)
_CRYPTO_MAX_POSITION_PCT = 0.005  # 0.5% of portfolio per crypto trade
_CRYPTO_HARD_STOP_PCT = 0.01      # 1% hard stop loss for crypto

_CRYPTO_SYMBOLS = frozenset(["BTC/USD", "ETH/USD", "SOL/USD", "BTCUSD", "ETHUSD", "SOLUSD"])


def _is_crypto(ticker: str) -> bool:
    return ticker in _CRYPTO_SYMBOLS or "/" in ticker


def _crypto_round(price: float) -> float:
    """Round a crypto price to enough decimal places (at least 4; more for sub-cent prices)."""
    if price <= 0:
        return 0.0
    decimals = max(4, -int(math.floor(math.log10(price))) + 4)
    return round(price, decimals)


class TradeExecutor:
    def __init__(self, alpaca: AlpacaWrapper, conn, alert_callback=None) -> None:
        self._alpaca = alpaca
        self._conn = conn
        self._alert = alert_callback

    # ── Main entry point ───────────────────────────────────────────────────────

    async def execute(
        self,
        decision_id: int,
        ticker: str,
        side: str,
        sizing: SizingResult,
        risk_profile: str,
    ) -> Trade | None:
        """
        Place an order and persist the trade record.
        Returns the Trade object, or None if execution was skipped.
        """
        if sizing.blocked or sizing.qty <= 0:
            log.info("executor_skipped_blocked", ticker=ticker, reason=sizing.block_reason)
            return None

        if _is_crypto(ticker):
            return await self._execute_crypto(decision_id, ticker, side, sizing, risk_profile)
        else:
            return await self._execute_equity(decision_id, ticker, side, sizing, risk_profile)

    # ── Equity (bracket order) ─────────────────────────────────────────────────

    async def _execute_equity(
        self,
        decision_id: int,
        ticker: str,
        side: str,
        sizing: SizingResult,
        risk_profile: str,
    ) -> Trade | None:
        # Alpaca bracket orders require whole-share quantities.
        # Round down; skip if that would be 0 shares.
        whole_qty = math.floor(sizing.qty)
        if whole_qty < 1:
            log.info(
                "executor_skipped_fractional",
                ticker=ticker,
                computed_qty=sizing.qty,
                reason="bracket orders require ≥1 whole share",
            )
            return None
        sizing = SizingResult(
            qty=float(whole_qty),
            stop_price=sizing.stop_price,
            take_profit_price=sizing.take_profit_price,
            position_pct=sizing.position_pct,
            sizing_method=sizing.sizing_method,
            blocked=sizing.blocked,
            block_reason=sizing.block_reason,
        )
        # Skip if Alpaca already holds an open position in this ticker.
        # Bracket orders are entry-only — submitting one on top of an existing
        # position is rejected with HTTP 422 "bracket orders must be entry orders".
        try:
            existing = await self._alpaca.get_all_positions()
            held = {p.symbol for p in existing}
            if ticker in held:
                log.info(
                    "executor_skipped_existing_position",
                    ticker=ticker,
                    reason="already holding position — bracket orders are entry-only",
                )
                return None
        except Exception as exc:  # noqa: BLE001
            log.warning("executor_position_check_failed", ticker=ticker, error=str(exc))
            # Proceed — Alpaca will reject if needed
        try:
            order = await self._alpaca.submit_bracket_order(
                ticker=ticker,
                qty=sizing.qty,
                side=side,
                stop_loss_price=sizing.stop_price,
                take_profit_price=sizing.take_profit_price,
            )
            log.info(
                "equity_order_submitted",
                ticker=ticker,
                side=side,
                qty=sizing.qty,
                order_id=str(order.id),
            )
        except APIError as exc:
            log.error("equity_order_failed", ticker=ticker, error=str(exc))
            if self._alert:
                await self._alert(f"❌ Order failed: {ticker} {side} — {exc}", priority=False)
            return None

        # Persist trade record
        trade = Trade(
            decision_id=decision_id,
            alpaca_order_id=str(order.id),
            ticker=ticker,
            side=side,
            qty=sizing.qty,
            entry_price=0.0,    # filled by poll
            stop_loss=sizing.stop_price,
            take_profit=sizing.take_profit_price,
            status="pending",
        )
        trade_id = await db_ops.insert_trade(self._conn, trade)
        trade.id = trade_id

        # Poll for fill
        filled_trade = await self._poll_fill(trade)
        if self._alert and filled_trade and filled_trade.status == "open":
            await self._alert(
                f"📈 Trade opened: {side.upper()} {ticker} "
                f"@ ${filled_trade.entry_price:.2f} qty={filled_trade.qty:.4f}",
                priority=False,
            )
        return filled_trade

    # ── Crypto (market + manual stop-limit) ───────────────────────────────────

    async def _execute_crypto(
        self,
        decision_id: int,
        ticker: str,
        side: str,
        sizing: SizingResult,
        risk_profile: str,
    ) -> Trade | None:
        try:
            order = await self._alpaca.submit_market_order(
                ticker=ticker,
                qty=sizing.qty,
                side=side,
            )
            log.info(
                "crypto_order_submitted",
                ticker=ticker,
                side=side,
                qty=sizing.qty,
                order_id=str(order.id),
            )
        except APIError as exc:
            log.error("crypto_order_failed", ticker=ticker, error=str(exc))
            return None

        # Persist pending trade
        trade = Trade(
            decision_id=decision_id,
            alpaca_order_id=str(order.id),
            ticker=ticker,
            side=side,
            qty=sizing.qty,
            entry_price=0.0,
            stop_loss=sizing.stop_price,
            take_profit=sizing.take_profit_price,
            status="pending",
        )
        trade_id = await db_ops.insert_trade(self._conn, trade)
        trade.id = trade_id

        # Poll for fill with short initial delay (crypto fills in <1s)
        filled_trade = await self._poll_fill(trade, first_poll_delay=_FILL_POLL_INTERVAL_CRYPTO)
        if filled_trade and filled_trade.status == "open" and filled_trade.entry_price > 0:
            await self._submit_crypto_stop(filled_trade)

        return filled_trade

    async def _submit_crypto_stop(self, trade: Trade) -> None:
        """After crypto entry fills: submit a separate stop-limit sell."""
        stop_price = _crypto_round(trade.entry_price * (1 - _CRYPTO_HARD_STOP_PCT))
        limit_price = _crypto_round(stop_price * 0.995)  # 0.5% slippage allowance
        if stop_price <= 0 or limit_price <= 0:
            log.error("crypto_stop_skipped_zero_price", ticker=trade.ticker, entry=trade.entry_price)
            return
        try:
            await self._alpaca.submit_stop_limit_order(
                ticker=trade.ticker,
                qty=trade.qty,
                side="sell",
                stop_price=stop_price,
                limit_price=limit_price,
            )
            log.info("crypto_stop_submitted", ticker=trade.ticker, stop_price=stop_price)
        except Exception as exc:  # noqa: BLE001
            log.error("crypto_stop_failed", ticker=trade.ticker, error=str(exc))
            # Fail-safe: immediately exit with market order
            log.warning("crypto_stop_fail_safe_market_exit", ticker=trade.ticker)
            try:
                exit_side = "buy" if trade.side == "sell" else "sell"
                await self._alpaca.submit_market_order(trade.ticker, trade.qty, exit_side)
            except Exception as inner_exc:  # noqa: BLE001
                log.error("crypto_market_exit_failed", ticker=trade.ticker, error=str(inner_exc))

    # ── Fill polling ───────────────────────────────────────────────────────────

    async def _poll_fill(self, trade: Trade, first_poll_delay: float = _FILL_POLL_INTERVAL) -> Trade:
        """
        Poll Alpaca for fill status every 10s for up to 60s.
        Updates the DB trade record on each status change.
        Pass first_poll_delay=_FILL_POLL_INTERVAL_CRYPTO for crypto (fills in <1s).
        """
        deadline = time.monotonic() + _FILL_POLL_TIMEOUT
        delay = first_poll_delay
        while time.monotonic() < deadline:
            await asyncio.sleep(delay)
            delay = _FILL_POLL_INTERVAL  # subsequent polls use standard interval
            try:
                order = await self._alpaca.get_order(trade.alpaca_order_id)
                status = getattr(order.status, "value", str(order.status)).lower()
                log.debug("order_poll_status", ticker=trade.ticker, status=status)

                if status == "filled":
                    fill_price = float(order.filled_avg_price or 0)
                    trade.entry_price = fill_price
                    trade.status = "open"
                    trade.qty = float(order.filled_qty or trade.qty)
                    await db_ops.update_trade(self._conn, trade)
                    log.info("order_filled", ticker=trade.ticker, fill_price=fill_price)
                    return trade

                if status == "partially_filled":
                    filled_qty = float(order.filled_qty or 0)
                    fill_price = float(order.filled_avg_price or 0)
                    trade.qty = filled_qty
                    trade.entry_price = fill_price
                    trade.status = "open"
                    await db_ops.update_trade(self._conn, trade)
                    log.info(
                        "order_partially_filled",
                        ticker=trade.ticker,
                        filled_qty=filled_qty,
                        fill_price=fill_price,
                    )
                    # Cancel the remaining unfilled portion
                    try:
                        await self._alpaca.cancel_order(trade.alpaca_order_id)
                    except Exception:  # noqa: BLE001
                        pass
                    return trade

                if status in ("cancelled", "expired", "rejected"):
                    trade.status = "cancelled"
                    await db_ops.update_trade(self._conn, trade)
                    log.warning("order_cancelled", ticker=trade.ticker, status=status)
                    return trade

            except Exception as exc:  # noqa: BLE001
                log.warning("fill_poll_error", ticker=trade.ticker, error=str(exc))

        # Timeout — cancel the order
        log.warning("order_fill_timeout", ticker=trade.ticker)
        try:
            await self._alpaca.cancel_order(trade.alpaca_order_id)
        except Exception:  # noqa: BLE001
            pass
        trade.status = "timeout_cancelled"
        await db_ops.update_trade(self._conn, trade)
        return trade

    # ── Startup reconciliation ─────────────────────────────────────────────────

    async def reconcile_open_orders(self) -> None:
        """
        On startup: compare Alpaca open orders to DB trades table.
        Sync any gaps (orders unknown to DB, or DB-open orders gone from Alpaca).
        Write serialization is handled by _wlock() inside insert_trade/update_trade.
        """
        log.info("reconciliation_start")
        try:

            alpaca_orders = await self._alpaca.get_open_orders()
            alpaca_ids = {str(o.id) for o in alpaca_orders}

            db_trades = await db_ops.get_open_trades(self._conn)
            db_ids = {t.alpaca_order_id for t in db_trades if t.alpaca_order_id}

            # Orders in Alpaca but not in DB — insert as pending
            for order in alpaca_orders:
                if str(order.id) not in db_ids:
                    t = Trade(
                        decision_id=None,
                        alpaca_order_id=str(order.id),
                        ticker=str(order.symbol),
                        side=str(order.side).lower(),
                        qty=float(order.qty or 0),
                        entry_price=float(order.filled_avg_price or 0),
                        stop_loss=0.0,
                        take_profit=0.0,
                        status="open" if order.filled_qty and float(order.filled_qty) > 0 else "pending",
                    )
                    await db_ops.insert_trade(self._conn, t)
                    log.info("reconciliation_inserted", order_id=str(order.id))

            # DB-open trades no longer in Alpaca open-orders list —
            # query actual status: filled orders are removed from open-orders
            # but still exist and are now live positions.
            for trade in db_trades:
                if not trade.alpaca_order_id or trade.alpaca_order_id in alpaca_ids:
                    continue
                try:
                    order = await self._alpaca.get_order(trade.alpaca_order_id)
                    status = getattr(order.status, "value", str(order.status)).lower()
                    if status == "filled":
                        # Order filled → remains an open position, not closed
                        trade.status = "open"
                        trade.entry_price = float(order.filled_avg_price or trade.entry_price)
                        trade.qty = float(order.filled_qty or trade.qty)
                        log.info("reconciliation_filled", order_id=trade.alpaca_order_id)
                    elif status in ("cancelled", "expired", "rejected"):
                        trade.status = "cancelled"
                        trade.closed_at = datetime.now(tz=timezone.utc)
                        log.info("reconciliation_cancelled", order_id=trade.alpaca_order_id,
                                 alpaca_status=status)
                    else:
                        # Unknown final state — mark closed conservatively
                        trade.status = "closed"
                        trade.closed_at = datetime.now(tz=timezone.utc)
                        log.warning("reconciliation_unknown_status", order_id=trade.alpaca_order_id,
                                    alpaca_status=status)
                except Exception as exc:  # noqa: BLE001
                    # Can't determine status — leave as-is and log
                    log.error("reconciliation_order_fetch_failed",
                              order_id=trade.alpaca_order_id, error=str(exc))
                    continue
                await db_ops.update_trade(self._conn, trade)

            await self._conn.commit()
            log.info("reconciliation_complete")
        except Exception as exc:  # noqa: BLE001
            await self._conn.rollback()
            log.error("reconciliation_failed", error=str(exc))
