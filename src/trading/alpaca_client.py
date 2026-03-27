"""
Thin wrapper around alpaca-py TradingClient and DataClient.

- Handles paper vs live mode selection
- Wraps Alpaca API errors with retry (via async_retry)
- Rate limit (429) gets exponential backoff: 2s, 4s, 8s; after 3 retries → skip
"""
from __future__ import annotations

import asyncio
import time
from functools import wraps
from typing import Any

import structlog
from alpaca.common.exceptions import APIError
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.trading.client import TradingClient
from alpaca.trading.enums import OrderSide, OrderType, TimeInForce
from alpaca.trading.requests import (
    GetOrdersRequest,
    LimitOrderRequest,
    MarketOrderRequest,
    StopLimitOrderRequest,
    StopOrderRequest,
    TakeProfitRequest,
    StopLossRequest,
    TrailingStopOrderRequest,
)

from src.config import Settings

log = structlog.get_logger(__name__)

_429_BACKOFFS = [2, 4, 8]


def _is_rate_limited(exc: Exception) -> bool:
    return isinstance(exc, APIError) and getattr(exc, "status_code", None) == 429


class AlpacaWrapper:
    def __init__(self, settings: Settings) -> None:
        paper = settings.trading_mode == "paper"
        self._trading = TradingClient(
            api_key=settings.alpaca_api_key,
            secret_key=settings.alpaca_secret_key,
            paper=paper,
        )
        self._data = StockHistoricalDataClient(
            api_key=settings.alpaca_api_key,
            secret_key=settings.alpaca_secret_key,
        )
        self._paper = paper

    # ── Account ────────────────────────────────────────────────────────────────

    async def get_account(self):
        return await asyncio.to_thread(self._trading.get_account)

    async def get_all_positions(self) -> list:
        t0 = time.monotonic()
        result = await asyncio.to_thread(self._trading.get_all_positions)
        log.debug("alpaca_get_positions", alpaca_api_ms=round((time.monotonic() - t0) * 1000))
        return result

    # ── Orders ──────────────────────────────────────────────────────────────────

    async def submit_bracket_order(
        self,
        ticker: str,
        qty: float,
        side: str,          # "buy" | "sell"
        stop_loss_price: float,
        take_profit_price: float,
    ):
        """Submit an equity bracket order (entry + stop + take-profit)."""
        from alpaca.trading.requests import OrderRequest
        order_side = OrderSide.BUY if side == "buy" else OrderSide.SELL
        req = MarketOrderRequest(
            symbol=ticker,
            qty=qty,
            side=order_side,
            time_in_force=TimeInForce.DAY,
            order_class="bracket",
            stop_loss=StopLossRequest(stop_price=round(stop_loss_price, 2)),
            take_profit=TakeProfitRequest(limit_price=round(take_profit_price, 2)),
        )
        return await self._with_429_backoff(
            lambda: asyncio.to_thread(self._trading.submit_order, req),
            label=f"submit_bracket_{ticker}",
        )

    async def submit_market_order(self, ticker: str, qty: float, side: str):
        order_side = OrderSide.BUY if side == "buy" else OrderSide.SELL
        req = MarketOrderRequest(
            symbol=ticker,
            qty=qty,
            side=order_side,
            time_in_force=TimeInForce.DAY,
        )
        return await self._with_429_backoff(
            lambda: asyncio.to_thread(self._trading.submit_order, req),
            label=f"submit_market_{ticker}",
        )

    async def submit_stop_limit_order(
        self,
        ticker: str,
        qty: float,
        side: str,
        stop_price: float,
        limit_price: float,
    ):
        """Crypto stop-loss: separate stop-limit sell order."""
        order_side = OrderSide.BUY if side == "buy" else OrderSide.SELL
        req = StopLimitOrderRequest(
            symbol=ticker,
            qty=qty,
            side=order_side,
            time_in_force=TimeInForce.GTC,
            stop_price=round(stop_price, 2),
            limit_price=round(limit_price, 2),
        )
        return await self._with_429_backoff(
            lambda: asyncio.to_thread(self._trading.submit_order, req),
            label=f"submit_stop_limit_{ticker}",
        )

    async def cancel_order(self, order_id: str) -> None:
        await asyncio.to_thread(self._trading.cancel_order_by_id, order_id)

    async def get_order(self, order_id: str):
        t0 = time.monotonic()
        result = await asyncio.to_thread(self._trading.get_order_by_id, order_id)
        log.debug("alpaca_get_order", alpaca_api_ms=round((time.monotonic() - t0) * 1000))
        return result

    async def get_open_orders(self) -> list:
        req = GetOrdersRequest(status="open")
        return await asyncio.to_thread(self._trading.get_orders, req)

    # ── Market data ────────────────────────────────────────────────────────────

    async def get_stock_bars(self, symbols: list[str], timeframe, start, end=None) -> Any:
        """Batch OHLCV fetch — one call for all symbols."""
        from alpaca.data.requests import StockBarsRequest
        t0 = time.monotonic()
        req = StockBarsRequest(
            symbol_or_symbols=symbols,
            timeframe=timeframe,
            start=start,
            end=end,
        )
        result = await asyncio.to_thread(self._data.get_stock_bars, req)
        log.debug(
            "alpaca_get_bars",
            alpaca_api_ms=round((time.monotonic() - t0) * 1000),
            symbols=len(symbols),
        )
        return result

    # ── Internal ───────────────────────────────────────────────────────────────

    @staticmethod
    async def _with_429_backoff(coro_fn, label: str) -> Any:
        t0 = time.monotonic()
        for i, delay in enumerate(_429_BACKOFFS + [None]):
            try:
                result = await coro_fn()
                log.debug("alpaca_order_submitted", alpaca_api_ms=round((time.monotonic() - t0) * 1000), label=label)
                return result
            except APIError as exc:
                if _is_rate_limited(exc) and delay is not None:
                    log.warning(
                        "alpaca_rate_limited",
                        label=label,
                        attempt=i + 1,
                        retry_in=delay,
                    )
                    await asyncio.sleep(delay)
                elif _is_rate_limited(exc):
                    log.error("alpaca_rate_limited_exhausted", label=label)
                    raise
                else:
                    raise
