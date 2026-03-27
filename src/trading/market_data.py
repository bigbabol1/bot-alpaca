"""
ATR / OHLCV cache — background task that refreshes every 15 minutes.

Architecture:
  ┌─────────────────────────────────┐
  │  MarketDataCache.run()          │  background task
  │    └─ every 15min:              │
  │         fetch all watchlist     │
  │         symbols in ONE batch    │
  │         call → store in         │
  │         _cache[ticker]=DataFrame│
  └─────────────────────────────────┘
  Risk engine reads get_atr(ticker) → no API calls on hot path.
"""
from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone

import pandas as pd
import structlog
import ta

from src.trading.alpaca_client import AlpacaWrapper

log = structlog.get_logger(__name__)

_REFRESH_MINUTES = 15
_BARS_LOOKBACK_DAYS = 30   # enough for ATR(14) to stabilize


class MarketDataCache:
    def __init__(self, alpaca: AlpacaWrapper, watchlist: list[str]) -> None:
        self._alpaca = alpaca
        self._watchlist = [s for s in watchlist if "/" not in s]  # equities only
        self._cache: dict[str, pd.DataFrame] = {}

    async def run(self) -> None:
        """Background task: refresh on startup then every 15 minutes."""
        while True:
            await self._refresh()
            await asyncio.sleep(_REFRESH_MINUTES * 60)

    async def _refresh(self) -> None:
        if not self._watchlist:
            return
        try:
            from alpaca.data.timeframe import TimeFrame
            start = datetime.now(tz=timezone.utc) - timedelta(days=_BARS_LOOKBACK_DAYS)
            bars_response = await self._alpaca.get_stock_bars(
                symbols=self._watchlist,
                timeframe=TimeFrame.Day,
                start=start,
            )
            data = bars_response.data if hasattr(bars_response, "data") else {}
            for ticker, bar_list in data.items():
                try:
                    df = pd.DataFrame([
                        {
                            "open": b.open,
                            "high": b.high,
                            "low": b.low,
                            "close": b.close,
                            "volume": b.volume,
                        }
                        for b in bar_list
                    ])
                    if len(df) >= 14:
                        df["atr"] = ta.volatility.AverageTrueRange(
                            high=df["high"],
                            low=df["low"],
                            close=df["close"],
                            window=14,
                        ).average_true_range()
                        self._cache[ticker] = df
                except Exception as exc:  # noqa: BLE001
                    log.warning("market_data_ticker_parse_error", ticker=ticker, error=str(exc))

            log.info("market_data_refreshed", tickers=len(self._cache))
        except Exception as exc:  # noqa: BLE001
            log.warning("market_data_refresh_failed", error=str(exc))

    def get_atr(self, ticker: str) -> float | None:
        """Return the latest ATR for a ticker, or None if not cached."""
        df = self._cache.get(ticker.upper())
        if df is None or df.empty or "atr" not in df.columns:
            return None
        val = df["atr"].iloc[-1]
        return float(val) if not pd.isna(val) else None

    def get_latest_close(self, ticker: str) -> float | None:
        df = self._cache.get(ticker.upper())
        if df is None or df.empty:
            return None
        return float(df["close"].iloc[-1])
