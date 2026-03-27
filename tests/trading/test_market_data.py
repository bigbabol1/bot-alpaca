"""
Tests for MarketDataCache — pure cache read logic (no external API calls).

Covers:
  - get_atr returns None for unknown ticker
  - get_atr returns None when ATR column missing or NaN
  - get_atr returns float after cache populated
  - get_latest_close returns None for unknown ticker
  - get_latest_close returns last bar's close price
  - Ticker lookup is case-insensitive (cache stores uppercase)
  - Constructor filters crypto tickers (contain '/') from watchlist
"""
from __future__ import annotations

import sys
from unittest.mock import MagicMock

# Stub out 'ta' before market_data.py is imported — 'ta' isn't installed in CI
if 'ta' not in sys.modules:
    sys.modules['ta'] = MagicMock()
    sys.modules['ta.volatility'] = MagicMock()

import numpy as np
import pandas as pd
import pytest

from src.trading.market_data import MarketDataCache


def _make_cache(ticker: str, bars: int = 20) -> MarketDataCache:
    """Build a MarketDataCache with a pre-seeded in-memory cache."""
    alpaca = MagicMock()  # not used in pure read path
    cache = MarketDataCache(alpaca=alpaca, watchlist=[ticker])

    # Build a synthetic OHLCV DataFrame with ATR column
    df = pd.DataFrame({
        "open":  np.full(bars, 100.0),
        "high":  np.full(bars, 102.0),
        "low":   np.full(bars, 98.0),
        "close": np.linspace(99.0, 105.0, bars),
        "volume": np.full(bars, 1_000_000),
    })
    # Compute a fake ATR (just use constant for test predictability)
    df["atr"] = 2.5
    cache._cache[ticker.upper()] = df
    return cache


# ── get_atr ───────────────────────────────────────────────────────────────────

def test_get_atr_unknown_ticker_returns_none():
    cache = MarketDataCache(MagicMock(), watchlist=[])
    assert cache.get_atr("AAPL") is None


def test_get_atr_returns_latest_atr():
    cache = _make_cache("AAPL", bars=20)
    atr = cache.get_atr("AAPL")
    assert atr == pytest.approx(2.5)


def test_get_atr_case_insensitive():
    """Cache stores tickers uppercase; lookup must normalise."""
    cache = _make_cache("AAPL")
    assert cache.get_atr("aapl") is not None
    assert cache.get_atr("AaPl") is not None


def test_get_atr_nan_last_row_returns_none():
    """If the last ATR value is NaN (e.g., warm-up period), return None."""
    alpaca = MagicMock()
    cache = MarketDataCache(alpaca=alpaca, watchlist=["TSLA"])
    df = pd.DataFrame({
        "open": [100.0], "high": [102.0], "low": [98.0],
        "close": [101.0], "volume": [1_000],
    })
    df["atr"] = float("nan")
    cache._cache["TSLA"] = df
    assert cache.get_atr("TSLA") is None


def test_get_atr_no_atr_column_returns_none():
    """DataFrame without ATR column (e.g., < 14 bars) → None."""
    alpaca = MagicMock()
    cache = MarketDataCache(alpaca=alpaca, watchlist=["SPY"])
    df = pd.DataFrame({"close": [100.0, 101.0]})
    cache._cache["SPY"] = df
    assert cache.get_atr("SPY") is None


# ── get_latest_close ──────────────────────────────────────────────────────────

def test_get_latest_close_unknown_ticker_returns_none():
    cache = MarketDataCache(MagicMock(), watchlist=[])
    assert cache.get_latest_close("NVDA") is None


def test_get_latest_close_returns_last_bar():
    """Returns the close price of the most recent bar."""
    cache = _make_cache("NVDA", bars=5)
    # linspace(99, 105, 5) → [99, 100.5, 102, 103.5, 105]
    close = cache.get_latest_close("NVDA")
    assert close == pytest.approx(105.0)


def test_get_latest_close_case_insensitive():
    cache = _make_cache("MSFT")
    assert cache.get_latest_close("msft") is not None


def test_watchlist_filters_crypto():
    """Constructor strips crypto tickers (contain '/') from watchlist."""
    alpaca = MagicMock()
    cache = MarketDataCache(alpaca=alpaca, watchlist=["AAPL", "BTC/USD", "ETH/USD"])
    assert "BTC/USD" not in cache._watchlist
    assert "ETH/USD" not in cache._watchlist
    assert "AAPL" in cache._watchlist
