"""
Tests for news filter pipeline.

Covers:
  - Happy path: relevant, recent, high-quality news passes
  - Duplicate detection (same URL = rejected)
  - Recency gate (intraday 4h, swing 48h)
  - Source quality tier assignment
  - Relevance scoring (ticker match, keyword match, both, neither)
  - Batch assembly grouping
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from src.news.filter import NewsFilter, assemble_batches
from src.history.models import NewsItem


def _make_item(
    headline: str = "Test headline",
    source: str = "reuters",
    url: str = "https://example.com/1",
    symbols: list[str] | None = None,
    age_minutes: int = 30,
    summary: str = "",
) -> NewsItem:
    received = datetime.now(tz=timezone.utc) - timedelta(minutes=age_minutes)
    return NewsItem(
        headline=headline,
        summary=summary,
        source=source,
        url=url,
        symbols=symbols or ["AAPL"],
        sentiment=0.0,
        relevance=0.0,
        received_at=received,
    )


# ── Happy path ─────────────────────────────────────────────────────────────────

def test_relevant_recent_news_passes():
    f = NewsFilter(watchlist=["AAPL"])
    item = _make_item(
        headline="Apple reports record earnings",
        symbols=["AAPL"],
        age_minutes=10,
    )
    result = f.check(item)
    assert result.passed
    assert result.reason == "ok"
    assert result.relevance >= 0.8


# ── Duplicate detection ────────────────────────────────────────────────────────

def test_duplicate_url_rejected():
    f = NewsFilter(watchlist=["AAPL"])
    item = _make_item(url="https://example.com/unique1")
    r1 = f.check(item)
    assert r1.passed

    # Same URL → duplicate
    r2 = f.check(item)
    assert not r2.passed
    assert r2.reason == "duplicate"


def test_different_url_same_headline_passes():
    """Different URL = not a duplicate (we hash URL + headline together)."""
    f = NewsFilter(watchlist=["AAPL"])
    item1 = _make_item(url="https://example.com/a")
    item2 = _make_item(url="https://example.com/b")
    assert f.check(item1).passed
    assert f.check(item2).passed


# ── Recency gate ───────────────────────────────────────────────────────────────

def test_intraday_4h_cutoff():
    f = NewsFilter(watchlist=["AAPL"])
    old_item = _make_item(age_minutes=241, url="https://example.com/old")  # > 4h
    result = f.check(old_item, mode="intraday")
    assert not result.passed
    assert result.reason == "too_old"


def test_intraday_recent_passes():
    f = NewsFilter(watchlist=["AAPL"])
    recent = _make_item(age_minutes=10, url="https://example.com/recent1")
    assert f.check(recent, mode="intraday").passed


def test_swing_48h_cutoff():
    f = NewsFilter(watchlist=["AAPL"])
    old = _make_item(age_minutes=2881, url="https://example.com/swing-old")  # > 48h
    result = f.check(old, mode="swing")
    assert not result.passed
    assert result.reason == "too_old"


def test_swing_24h_passes():
    f = NewsFilter(watchlist=["AAPL"])
    item = _make_item(age_minutes=1440, url="https://example.com/24h")  # 24h
    result = f.check(item, mode="swing")
    assert result.passed


# ── Source quality ─────────────────────────────────────────────────────────────

def test_tier1_source_quality():
    f = NewsFilter(watchlist=["AAPL"])
    result = f.check(_make_item(source="Reuters", url="https://r.com/1"))
    assert result.quality == 1.0


def test_tier2_source_quality():
    f = NewsFilter(watchlist=["AAPL"])
    result = f.check(_make_item(source="benzinga", url="https://b.com/1"))
    assert result.quality == 0.7


def test_tier3_source_quality():
    f = NewsFilter(watchlist=["AAPL"])
    result = f.check(_make_item(source="random_blog", url="https://rand.com/1"))
    assert result.quality == 0.4


# ── Relevance scoring ──────────────────────────────────────────────────────────

def test_ticker_and_keyword_match_max_relevance():
    f = NewsFilter(watchlist=["AAPL"])
    item = _make_item(
        headline="Apple Q4 earnings beat estimates",   # keyword: earnings
        symbols=["AAPL"],                              # ticker match
        url="https://example.com/max-rel",
    )
    result = f.check(item)
    assert result.relevance == 1.0


def test_ticker_only_relevance():
    f = NewsFilter(watchlist=["AAPL"])
    item = _make_item(
        headline="Apple announces new product",        # no keyword match
        symbols=["AAPL"],
        url="https://example.com/ticker-only",
    )
    result = f.check(item)
    assert result.relevance == 0.8


def test_keyword_only_relevance():
    f = NewsFilter(watchlist=["AAPL"])
    item = _make_item(
        headline="Fed raises interest rates",          # keyword but no watched ticker
        symbols=["OTHER"],
        url="https://example.com/keyword-only",
    )
    result = f.check(item)
    assert result.relevance == 0.5


def test_low_relevance_filtered():
    f = NewsFilter(watchlist=["AAPL"])
    item = _make_item(
        headline="Random unrelated news",
        symbols=["OTHER"],
        url="https://example.com/low-rel",
    )
    result = f.check(item)
    assert not result.passed
    assert result.reason == "low_relevance"


# ── Batch assembly ────────────────────────────────────────────────────────────

def test_batch_assembly_groups_correctly():
    items = [_make_item(url=f"https://x.com/{i}") for i in range(13)]
    batches = assemble_batches(items, batch_size=5)
    assert len(batches) == 3
    assert len(batches[0]) == 5
    assert len(batches[1]) == 5
    assert len(batches[2]) == 3


def test_batch_assembly_empty():
    assert assemble_batches([], batch_size=5) == []


def test_batch_assembly_smaller_than_batch():
    items = [_make_item(url=f"https://x.com/{i}") for i in range(3)]
    batches = assemble_batches(items, batch_size=5)
    assert len(batches) == 1
    assert len(batches[0]) == 3
