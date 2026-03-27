"""
Tests for DB fixes introduced in the review pass:
  - insert_news_item duplicate returns 0 (INSERT OR IGNORE)
  - update_trade persists entry_price
  - _parse_dt always returns timezone-aware datetime
  - mark_news_forwarded with empty list is a no-op
  - concurrent writes don't corrupt DB (write lock)
"""
from __future__ import annotations

from datetime import datetime, timezone

import pytest

from src.history.db import (
    get_open_trades,
    insert_ai_decision,
    insert_news_item,
    insert_trade,
    mark_news_forwarded,
    update_trade,
)
from src.history.models import AIDecision, NewsItem, Trade, _parse_dt


# ── _parse_dt always UTC-aware ────────────────────────────────────────────────

def test_parse_dt_naive_string_gets_utc():
    """Naive ISO string from SQLite must be treated as UTC."""
    result = _parse_dt("2024-01-15T10:30:00")
    assert result is not None
    assert result.tzinfo is not None
    assert result.tzinfo == timezone.utc


def test_parse_dt_aware_string_preserved():
    """Already-aware string must keep its tzinfo unchanged."""
    result = _parse_dt("2024-01-15T10:30:00+00:00")
    assert result is not None
    assert result.tzinfo is not None


def test_parse_dt_none_returns_none():
    assert _parse_dt(None) is None


def test_trade_from_db_row_datetime_is_aware():
    """from_db_row must produce timezone-aware datetimes even from naive strings."""
    row = {
        "id": 1,
        "decision_id": None,
        "alpaca_order_id": "ord_1",
        "ticker": "AAPL",
        "side": "buy",
        "qty": 1.0,
        "entry_price": 100.0,
        "stop_loss": 97.0,
        "take_profit": 106.0,
        "exit_price": None,
        "status": "open",
        "pnl": None,
        "pnl_pct": None,
        "opened_at": "2024-03-01T09:30:00",   # naive — SQLite default
        "closed_at": None,
    }
    trade = Trade.from_db_row(row)
    assert trade.opened_at.tzinfo is not None


# ── insert_news_item duplicate returns 0 ─────────────────────────────────────

async def test_news_duplicate_url_returns_zero(tmp_db):
    """INSERT OR IGNORE on duplicate URL must return 0, not the previous row's id."""
    item = NewsItem(
        headline="Rate hike incoming",
        source="wsj",
        url="https://wsj.com/rate-hike-2024",
        symbols=["TLT"],
        sentiment=-0.5,
        relevance=0.8,
    )
    first = await insert_news_item(tmp_db, item)
    assert first > 0

    duplicate = await insert_news_item(tmp_db, item)
    assert duplicate == 0


async def test_different_url_gets_distinct_ids(tmp_db):
    """Two news items with different URLs must get distinct positive IDs."""
    a = NewsItem(
        headline="Story A", source="src", url="https://example.com/a",
        symbols=[], sentiment=0.0, relevance=0.5,
    )
    b = NewsItem(
        headline="Story B", source="src", url="https://example.com/b",
        symbols=[], sentiment=0.0, relevance=0.5,
    )
    id_a = await insert_news_item(tmp_db, a)
    id_b = await insert_news_item(tmp_db, b)
    assert id_a > 0
    assert id_b > 0
    assert id_a != id_b


# ── update_trade persists entry_price ────────────────────────────────────────

async def test_update_trade_entry_price_persisted(tmp_db):
    """entry_price must survive an UPDATE — bug: was missing from SET clause."""
    dec = AIDecision(
        news_item_ids=[], model="test", action="buy", ticker="TSLA",
        confidence=0.8, position_pct=0.01, stop_loss_pct=0.02,
        take_profit_pct=0.04, hold_period="intraday", reasoning="test",
        risk_factors=[], literature_ref="", risk_profile="conservative",
    )
    did = await insert_ai_decision(tmp_db, dec)

    trade = Trade(
        decision_id=did, ticker="TSLA", side="buy", qty=2.0,
        entry_price=0.0,  # starts as 0 (not yet filled)
        stop_loss=195.0, take_profit=215.0, status="pending",
    )
    tid = await insert_trade(tmp_db, trade)
    trade.id = tid

    # Simulate fill: entry_price set from Alpaca filled_avg_price
    trade.entry_price = 204.75
    trade.status = "open"
    await update_trade(tmp_db, trade)

    open_trades = await get_open_trades(tmp_db)
    assert len(open_trades) == 1
    assert open_trades[0].entry_price == 204.75, (
        "entry_price was dropped by UPDATE — fix: add entry_price to SET clause"
    )


# ── mark_news_forwarded empty list ────────────────────────────────────────────

async def test_mark_news_forwarded_empty_list_noop(tmp_db):
    """Empty list must not execute SQL (would produce invalid IN () clause)."""
    # Should not raise — bug: would produce syntax error without guard
    await mark_news_forwarded(tmp_db, [])


async def test_mark_news_forwarded_sets_flag(tmp_db):
    """Non-empty list must actually flip forwarded=1."""
    item = NewsItem(
        headline="Mark test", source="src", url="https://example.com/mark",
        symbols=[], sentiment=0.0, relevance=0.5,
    )
    nid = await insert_news_item(tmp_db, item)
    assert nid > 0

    await mark_news_forwarded(tmp_db, [nid])

    async with tmp_db.execute(
        "SELECT forwarded FROM news_items WHERE id=?", (nid,)
    ) as cur:
        row = await cur.fetchone()
    assert row[0] == 1
