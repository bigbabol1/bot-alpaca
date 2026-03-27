"""
Tests for DB row models.
  - to_db_row / from_db_row round-trips for all JSON fields
  - DB insert and fetch round-trips for all models
"""
from __future__ import annotations

import json
from datetime import datetime, timezone

import pytest

from src.history.db import (
    count_closed_trades,
    get_open_trades,
    get_recent_decisions,
    insert_ai_decision,
    insert_news_item,
    insert_portfolio_snapshot,
    insert_trade,
    update_trade,
)
from src.history.models import AIDecision, NewsItem, PortfolioSnapshot, Trade


# ── NewsItem ──────────────────────────────────────────────────────────────────

def test_news_item_round_trip():
    item = NewsItem(
        headline="Apple beats earnings",
        summary="Apple reported Q4 results.",
        author="Jane Doe",
        source="reuters",
        url="https://reuters.com/article/1",
        symbols=["AAPL", "MSFT"],
        sentiment=0.85,
        relevance=0.9,
    )
    row = item.to_db_row()
    # JSON-encoded symbols
    decoded = json.loads(row["symbols"])
    assert decoded == ["AAPL", "MSFT"]

    # Round-trip via from_db_row
    row["id"] = 1
    reconstructed = NewsItem.from_db_row(row)
    assert reconstructed.symbols == ["AAPL", "MSFT"]
    assert reconstructed.headline == item.headline
    assert reconstructed.sentiment == item.sentiment


async def test_news_item_db_insert(tmp_db):
    item = NewsItem(
        headline="Fed raises rates",
        source="bloomberg",
        url="https://bloomberg.com/article/1",
        symbols=["SPY", "TLT"],
        sentiment=-0.6,
        relevance=0.95,
    )
    nid = await insert_news_item(tmp_db, item)
    assert nid > 0

    # Duplicate URL → INSERT OR IGNORE returns 0
    nid2 = await insert_news_item(tmp_db, item)
    assert nid2 == 0


# ── AIDecision ────────────────────────────────────────────────────────────────

def test_ai_decision_round_trip():
    dec = AIDecision(
        news_item_ids=[1, 2, 3],
        model="qwen2.5:32b",
        action="buy",
        ticker="AAPL",
        confidence=0.82,
        position_pct=0.01,
        stop_loss_pct=0.02,
        take_profit_pct=0.04,
        hold_period="swing",
        reasoning="Strong sentiment + earnings beat",
        risk_factors=["macro uncertainty", "high valuation"],
        literature_ref="Tetlock 2007",
        risk_profile="conservative",
    )
    row = dec.to_db_row()
    assert json.loads(row["news_item_ids"]) == [1, 2, 3]
    assert json.loads(row["risk_factors"]) == ["macro uncertainty", "high valuation"]

    row["id"] = 1
    r2 = AIDecision.from_db_row(row)
    assert r2.news_item_ids == [1, 2, 3]
    assert r2.risk_factors == ["macro uncertainty", "high valuation"]
    assert r2.action == "buy"


async def test_ai_decision_db_insert(tmp_db):
    dec = AIDecision(
        news_item_ids=[1],
        model="qwen2.5:32b",
        action="hold",
        ticker=None,
        confidence=0.4,
        position_pct=0.0,
        stop_loss_pct=0.0,
        take_profit_pct=0.0,
        hold_period="intraday",
        reasoning="Insufficient signal",
        risk_factors=[],
        literature_ref="",
        risk_profile="conservative",
    )
    did = await insert_ai_decision(tmp_db, dec)
    assert did > 0

    decisions = await get_recent_decisions(tmp_db, limit=5)
    assert len(decisions) == 1
    assert decisions[0].action == "hold"


# ── Trade ─────────────────────────────────────────────────────────────────────

def test_trade_round_trip():
    t = Trade(
        decision_id=1,
        alpaca_order_id="ord_123",
        ticker="NVDA",
        side="buy",
        qty=5.0,
        entry_price=850.0,
        stop_loss=830.0,
        take_profit=900.0,
        status="open",
    )
    row = t.to_db_row()
    row["id"] = 1
    r2 = Trade.from_db_row(row)
    assert r2.ticker == "NVDA"
    assert r2.entry_price == 850.0
    assert r2.status == "open"


async def test_trade_insert_and_update(tmp_db):
    # Insert a decision first (FK)
    dec = AIDecision(
        news_item_ids=[],
        model="test",
        action="buy",
        ticker="TSLA",
        confidence=0.8,
        position_pct=0.01,
        stop_loss_pct=0.02,
        take_profit_pct=0.04,
        hold_period="intraday",
        reasoning="test",
        risk_factors=[],
        literature_ref="",
        risk_profile="conservative",
    )
    did = await insert_ai_decision(tmp_db, dec)

    t = Trade(
        decision_id=did,
        alpaca_order_id="ord_456",
        ticker="TSLA",
        side="buy",
        qty=2.0,
        entry_price=200.0,
        stop_loss=195.0,
        take_profit=210.0,
        status="pending",
    )
    tid = await insert_trade(tmp_db, t)
    assert tid > 0

    # Update to open
    t.id = tid
    t.status = "open"
    t.entry_price = 201.5
    await update_trade(tmp_db, t)

    open_trades = await get_open_trades(tmp_db)
    assert len(open_trades) == 1
    assert open_trades[0].entry_price == 201.5

    # Close the trade
    t.status = "closed"
    t.pnl = 3.0
    t.pnl_pct = 1.5
    await update_trade(tmp_db, t)

    count = await count_closed_trades(tmp_db)
    assert count == 1


# ── PortfolioSnapshot ──────────────────────────────────────────────────────────

def test_portfolio_snapshot_round_trip():
    snap = PortfolioSnapshot(
        equity=10000.0,
        cash=5000.0,
        positions=[{"ticker": "AAPL", "qty": 10, "market_value": 1800.0, "unrealized_pnl": 50.0}],
        daily_pnl=120.0,
        daily_pnl_pct=1.2,
    )
    row = snap.to_db_row()
    positions = json.loads(row["positions"])
    assert positions[0]["ticker"] == "AAPL"

    row["id"] = 1
    r2 = PortfolioSnapshot.from_db_row(row)
    assert r2.positions[0]["ticker"] == "AAPL"
    assert r2.equity == 10000.0


async def test_portfolio_snapshot_insert(tmp_db):
    snap = PortfolioSnapshot(
        equity=50000.0,
        cash=20000.0,
        positions=[],
        daily_pnl=500.0,
        daily_pnl_pct=1.0,
    )
    sid = await insert_portfolio_snapshot(tmp_db, snap)
    assert sid > 0
