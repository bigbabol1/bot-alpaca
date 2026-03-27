"""
Async SQLite interface with WAL mode, migrations, and typed accessors.

WAL mode is enabled on every new connection (required for concurrent writers).

Write serialization: all write operations acquire _WRITE_LOCK to prevent
interleaved commits when multiple async tasks share a single connection.
Reads do not acquire the lock (WAL mode supports concurrent readers).
"""
from __future__ import annotations

import asyncio
from pathlib import Path

import aiosqlite
import structlog

from src.history.models import AIDecision, NewsItem, PortfolioSnapshot, Trade

log = structlog.get_logger(__name__)

# Serializes all write operations on the shared connection.
# Created lazily to avoid event-loop issues at import time.
_write_lock: asyncio.Lock | None = None


def _wlock() -> asyncio.Lock:
    global _write_lock
    if _write_lock is None:
        _write_lock = asyncio.Lock()
    return _write_lock

_SCHEMA = """
PRAGMA journal_mode=WAL;

CREATE TABLE IF NOT EXISTS news_items (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    headline    TEXT NOT NULL,
    summary     TEXT,
    author      TEXT,
    source      TEXT,
    url         TEXT UNIQUE,
    symbols     TEXT,
    sentiment   REAL,
    relevance   REAL,
    received_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    forwarded   BOOLEAN DEFAULT 0
);

CREATE TABLE IF NOT EXISTS ai_decisions (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    news_item_ids   TEXT,
    model           TEXT,
    action          TEXT,
    ticker          TEXT,
    confidence      REAL,
    position_pct    REAL,
    stop_loss_pct   REAL,
    take_profit_pct REAL,
    hold_period     TEXT,
    reasoning       TEXT,
    risk_factors    TEXT,
    literature_ref  TEXT,
    risk_profile    TEXT,
    created_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS trades (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    decision_id     INTEGER REFERENCES ai_decisions(id),
    alpaca_order_id TEXT UNIQUE,
    ticker          TEXT,
    side            TEXT,
    qty             REAL,
    entry_price     REAL,
    stop_loss       REAL,
    take_profit     REAL,
    exit_price      REAL,
    status          TEXT,
    pnl             REAL,
    pnl_pct         REAL,
    opened_at       TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    closed_at       TIMESTAMP
);

CREATE TABLE IF NOT EXISTS portfolio_snapshots (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    equity      REAL,
    cash        REAL,
    positions   TEXT,
    daily_pnl   REAL,
    daily_pnl_pct REAL,
    snapped_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS performance_metrics (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    period          TEXT,
    total_trades    INTEGER,
    winning_trades  INTEGER,
    win_rate        REAL,
    total_pnl       REAL,
    sharpe_ratio    REAL,
    max_drawdown    REAL,
    avg_hold_hours  REAL,
    calculated_at   TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_news_received ON news_items(received_at);
CREATE INDEX IF NOT EXISTS idx_news_symbols ON news_items(symbols);
CREATE INDEX IF NOT EXISTS idx_decisions_ticker ON ai_decisions(ticker, created_at);
CREATE INDEX IF NOT EXISTS idx_trades_status ON trades(status, opened_at);
CREATE INDEX IF NOT EXISTS idx_trades_ticker ON trades(ticker, opened_at);
"""


async def open_db(db_path: Path) -> aiosqlite.Connection:
    """Open a connection with WAL mode enabled. Caller is responsible for closing."""
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = await aiosqlite.connect(str(db_path))
    conn.row_factory = aiosqlite.Row
    await conn.execute("PRAGMA journal_mode=WAL")
    await conn.execute("PRAGMA foreign_keys=ON")
    return conn


async def init_db(db_path: Path) -> aiosqlite.Connection:
    """Open DB, run migrations, return connection."""
    conn = await open_db(db_path)
    await conn.executescript(_SCHEMA)
    await conn.commit()
    log.info("db_initialized", path=str(db_path))
    return conn


# ── News ──────────────────────────────────────────────────────────────────────

async def insert_news_item(conn: aiosqlite.Connection, item: NewsItem) -> int:
    row = item.to_db_row()
    async with _wlock():
        async with conn.execute(
            """INSERT OR IGNORE INTO news_items
               (headline, summary, author, source, url, symbols, sentiment, relevance, received_at, forwarded)
               VALUES (:headline, :summary, :author, :source, :url, :symbols, :sentiment, :relevance, :received_at, :forwarded)""",
            row,
        ) as cur:
            await conn.commit()
            return cur.lastrowid or 0


async def mark_news_forwarded(conn: aiosqlite.Connection, news_ids: list[int]) -> None:
    if not news_ids:
        return
    async with _wlock():
        await conn.execute(
            f"UPDATE news_items SET forwarded=1 WHERE id IN ({','.join('?' * len(news_ids))})",
            news_ids,
        )
        await conn.commit()


# ── AI Decisions ───────────────────────────────────────────────────────────────

async def insert_ai_decision(conn: aiosqlite.Connection, decision: AIDecision) -> int:
    row = decision.to_db_row()
    async with _wlock():
        async with conn.execute(
            """INSERT INTO ai_decisions
               (news_item_ids, model, action, ticker, confidence, position_pct,
                stop_loss_pct, take_profit_pct, hold_period, reasoning,
                risk_factors, literature_ref, risk_profile, created_at)
               VALUES (:news_item_ids, :model, :action, :ticker, :confidence, :position_pct,
                       :stop_loss_pct, :take_profit_pct, :hold_period, :reasoning,
                       :risk_factors, :literature_ref, :risk_profile, :created_at)""",
            row,
        ) as cur:
            await conn.commit()
            return cur.lastrowid or 0


async def get_recent_decisions(
    conn: aiosqlite.Connection, limit: int = 10
) -> list[AIDecision]:
    async with conn.execute(
        "SELECT * FROM ai_decisions ORDER BY created_at DESC LIMIT ?", (limit,)
    ) as cur:
        rows = await cur.fetchall()
    return [AIDecision.from_db_row(dict(r)) for r in rows]


async def count_closed_trades(conn: aiosqlite.Connection) -> int:
    async with conn.execute(
        "SELECT COUNT(*) FROM trades WHERE status='closed'"
    ) as cur:
        row = await cur.fetchone()
    return row[0] if row else 0


# ── Trades ────────────────────────────────────────────────────────────────────

async def insert_trade(conn: aiosqlite.Connection, trade: Trade) -> int:
    row = trade.to_db_row()
    async with _wlock():
        async with conn.execute(
            """INSERT INTO trades
               (decision_id, alpaca_order_id, ticker, side, qty, entry_price,
                stop_loss, take_profit, exit_price, status, pnl, pnl_pct, opened_at, closed_at)
               VALUES (:decision_id, :alpaca_order_id, :ticker, :side, :qty, :entry_price,
                       :stop_loss, :take_profit, :exit_price, :status, :pnl, :pnl_pct, :opened_at, :closed_at)""",
            row,
        ) as cur:
            await conn.commit()
            return cur.lastrowid or 0


async def update_trade(conn: aiosqlite.Connection, trade: Trade) -> None:
    row = trade.to_db_row()
    row["id"] = trade.id
    async with _wlock():
        await conn.execute(
            """UPDATE trades SET
               alpaca_order_id=:alpaca_order_id, status=:status, entry_price=:entry_price,
               exit_price=:exit_price, pnl=:pnl, pnl_pct=:pnl_pct, closed_at=:closed_at, qty=:qty
               WHERE id=:id""",
            row,
        )
        await conn.commit()


async def get_open_trades(conn: aiosqlite.Connection) -> list[Trade]:
    async with conn.execute(
        "SELECT * FROM trades WHERE status IN ('pending', 'open') ORDER BY opened_at"
    ) as cur:
        rows = await cur.fetchall()
    return [Trade.from_db_row(dict(r)) for r in rows]


async def get_trade_by_order_id(
    conn: aiosqlite.Connection, alpaca_order_id: str
) -> Trade | None:
    async with conn.execute(
        "SELECT * FROM trades WHERE alpaca_order_id=?", (alpaca_order_id,)
    ) as cur:
        row = await cur.fetchone()
    return Trade.from_db_row(dict(row)) if row else None


async def get_closed_trades(
    conn: aiosqlite.Connection, limit: int = 200
) -> list[Trade]:
    async with conn.execute(
        "SELECT * FROM trades WHERE status='closed' ORDER BY closed_at DESC LIMIT ?",
        (limit,),
    ) as cur:
        rows = await cur.fetchall()
    return [Trade.from_db_row(dict(r)) for r in rows]


# ── Portfolio ──────────────────────────────────────────────────────────────────

async def insert_portfolio_snapshot(
    conn: aiosqlite.Connection, snapshot: PortfolioSnapshot
) -> int:
    row = snapshot.to_db_row()
    async with _wlock():
        async with conn.execute(
            """INSERT INTO portfolio_snapshots
               (equity, cash, positions, daily_pnl, daily_pnl_pct, snapped_at)
               VALUES (:equity, :cash, :positions, :daily_pnl, :daily_pnl_pct, :snapped_at)""",
            row,
        ) as cur:
            await conn.commit()
            return cur.lastrowid or 0


async def get_latest_snapshot(
    conn: aiosqlite.Connection,
) -> PortfolioSnapshot | None:
    async with conn.execute(
        "SELECT * FROM portfolio_snapshots ORDER BY snapped_at DESC LIMIT 1"
    ) as cur:
        row = await cur.fetchone()
    return PortfolioSnapshot.from_db_row(dict(row)) if row else None
