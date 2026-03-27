"""
DB row models with centralized JSON serialization.

Every model exposes:
  - to_db_row()   → dict ready for INSERT (JSON-encodes list/dict fields)
  - from_db_row() → classmethod constructing from a sqlite3.Row / dict
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any


def _dumps(v: Any) -> str:
    return json.dumps(v, default=str)


def _loads(v: str | None) -> Any:
    if v is None:
        return None
    if isinstance(v, (list, dict)):
        return v  # already deserialized (e.g. from test fixtures)
    return json.loads(v)


@dataclass
class NewsItem:
    headline: str
    source: str
    url: str
    symbols: list[str]
    sentiment: float          # -1.0 to 1.0 (FinBERT pre-score)
    relevance: float          # 0.0 to 1.0
    summary: str = ""
    author: str = ""
    received_at: datetime = field(default_factory=datetime.utcnow)
    forwarded: bool = False
    id: int | None = None

    def to_db_row(self) -> dict:
        return {
            "headline": self.headline,
            "summary": self.summary,
            "author": self.author,
            "source": self.source,
            "url": self.url,
            "symbols": _dumps(self.symbols),
            "sentiment": self.sentiment,
            "relevance": self.relevance,
            "received_at": self.received_at.isoformat(),
            "forwarded": int(self.forwarded),
        }

    @classmethod
    def from_db_row(cls, row: dict) -> "NewsItem":
        return cls(
            id=row["id"],
            headline=row["headline"],
            summary=row.get("summary") or "",
            author=row.get("author") or "",
            source=row["source"],
            url=row["url"],
            symbols=_loads(row.get("symbols")) or [],
            sentiment=row["sentiment"],
            relevance=row["relevance"],
            received_at=datetime.fromisoformat(row["received_at"]),
            forwarded=bool(row.get("forwarded", 0)),
        )


@dataclass
class AIDecision:
    news_item_ids: list[int]
    model: str
    action: str               # buy | sell | hold
    ticker: str | None
    confidence: float
    position_pct: float
    stop_loss_pct: float
    take_profit_pct: float
    hold_period: str          # intraday | swing | position
    reasoning: str
    risk_factors: list[str]
    literature_ref: str
    risk_profile: str
    created_at: datetime = field(default_factory=datetime.utcnow)
    id: int | None = None

    def to_db_row(self) -> dict:
        return {
            "news_item_ids": _dumps(self.news_item_ids),
            "model": self.model,
            "action": self.action,
            "ticker": self.ticker,
            "confidence": self.confidence,
            "position_pct": self.position_pct,
            "stop_loss_pct": self.stop_loss_pct,
            "take_profit_pct": self.take_profit_pct,
            "hold_period": self.hold_period,
            "reasoning": self.reasoning,
            "risk_factors": _dumps(self.risk_factors),
            "literature_ref": self.literature_ref,
            "risk_profile": self.risk_profile,
            "created_at": self.created_at.isoformat(),
        }

    @classmethod
    def from_db_row(cls, row: dict) -> "AIDecision":
        return cls(
            id=row["id"],
            news_item_ids=_loads(row["news_item_ids"]) or [],
            model=row["model"],
            action=row["action"],
            ticker=row.get("ticker"),
            confidence=row["confidence"],
            position_pct=row["position_pct"],
            stop_loss_pct=row["stop_loss_pct"],
            take_profit_pct=row["take_profit_pct"],
            hold_period=row["hold_period"],
            reasoning=row["reasoning"],
            risk_factors=_loads(row["risk_factors"]) or [],
            literature_ref=row.get("literature_ref") or "",
            risk_profile=row["risk_profile"],
            created_at=datetime.fromisoformat(row["created_at"]),
        )


@dataclass
class Trade:
    decision_id: int
    ticker: str
    side: str               # buy | sell
    qty: float
    entry_price: float
    stop_loss: float
    take_profit: float
    alpaca_order_id: str = ""
    exit_price: float | None = None
    status: str = "pending"  # pending | open | closed | cancelled | rate_limited | timeout_cancelled | abandoned
    pnl: float | None = None
    pnl_pct: float | None = None
    opened_at: datetime = field(default_factory=datetime.utcnow)
    closed_at: datetime | None = None
    id: int | None = None

    def to_db_row(self) -> dict:
        return {
            "decision_id": self.decision_id,
            "alpaca_order_id": self.alpaca_order_id,
            "ticker": self.ticker,
            "side": self.side,
            "qty": self.qty,
            "entry_price": self.entry_price,
            "stop_loss": self.stop_loss,
            "take_profit": self.take_profit,
            "exit_price": self.exit_price,
            "status": self.status,
            "pnl": self.pnl,
            "pnl_pct": self.pnl_pct,
            "opened_at": self.opened_at.isoformat(),
            "closed_at": self.closed_at.isoformat() if self.closed_at else None,
        }

    @classmethod
    def from_db_row(cls, row: dict) -> "Trade":
        return cls(
            id=row["id"],
            decision_id=row["decision_id"],
            alpaca_order_id=row.get("alpaca_order_id") or "",
            ticker=row["ticker"],
            side=row["side"],
            qty=row["qty"],
            entry_price=row["entry_price"],
            stop_loss=row["stop_loss"],
            take_profit=row["take_profit"],
            exit_price=row.get("exit_price"),
            status=row["status"],
            pnl=row.get("pnl"),
            pnl_pct=row.get("pnl_pct"),
            opened_at=datetime.fromisoformat(row["opened_at"]),
            closed_at=datetime.fromisoformat(row["closed_at"]) if row.get("closed_at") else None,
        )


@dataclass
class PortfolioSnapshot:
    equity: float
    cash: float
    positions: list[dict]    # [{ticker, qty, market_value, unrealized_pnl}]
    daily_pnl: float
    daily_pnl_pct: float
    snapped_at: datetime = field(default_factory=datetime.utcnow)
    id: int | None = None

    def to_db_row(self) -> dict:
        return {
            "equity": self.equity,
            "cash": self.cash,
            "positions": _dumps(self.positions),
            "daily_pnl": self.daily_pnl,
            "daily_pnl_pct": self.daily_pnl_pct,
            "snapped_at": self.snapped_at.isoformat(),
        }

    @classmethod
    def from_db_row(cls, row: dict) -> "PortfolioSnapshot":
        return cls(
            id=row["id"],
            equity=row["equity"],
            cash=row["cash"],
            positions=_loads(row["positions"]) or [],
            daily_pnl=row["daily_pnl"],
            daily_pnl_pct=row["daily_pnl_pct"],
            snapped_at=datetime.fromisoformat(row["snapped_at"]),
        )
