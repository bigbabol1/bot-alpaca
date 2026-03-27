"""
News filter pipeline — runs before FinBERT and Ollama.

Steps:
  1. Dedup (URL + headline hash, 24h window)
  2. Source quality tier
  3. Relevance scoring (ticker mention + keyword match)
  4. Recency gate (intraday: 4h, swing: 48h)
  5. Batch assembly (3–5 items per Ollama call)
"""
from __future__ import annotations

import hashlib
from datetime import datetime, timedelta, timezone
from typing import NamedTuple

import structlog

from src.history.models import NewsItem

log = structlog.get_logger(__name__)

# ── Source tiers ──────────────────────────────────────────────────────────────
_TIER1 = frozenset(["wsj", "bloomberg", "reuters", "ft", "financial times"])
_TIER2 = frozenset(["marketwatch", "benzinga", "cnbc", "seekingalpha"])
# Everything else = tier 3

_SOURCE_QUALITY: dict[int, float] = {1: 1.0, 2: 0.7, 3: 0.4}

# ── Relevance keywords ────────────────────────────────────────────────────────
_KEYWORDS = frozenset([
    "earnings", "revenue", "profit", "loss", "guidance", "forecast",
    "fda", "approval", "drug", "clinical", "trial",
    "acquisition", "merger", "buyout", "takeover", "deal",
    "rate", "inflation", "fed", "federal reserve", "interest",
    "gdp", "recession", "jobs", "unemployment",
    "dividend", "buyback", "ipo", "offering",
    "lawsuit", "settlement", "investigation", "fine",
    "upgrade", "downgrade", "target", "analyst",
    "beat", "miss", "above", "below", "estimate",
    "crypto", "bitcoin", "ethereum",
])


class FilterResult(NamedTuple):
    passed: bool
    reason: str
    quality: float
    relevance: float


class NewsFilter:
    """
    Stateful filter that tracks seen URLs/headlines in a rolling 24h window.
    """

    def __init__(self, watchlist: list[str]) -> None:
        self._seen: dict[str, datetime] = {}   # hash → first_seen_utc
        self._watchlist: frozenset[str] = frozenset(t.upper() for t in watchlist)

    def _hash(self, item: NewsItem) -> str:
        key = f"{item.url}|{item.headline.lower().strip()}"
        return hashlib.sha256(key.encode()).hexdigest()[:16]

    def _source_tier(self, source: str) -> int:
        s = source.lower()
        if any(t in s for t in _TIER1):
            return 1
        if any(t in s for t in _TIER2):
            return 2
        return 3

    def _relevance(self, item: NewsItem) -> float:
        headline_lower = item.headline.lower()
        summary_lower = (item.summary or "").lower()

        ticker_hit = any(
            sym.upper() in self._watchlist for sym in (item.symbols or [])
        )
        keyword_hit = any(kw in headline_lower or kw in summary_lower for kw in _KEYWORDS)

        if ticker_hit and keyword_hit:
            return 1.0
        if ticker_hit:
            return 0.8
        if keyword_hit:
            return 0.5
        return 0.1

    def check(self, item: NewsItem, mode: str = "intraday") -> FilterResult:
        """
        Args:
            item:  News item to evaluate.
            mode:  "intraday" (4h cutoff) or "swing" (48h cutoff).

        Returns:
            FilterResult with passed=True if the item should be forwarded.
        """
        now = datetime.now(tz=timezone.utc)

        # 1. Recency gate
        max_age = timedelta(hours=4) if mode == "intraday" else timedelta(hours=48)
        received = item.received_at
        if received.tzinfo is None:
            received = received.replace(tzinfo=timezone.utc)
        if now - received > max_age:
            return FilterResult(False, "too_old", 0.0, 0.0)

        # 2. Dedup
        h = self._hash(item)
        self._prune_seen(now)
        if h in self._seen:
            return FilterResult(False, "duplicate", 0.0, 0.0)
        self._seen[h] = now

        # 3. Source quality
        tier = self._source_tier(item.source or "")
        quality = _SOURCE_QUALITY[tier]

        # 4. Relevance
        relevance = self._relevance(item)
        if relevance < 0.4:
            return FilterResult(False, "low_relevance", quality, relevance)

        return FilterResult(True, "ok", quality, relevance)

    def _prune_seen(self, now: datetime) -> None:
        cutoff = now - timedelta(hours=24)
        to_delete = [h for h, ts in self._seen.items() if ts < cutoff]
        for h in to_delete:
            del self._seen[h]


def assemble_batches(items: list[NewsItem], batch_size: int = 5) -> list[list[NewsItem]]:
    """Group filtered news items into batches for Ollama."""
    return [items[i:i + batch_size] for i in range(0, len(items), batch_size)]
