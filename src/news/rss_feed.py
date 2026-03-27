"""
RSS feed aggregator — supplement/fallback when Alpaca WS is down.

Polls Reuters, AP, MarketWatch, FT (free tier) every 5 minutes.
Degrades gracefully on individual feed failures.
"""
from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime

import aiohttp
import feedparser
import structlog

from src.history.models import NewsItem

log = structlog.get_logger(__name__)

_RSS_FEEDS = {
    "reuters": "https://feeds.reuters.com/reuters/businessNews",
    "ap": "https://feeds.apnews.com/rss/apf-business",
    "marketwatch": "https://feeds.marketwatch.com/marketwatch/topstories",
}

_POLL_INTERVAL = 300   # 5 minutes


class RSSFeedAggregator:
    def __init__(self, news_queue: asyncio.Queue) -> None:
        self._queue = news_queue

    async def run(self) -> None:
        """Poll all RSS feeds on a fixed interval."""
        while True:
            await self._poll_all()
            await asyncio.sleep(_POLL_INTERVAL)

    async def _poll_all(self) -> None:
        async with aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=15)
        ) as session:
            tasks = [
                self._poll_feed(session, name, url)
                for name, url in _RSS_FEEDS.items()
            ]
            await asyncio.gather(*tasks, return_exceptions=True)

    async def _poll_feed(
        self, session: aiohttp.ClientSession, name: str, url: str
    ) -> None:
        try:
            async with session.get(url) as resp:
                text = await resp.text()
            feed = feedparser.parse(text)
            count = 0
            for entry in feed.entries:
                item = self._parse_entry(entry, source=name)
                if item:
                    if not self._queue.full():
                        await self._queue.put(item)
                        count += 1
            log.debug("rss_poll_complete", source=name, items=count)
        except Exception as exc:  # noqa: BLE001
            log.warning("rss_poll_failed", source=name, error=str(exc))

    @staticmethod
    def _parse_entry(entry, source: str) -> NewsItem | None:
        try:
            headline = entry.get("title") or ""
            url = entry.get("link") or ""
            summary = entry.get("summary") or ""

            # Parse published date
            published_str = entry.get("published") or entry.get("updated")
            if published_str:
                try:
                    received = parsedate_to_datetime(published_str)
                    if received.tzinfo is None:
                        received = received.replace(tzinfo=timezone.utc)
                except Exception:
                    received = datetime.now(tz=timezone.utc)
            else:
                received = datetime.now(tz=timezone.utc)

            # Extract tickers from tags if available
            symbols: list[str] = []
            for tag in entry.get("tags") or []:
                term = (tag.get("term") or "").upper()
                if term and len(term) <= 5 and term.isalpha():
                    symbols.append(term)

            if not headline or not url:
                return None

            return NewsItem(
                headline=headline,
                summary=summary,
                author=entry.get("author") or "",
                source=source,
                url=url,
                symbols=symbols,
                sentiment=0.0,
                relevance=0.0,
                received_at=received,
            )
        except Exception as exc:  # noqa: BLE001
            log.warning("rss_entry_parse_error", source=source, error=str(exc))
            return None
