"""
Alpaca News feed — WebSocket (real-time) with REST backfill on reconnect.

Architecture:
  ┌─────────────────────┐
  │  AlpacaNewsStream   │──── WebSocket (real-time push) ────▶ news_queue
  │                     │──── REST backfill on reconnect ────▶ news_queue
  │                     │──── Exponential backoff reconnect
  └─────────────────────┘

On disconnect: marks ws_connected=False, backoff reconnects, runs REST backfill
               before resuming to avoid missing news during the gap.
"""
from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone

import structlog
from alpaca.data.live import NewsDataStream
from alpaca.data.historical import NewsClient
from alpaca.data.requests import NewsRequest

from src.config import Settings
from src.history.models import NewsItem

log = structlog.get_logger(__name__)

_BACKOFF = [2, 4, 8, 16, 32, 60, 120, 300]   # seconds; capped at 5min


class AlpacaNewsStream:
    def __init__(
        self,
        settings: Settings,
        news_queue: asyncio.Queue,
        watchlist: list[str],
    ) -> None:
        self._settings = settings
        self._queue = news_queue
        self._watchlist = watchlist
        self._ws_connected = False
        self._stream: NewsDataStream | None = None
        self._news_client: NewsClient | None = None

    # ── Public API ─────────────────────────────────────────────────────────────

    async def run(self) -> None:
        """Main loop — connect, subscribe, reconnect on failure."""
        self._news_client = NewsClient(
            api_key=self._settings.alpaca_api_key,
            secret_key=self._settings.alpaca_secret_key,
        )
        attempt = 0
        while True:
            try:
                await self._connect_and_subscribe()
                attempt = 0   # reset on successful connection
            except Exception as exc:  # noqa: BLE001
                self._ws_connected = False
                delay = _BACKOFF[min(attempt, len(_BACKOFF) - 1)]
                log.warning(
                    "alpaca_ws_disconnected",
                    error=str(exc),
                    reconnect_in=delay,
                )
                await asyncio.sleep(delay)
                attempt += 1

    @property
    def ws_connected(self) -> bool:
        return self._ws_connected

    # ── Private ────────────────────────────────────────────────────────────────

    async def _connect_and_subscribe(self) -> None:
        self._stream = NewsDataStream(
            api_key=self._settings.alpaca_api_key,
            secret_key=self._settings.alpaca_secret_key,
        )

        async def _on_news(data) -> None:
            item = self._parse(data)
            if item:
                await self._enqueue(item)

        self._stream.subscribe_news(_on_news, *self._watchlist)
        log.info("alpaca_ws_connecting")
        self._ws_connected = True

        # Backfill recent news missed during any prior disconnect
        await self._backfill(minutes=30)

        await self._stream._run_forever()  # blocks until disconnect

    async def _backfill(self, minutes: int = 30) -> None:
        """Fetch news from REST API to fill any gap since last connection."""
        if not self._news_client:
            return
        since = datetime.now(tz=timezone.utc) - timedelta(minutes=minutes)
        try:
            req = NewsRequest(
                symbols=",".join(self._watchlist),
                start=since,
                limit=50,
            )
            news = self._news_client.get_news(req)
            count = 0
            for article in (news.data.get("news") or []):
                item = self._parse_rest(article)
                if item:
                    await self._enqueue(item)
                    count += 1
            log.info("alpaca_backfill_complete", items=count, minutes=minutes)
        except Exception as exc:  # noqa: BLE001
            log.warning("alpaca_backfill_failed", error=str(exc))

    async def _enqueue(self, item: NewsItem) -> None:
        if self._queue.full():
            depth = self._queue.qsize()
            log.warning("news_queue_full", queue_depth=depth, dropped_headline=item.headline[:80])
            return
        await self._queue.put(item)

        # Alert at 80% capacity
        depth = self._queue.qsize()
        if depth >= self._queue.maxsize * 0.8:
            log.warning("news_queue_near_capacity", queue_depth=depth, maxsize=self._queue.maxsize)

    @staticmethod
    def _parse(data) -> NewsItem | None:
        """Parse a live WebSocket news message."""
        try:
            return NewsItem(
                headline=data.headline or "",
                summary=data.summary or "",
                author=data.author or "",
                source=data.source or "",
                url=data.url or "",
                symbols=list(data.symbols or []),
                sentiment=0.0,   # will be filled by sentiment.py
                relevance=0.0,   # will be filled by filter.py
                received_at=datetime.now(tz=timezone.utc),
            )
        except Exception as exc:  # noqa: BLE001
            log.warning("alpaca_ws_parse_error", error=str(exc))
            return None

    @staticmethod
    def _parse_rest(article) -> NewsItem | None:
        """Parse a REST API news article."""
        try:
            published = article.get("created_at") or article.get("updated_at")
            if isinstance(published, str):
                received = datetime.fromisoformat(published.replace("Z", "+00:00"))
            elif isinstance(published, datetime):
                received = published
            else:
                received = datetime.now(tz=timezone.utc)

            return NewsItem(
                headline=article.get("headline") or "",
                summary=article.get("summary") or "",
                author=article.get("author") or "",
                source=article.get("source") or "alpaca",
                url=article.get("url") or "",
                symbols=list(article.get("symbols") or []),
                sentiment=0.0,
                relevance=0.0,
                received_at=received,
            )
        except Exception as exc:  # noqa: BLE001
            log.warning("alpaca_rest_parse_error", error=str(exc))
            return None
