"""
Telegram alert system with burst guard.

Alert taxonomy (from CEO plan):
  - Trade opened/closed    → batched, max 1 msg/5s
  - Daily P&L summary      → 16:05 ET, once/day
  - Ollama circuit breaker → priority (bypasses queue)
  - Daily loss limit hit   → priority (bypasses queue)
  - Bot startup/shutdown   → priority (bypasses queue)
  - Telegram API failure   → log to file, retry 3x with 5s backoff; skip silently

Burst guard: global max 1 message per 5s.
Priority alerts bypass the queue and are sent immediately.
"""
from __future__ import annotations

import asyncio
import time

import aiohttp
import structlog

log = structlog.get_logger(__name__)

_BURST_INTERVAL = 5.0     # minimum seconds between messages
_TELEGRAM_API = "https://api.telegram.org/bot{token}/sendMessage"
_MAX_RETRIES = 3
_RETRY_DELAY = 5.0


class TelegramAlerter:
    def __init__(self, bot_token: str, chat_id: str) -> None:
        self._token = bot_token
        self._chat_id = chat_id
        self._queue: asyncio.Queue[tuple[str, bool]] = asyncio.Queue()
        self._last_sent = 0.0
        self._enabled = bool(bot_token and chat_id)

    async def send(self, message: str, priority: bool = False) -> None:
        """
        Enqueue a message for sending.

        Priority messages bypass the queue and respect only the burst guard.
        """
        if not self._enabled:
            return

        if priority:
            await self._send_now(message)
        else:
            await self._queue.put((message, False))

    async def run(self) -> None:
        """Background task — drains the queue with burst guard."""
        while True:
            message, _ = await self._queue.get()
            await self._send_now(message)

    async def _send_now(self, message: str) -> None:
        """Send immediately, respecting burst guard and retrying on failure."""
        # Burst guard: wait until enough time has passed
        now = time.monotonic()
        wait = _BURST_INTERVAL - (now - self._last_sent)
        if wait > 0:
            await asyncio.sleep(wait)

        self._last_sent = time.monotonic()

        url = _TELEGRAM_API.format(token=self._token)
        payload = {
            "chat_id": self._chat_id,
            "text": message,
            "parse_mode": "HTML",
        }

        for attempt in range(1, _MAX_RETRIES + 1):
            try:
                async with aiohttp.ClientSession(
                    timeout=aiohttp.ClientTimeout(total=10)
                ) as session:
                    async with session.post(url, json=payload) as resp:
                        if resp.status == 200:
                            log.debug("telegram_sent", chars=len(message))
                            return
                        body = await resp.text()
                        # 4xx = permanent client error (wrong token/chat_id) — disable immediately
                        if 400 <= resp.status < 500:
                            log.warning(
                                "telegram_disabled",
                                reason=f"HTTP {resp.status} — check TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID",
                                body=body[:120],
                            )
                            self._enabled = False
                            return
                        log.warning(
                            "telegram_http_error",
                            status=resp.status,
                            attempt=attempt,
                            body=body[:200],
                        )
            except Exception as exc:  # noqa: BLE001
                log.warning("telegram_send_error", attempt=attempt, error=str(exc))

            if attempt < _MAX_RETRIES:
                await asyncio.sleep(_RETRY_DELAY)

        log.error("telegram_all_retries_failed", message_preview=message[:100])
