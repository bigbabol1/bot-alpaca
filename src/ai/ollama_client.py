"""
Ollama client with circuit breaker, retry, and stripped-down fallback prompt.

Circuit breaker states:
  CLOSED  → normal operation
  OPEN    → safe mode: no new trades; retry health-check every 5min
  HALF    → one successful health check; send test prompt; if OK → CLOSED

Flow on failure:
  1st failure → retry with same prompt (async_retry, 2 attempts)
  2nd failure → retry with stripped-down prompt (headline only)
  3rd consecutive failure → OPEN circuit; Telegram alert
"""
from __future__ import annotations

import asyncio
import time
from enum import Enum

import structlog
from openai import AsyncOpenAI

from src.ai.decision import ParsedDecision, parse_decision
from src.ai.prompts import (
    STRIPPED_SYSTEM,
    SYSTEM_PROMPT,
    build_stripped_prompt,
)
from src.utils.retry import async_retry

log = structlog.get_logger(__name__)

_OLLAMA_TIMEOUT = 30.0         # seconds per request
_CIRCUIT_RETRY_INTERVAL = 300  # 5 min health check when open
_CIRCUIT_THRESHOLD = 3         # consecutive failures to open circuit
_HEALTH_PATH = "/api/tags"


class CircuitState(Enum):
    CLOSED = "closed"
    OPEN = "open"
    HALF = "half"


class OllamaClient:
    def __init__(self, host: str, decision_model: str) -> None:
        self._host = host
        self._decision_model = decision_model
        self._client = AsyncOpenAI(
            base_url=f"{host}/v1",
            api_key="ollama",   # Ollama ignores the key value
            timeout=_OLLAMA_TIMEOUT,
        )
        self._state = CircuitState.CLOSED
        self._consecutive_failures = 0
        self._alert_callback = None   # injected by main.py

    def set_alert_callback(self, fn) -> None:
        """Inject Telegram alert function (avoids circular import)."""
        self._alert_callback = fn

    @property
    def is_safe_mode(self) -> bool:
        return self._state == CircuitState.OPEN

    # ── Main decision call ─────────────────────────────────────────────────────

    async def decide(
        self,
        user_prompt: str,
        news_ticker: str = "",
        news_headline: str = "",
    ) -> ParsedDecision:
        """
        Request a trading decision from Ollama.

        Retry strategy:
          - Attempt 1 & 2: full prompt (async_retry with backoff)
          - Attempt 3: stripped-down prompt (headline only)
          - All failed: circuit breaker incremented; return hold

        Returns a ParsedDecision (may have validation_failed=True on parse error).
        """
        if self._state == CircuitState.OPEN:
            log.warning("ollama_circuit_open_skipping")
            return _hold_decision("circuit breaker open — trading paused")

        # Attempt 1–2: full prompt
        try:
            result = await async_retry(
                fn=lambda: self._call(SYSTEM_PROMPT, user_prompt),
                attempts=2,
                backoff_base=2.0,
                max_backoff=8.0,
                label="ollama_decision_full",
            )
            self._on_success()
            return result
        except Exception:  # noqa: BLE001
            pass

        # Attempt 3: stripped-down prompt
        log.warning("ollama_fallback_stripped_prompt", ticker=news_ticker)
        try:
            stripped = build_stripped_prompt(news_ticker, news_headline)
            result = await self._call(STRIPPED_SYSTEM, stripped)
            self._on_success()
            return result
        except Exception as exc:  # noqa: BLE001
            log.error("ollama_all_attempts_failed", error=str(exc))
            await self._on_failure()
            return _hold_decision(f"all Ollama attempts failed: {exc}")

    # ── Health check (used by circuit breaker recovery) ────────────────────────

    async def health_check(self) -> bool:
        """Return True if Ollama is reachable."""
        import aiohttp
        try:
            async with aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=10)
            ) as session:
                async with session.get(f"{self._host}{_HEALTH_PATH}") as resp:
                    return resp.status == 200
        except Exception:  # noqa: BLE001
            return False

    async def run_recovery_loop(self) -> None:
        """Background task: attempt to exit safe mode every 5 min."""
        while True:
            await asyncio.sleep(_CIRCUIT_RETRY_INTERVAL)
            if self._state != CircuitState.OPEN:
                continue

            log.info("ollama_circuit_health_check")
            if not await self.health_check():
                log.warning("ollama_circuit_still_open")
                continue

            # Health check passed — send test prompt
            self._state = CircuitState.HALF
            test_result = await self.decide(
                user_prompt="News to evaluate: []\nRespond with a hold decision.",
            )
            if not test_result.validation_failed:
                self._state = CircuitState.CLOSED
                self._consecutive_failures = 0
                log.info("ollama_circuit_recovered")
                await self._send_alert("✅ Ollama recovered — trading resumed")
            else:
                self._state = CircuitState.OPEN
                log.warning("ollama_circuit_test_failed")

    # ── Internal ───────────────────────────────────────────────────────────────

    async def _call(self, system: str, user: str) -> ParsedDecision:
        t0 = time.monotonic()
        try:
            response = await self._client.chat.completions.create(
                model=self._decision_model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                temperature=0.1,
            )
            ollama_ms = round((time.monotonic() - t0) * 1000)
            raw = response.choices[0].message.content or ""
            log.info("ollama_response", ollama_latency_ms=ollama_ms, model=self._decision_model)
            return parse_decision(raw)
        except Exception:
            ollama_ms = round((time.monotonic() - t0) * 1000)
            log.warning("ollama_call_failed", ollama_latency_ms=ollama_ms)
            raise

    async def _on_failure(self) -> None:
        self._consecutive_failures += 1
        if self._consecutive_failures >= _CIRCUIT_THRESHOLD:
            if self._state != CircuitState.OPEN:
                self._state = CircuitState.OPEN
                log.error(
                    "ollama_circuit_opened",
                    consecutive_failures=self._consecutive_failures,
                )
                await self._send_alert("⚠️ Ollama unreachable — trading paused")

    def _on_success(self) -> None:
        self._consecutive_failures = 0
        if self._state == CircuitState.HALF:
            self._state = CircuitState.CLOSED

    async def _send_alert(self, message: str) -> None:
        if self._alert_callback:
            try:
                await self._alert_callback(message, priority=True)
            except Exception:  # noqa: BLE001
                pass


def _hold_decision(reason: str) -> ParsedDecision:
    from src.ai.decision import ParsedDecision
    return ParsedDecision(
        action="hold",
        ticker=None,
        confidence=0.0,
        position_size_pct=0.0,
        stop_loss_pct=0.0,
        take_profit_pct=0.0,
        hold_period="intraday",
        reasoning=reason,
        risk_factors=[],
        literature_basis="",
    )
