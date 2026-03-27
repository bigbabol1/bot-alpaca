"""
Parse and validate JSON decisions from Ollama.

Validation rules:
  - Required fields: action, ticker (unless hold), confidence, stop_loss_pct, take_profit_pct
  - None/NaN on any required field → hold + WARNING log
  - Unknown action values → hold + WARNING log
  - Validation failures are written to ai_decisions table with action="validation_failed"
"""
from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass

import structlog

log = structlog.get_logger(__name__)

_VALID_ACTIONS = frozenset(["buy", "sell", "hold"])
_VALID_HOLD_PERIODS = frozenset(["intraday", "swing", "position"])
_MAX_RAW_LOG = 500   # cap logged raw response length


@dataclass
class ParsedDecision:
    action: str
    ticker: str | None
    confidence: float
    position_size_pct: float
    stop_loss_pct: float
    take_profit_pct: float
    hold_period: str
    reasoning: str
    risk_factors: list[str]
    literature_basis: str
    validation_failed: bool = False
    validation_error: str = ""


def _is_null_or_nan(v) -> bool:
    if v is None:
        return True
    try:
        return math.isnan(float(v))
    except (TypeError, ValueError):
        return False


def _extract_json(text: str) -> str | None:
    """Extract the first JSON object from a response that may contain extra text."""
    # Try to find a {...} block
    match = re.search(r'\{.*\}', text, re.DOTALL)
    if match:
        return match.group(0)
    return None


def parse_decision(raw: str) -> ParsedDecision:
    """
    Parse Ollama's JSON response into a validated ParsedDecision.

    On any validation failure: returns a hold decision with validation_failed=True
    and logs a WARNING with the failed field and raw response excerpt.
    """
    raw_excerpt = raw[:_MAX_RAW_LOG]

    # Step 1: JSON decode
    try:
        extracted = _extract_json(raw)
        if not extracted:
            raise ValueError("No JSON object found in response")
        data = json.loads(extracted)
    except (json.JSONDecodeError, ValueError) as exc:
        log.warning(
            "decision_json_parse_failed",
            error_code="ERR_JSON_DECODE",
            error=str(exc),
            raw_excerpt=raw_excerpt,
        )
        return _failed("ERR_JSON_DECODE", str(exc))

    # Step 2: action validation
    action = str(data.get("action") or "").lower().strip()
    if action not in _VALID_ACTIONS:
        log.warning(
            "decision_invalid_action",
            error_code="ERR_INVALID_ACTION",
            field="action",
            value=action,
            raw_excerpt=raw_excerpt,
        )
        return _failed("ERR_INVALID_ACTION", f"invalid action: {action!r}")

    # Step 3: required numeric fields (skip ticker check for hold)
    required_numeric = ["confidence", "stop_loss_pct", "take_profit_pct"]
    for field_name in required_numeric:
        val = data.get(field_name)
        if _is_null_or_nan(val):
            log.warning(
                "decision_null_required_field",
                error_code="ERR_NULL_FIELD",
                field=field_name,
                raw_excerpt=raw_excerpt,
            )
            return _failed("ERR_NULL_FIELD", f"null/NaN required field: {field_name}")

    # Step 4: ticker required for buy/sell
    ticker = data.get("ticker")
    if action in ("buy", "sell") and not ticker:
        log.warning(
            "decision_missing_ticker",
            error_code="ERR_MISSING_TICKER",
            field="ticker",
            raw_excerpt=raw_excerpt,
        )
        return _failed("ERR_MISSING_TICKER", "missing ticker for buy/sell")

    # Step 5: hold_period default
    hold_period = str(data.get("hold_period") or "intraday").lower()
    if hold_period not in _VALID_HOLD_PERIODS:
        hold_period = "intraday"

    return ParsedDecision(
        action=action,
        ticker=str(ticker).upper() if ticker else None,
        confidence=max(0.0, min(1.0, float(data.get("confidence") or 0.0))),
        position_size_pct=float(data.get("position_size_pct") or 0.0),
        stop_loss_pct=float(data.get("stop_loss_pct") or 0.0),
        take_profit_pct=float(data.get("take_profit_pct") or 0.0),
        hold_period=hold_period,
        reasoning=str(data.get("reasoning") or ""),
        risk_factors=list(data.get("risk_factors") or []),
        literature_basis=str(data.get("literature_basis") or ""),
    )


def _failed(error_code: str, message: str) -> ParsedDecision:
    return ParsedDecision(
        action="hold",
        ticker=None,
        confidence=0.0,
        position_size_pct=0.0,
        stop_loss_pct=0.0,
        take_profit_pct=0.0,
        hold_period="intraday",
        reasoning=f"Validation failed: {message}",
        risk_factors=[],
        literature_basis="",
        validation_failed=True,
        validation_error=f"{error_code}: {message}",
    )
