"""
Tests for AI decision parsing and validation.

Covers:
  - Happy path: all fields present and valid
  - Invalid/missing action
  - Null/NaN required fields
  - Missing ticker on buy/sell
  - JSON embedded in extra text
  - hold_period defaults
  - validation_failed flag set correctly
"""
from __future__ import annotations

import json

import pytest

from src.ai.decision import parse_decision


def _make_json(**kwargs) -> str:
    base = {
        "action": "buy",
        "ticker": "AAPL",
        "confidence": 0.82,
        "position_size_pct": 0.01,
        "stop_loss_pct": 0.02,
        "take_profit_pct": 0.04,
        "hold_period": "swing",
        "reasoning": "Strong earnings beat",
        "risk_factors": ["macro risk"],
        "literature_basis": "Tetlock 2007",
    }
    base.update(kwargs)
    return json.dumps(base)


# ── Happy path ────────────────────────────────────────────────────────────────

def test_parse_valid_buy():
    d = parse_decision(_make_json())
    assert d.action == "buy"
    assert d.ticker == "AAPL"
    assert d.confidence == 0.82
    assert d.stop_loss_pct == 0.02
    assert d.take_profit_pct == 0.04
    assert d.hold_period == "swing"
    assert not d.validation_failed


def test_parse_valid_hold():
    raw = json.dumps({
        "action": "hold",
        "ticker": None,
        "confidence": 0.5,
        "position_size_pct": 0.0,
        "stop_loss_pct": 0.01,
        "take_profit_pct": 0.02,
        "hold_period": "intraday",
        "reasoning": "Insufficient signal",
        "risk_factors": [],
        "literature_basis": "",
    })
    d = parse_decision(raw)
    assert d.action == "hold"
    assert not d.validation_failed


def test_parse_sell():
    d = parse_decision(_make_json(action="sell", ticker="TSLA"))
    assert d.action == "sell"
    assert d.ticker == "TSLA"


# ── JSON extraction ────────────────────────────────────────────────────────────

def test_parse_json_with_surrounding_text():
    """Model may include preamble text before the JSON blob."""
    raw = "Sure, here is my analysis:\n" + _make_json() + "\nHope this helps!"
    d = parse_decision(raw)
    assert d.action == "buy"
    assert not d.validation_failed


def test_parse_no_json_returns_failed():
    d = parse_decision("This is plain text with no JSON.")
    assert d.validation_failed
    assert d.action == "hold"
    assert "ERR_JSON_DECODE" in d.validation_error


# ── Invalid action ─────────────────────────────────────────────────────────────

def test_parse_invalid_action():
    d = parse_decision(_make_json(action="short"))
    assert d.validation_failed
    assert "ERR_INVALID_ACTION" in d.validation_error
    assert d.action == "hold"


def test_parse_empty_action():
    d = parse_decision(_make_json(action=""))
    assert d.validation_failed


# ── Null/NaN required fields ──────────────────────────────────────────────────

def test_parse_null_confidence():
    d = parse_decision(_make_json(confidence=None))
    assert d.validation_failed
    assert "ERR_NULL_FIELD" in d.validation_error


def test_parse_null_stop_loss():
    d = parse_decision(_make_json(stop_loss_pct=None))
    assert d.validation_failed


def test_parse_null_take_profit():
    d = parse_decision(_make_json(take_profit_pct=None))
    assert d.validation_failed


# ── Missing ticker ─────────────────────────────────────────────────────────────

def test_parse_buy_without_ticker():
    d = parse_decision(_make_json(action="buy", ticker=None))
    assert d.validation_failed
    assert "ERR_MISSING_TICKER" in d.validation_error


def test_parse_sell_without_ticker():
    d = parse_decision(_make_json(action="sell", ticker=""))
    assert d.validation_failed


def test_parse_hold_without_ticker_ok():
    """hold action doesn't require a ticker."""
    raw = json.dumps({
        "action": "hold",
        "ticker": None,
        "confidence": 0.5,
        "position_size_pct": 0.0,
        "stop_loss_pct": 0.01,
        "take_profit_pct": 0.02,
        "hold_period": "intraday",
        "reasoning": "test",
        "risk_factors": [],
        "literature_basis": "",
    })
    d = parse_decision(raw)
    assert not d.validation_failed


# ── hold_period defaults ───────────────────────────────────────────────────────

def test_parse_missing_hold_period_defaults_intraday():
    raw = json.dumps({
        "action": "buy",
        "ticker": "MSFT",
        "confidence": 0.7,
        "position_size_pct": 0.01,
        "stop_loss_pct": 0.02,
        "take_profit_pct": 0.04,
        "reasoning": "test",
        "risk_factors": [],
        "literature_basis": "",
    })
    d = parse_decision(raw)
    assert d.hold_period == "intraday"
    assert not d.validation_failed


def test_parse_invalid_hold_period_defaults_intraday():
    d = parse_decision(_make_json(hold_period="weekly"))
    assert d.hold_period == "intraday"
    assert not d.validation_failed


# ── Ticker uppercase normalization ────────────────────────────────────────────

def test_parse_ticker_uppercased():
    d = parse_decision(_make_json(ticker="aapl"))
    assert d.ticker == "AAPL"
