"""
Tests for Settings — pydantic-settings configuration class.

Covers:
  - Defaults are sane (paper mode, conservative profile)
  - allowed_ip_list parses comma-separated IPs correctly
  - allowed_ip_list returns empty list when allowed_ips is empty
  - telegram_enabled requires both token and chat_id
  - Sensitive fields masked in repr/str
  - live_mode_confirmed_path under data_dir
"""
from __future__ import annotations

from pathlib import Path

import pytest

from src.config import Settings


def _s(**overrides) -> Settings:
    """Construct a Settings instance with overrides, bypassing .env file."""
    return Settings(_env_file=None, **overrides)


# ── Defaults ─────────────────────────────────────────────────────────────────

def test_defaults_paper_mode():
    s = _s()
    assert s.trading_mode == "paper"


def test_defaults_conservative_profile():
    s = _s()
    assert s.risk_profile == "conservative"


def test_defaults_allowed_ips_empty():
    s = _s()
    assert s.allowed_ips == ""


# ── allowed_ip_list ───────────────────────────────────────────────────────────

def test_allowed_ip_list_empty_string():
    s = _s(allowed_ips="")
    assert s.allowed_ip_list == []


def test_allowed_ip_list_single_ip():
    s = _s(allowed_ips="192.168.1.100")
    assert s.allowed_ip_list == ["192.168.1.100"]


def test_allowed_ip_list_multiple_ips():
    s = _s(allowed_ips="10.0.0.1, 10.0.0.2, 192.168.1.1")
    assert s.allowed_ip_list == ["10.0.0.1", "10.0.0.2", "192.168.1.1"]


def test_allowed_ip_list_strips_whitespace():
    s = _s(allowed_ips="  127.0.0.1  ,  ::1  ")
    assert s.allowed_ip_list == ["127.0.0.1", "::1"]


# ── telegram_enabled ──────────────────────────────────────────────────────────

def test_telegram_disabled_when_both_empty():
    s = _s()
    assert s.telegram_enabled is False


def test_telegram_disabled_when_only_token():
    s = _s(telegram_bot_token="abc123", telegram_chat_id="")
    assert s.telegram_enabled is False


def test_telegram_disabled_when_only_chat_id():
    s = _s(telegram_bot_token="", telegram_chat_id="12345")
    assert s.telegram_enabled is False


def test_telegram_enabled_when_both_set():
    s = _s(telegram_bot_token="abc123", telegram_chat_id="12345")
    assert s.telegram_enabled is True


# ── live_mode_confirmed_path ──────────────────────────────────────────────────

def test_live_mode_path_under_data_dir():
    s = _s(data_dir=Path("/custom/data"))
    assert s.live_mode_confirmed_path == Path("/custom/data/.live_mode_confirmed")


# ── Secret masking ────────────────────────────────────────────────────────────

def test_repr_masks_api_key():
    s = _s(alpaca_api_key="REAL_KEY_VALUE", alpaca_secret_key="REAL_SECRET")
    r = repr(s)
    assert "REAL_KEY_VALUE" not in r
    assert "REAL_SECRET" not in r
    assert "***" in r


def test_str_masks_secrets():
    s = _s(alpaca_api_key="DO_NOT_LOG_ME")
    assert "DO_NOT_LOG_ME" not in str(s)
