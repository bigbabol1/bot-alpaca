"""
Central configuration — loaded from .env via pydantic-settings.

API keys are masked in repr/str/logging to prevent accidental leakage.
"""
from __future__ import annotations

import re
from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

import structlog

log = structlog.get_logger(__name__)

_MASK = "***"
_SECRET_PATTERN = re.compile(r"(key|secret|token|password|passwd|pwd)", re.IGNORECASE)


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ── Alpaca ────────────────────────────────────────────────────────────────
    alpaca_api_key: str = ""
    alpaca_secret_key: str = ""
    trading_mode: Literal["paper", "live"] = "paper"

    # ── Ollama ─────────────────────────────────────────────────────────────────
    ollama_host: str = "http://192.168.0.66:11434"
    ollama_decision_model: str = "qwen2.5:32b"
    ollama_triage_model: str = "qwen2.5:7b"
    two_stage_llm: bool = False

    # ── Risk ───────────────────────────────────────────────────────────────────
    risk_profile: Literal["conservative", "moderate", "aggressive"] = "conservative"

    # ── Telegram ───────────────────────────────────────────────────────────────
    telegram_bot_token: str = ""
    telegram_chat_id: str = ""

    # ── Dashboard ──────────────────────────────────────────────────────────────
    dashboard_port: int = 8080
    allowed_ips: str = ""  # comma-separated; empty = any LAN IP

    # ── Paths ──────────────────────────────────────────────────────────────────
    data_dir: Path = Path("/app/data")
    log_dir: Path = Path("/app/logs")
    config_dir: Path = Path("/app/config")

    @field_validator("trading_mode")
    @classmethod
    def validate_live_gate(cls, v: str) -> str:
        # Actual file-existence check done at runtime in main.py so we can
        # test this class without the file present.
        return v

    @model_validator(mode="after")
    def warn_two_stage_single_gpu(self) -> "Settings":
        if self.two_stage_llm:
            log.warning(
                "two_stage_llm_single_gpu_warning",
                message=(
                    "TWO_STAGE_LLM=true on a single GPU will increase latency via "
                    "model eviction. Enable only when triage_model runs on a separate "
                    "GPU/host."
                ),
            )
        return self

    # ── Masking ────────────────────────────────────────────────────────────────
    def _masked_dict(self) -> dict:
        d = {}
        for field_name, value in self.__dict__.items():
            if _SECRET_PATTERN.search(field_name):
                d[field_name] = _MASK
            else:
                d[field_name] = value
        return d

    def __repr__(self) -> str:
        return f"Settings({self._masked_dict()})"

    def __str__(self) -> str:
        return repr(self)

    @property
    def allowed_ip_list(self) -> list[str]:
        if not self.allowed_ips:
            return []
        return [ip.strip() for ip in self.allowed_ips.split(",") if ip.strip()]

    @property
    def live_mode_confirmed_path(self) -> Path:
        return self.data_dir / ".live_mode_confirmed"

    @property
    def telegram_enabled(self) -> bool:
        return bool(self.telegram_bot_token and self.telegram_chat_id)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
