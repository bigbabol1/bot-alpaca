"""
Alpaca Trading Bot — async entry point.

Startup sequence:
  1. Validate config (paper→live gate)
  2. Initialize DB (WAL mode, migrations)
  3. Run order reconciliation (sync DB ↔ Alpaca open orders)
  4. Start background tasks:
     - Market data cache refresh (every 15min)
     - Config hot-reload watcher (every 5s)
     - RSS feed aggregator (every 5min)
     - Ollama circuit breaker recovery loop
     - Telegram alert queue drainer
     - Daily metrics job
  5. Connect Alpaca News WebSocket
  6. Main news processing loop

Signal flow:
  news_queue → filter → FinBERT → batch → Ollama → risk engine → executor → DB
"""
from __future__ import annotations

import asyncio
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import structlog
import yaml

from src.ai.ollama_client import OllamaClient
from src.ai.prompts import build_context_prompt
from src.config import Settings, get_settings
from src.history import db as db_ops
from src.history.context import build_context
from src.history.db import init_db, insert_ai_decision
from src.history.models import AIDecision, PortfolioSnapshot
from src.monitoring.metrics import DailyMetricsJob
from src.monitoring.telegram import TelegramAlerter
from src.news.alpaca_news import AlpacaNewsStream
from src.news.filter import NewsFilter
from src.news.rss_feed import RSSFeedAggregator
from src.news.sentiment import passes_threshold, score_sentiment
from src.trading.alpaca_client import AlpacaWrapper
from src.trading.executor import TradeExecutor, _is_crypto
from src.trading.market_data import MarketDataCache
from src.trading.risk import PROFILES, RiskEngine
import pandas_market_calendars as mcal

log = structlog.get_logger(__name__)


def _read_version() -> str:
    try:
        return (Path(__file__).parent.parent / "VERSION").read_text().strip()
    except Exception:
        return "unknown"


# ── Logging setup ──────────────────────────────────────────────────────────────

class _PipeRenderer:
    """Formats log lines as:  YYYY-MM-DD HH:MM:SS | LEVEL    | event  key=val ..."""

    _RESET  = "\033[0m"
    _GREEN  = "\033[32m"
    _YELLOW = "\033[33m"
    _RED    = "\033[31m"
    _CYAN   = "\033[36m"
    _BOLD   = "\033[1m"

    _LEVEL_COLOR = {
        "debug":    "\033[36m",   # cyan
        "info":     "\033[32m",   # green
        "warning":  "\033[33m",   # yellow
        "error":    "\033[31m",   # red
        "critical": "\033[35m",   # magenta
    }

    def __call__(self, logger, method, event_dict: dict) -> str:
        ts    = event_dict.pop("timestamp", "")
        level = event_dict.pop("level", method).upper()
        event_dict.pop("logger", None)
        event_dict.pop("_record", None)
        event = event_dict.pop("event", "")

        # Build key=val pairs (skip internal structlog keys)
        kv = "  ".join(
            f"{k}={v}"
            for k, v in event_dict.items()
            if not k.startswith("_")
        )
        message = f"{event}  {kv}" if kv else event

        lc = self._LEVEL_COLOR.get(level.lower(), "")
        return (
            f"{self._GREEN}{ts}{self._RESET}"
            f" | {lc}{self._BOLD}{level:<8}{self._RESET}"
            f" | {message}"
        )


def _setup_logging(log_dir: Path) -> None:
    import logging
    log_dir.mkdir(parents=True, exist_ok=True)
    # Configure stdlib root logger so structlog.stdlib.LoggerFactory has a handler.
    # Root at DEBUG — explicitly silence noisy third-party libs.
    logging.basicConfig(
        format="%(message)s",
        stream=__import__("sys").stdout,
        level=logging.DEBUG,
    )
    for noisy in ("aiosqlite", "urllib3", "websockets", "asyncio", "hpack",
                  "alpaca", "httpcore", "httpx"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    logging.getLogger("src").setLevel(logging.DEBUG)
    structlog.configure(
        processors=[
            structlog.stdlib.add_log_level,
            structlog.stdlib.add_logger_name,
            structlog.processors.TimeStamper(fmt="%Y-%m-%d %H:%M:%S", utc=True),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            _PipeRenderer(),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(logging.DEBUG),
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
    )


# ── Market hours ───────────────────────────────────────────────────────────────

_nyse = mcal.get_calendar("NYSE")


def _market_is_open() -> bool:
    now = datetime.now(tz=timezone.utc)
    schedule = _nyse.schedule(
        start_date=now.strftime("%Y-%m-%d"),
        end_date=now.strftime("%Y-%m-%d"),
    )
    if schedule.empty:
        return False
    open_t = schedule.iloc[0]["market_open"].to_pydatetime()
    close_t = schedule.iloc[0]["market_close"].to_pydatetime()
    return open_t <= now <= close_t


# ── Watchlist loading ──────────────────────────────────────────────────────────

def _load_watchlist(config_dir: Path) -> tuple[list[str], list[str], dict]:
    path = config_dir / "watchlist.yml"
    with open(path) as f:
        data = yaml.safe_load(f)
    equities = data.get("equities") or []
    crypto = data.get("crypto") or []
    macro = data.get("macro_mappings") or {}
    return equities, crypto, macro


def _load_trading_params(config_dir: Path) -> dict:
    path = config_dir / "trading_params.yml"
    with open(path) as f:
        return yaml.safe_load(f)


# ── Config hot-reload ──────────────────────────────────────────────────────────

class HotReloadWatcher:
    """Polls trading_params.yml every 5s and atomically swaps the risk engine profile.

    Also refreshes the shared ``params`` dict so that operational settings
    (finbert_threshold, batch_size, etc.) take effect without a restart.
    """

    def __init__(
        self,
        config_dir: Path,
        risk_engine: RiskEngine,
        alert_fn,
        params: dict,
    ) -> None:
        self._path = config_dir / "trading_params.yml"
        self._risk_engine = risk_engine
        self._alert = alert_fn
        self._params = params
        self._last_mtime: float = 0.0

    async def run(self) -> None:
        while True:
            await asyncio.sleep(5)
            try:
                mtime = self._path.stat().st_mtime
                if mtime == self._last_mtime:
                    continue
                self._last_mtime = mtime
                await self._reload()
            except Exception as exc:  # noqa: BLE001
                log.warning("hot_reload_stat_error", error=str(exc))

    async def _reload(self) -> None:
        try:
            with open(self._path) as f:
                new_params = yaml.safe_load(f)
            new_profile = new_params.get("risk_profile", "conservative")
            if new_profile not in PROFILES:
                raise ValueError(f"Unknown profile: {new_profile!r}")
            old_profile = self._risk_engine.profile.name
            if new_profile != old_profile:
                self._risk_engine.set_profile(new_profile)
                log.info("hot_reload_profile_changed", old=old_profile, new=new_profile)
                if self._alert:
                    await self._alert(
                        f"⚙️ Risk profile changed: {old_profile} → {new_profile}",
                        priority=False,
                    )
            # Refresh shared params dict so process_news_loop picks up new values
            self._params.clear()
            self._params.update(new_params)
            log.debug("hot_reload_params_refreshed")
        except Exception as exc:  # noqa: BLE001
            log.error("hot_reload_failed", error=str(exc))
            if self._alert:
                await self._alert(
                    f"⚠️ Config reload failed — continuing with previous config\n{exc}",
                    priority=True,
                )


# ── Portfolio snapshot ─────────────────────────────────────────────────────────

async def _snap_portfolio(
    alpaca: AlpacaWrapper,
    conn,
    prev_equity: float,
) -> PortfolioSnapshot | None:
    try:
        account = await alpaca.get_account()
        equity = float(account.equity or 0)
        cash = float(account.cash or 0)
        positions_raw = await alpaca.get_all_positions()
        positions = [
            {
                "ticker": str(p.symbol),
                "qty": float(p.qty or 0),
                "market_value": float(p.market_value or 0),
                "unrealized_pnl": float(p.unrealized_pl or 0),
            }
            for p in positions_raw
        ]
        daily_pnl = equity - prev_equity if prev_equity > 0 else 0.0
        daily_pnl_pct = (daily_pnl / prev_equity * 100) if prev_equity > 0 else 0.0
        snap = PortfolioSnapshot(
            equity=equity,
            cash=cash,
            positions=positions,
            daily_pnl=daily_pnl,
            daily_pnl_pct=daily_pnl_pct,
        )
        await db_ops.insert_portfolio_snapshot(conn, snap)
        return snap
    except Exception as exc:  # noqa: BLE001
        log.warning("portfolio_snap_failed", error=str(exc))
        return None


# ── Main processing loop ───────────────────────────────────────────────────────

async def process_news_loop(
    news_queue: asyncio.Queue,
    conn,
    news_filter: NewsFilter,
    ollama: OllamaClient,
    risk_engine: RiskEngine,
    executor: TradeExecutor,
    market_data: MarketDataCache,
    alpaca: AlpacaWrapper,
    settings: Settings,
    params: dict,
) -> None:
    """
    Consumes news_queue, filters, batches, calls Ollama, executes trades.
    """
    pending: list = []
    prev_equity = 0.0

    while True:
        # Drain up to batch_size items from queue (non-blocking after first)
        batch_size = (params.get("news") or {}).get("batch_size", 5)
        try:
            item = await asyncio.wait_for(news_queue.get(), timeout=2.0)
            pending.append(item)
        except asyncio.TimeoutError:
            pass

        # Drain any additional items already queued
        while not news_queue.empty() and len(pending) < batch_size:
            try:
                pending.append(news_queue.get_nowait())
            except asyncio.QueueEmpty:
                break

        if not pending:
            continue

        # Filter + FinBERT
        filtered = []
        finbert_threshold = (params.get("news") or {}).get("finbert_threshold", 0.3)
        for raw_item in pending:
            fres = news_filter.check(raw_item)
            if not fres.passed:
                continue
            raw_item.relevance = fres.relevance

            # FinBERT sentiment pre-score
            text = f"{raw_item.headline}. {raw_item.summary or ''}"
            sentiment = await score_sentiment(text)
            raw_item.sentiment = sentiment

            if not passes_threshold(sentiment, finbert_threshold):
                log.debug("news_filtered_neutral", headline=raw_item.headline[:60], sentiment=sentiment)
                continue

            filtered.append(raw_item)

        pending = []

        if not filtered:
            continue

        # Insert news items into DB
        news_ids = []
        for item in filtered:
            nid = await db_ops.insert_news_item(conn, item)
            if nid:
                news_ids.append(nid)

        # Market hours gate
        equity_items = [i for i in filtered if not _is_crypto(str(i.symbols[0] if i.symbols else ""))]
        crypto_items = [i for i in filtered if _is_crypto(str(i.symbols[0] if i.symbols else ""))]

        market_open = _market_is_open()
        batch_to_process = list(crypto_items)
        if market_open:
            batch_to_process.extend(equity_items)
        else:
            if equity_items:
                log.debug(
                    "equity_signals_queued_market_closed",
                    count=len(equity_items),
                )
            # Only crypto goes through when market is closed

        if not batch_to_process:
            continue

        # Build AI context
        ctx = await build_context(conn)
        snap = await db_ops.get_latest_snapshot(conn)
        if snap:
            prev_equity = snap.equity

        # Build prompt and call Ollama
        profile = risk_engine.profile
        user_prompt = build_context_prompt(
            news_batch=batch_to_process,
            ctx=ctx,
            risk_profile_name=profile.name,
            max_position_pct=profile.max_position_pct * 100,
            max_daily_loss_pct=profile.max_daily_loss_pct * 100,
            min_confidence=profile.min_confidence,
        )

        first_item = batch_to_process[0]
        headline = first_item.headline
        ticker_hint = str(first_item.symbols[0]) if first_item.symbols else ""

        parsed = await ollama.decide(
            user_prompt=user_prompt,
            news_ticker=ticker_hint,
            news_headline=headline,
        )

        # Persist AI decision
        daily_pnl_pct = snap.daily_pnl_pct if snap else 0.0
        decision = AIDecision(
            news_item_ids=news_ids,
            model=settings.ollama_decision_model,
            action=parsed.action,
            ticker=parsed.ticker,
            confidence=parsed.confidence,
            position_pct=parsed.position_size_pct,
            stop_loss_pct=parsed.stop_loss_pct,
            take_profit_pct=parsed.take_profit_pct,
            hold_period=parsed.hold_period,
            reasoning=parsed.reasoning,
            risk_factors=parsed.risk_factors,
            literature_ref=parsed.literature_basis,
            risk_profile=profile.name,
        )
        decision_id = await insert_ai_decision(conn, decision)
        await db_ops.mark_news_forwarded(conn, news_ids)

        # Skip holds and validation failures
        if parsed.action == "hold" or parsed.validation_failed:
            continue

        if ollama.is_safe_mode:
            log.warning("ollama_safe_mode_skip_trade")
            continue

        # Risk engine
        ticker = parsed.ticker
        if not ticker:
            continue

        atr = market_data.get_atr(ticker)
        entry_price = market_data.get_latest_close(ticker)
        if not entry_price:
            log.warning("no_entry_price", ticker=ticker)
            continue

        equity = snap.equity if snap else 0.0
        open_positions = len(snap.positions if snap else [])

        sizing = await risk_engine.compute_sizing(
            conn=conn,
            equity=equity,
            entry_price=entry_price,
            atr=atr,
            ai_confidence=parsed.confidence,
            ai_position_pct=parsed.position_size_pct,
            open_positions_count=open_positions,
            daily_loss_pct=daily_pnl_pct / 100.0,
        )

        # Execute
        await executor.execute(
            decision_id=decision_id,
            ticker=ticker,
            side=parsed.action,
            sizing=sizing,
            risk_profile=profile.name,
        )


# ── Health check endpoint ──────────────────────────────────────────────────────

def _client_ip(request) -> str:
    """
    Resolve the real client IP, honouring proxy headers.
    Checks X-Real-IP then X-Forwarded-For first (set by nginx/Caddy/Traefik),
    falling back to the raw TCP connection address.
    """
    real_ip = request.headers.get("X-Real-IP", "").strip()
    if real_ip:
        return real_ip
    forwarded_for = request.headers.get("X-Forwarded-For", "").strip()
    if forwarded_for:
        return forwarded_for.split(",")[0].strip()
    return request.client.host if request.client else ""


def _build_app(state: dict, settings: Settings):
    from fastapi import FastAPI, Request, Response
    app = FastAPI(title="Alpaca Bot Health")

    allowed = settings.allowed_ip_list
    if not allowed:
        log.warning(
            "dashboard_no_ip_allowlist",
            message="ALLOWED_IPS is empty — dashboard is accessible from any host",
        )

    @app.middleware("http")
    async def ip_filter(request: Request, call_next):
        if allowed:
            if _client_ip(request) not in allowed:
                return Response(status_code=403, content="Forbidden")
        return await call_next(request)

    @app.get("/health")
    async def health():
        return {
            "status": "ok",
            "db": state.get("db_ok", False),
            "ws_connected": state.get("ws_connected", False),
            "circuit_breaker": state.get("circuit_state", "unknown"),
        }

    @app.get("/status")
    async def status():
        return state

    @app.post("/risk-profile/{profile}")
    async def set_risk_profile(profile: str, request: Request):
        # IP check already enforced by middleware; this is belt-and-suspenders
        if profile not in PROFILES:
            return Response(status_code=400, content=f"Unknown profile: {profile}")
        # Write to trading_params.yml — hot-reload watcher will pick it up
        params_path = settings.config_dir / "trading_params.yml"
        with open(params_path) as f:
            data = yaml.safe_load(f)
        data["risk_profile"] = profile
        with open(params_path, "w") as f:
            yaml.safe_dump(data, f)
        return {"ok": True, "profile": profile}

    return app


# ── Entry point ────────────────────────────────────────────────────────────────

async def main() -> None:
    settings = get_settings()
    _setup_logging(settings.log_dir)

    # ── Startup config banner ──────────────────────────────────────────────────
    telegram_enabled = bool(settings.telegram_bot_token and settings.telegram_chat_id)
    log.info(
        "bot_starting",
        version=_read_version(),
        trading_mode=settings.trading_mode,
        risk_profile=settings.risk_profile,
        ollama_host=settings.ollama_host,
        ollama_model=settings.ollama_decision_model,
        two_stage_llm=settings.two_stage_llm,
        telegram=telegram_enabled,
        dashboard_port=settings.dashboard_port,
        allowed_ips=settings.allowed_ips or "(any)",
    )

    # ── Paper→live gate ────────────────────────────────────────────────────────
    if settings.trading_mode == "live":
        gate_file = settings.live_mode_confirmed_path
        if not gate_file.exists():
            log.error(
                "live_mode_gate_failed",
                message=(
                    f"TRADING_MODE=live requires a confirmation file. "
                    f"Run: touch {gate_file}"
                ),
            )
            sys.exit(1)
        log.warning("live_mode_active", gate_file=str(gate_file))

    # ── DB ─────────────────────────────────────────────────────────────────────
    db_path = settings.data_dir / "trading.db"
    conn = await init_db(db_path)
    state = {"db_ok": True, "ws_connected": False, "circuit_state": "closed"}

    # ── Telegram ───────────────────────────────────────────────────────────────
    telegram = TelegramAlerter(settings.telegram_bot_token, settings.telegram_chat_id)

    async def alert(message: str, priority: bool = False) -> None:
        await telegram.send(message, priority=priority)

    # ── Alpaca client ──────────────────────────────────────────────────────────
    alpaca = AlpacaWrapper(settings)

    # ── Watchlist ──────────────────────────────────────────────────────────────
    equities, crypto, macro_map = _load_watchlist(settings.config_dir)
    all_symbols = equities + crypto
    log.info(
        "watchlist_loaded",
        equities=len(equities),
        crypto=len(crypto),
        total=len(all_symbols),
        tickers=", ".join(all_symbols[:8]) + ("..." if len(all_symbols) > 8 else ""),
    )

    # ── Market data cache ──────────────────────────────────────────────────────
    market_data = MarketDataCache(alpaca, equities)

    # ── Risk engine ────────────────────────────────────────────────────────────
    risk_engine = RiskEngine(settings.risk_profile)

    # ── Ollama client ──────────────────────────────────────────────────────────
    ollama = OllamaClient(settings.ollama_host, settings.ollama_decision_model)
    ollama.set_alert_callback(alert)
    ollama_ok = await ollama.health_check()
    log.info("ollama_ready" if ollama_ok else "ollama_unreachable",
             host=settings.ollama_host, model=settings.ollama_decision_model,
             status="ok" if ollama_ok else "CIRCUIT WILL OPEN — bot trades safely in hold mode")

    # ── Executor ───────────────────────────────────────────────────────────────
    executor = TradeExecutor(alpaca, conn, alert)

    # ── Startup reconciliation ─────────────────────────────────────────────────
    await executor.reconcile_open_orders()

    # ── Seed opening equity for day-1 daily loss calculation ───────────────────
    # If no snapshot exists yet, take one now so prev_equity is never 0.
    existing_snap = await db_ops.get_latest_snapshot(conn)
    if not existing_snap:
        await _snap_portfolio(alpaca, conn, prev_equity=0.0)
        log.info("startup_equity_snapshot_taken")

    # ── Queues ─────────────────────────────────────────────────────────────────
    news_queue: asyncio.Queue = asyncio.Queue(maxsize=200)

    # ── News sources ───────────────────────────────────────────────────────────
    news_stream = AlpacaNewsStream(settings, news_queue, all_symbols)
    rss_aggregator = RSSFeedAggregator(news_queue)
    news_filter = NewsFilter(all_symbols)

    # ── Config hot-reload ──────────────────────────────────────────────────────
    params = _load_trading_params(settings.config_dir)
    hot_reload = HotReloadWatcher(settings.config_dir, risk_engine, alert, params)

    # ── Health check app ───────────────────────────────────────────────────────
    app = _build_app(state, settings)

    # ── Send startup alert ─────────────────────────────────────────────────────
    await alert(
        f"🚀 Alpaca Bot started — mode: {settings.trading_mode}, "
        f"profile: {settings.risk_profile}",
        priority=True,
    )

    # ── Launch all background tasks ────────────────────────────────────────────
    import uvicorn

    async def run_uvicorn():
        config = uvicorn.Config(
            app,
            host="0.0.0.0",
            port=settings.dashboard_port,
            log_level="warning",
        )
        server = uvicorn.Server(config)
        await server.serve()

    await asyncio.gather(
        # Health check / dashboard
        run_uvicorn(),
        # Market data refresh
        market_data.run(),
        # Config hot-reload
        hot_reload.run(),
        # News sources
        news_stream.run(),
        rss_aggregator.run(),
        # Ollama circuit breaker recovery
        ollama.run_recovery_loop(),
        # Telegram queue drainer
        telegram.run(),
        # Daily metrics
        DailyMetricsJob(conn, alert).run(),
        # Main processing loop
        process_news_loop(
            news_queue=news_queue,
            conn=conn,
            news_filter=news_filter,
            ollama=ollama,
            risk_engine=risk_engine,
            executor=executor,
            market_data=market_data,
            alpaca=alpaca,
            settings=settings,
            params=params,
        ),
    )
    # Note: return_exceptions intentionally removed — any task crash propagates
    # to run(), which logs the error and exits. A process supervisor (systemd,
    # docker restart=always) should restart the bot. Swallowing crashes here
    # would leave the bot running with silent broken state.


def run() -> None:
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        log.info("bot_shutdown_requested")
    except Exception as exc:
        log.error("bot_crashed", error=str(exc))
        sys.exit(1)


if __name__ == "__main__":
    run()
