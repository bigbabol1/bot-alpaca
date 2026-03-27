# Changelog

All notable changes to this project will be documented in this file.

Format: [Semantic versioning](https://semver.org) — MAJOR.MINOR.PATCH.MICRO

## [0.1.0.0] - 2026-03-27

### Added
- Phase 1 trading bot: full async pipeline from news ingestion to order execution
  - Alpaca News WebSocket feed with real-time ticker-tagged articles (`src/news/alpaca_news.py`)
  - RSS supplement feed (Reuters, AP, MarketWatch) for Alpaca WS fallback (`src/news/rss_feed.py`)
  - News filter pipeline: dedup, source quality scoring, relevance scoring, recency gate, batch assembly (`src/news/filter.py`)
  - FinBERT sentiment pre-scoring to prioritize high-signal items (`src/news/sentiment.py`)
  - Ollama client with circuit breaker, 3-level retry (full prompt → stripped prompt → hold), and 5-minute recovery loop (`src/ai/ollama_client.py`)
  - LLM prompt engine with context injection and `_sanitize()` to strip prompt injection vectors (`src/ai/prompts.py`)
  - JSON decision parser with validation of all required fields and `validation_failed` error codes (`src/ai/decision.py`)
  - Risk engine with 3 profiles (conservative/moderate/aggressive), Kelly Criterion sizing (activates after 50 trades), ATR stops, daily loss limit, and Kelly guard (`src/trading/risk.py`)
  - Trade executor for equity (bracket orders) and crypto (market + manual stop-limit), with async fill polling (`src/trading/executor.py`)
  - Market data cache: background refresh every 15 min, ATR(14) pre-computed for all watchlist tickers (`src/trading/market_data.py`)
  - SQLite persistence with WAL mode, write-lock serialization, and full CRUD for news, decisions, trades, and portfolio snapshots (`src/history/db.py`, `src/history/models.py`)
  - Portfolio context window builder for LLM prompt injection (`src/history/context.py`)
  - Daily metrics job: Sharpe, max drawdown, win rate logged at 16:05 ET (`src/monitoring/metrics.py`)
  - Telegram notifier for trade alerts and daily P&L (`src/monitoring/telegram.py`)
  - FastAPI health dashboard: `/health`, `/status`, `/risk-profile/{name}` with IP allowlist (`src/main.py`)
  - Config hot-reload watcher: trades pause during reload, resume with new params (`src/main.py`)
  - Curated watchlist: Mag7, sector ETFs, crypto (`config/watchlist.yml`)
  - Docker + docker-compose with FinBERT model baked into image (`Dockerfile`, `docker-compose.yml`)

### Fixed (review pass — 21 issues)
- `asyncio.Lock` serialization for all DB writes — prevents interleaved commits when multiple async tasks write concurrently
- Reconciliation correctly identifies filled orders via `get_order(id)` — the old `get_open_orders()` call was marking filled trades as still open
- Kelly `b` parameter now derived from actual trade history (mean winner / mean loser) — was accidentally using `ai_position_pct` (≈0.01), producing near-zero position sizes
- IP allowlist reads `X-Real-IP` / `X-Forwarded-For` headers — `request.client.host` returns the proxy IP behind nginx, not the client IP
- Day-1 equity seed taken at startup so `prev_equity` is never `0.0` on the first trading day
- Circuit breaker recovery gated on `not validation_failed` — the old condition (`not failed OR action=="hold"`) closed the circuit on LLM parse failures, not just genuine holds
- Daily loss halt fires only on losses (`pct <= -limit`) — the old `abs(pct) >= limit` halted trading on big winning days too
- Kelly guard date-gated to once per calendar day — was incrementing on every trade call, not just the first of the day
- `update_trade` now persists `entry_price` — was silently omitted from the `UPDATE SET` clause since day one
- Crypto fail-safe exit side computed from `trade.side` — was hardcoded `"sell"`, which is wrong for short positions
- Crypto first fill poll: 1-second delay — market orders typically fill in under a second; the old 10-second wait was just unnecessary latency
- `_sanitize()` applied to `symbols`, `positions_json`, `decisions_json` in LLM context — prompt injection via ticker names or position data
- All model datetimes use `datetime.now(timezone.utc)`; `_parse_dt()` helper ensures tz-aware on DB read
- `mark_news_forwarded` guarded against empty list — an empty `IN ()` clause is invalid SQL and would crash the batch
- `decision_id=None` for reconciled orphan orders — `0` violated the FK constraint when `PRAGMA foreign_keys=ON`

### Tests
- 115 tests across 11 test files covering: decision parsing, prompt sanitization, DB operations, risk engine, Kelly criterion, market data cache, portfolio context, configuration, news filtering, and retry logic
