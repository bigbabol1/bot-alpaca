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
- `asyncio.Lock` serialization for all DB writes — prevents interleaved commits under concurrent async tasks
- Reconciliation correctly identifies filled orders via `get_order(id)` — was wrongly marking filled trades as closed
- Kelly `b` parameter derived from actual trade history (mean winner / mean loser) — was receiving `ai_position_pct` fraction (≈0.01), producing near-zero sizing
- IP allowlist reads `X-Real-IP` / `X-Forwarded-For` headers — was using `request.client.host` (proxy IP) behind nginx
- Day-1 equity seed taken at startup so `prev_equity` is never 0.0 on first day
- Circuit breaker recovery gated on `not validation_failed` — was `not failed OR action=="hold"`, closing circuit on parse failures
- Daily loss halt fires only on losses (`pct <= -limit`) — was `abs(pct) >= limit`, halting on winning days too
- Kelly guard date-gated to once per calendar day — was incrementing per trade call
- `update_trade` now persists `entry_price` — was silently omitted from the UPDATE SET clause
- Crypto fail-safe exit side computed from `trade.side` — was hardcoded `"sell"` for all positions
- Crypto first fill poll: 1-second delay (market orders fill in <1s) — was 10 seconds
- `_sanitize()` applied to `symbols`, `positions_json`, `decisions_json` in LLM context
- All model datetimes use `datetime.now(timezone.utc)` + `_parse_dt()` helper ensures tz-aware on DB read
- `mark_news_forwarded` guarded against empty list (would produce invalid `IN ()` SQL)
- `decision_id=None` for reconciled orphan orders — was `0`, violating FK constraint with `foreign_keys=ON`

### Tests
- 115 tests across 11 test files covering: decision parsing, prompt sanitization, DB operations, risk engine, Kelly criterion, market data cache, portfolio context, configuration, news filtering, and retry logic
