# Alpaca Trading Bot

An async news-to-trade pipeline. Reads Alpaca and RSS news feeds, scores sentiment with FinBERT, runs each article through a local Ollama LLM for a buy/sell/hold decision, sizes the position with a risk engine, and submits the order to Alpaca — all without human intervention.

Built to run 24/7 on a NAS box with a GPU machine on the LAN for Ollama.

## What it does

1. **News ingestion** — Alpaca WebSocket (primary) + Reuters/AP/MarketWatch RSS (fallback)
2. **Sentiment pre-filter** — FinBERT scores each item; weak-signal items are dropped before hitting the LLM
3. **LLM decision** — Ollama (`qwen2.5:32b` by default) returns a structured JSON decision: action, ticker, confidence, stop/take-profit pcts, reasoning
4. **Risk engine** — Kelly Criterion sizing (activates after 50 closed trades), ATR-based stops, daily loss limit, 3 risk profiles
5. **Trade execution** — bracket orders for equities, market + manual stop-limit for crypto
6. **Persistence** — SQLite (WAL mode) for news items, decisions, trades, portfolio snapshots
7. **Dashboard** — FastAPI at `localhost:8080` for health checks and live status
8. **Alerts** — Telegram notifications on fills and daily P&L summary

## Prerequisites

- Python 3.11+
- [Alpaca](https://alpaca.markets) account (paper or live)
- [Ollama](https://ollama.ai) running on LAN with `qwen2.5:32b` pulled
- Docker + docker-compose (for production deployment)
- (Optional) Telegram bot token for alerts

## Quick start

```bash
# 1. Copy env template and fill in your keys
cp .env.example .env
# Edit .env with your Alpaca keys, Ollama host, Telegram token

# 2. Run with Docker (recommended)
docker-compose up -d

# 3. Check the dashboard
curl http://localhost:8080/health
```

## Configuration

All config lives in `.env`. Key settings:

| Variable | Default | Description |
|---|---|---|
| `ALPACA_API_KEY` | — | Alpaca API key |
| `ALPACA_SECRET_KEY` | — | Alpaca secret key |
| `TRADING_MODE` | `paper` | `paper` or `live` |
| `OLLAMA_HOST` | `http://192.168.0.66:11434` | Ollama base URL |
| `OLLAMA_DECISION_MODEL` | `qwen2.5:32b` | LLM for trade decisions |
| `RISK_PROFILE` | `conservative` | `conservative`, `moderate`, or `aggressive` |
| `TELEGRAM_BOT_TOKEN` | — | Optional — enables trade alerts |
| `TELEGRAM_CHAT_ID` | — | Telegram chat ID for alerts |
| `DASHBOARD_PORT` | `8080` | FastAPI dashboard port |
| `ALLOWED_IPS` | (any LAN IP) | Comma-separated IP allowlist for dashboard |

### Going live

The bot defaults to paper trading. To switch to live:

1. Set `TRADING_MODE=live` in `.env`
2. Create the gate file: `touch data/.live_mode_confirmed`

Both conditions must be true at startup. The gate file is a deliberate manual step — no typo in `.env` can accidentally trigger live trading.

## Risk profiles

| Profile | Daily loss limit | Position cap | Kelly |
|---|---|---|---|
| `conservative` | 1% | 1% | 1% max |
| `moderate` | 2% | 2% | 2% max |
| `aggressive` | 10% | 10% | 10% max |

Kelly Criterion activates after 50 closed trades (enough history to estimate win rate). Before that, a fixed fraction is used. You can change the profile at runtime:

```bash
curl -X POST http://localhost:8080/risk-profile/moderate
```

## Dashboard

| Endpoint | Description |
|---|---|
| `GET /health` | Returns 200 when DB + Ollama + Alpaca WS are connected |
| `GET /status` | Current equity, open positions, today's P&L, risk profile |
| `POST /risk-profile/{name}` | Switch risk profile without restart |

## Watchlist

Curated in `config/watchlist.yml`: Mag7, sector ETFs (SPY, QQQ, XLF, XLE, XLK, GLD, TLT), and top S&P names by volume. Crypto (BTC/USD, ETH/USD, SOL/USD) can be added. Manual quarterly review only — no auto-adds in v1.

## Architecture

```
Alpaca WS / RSS feeds
        ↓
  News filter (dedup, relevance, recency)
        ↓
  FinBERT sentiment (ThreadPoolExecutor)
        ↓
  Ollama LLM (circuit breaker, 3-level retry)
        ↓
  Risk engine (Kelly, ATR stops, daily limit)
        ↓
  Trade executor → Alpaca orders
        ↓
  SQLite (WAL, asyncio.Lock serialization)
```

See [CLAUDE.md](CLAUDE.md) for module-by-module breakdown and [CHANGELOG.md](CHANGELOG.md) for full history.

## Development

```bash
# Install test dependencies (excludes torch/transformers — mocked in tests)
pip install -r requirements-ci.txt

# Run tests
python -m pytest tests/ -v

# Run with live reload (for local dev, not Docker)
python -m src.main
```

115 tests covering decision parsing, risk engine, DB operations, prompt sanitization, news filtering, and more. External API wrappers (Alpaca, Ollama, Telegram) use `unittest.mock.MagicMock`.

CI runs on every push and PR via GitHub Actions (`.github/workflows/ci.yml`).

## Project structure

```
src/
  main.py              asyncio entry point, FastAPI dashboard, hot-reload
  config.py            pydantic-settings config (reads .env)
  ai/
    ollama_client.py   circuit breaker, 3-level retry, recovery loop
    prompts.py         context injection, _sanitize() for prompt injection
    decision.py        JSON parser + validator for LLM responses
  news/
    alpaca_news.py     Alpaca WebSocket news feed
    rss_feed.py        RSS supplement (Reuters, AP, MarketWatch)
    filter.py          dedup, relevance scoring, recency gate, batching
    sentiment.py       FinBERT pre-scoring (ThreadPoolExecutor)
  trading/
    risk.py            Kelly sizing, ATR stops, daily loss limit
    executor.py        bracket orders (equity), market + stop-limit (crypto)
    market_data.py     ATR(14) cache, 15-min background refresh
    alpaca_client.py   Alpaca REST + WS wrapper
  history/
    db.py              SQLite CRUD, WAL, write-lock serialization
    models.py          dataclasses for NewsItem, AIDecision, Trade, Portfolio
    context.py         portfolio context builder for LLM prompt
  monitoring/
    metrics.py         daily Sharpe, drawdown, win rate (16:05 ET)
    telegram.py        trade alerts, daily P&L summary
config/
  watchlist.yml        curated ticker list
tests/                 mirrors src/ structure
```
