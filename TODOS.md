# TODOS — Alpaca Trading Bot

Deferred items from CEO review (2026-03-26) and Eng review (2026-03-27).
These are explicitly out of scope for v1. Do not implement until Phase 1 is live and validated.

## Post-v1 (after first 3 months live)

- [ ] **Adaptive prompt refinement** — feed trade outcomes back into system prompt; self-improvement loop based on which news sources/event types the AI mis-called
- [ ] **Sector-rotation awareness** — macro regime detection (e.g. risk-on vs risk-off based on VIX + yield curve)
- [ ] **VIX-based position sizing** — reduce per-trade size when VIX > 30; requires VIX data feed
- [ ] **Multi-account support** — separate paper and live accounts managed in parallel
- [ ] **Auto-watchlist additions** — currently manual quarterly review for adds; auto-add via liquidity screen
- [ ] **API-hosted model fallback** — when local Ollama p95 > target sustained 30min, route to a cloud API (OpenAI/Anthropic) as fallback
- [ ] **Two-stage LLM pipeline activation** — `TWO_STAGE_LLM=true` only becomes meaningful when a second GPU (≥8GB VRAM) is available on a separate Ollama host; see EUREKA note in design doc

## Phase 2 (after Phase 1 core is stable)

- [ ] RSS feed backup (Reuters, AP, MarketWatch) — supplement/fallback when Alpaca WS is down
- [ ] Daily performance metrics job — Sharpe, max drawdown, win rate, avg hold — logged to `performance_metrics` table at 16:05 ET
- [ ] Position monitoring loop — poll open positions every 30s for crypto stop-loss; equity uses server-side bracket stops
- [ ] Health check endpoint — `GET /health` returns 200 when DB + Ollama + Alpaca WS are all connected

## Phase 3 (intelligence)

- [ ] Backtesting harness (`src/backtest/`) — Alpaca historical news replay; verify ≥2yr lookback available on free tier before building
- [ ] Live → paper flip switch — environment toggle without restart
- [ ] Telegram alert on major trades — already in scope, track completion here
