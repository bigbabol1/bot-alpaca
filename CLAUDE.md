# Project: Alpaca Trading Bot

See design doc: `~/.gstack/projects/bigbabol1-bot-alpaca/2026-03-25-design-trading-bot.md`
CEO plan: `~/.gstack/projects/bigbabol1-bot-alpaca/ceo-plans/2026-03-26-trading-bot.md`

## Architecture

Async pipeline: News ingestion → FinBERT sentiment → Ollama LLM decision → Risk engine → Trade executor → SQLite persistence.

Key modules:
- `src/main.py` — asyncio entry point, FastAPI dashboard, hot-reload watcher
- `src/ai/ollama_client.py` — circuit breaker, 3-level retry, recovery loop
- `src/trading/risk.py` — Kelly sizing (activates at 50 closed trades), ATR stops, daily loss limit
- `src/trading/executor.py` — equity (bracket) and crypto (market + manual stop) execution
- `src/history/db.py` — SQLite WAL, asyncio.Lock write serialization

## Testing

Run: `python -m pytest tests/ -v`

Framework: pytest + pytest-asyncio (`asyncio_mode = auto`). Test directory: `tests/` — mirrors `src/`.

**Expectations:**
- 100% coverage of all testable-without-external-deps code
- Write a test for every new function with conditional logic
- Write a regression test for every bug fix
- External API wrappers (Alpaca, Ollama, Telegram) require `unittest.mock.MagicMock`
- Never commit code that makes existing tests fail

## Test Coverage Minimum

Minimum: 60%
Target: 80%

# gstack

Use the `/browse` skill from gstack for all web browsing. Never use `mcp__claude-in-chrome__*` tools.

Available gstack skills:
- `/office-hours`
- `/plan-ceo-review`
- `/plan-eng-review`
- `/plan-design-review`
- `/design-consultation`
- `/review`
- `/ship`
- `/land-and-deploy`
- `/canary`
- `/benchmark`
- `/browse`
- `/qa`
- `/qa-only`
- `/design-review`
- `/setup-browser-cookies`
- `/setup-deploy`
- `/retro`
- `/investigate`
- `/document-release`
- `/codex`
- `/cso`
- `/autoplan`
- `/careful`
- `/freeze`
- `/guard`
- `/unfreeze`
- `/gstack-upgrade`
