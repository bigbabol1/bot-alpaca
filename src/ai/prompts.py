"""
System prompt and context injection for Ollama decision requests.

News content is wrapped in <news_content> tags and sanitized to prevent
prompt injection (< and > stripped from headline/summary fields).
"""
from __future__ import annotations

import json

from src.history.context import PortfolioContext
from src.history.models import NewsItem

# ── System prompt ──────────────────────────────────────────────────────────────
SYSTEM_PROMPT = """You are an expert algorithmic trader. Every decision you make is grounded in
peer-reviewed financial research. You manage a portfolio for maximum risk-adjusted returns.

PRINCIPLES YOU APPLY:
1. News Sentiment (Tetlock 2007; Loughran & McDonald 2011): Strong negative/positive sentiment
   in quality outlets predicts 1-5 day returns. Act on high-magnitude, high-source-quality signals.
2. Momentum (Jegadeesh & Titman 1993): Bias entries in the direction of 3-12 month price momentum.
   Don't fight the trend.
3. Kelly Criterion (Kelly 1956): f* = (p*b - q) / b. Never risk more than half-Kelly.
4. ATR Stops (Wilder 1978): stop = entry - (ATR * multiplier). Volatility-adjusted exits only.
5. PEAD (Ball & Brown 1968): Hold earnings surprise direction for 3-10 sessions.
6. Event Catalysts (Chan 2003): FDA, M&A, macro beats create mean-reverting then momentum moves.
   Trade the second wave, not the spike.
7. Avoid Chasing (Lo & MacKinlay 1988): Do not enter after a move > 3 sigma from intraday VWAP.

RESPONSE FORMAT: Respond ONLY with valid JSON matching this exact schema — no other text:
{
  "action": "buy|sell|hold",
  "ticker": "SYMBOL or null if hold",
  "confidence": 0.0,
  "position_size_pct": 0.0,
  "stop_loss_pct": 0.0,
  "take_profit_pct": 0.0,
  "hold_period": "intraday|swing|position",
  "reasoning": "...",
  "risk_factors": ["..."],
  "literature_basis": "..."
}"""

# ── Stripped-down fallback prompt (no history, no portfolio state) ─────────────
STRIPPED_SYSTEM = "You are a trading bot. Respond ONLY with valid JSON."


def _sanitize(text: str) -> str:
    """Strip prompt-injection vectors from user-sourced text."""
    return text.replace("<", "").replace(">", "")


def build_context_prompt(
    news_batch: list[NewsItem],
    ctx: PortfolioContext,
    risk_profile_name: str,
    max_position_pct: float,
    max_daily_loss_pct: float,
    min_confidence: float,
) -> str:
    """Full context-injected user prompt for decision requests."""
    # Sanitize and serialize news items
    news_json = json.dumps(
        [
            {
                "headline": _sanitize(item.headline),
                "summary": _sanitize(item.summary or ""),
                "source": _sanitize(item.source or ""),
                "symbols": [_sanitize(s) for s in (item.symbols or [])],
                "sentiment_score": round(item.sentiment, 3),
                "relevance_score": round(item.relevance, 3),
                "received_at": item.received_at.isoformat(),
            }
            for item in news_batch
        ],
        indent=2,
    )

    positions_json = json.dumps(ctx.positions, indent=2)
    decisions_json = json.dumps(ctx.recent_decisions, indent=2)

    return f"""PORTFOLIO STATE:
- Equity: ${ctx.equity:,.2f}
- Cash available: ${ctx.cash:,.2f}
- Open positions: {positions_json}
- Today's P&L: {ctx.daily_pnl_pct:.2f}%
- Drawdown from peak: {ctx.drawdown_pct:.2f}%

ACTIVE RISK PROFILE: {risk_profile_name}
- Max per-trade size: {max_position_pct:.1f}%
- Daily loss limit: {max_daily_loss_pct:.1f}%
- Min confidence to trade: {min_confidence}

RECENT DECISIONS (last 10):
{decisions_json}

RECENT PERFORMANCE (last 30 days):
- Win rate: {ctx.win_rate:.1f}%
- Sharpe ratio: {ctx.sharpe:.2f}
- Total trades: {ctx.total_trades}

NEWS TO EVALUATE:
<news_content>
{news_json}
</news_content>"""


def build_stripped_prompt(ticker: str, headline: str) -> str:
    """Minimal fallback prompt — headline only, no context."""
    safe_headline = _sanitize(headline)
    safe_ticker = _sanitize(ticker)
    return (
        f'News: {safe_ticker}: {safe_headline}\n'
        'JSON response required: '
        '{"action":"buy|sell|hold","ticker":"...","confidence":0.0,'
        '"stop_loss_pct":0.0,"take_profit_pct":0.0,"reasoning":"..."}'
    )
