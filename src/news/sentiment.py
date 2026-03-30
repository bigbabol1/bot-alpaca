"""
FinBERT sentiment pre-scorer.

Runs in a ThreadPoolExecutor so PyTorch inference never blocks the async loop.
Only forwards items to Ollama if abs(sentiment_score) > threshold (default 0.3).

FinBERT labels: positive → +score, negative → -score, neutral → near 0.
"""
from __future__ import annotations

import asyncio
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import structlog

log = structlog.get_logger(__name__)

# Module-level pipeline singleton — loaded once on first use
_pipeline = None
_pipeline_lock = threading.Lock()  # threading.Lock because _run_inference runs in ThreadPoolExecutor
_executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="finbert")


def _load_pipeline():
    """Load FinBERT (blocking — runs in thread pool)."""
    import io
    import logging
    import os
    import sys
    # Suppress HuggingFace Hub unauthenticated-request warnings and
    # transformers weight-mismatch LOAD REPORT printed to stdout.
    os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")
    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
    logging.getLogger("transformers").setLevel(logging.ERROR)
    logging.getLogger("huggingface_hub").setLevel(logging.ERROR)

    from transformers import pipeline as hf_pipeline

    _sink = io.StringIO()
    _old_out, _old_err = sys.stdout, sys.stderr
    sys.stdout = sys.stderr = _sink
    try:
        result = hf_pipeline(
            "sentiment-analysis",
            model="ProsusAI/finbert",
            device=-1,          # CPU — NAS container has no GPU
            truncation=True,
            max_length=512,
        )
    finally:
        sys.stdout, sys.stderr = _old_out, _old_err
    return result


def _run_inference(text: str) -> float:
    """
    Score a single headline/summary. Returns float in [-1.0, 1.0].
    Runs in thread pool (blocking PyTorch call).
    """
    global _pipeline
    if _pipeline is None:
        with _pipeline_lock:
            if _pipeline is None:  # double-checked locking
                _pipeline = _load_pipeline()

    result = _pipeline(text[:512])[0]
    label = result["label"].lower()   # "positive", "negative", "neutral"
    score = float(result["score"])

    if label == "positive":
        return score
    if label == "negative":
        return -score
    return 0.0


async def score_sentiment(text: str) -> float:
    """
    Async wrapper — dispatches to thread pool, returns sentiment score.
    Includes per-call latency logging (finbert_ms).
    """
    loop = asyncio.get_event_loop()
    t0 = time.monotonic()
    try:
        score = await loop.run_in_executor(_executor, _run_inference, text)
    except Exception as exc:  # noqa: BLE001
        log.warning("finbert_error", error=str(exc))
        return 0.0
    finally:
        finbert_ms = round((time.monotonic() - t0) * 1000)
        log.debug("finbert_scored", finbert_ms=finbert_ms)

    return score


def passes_threshold(score: float, threshold: float = 0.3) -> bool:
    """True if the sentiment score is strong enough to forward to Ollama."""
    return abs(score) > threshold
