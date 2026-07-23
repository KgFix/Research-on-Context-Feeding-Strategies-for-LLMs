"""S2 — Query-Intent-Driven Windowing Strategy.

Classifies the query into one of four intent categories via a zero-shot
LLM call, then dispatches to a specialised sub-strategy that selects
contextually appropriate data.

Per strategy_specs.md:
- RECENCY: Last 30 rows for the queried ticker
- TREND:   Last 5 rows + stratified samples at 30/60/90 days back (3 per point)
- ANOMALY: 10 rows before + anomaly row + 5 rows after the largest daily move
- COMPARISON: Last 20 rows per queried ticker, interleaved chronologically
"""

from __future__ import annotations

import logging
import re
from datetime import datetime, timedelta
from typing import Optional

import pandas as pd

from ctxfeed.config import Config
from ctxfeed.formatting.markdown_table import df_to_markdown
from ctxfeed.llm.ollama_client import OllamaClient
from ctxfeed.storage.historical_store import HistoricalStore
from ctxfeed.storage.stream_buffer import StreamBuffer
from ctxfeed.strategies.base import ContextResult, Strategy
from ctxfeed.strategies.intent_classifier import QueryIntent, classify_intent

logger = logging.getLogger(__name__)


class IntentDrivenStrategy(Strategy):
    """Classifies query intent, then applies a matched sub-strategy.

    Args:
        llm_client: Ollama client for the intent classification call.
    """

    def __init__(self, llm_client: OllamaClient) -> None:
        self._client = llm_client

    @property
    def name(self) -> str:
        return "s2_intent"

    def select_context(
        self,
        query: str,
        buffer: StreamBuffer,
        store: HistoricalStore,
        config: Config,
    ) -> ContextResult:
        """Classify the query intent and dispatch to the matching sub-strategy."""
        intent = classify_intent(query, self._client)
        cfg = config.strategies.s2_intent

        if intent == QueryIntent.RECENCY:
            return self._recency(query, buffer, cfg)
        elif intent == QueryIntent.TREND:
            return self._trend(query, buffer, store, cfg)
        elif intent == QueryIntent.ANOMALY:
            return self._anomaly(query, store, cfg)
        elif intent == QueryIntent.COMPARISON:
            return self._comparison(query, buffer, store, cfg, config)
        else:
            # Fallback — should not reach here
            return self._recency(query, buffer, cfg)

    # ── Sub-strategies ───────────────────────────────────────

    def _recency(self, query: str, buffer: StreamBuffer, cfg) -> ContextResult:
        """RECENCY: Last N rows for the queried ticker."""
        ticker = _extract_ticker(query)
        if ticker:
            df = buffer.get_latest_for_ticker(ticker, n=cfg.recency_window)
        else:
            df = buffer.get_latest(n=cfg.recency_window)

        return ContextResult(
            context_text=df_to_markdown(df),
            rows_selected=len(df),
            metadata={"intent": "RECENCY", "ticker": ticker},
        )

    def _trend(
        self, query: str, buffer: StreamBuffer, store: HistoricalStore, cfg
    ) -> ContextResult:
        """TREND: Recent rows + stratified historical samples."""
        ticker = _extract_ticker(query)

        # Recent rows from buffer
        if ticker:
            recent = buffer.get_latest_for_ticker(ticker, n=cfg.trend_recent)
        else:
            recent = buffer.get_latest(n=cfg.trend_recent)

        # Historical samples at lookback points
        samples: list[pd.DataFrame] = []
        if ticker and not recent.empty:
            latest_date = recent["date"].max()
            for days_back in cfg.trend_lookback_days:
                target = _date_offset(latest_date, -days_back)
                sample = store.get_nearest_rows(
                    ticker, target, n=cfg.trend_samples_per_lookback
                )
                samples.append(sample)

        # Combine: recent + historical samples
        parts = [recent] + samples
        combined = pd.concat(parts, ignore_index=True).drop_duplicates(
            subset=["ticker", "date"]
        )
        combined = combined.sort_values("date").reset_index(drop=True)

        # Format with separator between recent and historical
        context_text = df_to_markdown(combined)

        return ContextResult(
            context_text=context_text,
            rows_selected=len(combined),
            metadata={
                "intent": "TREND",
                "ticker": ticker,
                "lookback_days": cfg.trend_lookback_days,
            },
        )

    def _anomaly(
        self, query: str, store: HistoricalStore, cfg
    ) -> ContextResult:
        """ANOMALY: Rows surrounding the largest single-day move."""
        ticker = _extract_ticker(query)

        # Get all data for this ticker
        if ticker:
            df = store.query(ticker=ticker)
        else:
            df = store.query()

        if len(df) < 2:
            return ContextResult(
                context_text=df_to_markdown(df),
                rows_selected=len(df),
                metadata={"intent": "ANOMALY", "ticker": ticker, "anomaly_found": False},
            )

        # Compute daily returns and find the anomaly
        df = df.sort_values("date").reset_index(drop=True)
        df["daily_return"] = df["close"].pct_change()
        anomaly_idx = df["daily_return"].abs().idxmax()

        # Extract surrounding rows
        start = max(0, anomaly_idx - cfg.anomaly_k_before)
        end = min(len(df), anomaly_idx + cfg.anomaly_k_after + 1)
        window = df.iloc[start:end].drop(columns=["daily_return"]).reset_index(drop=True)

        # Calculate the position of the anomaly row within the window for highlighting
        anomaly_pos = anomaly_idx - start
        highlight = {anomaly_pos}

        return ContextResult(
            context_text=df_to_markdown(window, highlight_rows=highlight),
            rows_selected=len(window),
            metadata={
                "intent": "ANOMALY",
                "ticker": ticker,
                "anomaly_date": str(df.iloc[anomaly_idx]["date"]),
                "anomaly_return": float(df.iloc[anomaly_idx]["daily_return"]),
                "anomaly_found": True,
            },
        )

    def _comparison(
        self,
        query: str,
        buffer: StreamBuffer,
        store: HistoricalStore,
        cfg,
        config: Config,
    ) -> ContextResult:
        """COMPARISON: Last N rows per queried ticker, interleaved."""
        tickers = _extract_tickers(query, config.data.tickers)

        frames: list[pd.DataFrame] = []
        for ticker in tickers:
            df = buffer.get_latest_for_ticker(ticker, n=cfg.comparison_per_ticker)
            if df.empty:
                # Fall back to historical store
                df = store.query(ticker=ticker, limit=cfg.comparison_per_ticker)
            frames.append(df)

        if not frames:
            combined = pd.DataFrame()
        else:
            combined = pd.concat(frames, ignore_index=True)
            combined = combined.sort_values("date").reset_index(drop=True)

        return ContextResult(
            context_text=df_to_markdown(combined),
            rows_selected=len(combined),
            metadata={"intent": "COMPARISON", "tickers": tickers},
        )


# ── Helper functions ─────────────────────────────────────────

_KNOWN_TICKERS = {"AAPL", "GOOG", "MSFT", "TSLA"}


def _extract_ticker(query: str) -> Optional[str]:
    """Extract a single ticker symbol from the query text.

    Returns the first known ticker found, or None.
    """
    upper = query.upper()
    for ticker in _KNOWN_TICKERS:
        if re.search(rf"\b{ticker}\b", upper):
            return ticker
    return None


def _extract_tickers(query: str, all_tickers: list[str]) -> list[str]:
    """Extract all ticker symbols mentioned in the query.

    If none found or query implies 'all', returns all configured tickers.
    """
    upper = query.upper()

    # Check for "all" / "which" / "every" — implies all tickers
    if any(word in upper for word in ["ALL", "WHICH", "EVERY", "EACH"]):
        return all_tickers

    found = [t for t in all_tickers if re.search(rf"\b{t}\b", upper)]
    return found if found else all_tickers


def _date_offset(date_str: str, days: int) -> str:
    """Offset an ISO date string by N days. Returns ISO format."""
    dt = datetime.strptime(str(date_str)[:10], "%Y-%m-%d")
    return (dt + timedelta(days=days)).strftime("%Y-%m-%d")
