# s4: summarisation-augmented context
# generates an llm summary of older historical data, cached per ticker, then adds recent raw rows

from __future__ import annotations

import logging
import re
from typing import Optional

from ctxfeed.config import Config
from ctxfeed.formatting.markdown_table import df_to_markdown
from ctxfeed.llm.ollama_client import OllamaClient
from ctxfeed.llm.prompts import SUMMARISATION_PROMPT, SUMMARISATION_SYSTEM
from ctxfeed.storage.historical_store import HistoricalStore
from ctxfeed.storage.stream_buffer import StreamBuffer
from ctxfeed.strategies.base import ContextResult, Strategy

logger = logging.getLogger(__name__)

# supported ticker symbols
_KNOWN_TICKERS = {"AAPL", "GOOG", "MSFT", "TSLA"}


class SummarisationStrategy(Strategy):
    def __init__(self, llm_client: OllamaClient) -> None:
        self._client = llm_client
        self._summary_cache: dict[str, str] = {}

    @property
    def name(self) -> str:
        return "s4_summary"

    def select_context(
        self,
        query: str,
        buffer: StreamBuffer,
        store: HistoricalStore,
        config: Config,
    ) -> ContextResult:
        # build context: cached summary + recent raw rows
        cfg = config.strategies.s4_summary

        ticker = self._extract_ticker(query)

        # Get recent rows
        if ticker:
            recent = buffer.get_latest_for_ticker(ticker, n=cfg.recent_window)
        else:
            recent = buffer.get_latest(n=cfg.recent_window)

        # Get or generate summary
        summary = ""
        if ticker:
            summary = self._get_or_generate_summary(ticker, store, recent, cfg, config)

        # Assemble context
        recent_table = df_to_markdown(recent)

        if summary:
            context_text = (
                f"## Historical Summary for {ticker}\n"
                f"{summary}\n\n"
                f"## Recent Data (last {len(recent)} trading days)\n"
                f"{recent_table}"
            )
        else:
            context_text = recent_table

        return ContextResult(
            context_text=context_text,
            rows_selected=len(recent),
            metadata={
                "ticker": ticker,
                "summary_cached": ticker in self._summary_cache if ticker else False,
                "summary_length": len(summary.split()) if summary else 0,
            },
        )

    def _get_or_generate_summary(
        self,
        ticker: str,
        store: HistoricalStore,
        recent: "pd.DataFrame",
        cfg,
        config: Config,
    ) -> str:
        # return cached summary or generate once per ticker
        if ticker in self._summary_cache:
            logger.debug("Using cached summary for %s", ticker)
            return self._summary_cache[ticker]

        # Get all historical data for this ticker
        all_data = store.query(ticker=ticker)

        if all_data.empty:
            return ""

        # Exclude the recent rows (already shown as raw data)
        if not recent.empty:
            recent_dates = set(recent["date"].astype(str))
            historical = all_data[~all_data["date"].astype(str).isin(recent_dates)]
        else:
            historical = all_data

        if historical.empty:
            return ""

        # Format historical data for the summarisation prompt
        historical_table = df_to_markdown(historical)

        # Generate summary
        system = SUMMARISATION_SYSTEM.format(max_words=cfg.summary_max_words)
        prompt = SUMMARISATION_PROMPT.format(ticker=ticker, context=historical_table)

        logger.info("Generating summary for %s (%d historical rows)", ticker, len(historical))
        summary, _ = self._client.generate(prompt, system=system)

        # Cache it
        self._summary_cache[ticker] = summary
        logger.info("Summary for %s cached (%d words)", ticker, len(summary.split()))

        return summary

    @staticmethod
    def _extract_ticker(query: str) -> Optional[str]:
        # pull single ticker from query if present
        upper = query.upper()
        for ticker in _KNOWN_TICKERS:
            if re.search(rf"\b{ticker}\b", upper):
                return ticker
        return None
