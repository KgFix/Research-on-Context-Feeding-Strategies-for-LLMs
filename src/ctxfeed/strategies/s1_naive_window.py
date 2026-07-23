"""S1 — Naive Fixed Window (Baseline Strategy).

Selects the most recent N rows from the stream buffer with no
awareness of the query content. This is the control condition —
the simplest possible approach.

Per strategy_specs.md:
- Window size: 50 rows (configurable via config.strategies.s1_naive.window_size)
- No query awareness
- No ticker filtering — all tickers in the window are included
- Rows sorted ascending (chronological)
"""

from __future__ import annotations

from ctxfeed.config import Config
from ctxfeed.formatting.markdown_table import df_to_markdown
from ctxfeed.storage.historical_store import HistoricalStore
from ctxfeed.storage.stream_buffer import StreamBuffer
from ctxfeed.strategies.base import ContextResult, Strategy


class NaiveWindowStrategy(Strategy):
    """Baseline: last N rows from the buffer, no query analysis."""

    @property
    def name(self) -> str:
        return "s1_naive"

    def select_context(
        self,
        query: str,
        buffer: StreamBuffer,
        store: HistoricalStore,
        config: Config,
    ) -> ContextResult:
        """Select the most recent N rows regardless of query content.

        Args:
            query: The natural language query (ignored by this strategy).
            buffer: The in-memory stream buffer.
            store: The historical store (unused by this strategy).
            config: Pipeline configuration.

        Returns:
            ContextResult with the formatted Markdown table.
        """
        n = config.strategies.s1_naive.window_size
        df = buffer.get_latest(n)
        context_text = df_to_markdown(df)

        return ContextResult(
            context_text=context_text,
            rows_selected=len(df),
            metadata={"window_size": n},
        )
