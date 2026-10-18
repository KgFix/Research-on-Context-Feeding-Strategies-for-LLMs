# s1: naive fixed window baseline
# grabs the most recent n rows from buffer regardless of the query

from __future__ import annotations

from ctxfeed.config import Config
from ctxfeed.formatting.markdown_table import df_to_markdown
from ctxfeed.storage.historical_store import HistoricalStore
from ctxfeed.storage.stream_buffer import StreamBuffer
from ctxfeed.strategies.base import ContextResult, Strategy


class NaiveWindowStrategy(Strategy):
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
        # take latest n rows and dump to markdown table

        n = config.strategies.s1_naive.window_size
        df = buffer.get_latest(n)
        context_text = df_to_markdown(df)

        return ContextResult(
            context_text=context_text,
            rows_selected=len(df),
            metadata={"window_size": n},
        )
