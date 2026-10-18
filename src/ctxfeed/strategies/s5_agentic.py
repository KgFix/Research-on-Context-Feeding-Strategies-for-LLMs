# s5: agentic critique-and-fallback strategy
# pulls context with s2 first, runs a quick sufficiency critique call, falls back to 100-row window if needed

from __future__ import annotations

import logging

from ctxfeed.config import Config
from ctxfeed.formatting.markdown_table import df_to_markdown
from ctxfeed.llm.ollama_client import OllamaClient
from ctxfeed.llm.prompts import CRITIQUE_PROMPT, CRITIQUE_SYSTEM
from ctxfeed.storage.historical_store import HistoricalStore
from ctxfeed.storage.stream_buffer import StreamBuffer
from ctxfeed.strategies.base import ContextResult, Strategy
from ctxfeed.strategies.s2_intent_driven import IntentDrivenStrategy

logger = logging.getLogger(__name__)

# expanded window size for fallback (2x default s1 window)
_FALLBACK_WINDOW = 100


class AgenticStrategy(Strategy):
    def __init__(self, client: OllamaClient) -> None:
        self._client = client
        self._primary = IntentDrivenStrategy(client)

    @property
    def name(self) -> str:
        return "s5_agentic"

    def select_context(
        self,
        query: str,
        buffer: StreamBuffer,
        store: HistoricalStore,
        config: Config,
    ) -> ContextResult:
        # try s2 first
        primary_ctx = self._primary.select_context(query, buffer, store, config)

        # check if current context has enough info
        sufficient, critique_text = self._critique(
            query, primary_ctx.context_text
        )

        if sufficient:
            logger.info("S5 critique: sufficient (using S2 context)")
            primary_ctx.metadata["used_fallback"] = False
            primary_ctx.metadata["critique_response"] = critique_text
            return primary_ctx

        # if insufficient, fall back to larger window
        logger.info("S5 critique: insufficient, falling back to expanded window")
        fallback_ctx = self._fallback(query, buffer, store)
        fallback_ctx.metadata["used_fallback"] = True
        fallback_ctx.metadata["critique_response"] = critique_text
        fallback_ctx.metadata["primary_rows"] = primary_ctx.rows_selected
        return fallback_ctx

    def _critique(self, query: str, context_text: str) -> tuple[bool, str]:
        prompt = CRITIQUE_PROMPT.format(query=query, context=context_text)
        response, _ = self._client.generate(
            prompt, system=CRITIQUE_SYSTEM
        )
        response_clean = response.strip().lower()

        # default to sufficient if unclear so we don't trigger fallback needlessly
        is_sufficient = "no" not in response_clean

        logger.debug("Critique response: '%s' -> sufficient=%s", response_clean, is_sufficient)
        return is_sufficient, response.strip()

    def _fallback(
        self,
        query: str,
        buffer: StreamBuffer,
        store: HistoricalStore,
    ) -> ContextResult:
        # fallback to larger buffer window (100 rows vs s1's 50)
        rows = buffer.get_latest(_FALLBACK_WINDOW)

        if rows.empty:
            return ContextResult(
                context_text="No data available.",
                rows_selected=0,
                metadata={"fallback_type": "expanded_naive"},
            )

        context_text = df_to_markdown(rows)
        return ContextResult(
            context_text=context_text,
            rows_selected=len(rows),
            metadata={"fallback_type": "expanded_naive"},
        )
