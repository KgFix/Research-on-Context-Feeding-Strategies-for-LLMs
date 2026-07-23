"""S5 — Agentic Routing Strategy (Dynamic Context Selection).

Transforms context selection from a static rule into a self-correcting
loop. Instead of blindly trusting one strategy's context, S5 first
pulls context using S2 (Intent-Driven), then makes a lightweight LLM
call to check if that context is sufficient to answer the query. If
not, it falls back to an expanded naive window to cast a wider net.

Agentic loop:
    1. S2 selects initial context (query-aware)
    2. Critique LLM call: "Is this data sufficient?" (Yes/No)
    3. If Yes → use S2 context
    4. If No  → fallback to expanded S1 window (100 rows)

Research question: Do the accuracy gains of self-correction justify
the extra latency of the critique call?
"""

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

# Expanded window size for fallback (2x the default S1 window)
_FALLBACK_WINDOW = 100


class AgenticStrategy(Strategy):
    """Self-correcting context selection via LLM critique.

    Uses S2 Intent-Driven as the primary strategy, with an LLM
    sufficiency check before proceeding. Falls back to a wider
    naive window if the critique determines the context is
    insufficient.
    """

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
        """Select context with a critique-and-fallback loop.

        1. Get context from S2 (intent-driven)
        2. Ask the LLM if the context is sufficient
        3. If not, fall back to an expanded naive window
        """
        # Step 1: Primary fetch via S2
        primary_ctx = self._primary.select_context(query, buffer, store, config)

        # Step 2: Critique — is this context sufficient?
        sufficient, critique_text = self._critique(
            query, primary_ctx.context_text
        )

        if sufficient:
            logger.info("S5 critique: sufficient (using S2 context)")
            primary_ctx.metadata["used_fallback"] = False
            primary_ctx.metadata["critique_response"] = critique_text
            return primary_ctx

        # Step 3: Fallback — expanded naive window
        logger.info("S5 critique: insufficient — falling back to expanded window")
        fallback_ctx = self._fallback(query, buffer, store)
        fallback_ctx.metadata["used_fallback"] = True
        fallback_ctx.metadata["critique_response"] = critique_text
        fallback_ctx.metadata["primary_rows"] = primary_ctx.rows_selected
        return fallback_ctx

    def _critique(self, query: str, context_text: str) -> tuple[bool, str]:
        """Ask the LLM if the context is sufficient to answer the query.

        Returns:
            (is_sufficient, raw_critique_response)
        """
        prompt = CRITIQUE_PROMPT.format(query=query, context=context_text)
        response, _ = self._client.generate(
            prompt, system=CRITIQUE_SYSTEM
        )
        response_clean = response.strip().lower()

        # Parse binary response — default to "sufficient" if unclear
        # to avoid unnecessary fallback on unparseable responses
        is_sufficient = "no" not in response_clean

        logger.debug("Critique response: '%s' -> sufficient=%s", response_clean, is_sufficient)
        return is_sufficient, response.strip()

    def _fallback(
        self,
        query: str,
        buffer: StreamBuffer,
        store: HistoricalStore,
    ) -> ContextResult:
        """Expanded naive window fallback: last N rows from the buffer.

        Uses a larger window than S1 (100 vs 50) to capture more
        historical context when the primary strategy's context was
        deemed insufficient.
        """
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
