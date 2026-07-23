"""Intent classifier for S2 — Query-Intent-Driven Windowing.

Uses a zero-shot LLM call to classify a natural language query into
one of four intent categories: RECENCY, TREND, ANOMALY, COMPARISON.

The classifier is a separate module so it can be tested independently
of the S2 strategy logic.
"""

from __future__ import annotations

import logging
from enum import Enum

from ctxfeed.llm.ollama_client import OllamaClient
from ctxfeed.llm.prompts import INTENT_CLASSIFIER_PROMPT, INTENT_CLASSIFIER_SYSTEM

logger = logging.getLogger(__name__)


class QueryIntent(Enum):
    """The four analytical intent categories from the research design."""

    RECENCY = "RECENCY"
    TREND = "TREND"
    ANOMALY = "ANOMALY"
    COMPARISON = "COMPARISON"


def classify_intent(query: str, client: OllamaClient) -> QueryIntent:
    """Classify a query into one of the four intent categories.

    Uses a zero-shot LLM call. If the LLM output doesn't match
    any category, defaults to RECENCY (the safest fallback since
    it behaves most like S1).

    Args:
        query: The natural language query to classify.
        client: The Ollama client for making the LLM call.

    Returns:
        The classified QueryIntent.
    """
    prompt = INTENT_CLASSIFIER_PROMPT.format(query=query)
    response, _ = client.generate(prompt, system=INTENT_CLASSIFIER_SYSTEM)

    # Extract the intent from the response (may contain extra text)
    cleaned = response.strip().upper()

    for intent in QueryIntent:
        if intent.value in cleaned:
            logger.info("Query intent: %s (query: '%s...')", intent.value, query[:50])
            return intent

    logger.warning(
        "Could not classify intent from '%s', defaulting to RECENCY", response
    )
    return QueryIntent.RECENCY
