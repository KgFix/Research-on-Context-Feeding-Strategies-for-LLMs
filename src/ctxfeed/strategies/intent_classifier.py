# intent classifier for s2: categorises query into recency/trend/anomaly/comparison
# zero-shot prompt via ollama

from __future__ import annotations

import logging
from enum import Enum

from ctxfeed.llm.ollama_client import OllamaClient
from ctxfeed.llm.prompts import INTENT_CLASSIFIER_PROMPT, INTENT_CLASSIFIER_SYSTEM

logger = logging.getLogger(__name__)


class QueryIntent(Enum):
    RECENCY = "RECENCY"
    TREND = "TREND"
    ANOMALY = "ANOMALY"
    COMPARISON = "COMPARISON"


def classify_intent(query: str, client: OllamaClient) -> QueryIntent:
    # prompt model to classify question into one of the 4 buckets

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
