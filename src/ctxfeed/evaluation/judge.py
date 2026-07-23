"""LLM-as-judge relevance scoring.

Makes a separate LLM call to rate the quality of an LLM response
on a 1-5 scale, independent from the accuracy metric.

Per strategy_specs.md:
- Same model (Mistral 7B) as judge
- Low temperature (0.1) for consistency
- Returns a single integer 1-5
- Self-judging bias acknowledged as a limitation
"""

from __future__ import annotations

import logging
from typing import Optional

from ctxfeed.config import Config
from ctxfeed.evaluation.answer_parser import parse_judge_score
from ctxfeed.llm.ollama_client import OllamaClient
from ctxfeed.llm.prompts import JUDGE_PROMPT, JUDGE_SYSTEM

logger = logging.getLogger(__name__)


def judge_response(
    query: str,
    response: str,
    ground_truth: str,
    client: OllamaClient,
    config: Config,
) -> tuple[Optional[int], str]:
    """Rate an LLM response using the LLM-as-judge method.

    Args:
        query: The original natural language query.
        response: The LLM's full response text.
        ground_truth: The correct answer for comparison.
        client: Ollama client for making the judge call.
        config: Pipeline configuration (for judge model override).

    Returns:
        Tuple of (score, justification).
        Score is 1-5 or None if parsing failed.
        Justification is the judge's reasoning text.
    """
    prompt = JUDGE_PROMPT.format(
        query=query,
        response=response,
        ground_truth=ground_truth,
    )

    model_override = config.judge.model  # None means use default model
    temp_override = config.judge.temperature

    judge_response_text, _ = client.generate(
        prompt,
        system=JUDGE_SYSTEM,
        model_override=model_override,
        temperature_override=temp_override,
    )

    score = parse_judge_score(judge_response_text)

    if score is None:
        logger.warning(
            "Could not parse judge score from response: '%s'",
            judge_response_text[:100],
        )

    return score, judge_response_text
