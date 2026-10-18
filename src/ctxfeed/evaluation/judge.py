# llm-as-judge relevance scoring on 1-5 scale
# calls mistral with low temperature (0.1) for consistency

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
    # returns (score_1_to_5, raw_justification_text)

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
