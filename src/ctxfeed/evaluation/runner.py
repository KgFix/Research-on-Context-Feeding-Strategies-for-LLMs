# evaluation sweep runner: tests benchmark queries across context strategies
# evaluates accuracy against ground truth, judge score, ragas, and latency

from __future__ import annotations

import logging
import time

import pandas as pd

from ctxfeed.config import Config
from ctxfeed.evaluation.answer_parser import parse_answer
from ctxfeed.evaluation.benchmark import BenchmarkQuery
from ctxfeed.evaluation.ground_truth import compute_ground_truth
from ctxfeed.evaluation.judge import judge_response
from ctxfeed.evaluation.metrics import score_accuracy
from ctxfeed.evaluation.ragas_relevance import compute_ragas_score
from ctxfeed.evaluation.results_store import ResultRow, ResultsStore
from ctxfeed.llm.ollama_client import OllamaClient
from ctxfeed.llm.prompts import ANALYST_SYSTEM_PROMPT, QUERY_PROMPT
from ctxfeed.storage.historical_store import HistoricalStore
from ctxfeed.storage.stream_buffer import StreamBuffer
from ctxfeed.strategies.base import Strategy

logger = logging.getLogger(__name__)


def run_evaluation(
    queries: list[BenchmarkQuery],
    strategies: dict[str, Strategy],
    buffer: StreamBuffer,
    store: HistoricalStore,
    data: pd.DataFrame,
    client: OllamaClient,
    config: Config,
    results_store: ResultsStore,
    skip_judge: bool = False,
    skip_ragas: bool = False,
) -> pd.DataFrame:
    # main eval loop: iterates over queries and strategies

    total = len(queries) * len(strategies)
    completed = 0

    logger.info(
        "Starting evaluation: %d queries × %d strategies = %d runs",
        len(queries), len(strategies), total,
    )

    for query in queries:
        # Compute ground truth for this query
        ground_truth = compute_ground_truth(query.ground_truth_fn, data)
        query.ground_truth = ground_truth

        logger.info(
            "Query [%s] (%s): '%s...' -> GT: %s",
            query.id, query.category, query.query[:40], ground_truth,
        )

        for strategy_name, strategy in strategies.items():
            completed += 1
            logger.info(
                "  [%d/%d] Strategy: %s", completed, total, strategy_name
            )

            try:
                result = _run_single(
                    query, strategy, buffer, store, client, config,
                    ground_truth, skip_judge, skip_ragas,
                )
                results_store.save(result)
                logger.info(
                    "    accuracy=%.1f, relevance=%s, latency=%.0fms",
                    result.accuracy or 0,
                    result.relevance_score or "N/A",
                    result.latency_ms,
                )
            except Exception as e:
                logger.error(
                    "    FAILED: %s - %s", type(e).__name__, e
                )

    logger.info("Evaluation complete: %d/%d runs", completed, total)
    return results_store.load_all()


def _run_single(
    query: BenchmarkQuery,
    strategy: Strategy,
    buffer: StreamBuffer,
    store: HistoricalStore,
    client: OllamaClient,
    config: Config,
    ground_truth: str,
    skip_judge: bool,
    skip_ragas: bool,
) -> ResultRow:
    # select context with active strategy
    ctx = strategy.select_context(query.query, buffer, store, config)

    # assemble full prompt
    full_prompt = QUERY_PROMPT.format(
        context=ctx.context_text,
        query=query.query,
    )

    # prompt llm and record inference latency
    start = time.perf_counter()
    response, _ = client.generate(full_prompt, system=ANALYST_SYSTEM_PROMPT)
    latency_ms = (time.perf_counter() - start) * 1000

    # parse predicted answer from response text
    parsed = parse_answer(response)

    # score accuracy against ground truth
    accuracy = score_accuracy(
        parsed, ground_truth, query.answer_type, config.evaluation.numeric_tolerance
    )

    # optional judge & ragas relevance scoring
    relevance_score = None
    judge_text = None
    if not skip_judge:
        relevance_score, judge_text = judge_response(
            query.query, response, ground_truth, client, config
        )

    ragas_score_val = None
    if not skip_ragas:
        ragas_score_val, _ = compute_ragas_score(
            query.query, response, client
        )

    return ResultRow(

        query_id=query.id,
        strategy_name=strategy.name,
        query_text=query.query,
        query_category=query.category,
        answer_type=query.answer_type,
        ground_truth=ground_truth,
        full_prompt=full_prompt,
        llm_response=response,
        parsed_answer=parsed,
        accuracy=accuracy,
        relevance_score=relevance_score,
        judge_response=judge_text,
        ragas_score=ragas_score_val,
        latency_ms=latency_ms,
        context_rows=ctx.rows_selected,
        strategy_metadata=ctx.metadata,
    )
