"""Script 02: Run the full evaluation sweep.

Usage:
    uv run python scripts/02_run_evaluation.py
    uv run python scripts/02_run_evaluation.py --skip-judge    # no judge scoring
    uv run python scripts/02_run_evaluation.py --skip-ragas    # no RAGAS scoring
    uv run python scripts/02_run_evaluation.py --skip-judge --skip-ragas  # fastest
"""

import sys
sys.path.insert(0, "src")

import argparse

from ctxfeed.config import Config
from ctxfeed.evaluation.benchmark import load_benchmark
from ctxfeed.evaluation.results_store import ResultsStore
from ctxfeed.evaluation.runner import run_evaluation
from ctxfeed.ingestion.fetch_data import fetch_stock_data
from ctxfeed.ingestion.replay_engine import replay_all
from ctxfeed.llm.ollama_client import OllamaClient
from ctxfeed.storage.historical_store import HistoricalStore
from ctxfeed.storage.stream_buffer import StreamBuffer
from ctxfeed.strategies.registry import get_all_strategies, register_strategy
from ctxfeed.strategies.s1_naive_window import NaiveWindowStrategy
from ctxfeed.strategies.s2_intent_driven import IntentDrivenStrategy
from ctxfeed.strategies.s3_embedding_retrieval import EmbeddingRetrievalStrategy
from ctxfeed.strategies.s4_summarization import SummarisationStrategy
from ctxfeed.strategies.s5_agentic import AgenticStrategy
from ctxfeed.strategies.s6_sql_agent import SqlAgentStrategy
from ctxfeed.utils.logging_config import setup_logging


def main():
    parser = argparse.ArgumentParser(description="Run evaluation sweep")
    parser.add_argument(
        "--skip-judge", action="store_true",
        help="Skip LLM-as-judge scoring (faster dev iteration)",
    )
    parser.add_argument(
        "--skip-ragas", action="store_true",
        help="Skip RAGAS answer relevancy scoring",
    )
    args = parser.parse_args()

    setup_logging()
    config = Config.from_yaml("config/config.yaml")

    # 1. Fetch data
    print("=== Fetching data ===")
    data = fetch_stock_data(
        tickers=config.data.tickers,
        start_date=config.data.start_date,
        end_date=config.data.end_date,
        cache_dir=config.data.cache_dir,
    )

    # 2. Set up storage
    buffer = StreamBuffer(max_size=config.buffer.max_size)
    store = HistoricalStore(config.storage.db_path)

    # 3. Replay data into buffer + store
    print("=== Replaying data ===")
    replay_all(data, buffer, store)
    print(f"Buffer: {buffer}, Store: {store}")

    # 4. Set up LLM client
    client = OllamaClient(config.llm)
    if not client.is_available():
        print(f"\nERROR: Ollama not available at {config.llm.base_url}")
        print(f"Model '{config.llm.model}' not found.")
        print("Run: ollama serve  and  ollama pull " + config.llm.model)
        sys.exit(1)

    # 5. Register strategies
    register_strategy(NaiveWindowStrategy())
    register_strategy(IntentDrivenStrategy(llm_client=client))
    register_strategy(EmbeddingRetrievalStrategy())
    register_strategy(SummarisationStrategy(llm_client=client))
    register_strategy(AgenticStrategy(client=client))
    register_strategy(SqlAgentStrategy(llm_client=client))
    strategies = get_all_strategies()
    print(f"Strategies: {list(strategies.keys())}")

    # 6. Load benchmark
    queries = load_benchmark("benchmarks/benchmark_queries.json")
    print(f"Benchmark: {len(queries)} queries")

    # 7. Run evaluation
    print("=== Running evaluation ===")
    results_store = ResultsStore(config.evaluation.results_db)
    results = run_evaluation(
        queries=queries,
        strategies=strategies,
        buffer=buffer,
        store=store,
        data=data,
        client=client,
        config=config,
        results_store=results_store,
        skip_judge=args.skip_judge,
        skip_ragas=args.skip_ragas,
    )

    # 8. Summary
    print(f"\n=== Results: {len(results)} rows ===")
    print(results.groupby("strategy_name")[["accuracy", "latency_ms"]].mean().to_string())

    # Cleanup
    store.close()
    results_store.close()


if __name__ == "__main__":
    main()
