# benchmark query loader and validator
# reads benchmarks/benchmark_queries.json into BenchmarkQuery objects

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass
class BenchmarkQuery:
    id: str
    category: str         # recency, trend, anomaly, comparison
    query: str            # question text
    answer_type: str      # exact, numeric, directional
    ground_truth_fn: str  # maps to ground_truth.py function
    tickers: list[str]    # target tickers
    ground_truth: Optional[str] = None  # computed during run


def load_benchmark(path: str | Path) -> list[BenchmarkQuery]:
    # load and validate json query benchmark

    path = Path(path)
    with path.open() as f:
        raw = json.load(f)

    queries = []
    required_fields = {"id", "category", "query", "answer_type", "ground_truth_fn"}
    valid_categories = {"recency", "trend", "anomaly", "comparison"}
    valid_types = {"exact", "numeric", "directional"}

    for i, entry in enumerate(raw):
        # Validate required fields
        missing = required_fields - set(entry.keys())
        if missing:
            raise ValueError(
                f"Query at index {i} missing fields: {missing}"
            )

        # Validate category
        if entry["category"] not in valid_categories:
            raise ValueError(
                f"Query '{entry['id']}' has invalid category '{entry['category']}'. "
                f"Must be one of: {valid_categories}"
            )

        # Validate answer_type
        if entry["answer_type"] not in valid_types:
            raise ValueError(
                f"Query '{entry['id']}' has invalid answer_type '{entry['answer_type']}'. "
                f"Must be one of: {valid_types}"
            )

        queries.append(
            BenchmarkQuery(
                id=entry["id"],
                category=entry["category"],
                query=entry["query"],
                answer_type=entry["answer_type"],
                ground_truth_fn=entry["ground_truth_fn"],
                tickers=entry.get("tickers", []),
            )
        )

    return queries
