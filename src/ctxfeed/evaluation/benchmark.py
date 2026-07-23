"""Benchmark query loader and validator.

Loads the benchmark_queries.json file, validates required fields,
and provides access to query objects for the evaluation runner.

Expected JSON structure:
[
  {
    "id": "recency_01",
    "category": "recency",
    "query": "What was AAPL's closing price on 2026-06-03?",
    "answer_type": "numeric",
    "ground_truth_fn": "aapl_close_2026_06_03",
    "tickers": ["AAPL"]
  },
  ...
]
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass
class BenchmarkQuery:
    """A single benchmark query with metadata."""

    id: str
    category: str         # recency, trend, anomaly, comparison
    query: str            # The natural language question
    answer_type: str      # exact, numeric, directional
    ground_truth_fn: str  # Key into ground_truth.py functions
    tickers: list[str]    # Tickers relevant to this query
    ground_truth: Optional[str] = None  # Populated at runtime


def load_benchmark(path: str | Path) -> list[BenchmarkQuery]:
    """Load and validate the benchmark query set.

    Args:
        path: Path to benchmark_queries.json.

    Returns:
        List of validated BenchmarkQuery objects.

    Raises:
        FileNotFoundError: If the benchmark file doesn't exist.
        ValueError: If any query is missing required fields.
    """
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
