"""Results storage: SQLite schema for evaluation results.

Stores one row per (query × strategy) combination, including:
- The full assembled prompt (for auditability)
- The LLM response
- Accuracy score, relevance score, latency
- Strategy metadata

This is the raw data that the analysis notebook reads.
"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import pandas as pd


_CREATE_TABLE = """
CREATE TABLE IF NOT EXISTS results (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    query_id        TEXT    NOT NULL,
    strategy_name   TEXT    NOT NULL,
    query_text      TEXT    NOT NULL,
    query_category  TEXT    NOT NULL,
    answer_type     TEXT    NOT NULL,
    ground_truth    TEXT    NOT NULL,
    full_prompt     TEXT    NOT NULL,
    llm_response    TEXT    NOT NULL,
    parsed_answer   TEXT,
    accuracy        REAL,
    relevance_score INTEGER,
    judge_response  TEXT,
    ragas_score     REAL,
    latency_ms      REAL    NOT NULL,
    context_rows    INTEGER NOT NULL,
    strategy_metadata TEXT,
    created_at      TEXT    DEFAULT (datetime('now'))
)
"""

_CREATE_INDEX = """
CREATE INDEX IF NOT EXISTS idx_results_query_strategy
ON results (query_id, strategy_name)
"""


@dataclass
class ResultRow:
    """A single evaluation result to be stored."""

    query_id: str
    strategy_name: str
    query_text: str
    query_category: str
    answer_type: str
    ground_truth: str
    full_prompt: str
    llm_response: str
    parsed_answer: Optional[str]
    accuracy: Optional[float]
    relevance_score: Optional[int]
    judge_response: Optional[str]
    ragas_score: Optional[float]
    latency_ms: float
    context_rows: int
    strategy_metadata: Optional[dict]


class ResultsStore:
    """SQLite store for evaluation results.

    Args:
        db_path: Path to the results SQLite database.
    """

    def __init__(self, db_path: str | Path) -> None:
        self._db_path = Path(db_path)
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self._db_path))
        self._conn.execute(_CREATE_TABLE)
        self._conn.execute(_CREATE_INDEX)
        self._conn.commit()

    def save(self, result: ResultRow) -> None:
        """Insert a single result row."""
        self._conn.execute(
            """INSERT INTO results
               (query_id, strategy_name, query_text, query_category,
                answer_type, ground_truth, full_prompt, llm_response,
                parsed_answer, accuracy, relevance_score, judge_response,
                ragas_score, latency_ms, context_rows, strategy_metadata)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                result.query_id,
                result.strategy_name,
                result.query_text,
                result.query_category,
                result.answer_type,
                result.ground_truth,
                result.full_prompt,
                result.llm_response,
                result.parsed_answer,
                result.accuracy,
                result.relevance_score,
                result.judge_response,
                result.ragas_score,
                result.latency_ms,
                result.context_rows,
                json.dumps(result.strategy_metadata) if result.strategy_metadata else None,
            ),
        )
        self._conn.commit()

    def load_all(self) -> pd.DataFrame:
        """Load all results as a DataFrame for analysis."""
        return pd.read_sql_query("SELECT * FROM results ORDER BY id", self._conn)

    def clear(self) -> None:
        """Delete all results."""
        self._conn.execute("DELETE FROM results")
        self._conn.commit()

    def close(self) -> None:
        """Close the database connection."""
        self._conn.close()
