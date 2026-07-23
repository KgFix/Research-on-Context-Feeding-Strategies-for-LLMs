"""Configuration loader for the context-feed evaluation pipeline.

Loads parameters from config/config.yaml into typed, frozen dataclasses.
All configurable values — buffer sizes, strategy parameters, model settings —
live in config.yaml and are accessed via these dataclasses.

Usage:
    config = Config.from_yaml("config/config.yaml")
    print(config.strategies.s3_embedding.top_k)  # 20
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import yaml


# ── Data source ──────────────────────────────────────────────


@dataclass(frozen=True)
class DataConfig:
    """Which tickers to fetch, over what date range, and where to cache."""

    tickers: list[str] = field(
        default_factory=lambda: ["AAPL", "GOOG", "MSFT", "TSLA"]
    )
    start_date: str = "2025-07-01"
    end_date: str = "2026-06-30"
    cache_dir: str = "data/raw"


# ── Stream buffer ────────────────────────────────────────────


@dataclass(frozen=True)
class BufferConfig:
    """In-memory stream buffer sizing."""

    max_size: int = 250


# ── Historical store ─────────────────────────────────────────


@dataclass(frozen=True)
class StorageConfig:
    """SQLite historical store path."""

    db_path: str = "data/db/historical_store.sqlite"


# ── Strategy parameters ─────────────────────────────────────


@dataclass(frozen=True)
class S1Config:
    """S1 — Naive Fixed Window."""

    window_size: int = 50


@dataclass(frozen=True)
class S2Config:
    """S2 — Query-Intent-Driven Windowing."""

    recency_window: int = 30
    trend_recent: int = 5
    trend_lookback_days: list[int] = field(default_factory=lambda: [30, 60, 90])
    trend_samples_per_lookback: int = 3
    anomaly_k_before: int = 10
    anomaly_k_after: int = 5
    comparison_per_ticker: int = 20


@dataclass(frozen=True)
class S3Config:
    """S3 — Embedding-Based Retrieval."""

    model_name: str = "all-MiniLM-L6-v2"
    top_k: int = 20


@dataclass(frozen=True)
class S4Config:
    """S4 — Summarisation-Augmented Context."""

    recent_window: int = 20
    summary_max_words: int = 150


@dataclass(frozen=True)
class StrategiesConfig:
    """Container for all four strategy configurations."""

    s1_naive: S1Config = field(default_factory=S1Config)
    s2_intent: S2Config = field(default_factory=S2Config)
    s3_embedding: S3Config = field(default_factory=S3Config)
    s4_summary: S4Config = field(default_factory=S4Config)


# ── LLM runtime ─────────────────────────────────────────────


@dataclass(frozen=True)
class LLMConfig:
    """Ollama model and decoding parameters.

    Temperature is fixed low for reproducibility — the strategy,
    not sampling randomness, should explain differences between runs.
    """

    model: str = "phi3:mini"
    base_url: str = "http://localhost:11434"
    temperature: float = 0.1
    top_p: float = 0.9
    seed: int = 42


# ── Evaluation ───────────────────────────────────────────────


@dataclass(frozen=True)
class EvaluationConfig:
    """Evaluation harness settings."""

    results_db: str = "data/db/results.sqlite"
    numeric_tolerance: float = 0.01  # ±1% for continuous values


@dataclass(frozen=True)
class JudgeConfig:
    """LLM-as-judge configuration.

    If model is None, the same model as llm.model is used.
    """

    model: Optional[str] = None
    temperature: float = 0.1


# ── Top-level config ─────────────────────────────────────────


@dataclass(frozen=True)
class Config:
    """Top-level configuration container.

    Load from YAML:
        config = Config.from_yaml("config/config.yaml")

    All strategy parameters, model settings, and paths are accessible
    through nested dataclass attributes:
        config.strategies.s3_embedding.top_k  →  20
        config.llm.temperature                →  0.1
    """

    data: DataConfig = field(default_factory=DataConfig)
    buffer: BufferConfig = field(default_factory=BufferConfig)
    storage: StorageConfig = field(default_factory=StorageConfig)
    strategies: StrategiesConfig = field(default_factory=StrategiesConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    judge: JudgeConfig = field(default_factory=JudgeConfig)

    @classmethod
    def from_yaml(cls, path: str | Path) -> Config:
        """Load configuration from a YAML file.

        Missing keys fall back to defaults (which match strategy_specs.md).

        Args:
            path: Path to the YAML configuration file.

        Returns:
            A fully populated Config instance.

        Raises:
            FileNotFoundError: If the YAML file does not exist.
        """
        path = Path(path)
        with path.open() as f:
            raw = yaml.safe_load(f) or {}

        return cls(
            data=DataConfig(**raw.get("data", {})),
            buffer=BufferConfig(**raw.get("buffer", {})),
            storage=StorageConfig(**raw.get("storage", {})),
            strategies=_parse_strategies(raw.get("strategies", {})),
            llm=LLMConfig(**raw.get("llm", {})),
            evaluation=EvaluationConfig(**raw.get("evaluation", {})),
            judge=JudgeConfig(**raw.get("judge", {})),
        )


def _parse_strategies(raw: dict) -> StrategiesConfig:
    """Parse the nested strategies configuration section."""
    return StrategiesConfig(
        s1_naive=S1Config(**raw.get("s1_naive", {})),
        s2_intent=S2Config(**raw.get("s2_intent", {})),
        s3_embedding=S3Config(**raw.get("s3_embedding", {})),
        s4_summary=S4Config(**raw.get("s4_summary", {})),
    )
