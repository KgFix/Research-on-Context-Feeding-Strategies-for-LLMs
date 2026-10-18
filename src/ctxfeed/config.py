# config loader for the evaluation pipeline
# pulls params from config/config.yaml into dataclasses

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import yaml


# --- data source ---
@dataclass(frozen=True)
class DataConfig:
    tickers: list[str] = field(
        default_factory=lambda: ["AAPL", "GOOG", "MSFT", "TSLA"]
    )
    start_date: str = "2025-07-01"
    end_date: str = "2026-06-30"
    cache_dir: str = "data/raw"


# --- stream buffer ---
@dataclass(frozen=True)
class BufferConfig:
    max_size: int = 250


# --- historical store ---
@dataclass(frozen=True)
class StorageConfig:
    db_path: str = "data/db/historical_store.sqlite"


# --- strategy parameters ---
@dataclass(frozen=True)
class S1Config:
    window_size: int = 50


@dataclass(frozen=True)
class S2Config:
    recency_window: int = 30
    trend_recent: int = 5
    trend_lookback_days: list[int] = field(default_factory=lambda: [30, 60, 90])
    trend_samples_per_lookback: int = 3
    anomaly_k_before: int = 10
    anomaly_k_after: int = 5
    comparison_per_ticker: int = 20


@dataclass(frozen=True)
class S3Config:
    model_name: str = "all-MiniLM-L6-v2"
    top_k: int = 20


@dataclass(frozen=True)
class S4Config:
    recent_window: int = 20
    summary_max_words: int = 150


@dataclass(frozen=True)
class StrategiesConfig:
    s1_naive: S1Config = field(default_factory=S1Config)
    s2_intent: S2Config = field(default_factory=S2Config)
    s3_embedding: S3Config = field(default_factory=S3Config)
    s4_summary: S4Config = field(default_factory=S4Config)


# --- llm runtime ---
@dataclass(frozen=True)
class LLMConfig:
    # low temp for deterministic / reproducible runs
    model: str = "phi3:mini"
    base_url: str = "http://localhost:11434"
    temperature: float = 0.1
    top_p: float = 0.9
    seed: int = 42


# --- eval settings ---
@dataclass(frozen=True)
class EvaluationConfig:
    results_db: str = "data/db/results.sqlite"
    numeric_tolerance: float = 0.01  # +-1% for numeric queries


@dataclass(frozen=True)
class JudgeConfig:
    # none = reuse llm.model
    model: Optional[str] = None
    temperature: float = 0.1


# --- main config ---
@dataclass(frozen=True)
class Config:
    data: DataConfig = field(default_factory=DataConfig)
    buffer: BufferConfig = field(default_factory=BufferConfig)
    storage: StorageConfig = field(default_factory=StorageConfig)
    strategies: StrategiesConfig = field(default_factory=StrategiesConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    judge: JudgeConfig = field(default_factory=JudgeConfig)

    @classmethod
    def from_yaml(cls, path: str | Path) -> Config:
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
    return StrategiesConfig(
        s1_naive=S1Config(**raw.get("s1_naive", {})),
        s2_intent=S2Config(**raw.get("s2_intent", {})),
        s3_embedding=S3Config(**raw.get("s3_embedding", {})),
        s4_summary=S4Config(**raw.get("s4_summary", {})),
    )
