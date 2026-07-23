"""Base classes for context-feeding strategies.

Every strategy implements the same Strategy ABC, which is the
architectural linchpin of the evaluation pipeline — it's what lets
one evaluation loop run all four strategies fairly.

The only method to implement:
    select_context(query, buffer, store, config) -> ContextResult
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from ctxfeed.config import Config
from ctxfeed.storage.historical_store import HistoricalStore
from ctxfeed.storage.stream_buffer import StreamBuffer


@dataclass
class ContextResult:
    """The output of a strategy's context selection.

    Attributes:
        context_text: The formatted context string (Markdown table or
            summary + table) ready for injection into the LLM prompt.
        rows_selected: Number of data rows included in the context.
        metadata: Strategy-specific metadata for logging and analysis
            (e.g. detected intent for S2, similarity scores for S3).
    """

    context_text: str
    rows_selected: int
    metadata: dict[str, Any] = field(default_factory=dict)


class Strategy(ABC):
    """Abstract base class for all context-feeding strategies.

    Subclasses implement select_context() to choose which data from
    the stream buffer and/or historical store to include in the LLM
    prompt for a given query.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Short identifier for this strategy (e.g. 's1_naive')."""
        ...

    @abstractmethod
    def select_context(
        self,
        query: str,
        buffer: StreamBuffer,
        store: HistoricalStore,
        config: Config,
    ) -> ContextResult:
        """Select context data for the given query.

        Args:
            query: The natural language query from the user.
            buffer: The in-memory stream buffer (recent data).
            store: The SQLite historical store (full history).
            config: The pipeline configuration.

        Returns:
            A ContextResult with the formatted context text,
            row count, and any strategy-specific metadata.
        """
        ...

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name='{self.name}')"
