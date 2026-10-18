# base interfaces for context selection strategies
# each strategy implements select_context() to pull data from buffer/store

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from ctxfeed.config import Config
from ctxfeed.storage.historical_store import HistoricalStore
from ctxfeed.storage.stream_buffer import StreamBuffer


@dataclass
class ContextResult:
    context_text: str
    rows_selected: int
    metadata: dict[str, Any] = field(default_factory=dict)


class Strategy(ABC):
    @property
    @abstractmethod
    def name(self) -> str:
        ...

    @abstractmethod
    def select_context(
        self,
        query: str,
        buffer: StreamBuffer,
        store: HistoricalStore,
        config: Config,
    ) -> ContextResult:
        ...

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name='{self.name}')"

