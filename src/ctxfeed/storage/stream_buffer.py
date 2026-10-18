# fixed-size sliding window buffer in memory using deque
# keeps the most recent rows from the simulated live stream

from __future__ import annotations

from collections import deque
from typing import Optional

import pandas as pd


class StreamBuffer:
    def __init__(self, max_size: int = 250) -> None:
        self._buffer: deque[dict] = deque(maxlen=max_size)

    # --- buffer updates ---

    def push(self, row: dict) -> None:
        self._buffer.append(row)

    def clear(self) -> None:
        self._buffer.clear()

    # --- queries ---

    def get_latest(self, n: Optional[int] = None) -> pd.DataFrame:
        # returns latest n rows sorted ascending by date
        rows = list(self._buffer)
        if n is not None:
            rows = rows[-n:]
        if not rows:
            return pd.DataFrame()
        return pd.DataFrame(rows).sort_values("date").reset_index(drop=True)

    def get_latest_for_ticker(
        self, ticker: str, n: Optional[int] = None
    ) -> pd.DataFrame:
        # filter buffer by ticker, take last n
        rows = [r for r in self._buffer if r.get("ticker") == ticker]
        if n is not None:
            rows = rows[-n:]
        if not rows:
            return pd.DataFrame()
        return pd.DataFrame(rows).sort_values("date").reset_index(drop=True)

    # --- properties ---

    @property
    def size(self) -> int:
        return len(self._buffer)

    @property
    def max_size(self) -> int:
        return self._buffer.maxlen or 0

    def __len__(self) -> int:
        return len(self._buffer)

    def __repr__(self) -> str:
        return f"StreamBuffer(size={len(self)}, max_size={self.max_size})"


