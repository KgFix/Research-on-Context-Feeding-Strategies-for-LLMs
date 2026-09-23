"""In-memory stream buffer backed by a fixed-size deque.

Represents the "most recent" portion of the data stream. Strategies
that need only recent data (S1, S2-RECENCY) read from here rather
than querying the full historical store.

Design note: single-threaded, sequential replay — no concurrency
handling needed (see implementation-plan §7).
"""

from __future__ import annotations

from collections import deque
from typing import Optional

import pandas as pd


class StreamBuffer:
    """Fixed-size, in-memory buffer for the most recent stream rows.

    Args:
        max_size: Maximum number of rows to retain. Oldest rows are
            dropped automatically when the buffer is full.
    """

    def __init__(self, max_size: int = 250) -> None:
        self._buffer: deque[dict] = deque(maxlen=max_size)

    # ── Write ────────────────────────────────────────────────

    def push(self, row: dict) -> None:
        """Append a single row (as a dict) to the buffer."""
        self._buffer.append(row)

    def clear(self) -> None:
        """Remove all rows from the buffer."""
        self._buffer.clear()

    # ── Read ─────────────────────────────────────────────────

    def get_latest(self, n: Optional[int] = None) -> pd.DataFrame:
        """Return the most recent *n* rows as a DataFrame.

        Rows are returned in chronological order (ascending by date).
        If *n* is None or exceeds the buffer size, all rows are returned.
        """
        rows = list(self._buffer)
        if n is not None:
            rows = rows[-n:]
        if not rows:
            return pd.DataFrame()
        return pd.DataFrame(rows).sort_values("date").reset_index(drop=True)

    def get_latest_for_ticker(
        self, ticker: str, n: Optional[int] = None
    ) -> pd.DataFrame:
        """Return the most recent *n* rows for a single ticker.

        Filters the buffer to rows matching *ticker*, then takes the
        last *n* of those. Returned in chronological order.
        """
        rows = [r for r in self._buffer if r.get("ticker") == ticker]
        if n is not None:
            rows = rows[-n:]
        if not rows:
            return pd.DataFrame()
        return pd.DataFrame(rows).sort_values("date").reset_index(drop=True)

    # ── Info ─────────────────────────────────────────────────

    @property
    def size(self) -> int:
        """Current number of rows in the buffer."""
        return len(self._buffer)

    @property
    def max_size(self) -> int:
        """Maximum capacity of the buffer."""
        return self._buffer.maxlen or 0

    def __len__(self) -> int:
        return len(self._buffer)

    def __repr__(self) -> str:
        return f"StreamBuffer(size={len(self)}, max_size={self.max_size})"

