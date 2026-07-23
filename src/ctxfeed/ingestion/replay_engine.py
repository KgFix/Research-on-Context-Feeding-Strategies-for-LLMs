"""Replay engine: simulates a live data stream from cached historical data.

Iterates through a DataFrame in chronological order, yielding one row
at a time. Each row is pushed to both the stream buffer and the
historical store, simulating a live feed.

This is a single-threaded, sequential generator — no concurrency.
"""

from __future__ import annotations

import logging
from typing import Generator

import pandas as pd

from ctxfeed.storage.historical_store import HistoricalStore
from ctxfeed.storage.stream_buffer import StreamBuffer

logger = logging.getLogger(__name__)


def replay(
    data: pd.DataFrame,
    buffer: StreamBuffer,
    store: HistoricalStore,
) -> Generator[dict, None, None]:
    """Replay historical data as a simulated stream.

    Iterates through *data* in chronological order. For each row:
    1. Converts the row to a dict
    2. Pushes it into the stream buffer
    3. Inserts it into the historical store
    4. Yields the row dict

    Args:
        data: DataFrame with columns: ticker, date, open, high, low, close, volume.
            Must be pre-sorted by date (ascending).
        buffer: The in-memory stream buffer to push rows into.
        store: The SQLite historical store to persist rows into.

    Yields:
        Each row as a dict with keys: ticker, date, open, high, low, close, volume.
    """
    sorted_data = data.sort_values("date").reset_index(drop=True)

    logger.info("Starting replay: %d rows", len(sorted_data))

    for idx, pandas_row in sorted_data.iterrows():
        row = pandas_row.to_dict()
        buffer.push(row)
        store.insert_row(row)
        yield row

    logger.info("Replay complete: %d rows processed", len(sorted_data))


def replay_all(
    data: pd.DataFrame,
    buffer: StreamBuffer,
    store: HistoricalStore,
) -> None:
    """Replay all data without yielding — fully populates buffer and store.

    Convenience function for the evaluation pipeline where you want
    all data loaded before running queries, rather than streaming
    row-by-row.
    """
    for _ in replay(data, buffer, store):
        pass
