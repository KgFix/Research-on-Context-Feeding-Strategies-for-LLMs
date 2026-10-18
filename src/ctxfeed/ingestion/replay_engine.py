# simulated streaming replay from cached historical data
# pushes each row into both in-memory buffer and sqlite store

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
    # iterate chronologically and feed buffer + db
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
    # fast load: replay everything into buffer & store upfront

    for _ in replay(data, buffer, store):
        pass
