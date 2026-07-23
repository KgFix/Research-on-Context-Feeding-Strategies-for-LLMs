"""SQLite-backed historical store for the full replayed data stream.

Every row that enters the replay engine is persisted here. Strategies
that need lookback beyond the stream buffer (S2-TREND, S2-ANOMALY,
S3, S4) query this store.

Schema:
    ohlcv(ticker TEXT, date TEXT, open REAL, high REAL, low REAL,
          close REAL, volume INTEGER,
          PRIMARY KEY (ticker, date))

    Index on (ticker, date) for fast range queries.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Optional

import pandas as pd


_CREATE_TABLE = """
CREATE TABLE IF NOT EXISTS ohlcv (
    ticker  TEXT    NOT NULL,
    date    TEXT    NOT NULL,
    open    REAL    NOT NULL,
    high    REAL    NOT NULL,
    low     REAL    NOT NULL,
    close   REAL    NOT NULL,
    volume  INTEGER NOT NULL,
    PRIMARY KEY (ticker, date)
)
"""

_CREATE_INDEX = """
CREATE INDEX IF NOT EXISTS idx_ticker_date ON ohlcv (ticker, date)
"""


class HistoricalStore:
    """SQLite store for full replayed OHLCV history.

    Args:
        db_path: Path to the SQLite database file.
            Parent directories are created automatically.
    """

    def __init__(self, db_path: str | Path) -> None:
        self._db_path = Path(db_path)
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self._db_path))
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute(_CREATE_TABLE)
        self._conn.execute(_CREATE_INDEX)
        self._conn.commit()

    # ── Write ────────────────────────────────────────────────

    def insert_row(self, row: dict) -> None:
        """Insert a single OHLCV row. Silently replaces duplicates."""
        self._conn.execute(
            """INSERT OR REPLACE INTO ohlcv
               (ticker, date, open, high, low, close, volume)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (
                row["ticker"],
                str(row["date"]),
                float(row["open"]),
                float(row["high"]),
                float(row["low"]),
                float(row["close"]),
                int(row["volume"]),
            ),
        )
        self._conn.commit()

    def insert_dataframe(self, df: pd.DataFrame) -> None:
        """Bulk-insert a DataFrame of OHLCV rows.

        Expected columns: ticker, date, open, high, low, close, volume.
        """
        records = df[["ticker", "date", "open", "high", "low", "close", "volume"]]
        records = records.copy()
        records["date"] = records["date"].astype(str)
        self._conn.executemany(
            """INSERT OR REPLACE INTO ohlcv
               (ticker, date, open, high, low, close, volume)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            records.values.tolist(),
        )
        self._conn.commit()

    # ── Read ─────────────────────────────────────────────────

    def query(
        self,
        ticker: Optional[str] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        limit: Optional[int] = None,
    ) -> pd.DataFrame:
        """Query OHLCV rows with optional filters.

        Results are always sorted ascending by date (chronological).

        Args:
            ticker: Filter to a single ticker. None returns all.
            start_date: Inclusive lower bound (ISO format string).
            end_date: Inclusive upper bound (ISO format string).
            limit: Maximum number of rows to return.

        Returns:
            DataFrame with columns: ticker, date, open, high, low, close, volume.
        """
        clauses: list[str] = []
        params: list = []

        if ticker is not None:
            clauses.append("ticker = ?")
            params.append(ticker)
        if start_date is not None:
            clauses.append("date >= ?")
            params.append(str(start_date))
        if end_date is not None:
            clauses.append("date <= ?")
            params.append(str(end_date))

        sql = "SELECT ticker, date, open, high, low, close, volume FROM ohlcv"
        if clauses:
            sql += " WHERE " + " AND ".join(clauses)
        sql += " ORDER BY date ASC"
        if limit is not None:
            sql += f" LIMIT {int(limit)}"

        return pd.read_sql_query(sql, self._conn, params=params)

    def get_nearest_rows(
        self, ticker: str, target_date: str, n: int = 3
    ) -> pd.DataFrame:
        """Return the *n* rows for *ticker* closest to *target_date*.

        Used by S2-TREND to get stratified historical samples.
        """
        sql = """
            SELECT ticker, date, open, high, low, close, volume,
                   ABS(JULIANDAY(date) - JULIANDAY(?)) AS dist
            FROM ohlcv
            WHERE ticker = ?
            ORDER BY dist ASC
            LIMIT ?
        """
        df = pd.read_sql_query(sql, self._conn, params=[target_date, ticker, n])
        return df.drop(columns=["dist"]).sort_values("date").reset_index(drop=True)

    def row_count(self) -> int:
        """Return total number of rows in the store."""
        cur = self._conn.execute("SELECT COUNT(*) FROM ohlcv")
        return cur.fetchone()[0]

    # ── Lifecycle ────────────────────────────────────────────

    def clear(self) -> None:
        """Delete all rows from the store."""
        self._conn.execute("DELETE FROM ohlcv")
        self._conn.commit()

    def close(self) -> None:
        """Close the database connection."""
        self._conn.close()

    def __repr__(self) -> str:
        return f"HistoricalStore(db_path='{self._db_path}', rows={self.row_count()})"
