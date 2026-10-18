# sqlite-backed store for historical ohlcv data
# replayed stream records are persisted here for trend/anomaly/sql queries

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
    def __init__(self, db_path: str | Path) -> None:
        self._db_path = Path(db_path)
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self._db_path))
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute(_CREATE_TABLE)
        self._conn.execute(_CREATE_INDEX)
        self._conn.commit()

    # --- write methods ---

    def insert_row(self, row: dict) -> None:
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
        records = df[["ticker", "date", "open", "high", "low", "close", "volume"]].copy()
        records["date"] = records["date"].astype(str)
        self._conn.executemany(
            """INSERT OR REPLACE INTO ohlcv
               (ticker, date, open, high, low, close, volume)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            records.values.tolist(),
        )
        self._conn.commit()

    # --- read queries ---

    def query(
        self,
        ticker: Optional[str] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        limit: Optional[int] = None,
    ) -> pd.DataFrame:
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
        # gets closest rows to target_date using sqlite julianday dist
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
        cur = self._conn.execute("SELECT COUNT(*) FROM ohlcv")
        return cur.fetchone()[0]

    # --- db cleanup ---

    def clear(self) -> None:
        self._conn.execute("DELETE FROM ohlcv")
        self._conn.commit()

    def close(self) -> None:
        self._conn.close()

    def __repr__(self) -> str:
        return f"HistoricalStore(db_path='{self._db_path}', rows={self.row_count()})"

