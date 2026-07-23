"""Download and cache historical OHLCV data from Yahoo Finance.

Uses yfinance to pull daily data for configured tickers. Data is
cached locally as CSV to avoid repeated network calls. If cached
files exist and cover the requested date range, they are loaded
directly without re-downloading.

Output schema per ticker (DataFrame columns):
    ticker, date, open, high, low, close, volume
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
import yfinance as yf

logger = logging.getLogger(__name__)


def fetch_stock_data(
    tickers: list[str],
    start_date: str,
    end_date: str,
    cache_dir: str | Path = "data/raw",
) -> pd.DataFrame:
    """Fetch OHLCV data for all tickers, using local cache when available.

    Args:
        tickers: List of ticker symbols (e.g. ["AAPL", "GOOG"]).
        start_date: Start date (inclusive), ISO format.
        end_date: End date (inclusive), ISO format.
        cache_dir: Directory for cached CSV files.

    Returns:
        Combined DataFrame for all tickers, sorted by (date, ticker).
        Columns: ticker, date, open, high, low, close, volume.
    """
    cache_path = Path(cache_dir)
    cache_path.mkdir(parents=True, exist_ok=True)

    frames: list[pd.DataFrame] = []

    for ticker in tickers:
        csv_path = cache_path / f"{ticker}.csv"

        if csv_path.exists():
            logger.info("Loading cached data: %s", csv_path)
            df = pd.read_csv(csv_path)
        else:
            logger.info("Downloading %s (%s to %s)", ticker, start_date, end_date)
            df = _download_ticker(ticker, start_date, end_date)
            df.to_csv(csv_path, index=False)
            logger.info("Cached %d rows to %s", len(df), csv_path)

        frames.append(df)

    combined = pd.concat(frames, ignore_index=True)
    combined = combined.sort_values(["date", "ticker"]).reset_index(drop=True)

    logger.info(
        "Total: %d rows across %d tickers", len(combined), len(tickers)
    )
    return combined


def _download_ticker(ticker: str, start: str, end: str) -> pd.DataFrame:
    """Download daily OHLCV for a single ticker via yfinance.

    Returns a DataFrame with standardised column names and an
    explicit 'ticker' column.
    """
    raw = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=True)

    if raw.empty:
        logger.warning("No data returned for %s", ticker)
        return pd.DataFrame(
            columns=["ticker", "date", "open", "high", "low", "close", "volume"]
        )

    # yfinance returns a MultiIndex when downloading a single ticker
    # with auto_adjust=True. Flatten if needed.
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)

    df = raw.reset_index()
    df = df.rename(columns={
        "Date": "date",
        "Open": "open",
        "High": "high",
        "Low": "low",
        "Close": "close",
        "Volume": "volume",
    })

    # Ensure date is a string (ISO format) for consistent storage
    df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
    df["ticker"] = ticker

    # Select and order columns
    df = df[["ticker", "date", "open", "high", "low", "close", "volume"]]

    # Round prices to 2 decimal places for clean display
    for col in ["open", "high", "low", "close"]:
        df[col] = df[col].round(2)

    return df
