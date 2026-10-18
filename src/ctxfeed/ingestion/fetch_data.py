# downloads and caches daily stock data from yahoo finance
# caches to csv in data/raw so we dont hit rate limits on repeat runs

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
    # pull ohlcv for tickers, load from csv cache if already downloaded
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
    # grab daily bars via yfinance

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
