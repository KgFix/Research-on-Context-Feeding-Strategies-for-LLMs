# pulls daily stock data from yahoo finance and caches locally
# uv run python scripts/01_fetch_data.py


import sys
sys.path.insert(0, "src")

from ctxfeed.config import Config
from ctxfeed.ingestion.fetch_data import fetch_stock_data
from ctxfeed.utils.logging_config import setup_logging


def main():
    setup_logging()
    config = Config.from_yaml("config/config.yaml")

    df = fetch_stock_data(
        tickers=config.data.tickers,
        start_date=config.data.start_date,
        end_date=config.data.end_date,
        cache_dir=config.data.cache_dir,
    )

    print(f"\nFetched {len(df)} total rows")
    print(f"Tickers: {df['ticker'].unique().tolist()}")
    print(f"Date range: {df['date'].min()} to {df['date'].max()}")
    print(f"\nSample:\n{df.head(10).to_string(index=False)}")


if __name__ == "__main__":
    main()
