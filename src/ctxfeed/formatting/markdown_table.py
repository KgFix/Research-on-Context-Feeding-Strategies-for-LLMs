# formats dataframes into markdown tables for prompt context
# also has helper to turn a row dict into a string for s3 embedding

from __future__ import annotations

import pandas as pd


def df_to_markdown(df: pd.DataFrame, highlight_rows: set[int] | None = None) -> str:
    # convert df to markdown table string for prompt injection
    if df.empty:
        return "(no data)"

    # standard column order across tables
    display_cols = ["date", "ticker", "open", "high", "low", "close", "volume"]
    cols = [c for c in display_cols if c in df.columns]
    table_df = df[cols].copy()

    # markdown header
    headers = "| " + " | ".join(col.title() for col in cols) + " |"
    separator = "| " + " | ".join("---" for _ in cols) + " |"

    # rows
    lines = [headers, separator]
    for idx, (_, row) in enumerate(table_df.iterrows()):
        values = []
        for col in cols:
            val = row[col]
            if col == "volume":
                values.append(f"{int(val):,}")
            elif col == "date":
                values.append(str(val))
            elif isinstance(val, float):
                values.append(f"{val:.2f}")
            else:
                values.append(str(val))

        line = "| " + " | ".join(values) + " |"

        # mark anomaly row with prefix if requested
        if highlight_rows and idx in highlight_rows:
            line = ">>> " + line

        lines.append(line)

    return "\n".join(lines)


def row_to_text(row: dict) -> str:
    # format single row into text representation for sentence-transformer

    return (
        f"{row['ticker']} | {row['date']} | "
        f"Open: {float(row['open']):.2f} | "
        f"High: {float(row['high']):.2f} | "
        f"Low: {float(row['low']):.2f} | "
        f"Close: {float(row['close']):.2f} | "
        f"Volume: {int(row['volume']):,}"
    )
