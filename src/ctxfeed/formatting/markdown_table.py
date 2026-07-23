"""Convert DataFrames and row dicts to Markdown tables and text strings.

All strategies format their selected context as Markdown tables before
injection into the LLM prompt. This module provides the shared formatting
logic so every strategy produces consistent output.

The row_to_text() function is used specifically by S3 (embedding retrieval)
to encode individual rows as text strings for embedding.
"""

from __future__ import annotations

import pandas as pd


def df_to_markdown(df: pd.DataFrame, highlight_rows: set[int] | None = None) -> str:
    """Convert a DataFrame to a Markdown table string.

    Args:
        df: DataFrame with OHLCV columns.
        highlight_rows: Optional set of integer index positions to prefix
            with '>>> ' (used by S2-ANOMALY to mark the anomaly row).

    Returns:
        A Markdown-formatted table string. Empty DataFrame returns
        "(no data)".
    """
    if df.empty:
        return "(no data)"

    # Standardise column display order
    display_cols = ["date", "ticker", "open", "high", "low", "close", "volume"]
    cols = [c for c in display_cols if c in df.columns]
    table_df = df[cols].copy()

    # Header
    headers = "| " + " | ".join(col.title() for col in cols) + " |"
    separator = "| " + " | ".join("---" for _ in cols) + " |"

    # Rows
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

        if highlight_rows and idx in highlight_rows:
            line = ">>> " + line

        lines.append(line)

    return "\n".join(lines)


def row_to_text(row: dict) -> str:
    """Convert a single OHLCV row dict to a text string for embedding.

    Format: "AAPL | 2026-06-03 | Open: 195.20 | High: 198.50 | ..."

    Used by S3 (embedding retrieval) to create text representations
    that are then encoded by the sentence transformer.
    """
    return (
        f"{row['ticker']} | {row['date']} | "
        f"Open: {float(row['open']):.2f} | "
        f"High: {float(row['high']):.2f} | "
        f"Low: {float(row['low']):.2f} | "
        f"Close: {float(row['close']):.2f} | "
        f"Volume: {int(row['volume']):,}"
    )
