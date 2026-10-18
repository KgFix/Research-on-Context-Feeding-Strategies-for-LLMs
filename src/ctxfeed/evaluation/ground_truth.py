# ground truth computation functions
# computes ground truth answers directly from raw dataframe data for all 48 queries

from __future__ import annotations

from typing import Callable

import pandas as pd

# --- registry ---
GROUND_TRUTH_REGISTRY: dict[str, Callable[[pd.DataFrame], str]] = {}


def register(name: str):
    def wrapper(fn: Callable[[pd.DataFrame], str]):
        GROUND_TRUTH_REGISTRY[name] = fn
        return fn

    return wrapper


def compute_ground_truth(fn_name: str, data: pd.DataFrame) -> str:
    if fn_name not in GROUND_TRUTH_REGISTRY:
        available = ", ".join(sorted(GROUND_TRUTH_REGISTRY.keys())) or "(none)"
        raise KeyError(
            f"No ground truth function '{fn_name}'. Available: {available}"
        )
    return GROUND_TRUTH_REGISTRY[fn_name](data)


# --- dataframe helpers ---

def _ticker(data: pd.DataFrame, t: str) -> pd.DataFrame:
    return data[data["ticker"] == t].sort_values("date").reset_index(drop=True)


def _latest(data: pd.DataFrame, t: str) -> pd.Series:
    return _ticker(data, t).iloc[-1]



def _tail(data: pd.DataFrame, t: str, n: int) -> pd.DataFrame:
    """Last n rows for a ticker."""
    return _ticker(data, t).tail(n)


# ═══════════════════════════════════════════════════════════════
# RECENCY queries (12)
# ═══════════════════════════════════════════════════════════════


@register("aapl_close_latest")
def _(data: pd.DataFrame) -> str:
    """AAPL's most recent closing price."""
    return f"{_latest(data, 'AAPL')['close']:.2f}"


@register("goog_close_latest")
def _(data: pd.DataFrame) -> str:
    """GOOG's most recent closing price."""
    return f"{_latest(data, 'GOOG')['close']:.2f}"


@register("msft_open_latest")
def _(data: pd.DataFrame) -> str:
    """MSFT's opening price on the most recent day."""
    return f"{_latest(data, 'MSFT')['open']:.2f}"


@register("tsla_volume_latest")
def _(data: pd.DataFrame) -> str:
    """TSLA's trading volume on the most recent day."""
    return str(int(_latest(data, "TSLA")["volume"]))


@register("aapl_high_latest")
def _(data: pd.DataFrame) -> str:
    """AAPL's highest price on the most recent day."""
    return f"{_latest(data, 'AAPL')['high']:.2f}"


@register("msft_low_latest")
def _(data: pd.DataFrame) -> str:
    """MSFT's lowest price on the most recent day."""
    return f"{_latest(data, 'MSFT')['low']:.2f}"


@register("tsla_close_latest")
def _(data: pd.DataFrame) -> str:
    """TSLA's most recent closing price."""
    return f"{_latest(data, 'TSLA')['close']:.2f}"


@register("goog_volume_latest")
def _(data: pd.DataFrame) -> str:
    """GOOG's trading volume on the most recent day."""
    return str(int(_latest(data, "GOOG")["volume"]))


@register("aapl_close_vs_open_latest")
def _(data: pd.DataFrame) -> str:
    """Did AAPL close higher than it opened on the most recent day?"""
    row = _latest(data, "AAPL")
    return "Yes" if row["close"] > row["open"] else "No"


@register("goog_range_latest")
def _(data: pd.DataFrame) -> str:
    """GOOG's price range (high - low) on the most recent day."""
    row = _latest(data, "GOOG")
    return f"{row['high'] - row['low']:.2f}"


@register("msft_close_latest")
def _(data: pd.DataFrame) -> str:
    """MSFT's most recent closing price."""
    return f"{_latest(data, 'MSFT')['close']:.2f}"


@register("tsla_close_vs_open_latest")
def _(data: pd.DataFrame) -> str:
    """Did TSLA close higher than it opened on the most recent day?"""
    row = _latest(data, "TSLA")
    return "Yes" if row["close"] > row["open"] else "No"


# ═══════════════════════════════════════════════════════════════
# TREND queries (12)
# ═══════════════════════════════════════════════════════════════


@register("msft_trend_last_month")
def _(data: pd.DataFrame) -> str:
    """Has MSFT trended upward over the last ~30 trading days?"""
    recent = _tail(data, "MSFT", 30)
    return "Yes" if recent.iloc[-1]["close"] > recent.iloc[0]["close"] else "No"


@register("aapl_trend_3_months")
def _(data: pd.DataFrame) -> str:
    """Has AAPL's price increased over the past ~60 trading days?"""
    recent = _tail(data, "AAPL", 60)
    return "Yes" if recent.iloc[-1]["close"] > recent.iloc[0]["close"] else "No"


@register("goog_trend_last_month")
def _(data: pd.DataFrame) -> str:
    """Has GOOG trended upward over the last ~30 trading days?"""
    recent = _tail(data, "GOOG", 30)
    return "Yes" if recent.iloc[-1]["close"] > recent.iloc[0]["close"] else "No"


@register("tsla_trend_3_months")
def _(data: pd.DataFrame) -> str:
    """Has TSLA's price increased over the past ~60 trading days?"""
    recent = _tail(data, "TSLA", 60)
    return "Yes" if recent.iloc[-1]["close"] > recent.iloc[0]["close"] else "No"


@register("aapl_pct_change_month")
def _(data: pd.DataFrame) -> str:
    """AAPL's percentage price change over the last ~30 trading days."""
    recent = _tail(data, "AAPL", 30)
    pct = (recent.iloc[-1]["close"] - recent.iloc[0]["close"]) / recent.iloc[0]["close"] * 100
    return f"{pct:.2f}"


@register("msft_pct_change_3_months")
def _(data: pd.DataFrame) -> str:
    """MSFT's percentage price change over the last ~60 trading days."""
    recent = _tail(data, "MSFT", 60)
    pct = (recent.iloc[-1]["close"] - recent.iloc[0]["close"]) / recent.iloc[0]["close"] * 100
    return f"{pct:.2f}"


@register("goog_volume_trend_month")
def _(data: pd.DataFrame) -> str:
    """Has GOOG's avg volume increased in the last 30 days vs the prior 30?"""
    goog = _ticker(data, "GOOG")
    recent_30 = goog.tail(30)["volume"].mean()
    prior_30 = goog.tail(60).head(30)["volume"].mean()
    return "Yes" if recent_30 > prior_30 else "No"


@register("tsla_pct_change_month")
def _(data: pd.DataFrame) -> str:
    """TSLA's percentage price change over the last ~30 trading days."""
    recent = _tail(data, "TSLA", 30)
    pct = (recent.iloc[-1]["close"] - recent.iloc[0]["close"]) / recent.iloc[0]["close"] * 100
    return f"{pct:.2f}"


@register("aapl_trend_last_week")
def _(data: pd.DataFrame) -> str:
    """Has AAPL trended upward over the last 5 trading days?"""
    recent = _tail(data, "AAPL", 5)
    return "Yes" if recent.iloc[-1]["close"] > recent.iloc[0]["close"] else "No"


@register("msft_avg_close_month")
def _(data: pd.DataFrame) -> str:
    """MSFT's average closing price over the last ~30 trading days."""
    recent = _tail(data, "MSFT", 30)
    return f"{recent['close'].mean():.2f}"


@register("tsla_trend_last_month")
def _(data: pd.DataFrame) -> str:
    """Has TSLA trended upward over the last ~30 trading days?"""
    recent = _tail(data, "TSLA", 30)
    return "Yes" if recent.iloc[-1]["close"] > recent.iloc[0]["close"] else "No"


@register("goog_avg_close_3_months")
def _(data: pd.DataFrame) -> str:
    """GOOG's average closing price over the last ~60 trading days."""
    recent = _tail(data, "GOOG", 60)
    return f"{recent['close'].mean():.2f}"


# ═══════════════════════════════════════════════════════════════
# ANOMALY queries (12)
# ═══════════════════════════════════════════════════════════════


@register("tsla_largest_drop_date")
def _(data: pd.DataFrame) -> str:
    """Date of TSLA's largest single-day percentage drop."""
    tsla = _ticker(data, "TSLA").copy()
    tsla["ret"] = tsla["close"].pct_change()
    return str(tsla.loc[tsla["ret"].idxmin(), "date"])


@register("aapl_largest_gain_date")
def _(data: pd.DataFrame) -> str:
    """Date of AAPL's largest single-day percentage gain."""
    aapl = _ticker(data, "AAPL").copy()
    aapl["ret"] = aapl["close"].pct_change()
    return str(aapl.loc[aapl["ret"].idxmax(), "date"])


@register("msft_highest_volume")
def _(data: pd.DataFrame) -> str:
    """MSFT's highest single-day trading volume."""
    return str(int(_ticker(data, "MSFT")["volume"].max()))


@register("goog_largest_drop_date")
def _(data: pd.DataFrame) -> str:
    """Date of GOOG's largest single-day percentage drop."""
    goog = _ticker(data, "GOOG").copy()
    goog["ret"] = goog["close"].pct_change()
    return str(goog.loc[goog["ret"].idxmin(), "date"])


@register("tsla_largest_gain_pct")
def _(data: pd.DataFrame) -> str:
    """TSLA's largest single-day percentage gain."""
    tsla = _ticker(data, "TSLA").copy()
    tsla["ret"] = tsla["close"].pct_change() * 100
    return f"{tsla['ret'].max():.2f}"


@register("aapl_highest_volume_date")
def _(data: pd.DataFrame) -> str:
    """Date of AAPL's highest single-day trading volume."""
    aapl = _ticker(data, "AAPL")
    return str(aapl.loc[aapl["volume"].idxmax(), "date"])


@register("msft_largest_drop_pct")
def _(data: pd.DataFrame) -> str:
    """MSFT's largest single-day percentage drop (as a positive number)."""
    msft = _ticker(data, "MSFT").copy()
    msft["ret"] = msft["close"].pct_change() * 100
    return f"{abs(msft['ret'].min()):.2f}"


@register("goog_largest_gain_date")
def _(data: pd.DataFrame) -> str:
    """Date of GOOG's largest single-day percentage gain."""
    goog = _ticker(data, "GOOG").copy()
    goog["ret"] = goog["close"].pct_change()
    return str(goog.loc[goog["ret"].idxmax(), "date"])


@register("aapl_lowest_close")
def _(data: pd.DataFrame) -> str:
    """AAPL's lowest closing price over the entire period."""
    return f"{_ticker(data, 'AAPL')['close'].min():.2f}"


@register("tsla_highest_close")
def _(data: pd.DataFrame) -> str:
    """TSLA's highest closing price over the entire period."""
    return f"{_ticker(data, 'TSLA')['close'].max():.2f}"


@register("msft_highest_close_date")
def _(data: pd.DataFrame) -> str:
    """Date when MSFT reached its highest closing price."""
    msft = _ticker(data, "MSFT")
    return str(msft.loc[msft["close"].idxmax(), "date"])


@register("tsla_highest_volume")
def _(data: pd.DataFrame) -> str:
    """TSLA's highest single-day trading volume."""
    return str(int(_ticker(data, "TSLA")["volume"].max()))


# ═══════════════════════════════════════════════════════════════
# COMPARISON queries (12)
# ═══════════════════════════════════════════════════════════════


@register("highest_avg_volume_ticker")
def _(data: pd.DataFrame) -> str:
    """Which ticker has the highest average volume overall?"""
    return str(data.groupby("ticker")["volume"].mean().idxmax())


@register("highest_close_latest_ticker")
def _(data: pd.DataFrame) -> str:
    """Which ticker had the highest closing price on the most recent day?"""
    latest_date = data["date"].max()
    last_day = data[data["date"] == latest_date]
    return str(last_day.loc[last_day["close"].idxmax(), "ticker"])


@register("largest_single_day_drop_ticker")
def _(data: pd.DataFrame) -> str:
    """Which ticker had the largest single-day percentage drop overall?"""
    worst = None
    worst_ticker = None
    for t in data["ticker"].unique():
        df = _ticker(data, t).copy()
        df["ret"] = df["close"].pct_change()
        min_ret = df["ret"].min()
        if worst is None or min_ret < worst:
            worst = min_ret
            worst_ticker = t
    return str(worst_ticker)


@register("best_total_return_ticker")
def _(data: pd.DataFrame) -> str:
    """Which ticker had the best return from start to end of the period?"""
    best = None
    best_ticker = None
    for t in data["ticker"].unique():
        df = _ticker(data, t)
        ret = (df.iloc[-1]["close"] - df.iloc[0]["close"]) / df.iloc[0]["close"]
        if best is None or ret > best:
            best = ret
            best_ticker = t
    return str(best_ticker)


@register("highest_avg_close_ticker")
def _(data: pd.DataFrame) -> str:
    """Which ticker has the highest average closing price?"""
    return str(data.groupby("ticker")["close"].mean().idxmax())


@register("most_volatile_ticker")
def _(data: pd.DataFrame) -> str:
    """Which ticker had the most volatile daily returns (highest std dev)?"""
    best = None
    best_ticker = None
    for t in data["ticker"].unique():
        df = _ticker(data, t).copy()
        df["ret"] = df["close"].pct_change()
        vol = df["ret"].std()
        if best is None or vol > best:
            best = vol
            best_ticker = t
    return str(best_ticker)


@register("aapl_vs_goog_avg_close")
def _(data: pd.DataFrame) -> str:
    """Did AAPL have a higher average closing price than GOOG?"""
    aapl_avg = _ticker(data, "AAPL")["close"].mean()
    goog_avg = _ticker(data, "GOOG")["close"].mean()
    return "Yes" if aapl_avg > goog_avg else "No"


@register("lowest_avg_volume_ticker")
def _(data: pd.DataFrame) -> str:
    """Which ticker has the lowest average volume overall?"""
    return str(data.groupby("ticker")["volume"].mean().idxmin())


@register("highest_ever_close_ticker")
def _(data: pd.DataFrame) -> str:
    """Which ticker reached the highest closing price at any point?"""
    return str(data.loc[data["close"].idxmax(), "ticker"])


@register("tsla_vs_msft_total_return")
def _(data: pd.DataFrame) -> str:
    """Did TSLA outperform MSFT in total return over the period?"""
    def _return(t):
        df = _ticker(data, t)
        return (df.iloc[-1]["close"] - df.iloc[0]["close"]) / df.iloc[0]["close"]
    return "Yes" if _return("TSLA") > _return("MSFT") else "No"


@register("smallest_price_range_ticker")
def _(data: pd.DataFrame) -> str:
    """Which ticker had the smallest price range (max close - min close)?"""
    best = None
    best_ticker = None
    for t in data["ticker"].unique():
        df = _ticker(data, t)
        rng = df["close"].max() - df["close"].min()
        if best is None or rng < best:
            best = rng
            best_ticker = t
    return str(best_ticker)


@register("highest_single_day_volume_ticker")
def _(data: pd.DataFrame) -> str:
    """Which ticker had the highest single-day volume at any point?"""
    return str(data.loc[data["volume"].idxmax(), "ticker"])
