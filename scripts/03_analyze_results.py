# analysis script: outputs tables and charts from results.sqlite
# uv run python scripts/03_analyze_results.py

import sys
sys.path.insert(0, "src")

from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # headless backend
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from ctxfeed.config import Config

# --- paths ---
RESULTS_DIR = Path("results")
PLOTS_DIR = RESULTS_DIR / "plots"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)


def load_results(db_path: str) -> pd.DataFrame:
    import sqlite3

    conn = sqlite3.connect(db_path)
    df = pd.read_sql_query("SELECT * FROM results ORDER BY id", conn)
    conn.close()
    return df


# --- table 1: overall strategy comparison ---

def table_overall(df: pd.DataFrame) -> pd.DataFrame:
    aggs = {
        "accuracy_mean": ("accuracy", "mean"),
        "accuracy_std": ("accuracy", "std"),
        "relevance_mean": ("relevance_score", "mean"),
        "relevance_std": ("relevance_score", "std"),
        "latency_mean": ("latency_ms", "mean"),
        "latency_std": ("latency_ms", "std"),
        "n": ("accuracy", "count"),
    }
    if "ragas_score" in df.columns:
        aggs["ragas_mean"] = ("ragas_score", "mean")
        aggs["ragas_std"] = ("ragas_score", "std")
    summary = df.groupby("strategy_name").agg(**aggs).round(3)
    return summary


# --- table 2: accuracy by category ---

def table_by_category(df: pd.DataFrame) -> pd.DataFrame:
    pivot = df.pivot_table(
        values="accuracy",
        index="strategy_name",
        columns="query_category",
        aggfunc="mean",
    ).round(3)
    pivot["overall"] = df.groupby("strategy_name")["accuracy"].mean().round(3)
    return pivot


# --- table 3: accuracy by answer type ---

def table_by_answer_type(df: pd.DataFrame) -> pd.DataFrame:
    pivot = df.pivot_table(
        values="accuracy",
        index="strategy_name",
        columns="answer_type",
        aggfunc="mean",
    ).round(3)
    return pivot


# --- figure 1: accuracy grouped bar chart ---



def plot_accuracy_by_category(df: pd.DataFrame) -> None:
    """Grouped bar chart: accuracy per strategy per category."""
    pivot = df.pivot_table(
        values="accuracy",
        index="query_category",
        columns="strategy_name",
        aggfunc="mean",
    )

    categories = ["recency", "trend", "anomaly", "comparison"]
    pivot = pivot.reindex(categories)
    strategies = sorted(pivot.columns)

    x = np.arange(len(categories))
    n_strats = len(strategies)
    width = 0.8 / n_strats
    colours = ["#4e79a7", "#f28e2b", "#e15759", "#76b7b2", "#59a14f", "#edc949"][:n_strats]

    fig, ax = plt.subplots(figsize=(10, 5))

    for i, (strat, colour) in enumerate(zip(strategies, colours)):
        vals = pivot[strat].values
        bars = ax.bar(x + i * width, vals, width, label=strat, color=colour)
        # Value labels on bars
        for bar, val in zip(bars, vals):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.02,
                f"{val:.2f}",
                ha="center",
                va="bottom",
                fontsize=8,
            )

    ax.set_xlabel("Query Category")
    ax.set_ylabel("Mean Accuracy")
    ax.set_title("Accuracy by Strategy and Query Category")
    ax.set_xticks(x + width * (n_strats - 1) / 2)
    ax.set_xticklabels([c.title() for c in categories])
    ax.set_ylim(0, 1.15)
    ax.legend(title="Strategy")
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "accuracy_by_category.png", dpi=150)
    plt.close()
    print(f"  Saved: {PLOTS_DIR / 'accuracy_by_category.png'}")


# --- figure 2: relevance scores box plot ---

def plot_relevance_boxplot(df: pd.DataFrame) -> None:

    if df["relevance_score"].isna().all():
        print("  Skipped relevance boxplot (no judge scores)")
        return

    strategies = sorted(df["strategy_name"].unique())
    data_per_strat = [
        df[df["strategy_name"] == s]["relevance_score"].dropna().values
        for s in strategies
    ]

    fig, ax = plt.subplots(figsize=(8, 5))
    bp = ax.boxplot(data_per_strat, tick_labels=strategies, patch_artist=True)

    colours = ["#4e79a7", "#f28e2b", "#e15759", "#76b7b2", "#59a14f", "#edc949"][:len(strategies)]
    for patch, colour in zip(bp["boxes"], colours):
        patch.set_facecolor(colour)
        patch.set_alpha(0.6)

    ax.set_ylabel("Relevance Score (1-5)")
    ax.set_title("LLM-as-Judge Relevance Scores by Strategy")
    ax.set_ylim(0.5, 5.5)
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "relevance_boxplot.png", dpi=150)
    plt.close()
    print(f"  Saved: {PLOTS_DIR / 'relevance_boxplot.png'}")


# --- figure 3: ragas answer relevancy box plot ---

def plot_ragas_boxplot(df: pd.DataFrame) -> None:
    if "ragas_score" not in df.columns or df["ragas_score"].isna().all():
        print("  Skipped RAGAS boxplot (no RAGAS scores)")
        return

    strategies = sorted(df["strategy_name"].unique())
    data_per_strat = [
        df[df["strategy_name"] == s]["ragas_score"].dropna().values
        for s in strategies
    ]

    fig, ax = plt.subplots(figsize=(8, 5))
    bp = ax.boxplot(data_per_strat, tick_labels=strategies, patch_artist=True)

    colours = ["#4e79a7", "#f28e2b", "#e15759", "#76b7b2", "#59a14f", "#edc949"][:len(strategies)]
    for patch, colour in zip(bp["boxes"], colours):
        patch.set_facecolor(colour)
        patch.set_alpha(0.6)

    ax.set_ylabel("RAGAS Answer Relevancy (0-1)")
    ax.set_title("RAGAS Answer Relevancy Scores by Strategy")
    ax.set_ylim(-0.05, 1.05)
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "ragas_boxplot.png", dpi=150)
    plt.close()
    print(f"  Saved: {PLOTS_DIR / 'ragas_boxplot.png'}")


# --- figure 4: latency comparison ---

def plot_latency(df: pd.DataFrame) -> None:
    stats_df = df.groupby("strategy_name")["latency_ms"].agg(["mean", "std"])
    strategies = sorted(stats_df.index)
    colours = ["#4e79a7", "#f28e2b", "#e15759", "#76b7b2", "#59a14f", "#edc949"][:len(strategies)]

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(
        strategies,
        stats_df.loc[strategies, "mean"],
        yerr=stats_df.loc[strategies, "std"],
        capsize=5,
        color=colours,
        alpha=0.8,
    )

    for bar, val in zip(bars, stats_df.loc[strategies, "mean"]):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 50,
            f"{val:.0f}ms",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    ax.set_ylabel("Latency (ms)")
    ax.set_title("Mean Query Latency by Strategy")
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "latency_comparison.png", dpi=150)
    print(f"  Saved: {PLOTS_DIR / 'latency_comparison.png'}")


# --- statistical significance tests ---

def run_statistical_tests(df: pd.DataFrame) -> None:
    baseline = "s1_naive"
    strategies = sorted(s for s in df["strategy_name"].unique() if s != baseline)

    print("\n=== Statistical Tests (Wilcoxon signed-rank vs S1 baseline) ===")
    print(f"{'Strategy':<20} {'Metric':<12} {'W-stat':>8} {'p-value':>10} {'Sig?':>6}")
    print("-" * 60)

    for strat in strategies:
        for metric in ["accuracy", "relevance_score", "ragas_score"]:
            # Align on query_id
            baseline_vals = df[df["strategy_name"] == baseline].groupby("query_id")[metric].last()
            strat_vals = df[df["strategy_name"] == strat].groupby("query_id")[metric].last()

            # Match on common queries
            common = baseline_vals.index.intersection(strat_vals.index)
            b = baseline_vals.loc[common].values
            s = strat_vals.loc[common].values

            # Drop NaN pairs
            b = pd.to_numeric(pd.Series(b), errors="coerce").values
            s = pd.to_numeric(pd.Series(s), errors="coerce").values
            mask = ~(np.isnan(b) | np.isnan(s))
            b, s = b[mask], s[mask]

            if len(b) < 5 or np.all(b == s):
                print(f"{strat:<20} {metric:<12} {'N/A':>8} {'N/A':>10} {'N/A':>6}")
                continue

            try:
                w_stat, p_val = stats.wilcoxon(b, s, alternative="two-sided")
                sig = "Yes" if p_val < 0.05 else "No"
                print(f"{strat:<20} {metric:<12} {w_stat:>8.1f} {p_val:>10.4f} {sig:>6}")
            except Exception as e:
                print(f"{strat:<20} {metric:<12} {'ERR':>8} {str(e)[:10]:>10}")


# --- main execution ---

def main():

    config = Config.from_yaml("config/config.yaml")
    db_path = config.evaluation.results_db

    if not Path(db_path).exists():
        print(f"Results database not found: {db_path}")
        print("Run scripts/02_run_evaluation.py first.")
        sys.exit(1)

    df = load_results(db_path)
    print(f"Loaded {len(df)} results from {db_path}")
    print(f"Strategies: {sorted(df['strategy_name'].unique())}")
    print(f"Queries: {df['query_id'].nunique()}")

    # Tables
    print("\n=== Table 1: Overall Strategy Comparison ===")
    print(table_overall(df).to_string())

    print("\n=== Table 2: Accuracy by Query Category ===")
    print(table_by_category(df).to_string())

    print("\n=== Table 3: Accuracy by Answer Type ===")
    print(table_by_answer_type(df).to_string())

    # Plots
    print("\n=== Generating Plots ===")
    plot_accuracy_by_category(df)
    plot_relevance_boxplot(df)
    plot_ragas_boxplot(df)
    plot_latency(df)

    # Stats
    run_statistical_tests(df)

    # Export summary CSV
    summary_path = RESULTS_DIR / "summary.csv"
    table_overall(df).to_csv(summary_path)
    print(f"\n  Saved: {summary_path}")

    print("\nAnalysis complete.")


if __name__ == "__main__":
    main()
