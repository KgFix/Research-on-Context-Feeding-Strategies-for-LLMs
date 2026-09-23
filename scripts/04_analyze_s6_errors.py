"""Empirical Error Taxonomy & Diagnostic Script for Strategy S6 (Text-to-SQL).

Analyzes all S6 execution logs from data/db/results.sqlite to classify failure modes,
producing paper-ready summary tables, CSV exports, and a LaTeX table for the IEEE paper.

Usage:
    uv run python scripts/04_analyze_s6_errors.py
"""

import json
import sqlite3
from collections import defaultdict
from pathlib import Path
import pandas as pd

DB_PATH = Path("data/db/results.sqlite")
OUTPUT_DIR = Path("results")
OUTPUT_CSV = OUTPUT_DIR / "s6_error_taxonomy.csv"


def classify_error(row: dict) -> str:
    """Classify the root cause of an inaccurate prediction (accuracy == 0.0)."""
    meta = row.get("strategy_metadata", {})
    used_fallback = meta.get("used_fallback", False)
    sql_error = (meta.get("sql_error") or "").lower()
    gen_sql = (meta.get("generated_sql") or "").upper()

    if row["accuracy"] == 1.0:
        return "Correct Prediction"

    if not used_fallback:
        # Valid SQL executed, but produced incorrect answer
        return "Semantic/Calculation Error (Valid SQL)"

    # Fallback was triggered due to SQL execution failure
    if "tabulate" in sql_error:
        return "Legacy Environment/Package Missing"
    elif "curdate" in sql_error or "date_sub" in sql_error or "interval" in sql_error:
        return "Dialect Mismatch (MySQL/PG in SQLite)"
    elif "window function" in sql_error:
        return "Illegal Window Function in WHERE Clause"
    elif "misuse of aggregate" in sql_error:
        return "Illegal Aggregate in WHERE Clause"
    elif "syntax error" in sql_error or "near" in sql_error:
        # Check if syntax error is due to interval / date functions
        if "interval" in gen_sql or "date_sub" in gen_sql or "curdate" in gen_sql:
            return "Dialect Mismatch (MySQL/PG in SQLite)"
        return "General SQL Syntax Error"
    else:
        return f"Other Execution Error ({sql_error[:30]})"


def main():
    if not DB_PATH.exists():
        print(f"Error: Database not found at {DB_PATH}")
        return

    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    c = conn.cursor()

    c.execute("""
        SELECT id, query_id, query_text, query_category, answer_type,
               ground_truth, parsed_answer, accuracy, strategy_metadata, latency_ms
        FROM results
        WHERE strategy_name = 's6_sql_agent'
        ORDER BY id
    """)
    rows = [dict(r) for r in c.fetchall()]
    conn.close()

    total_runs = len(rows)
    print(f"\n=======================================================")
    print(f"   S6 (Text-to-SQL) Empirical Error Analysis (N={total_runs})")
    print(f"=======================================================\n")

    # Parse metadata JSON
    for r in rows:
        if isinstance(r["strategy_metadata"], str):
            try:
                r["strategy_metadata"] = json.loads(r["strategy_metadata"])
            except Exception:
                r["strategy_metadata"] = {}
        elif r["strategy_metadata"] is None:
            r["strategy_metadata"] = {}

        r["error_category"] = classify_error(r)

    df = pd.DataFrame(rows)

    # ── Summary by Error Type ─────────────────────────────────────
    counts = df["error_category"].value_counts()
    print("--- Distribution of Execution & Error Types ---")
    for err_type, count in counts.items():
        pct = (count / total_runs) * 100
        print(f"  {err_type:<45} : {count:>3} ({pct:>5.1f}%)")

    # ── Category Breakdown Table ──────────────────────────────────
    pivot = pd.crosstab(
        df["query_category"],
        df["error_category"],
        margins=True,
        margins_name="Total",
    )
    print("\n--- Error Taxonomy by Query Category ---")
    print(pivot.to_string())

    # Save to CSV
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    pivot.to_csv(OUTPUT_CSV)
    print(f"\nSaved CSV taxonomy to: {OUTPUT_CSV}")

    # ── Generate LaTeX Table for the IEEE Paper ───────────────────
    print("\n--- LaTeX Table for Research Paper ---")
    latex_code = [
        "\\begin{table}[htbp]",
        "\\centering",
        "\\caption{Empirical Failure Modes of Text-to-SQL Agent (S6) Across Intent Categories}",
        "\\label{tab:s6_error_taxonomy}",
        "\\begin{tabular}{lrrrrr}",
        "\\hline",
        "\\textbf{Failure Mode} & \\textbf{Recency} & \\textbf{Trend} & \\textbf{Anomaly} & \\textbf{Comparison} & \\textbf{Total} \\\\",
        "\\hline",
    ]

    error_types = [
        ("Correct Prediction", "Correct (Accurate)"),
        ("Dialect Mismatch (MySQL/PG in SQLite)", "Dialect Mismatch (CURDATE/DATE\\_SUB)"),
        ("Illegal Window Function in WHERE Clause", "Window Function in WHERE Clause"),
        ("Illegal Aggregate in WHERE Clause", "Aggregate in WHERE Clause"),
        ("Semantic/Calculation Error (Valid SQL)", "Semantic/Calculation Error"),
        ("General SQL Syntax Error", "General SQL Syntax Error"),
        ("Legacy Environment/Package Missing", "Environment Formatting Fallback"),
    ]

    for err_key, label in error_types:
        row_vals = []
        for cat in ["recency", "trend", "anomaly", "comparison"]:
            val = len(df[(df["query_category"] == cat) & (df["error_category"] == err_key)])
            row_vals.append(str(val))
        tot = len(df[df["error_category"] == err_key])
        latex_code.append(f"{label} & {' & '.join(row_vals)} & {tot} \\\\")

    latex_code.extend([
        "\\hline",
        "\\textbf{Total Inferences} & "
        + " & ".join([str(len(df[df["query_category"] == cat])) for cat in ["recency", "trend", "anomaly", "comparison"]])
        + f" & {total_runs} \\\\",
        "\\hline",
        "\\end{tabular}",
        "\\end{table}",
    ])

    print("\n".join(latex_code))
    print("\nAnalysis complete.")


if __name__ == "__main__":
    main()
