# quick sanity check to verify config.yaml loads without issues
# usage: uv run python scratch/verify_config.py

import sys
sys.path.insert(0, "src")

from ctxfeed.config import Config

cfg = Config.from_yaml("config/config.yaml")

# expected config params for the eval runs
checks = [
    ("buffer.max_size", cfg.buffer.max_size, 250),
    ("s1_naive.window_size", cfg.strategies.s1_naive.window_size, 50),
    ("s2_intent.recency_window", cfg.strategies.s2_intent.recency_window, 30),
    ("s2_intent.trend_recent", cfg.strategies.s2_intent.trend_recent, 5),
    ("s2_intent.trend_lookback_days", cfg.strategies.s2_intent.trend_lookback_days, [30, 60, 90]),
    ("s2_intent.trend_samples_per_lookback", cfg.strategies.s2_intent.trend_samples_per_lookback, 3),
    ("s2_intent.anomaly_k_before", cfg.strategies.s2_intent.anomaly_k_before, 10),
    ("s2_intent.anomaly_k_after", cfg.strategies.s2_intent.anomaly_k_after, 5),
    ("s2_intent.comparison_per_ticker", cfg.strategies.s2_intent.comparison_per_ticker, 20),
    ("s3_embedding.model_name", cfg.strategies.s3_embedding.model_name, "all-MiniLM-L6-v2"),
    ("s3_embedding.top_k", cfg.strategies.s3_embedding.top_k, 20),
    ("s4_summary.recent_window", cfg.strategies.s4_summary.recent_window, 20),
    ("s4_summary.summary_max_words", cfg.strategies.s4_summary.summary_max_words, 150),
    ("llm.model", cfg.llm.model, "mistral"),
    ("llm.temperature", cfg.llm.temperature, 0.1),
    ("llm.top_p", cfg.llm.top_p, 0.9),
    ("llm.seed", cfg.llm.seed, 42),
    ("evaluation.numeric_tolerance", cfg.evaluation.numeric_tolerance, 0.01),
    ("judge.model", cfg.judge.model, None),
    ("data.tickers", cfg.data.tickers, ["AAPL", "GOOG", "MSFT", "TSLA"]),
]

passed = 0
failed = 0
for name, actual, expected in checks:
    if actual == expected:
        print(f"  OK    {name} = {actual}")
        passed += 1
    else:
        print(f"  FAIL  {name}: expected {expected}, got {actual}")
        failed += 1

print(f"\n{'-'*45}")
print(f"sanity check: {passed} passed, {failed} failed")
if failed:
    sys.exit(1)
else:
    print("all config parameters look good")

