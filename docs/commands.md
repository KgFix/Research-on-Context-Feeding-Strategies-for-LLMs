# Commands Reference

Everything you need to run this project from the command line.

---

## Prerequisites

### `uv` must be on your PATH

`uv` is a Python package manager. It's installed at `C:\Users\kiron\.local\bin\uv.exe`. If your terminal says `'uv' is not recognized`, you need to add it to your PATH.

**Option A — Add permanently (do once, lasts forever):**

Open PowerShell and run:

```powershell
[Environment]::SetEnvironmentVariable("Path", "$env:Path;C:\Users\kiron\.local\bin", "User")
```

Close and reopen your terminal. `uv` will now always work.

**Option B — Add per session (temporary):**

Run this at the start of every terminal session:

```powershell
$env:Path = "C:\Users\kiron\.local\bin;$env:Path"
```

### You must be in the project directory

All commands assume you are inside the project folder:

```powershell
cd "C:\Users\kiron\OneDrive\Desktop\Honours Project\context-feed-eval"
```

---

## Core Commands

### 1. Install/sync dependencies

Downloads all Python packages (93 total). Run once, or after changing `pyproject.toml`.

```
uv sync
```

### 2. Fetch stock data

Downloads 1 year of daily OHLCV for AAPL, GOOG, MSFT, TSLA from Yahoo Finance. Saves CSVs to `data/raw/`. Skips download if files already exist.

```
uv run python scripts/01_fetch_data.py
```

### 3. Run the evaluation

Runs all 48 queries through all 5 strategies. **Requires Ollama running** (see [ollama-setup.md](ollama-setup.md)).

```bash
# Fastest — accuracy + RAGAS only, no judge scoring (240 + 48 RAGAS LLM calls)
uv run python scripts/02_run_evaluation.py --skip-judge

# Absolute fastest — accuracy only (240 LLM calls)
uv run python scripts/02_run_evaluation.py --skip-judge --skip-ragas

# Full run — accuracy + judge + RAGAS (240 + 96 LLM calls)
uv run python scripts/02_run_evaluation.py
```

Results are saved to `data/db/results.sqlite`.

> **Note:** S5 makes 1–2 extra LLM calls per query (critique + possible fallback), so actual call count is higher than 240.

### 4. Analyze results

Generates tables, plots, and statistical tests from `results.sqlite`.

```
uv run python scripts/03_analyze_results.py
```

Outputs:
- **Terminal:** 3 summary tables + Wilcoxon test results (accuracy, judge, RAGAS)
- **`results/plots/`:** 4 PNG charts (accuracy bars, judge boxplot, RAGAS boxplot, latency)
- **`results/summary.csv`:** Exportable summary

### 5. Launch the demo

Interactive Streamlit UI for querying with any strategy.

```
uv run streamlit run app/demo.py
```

Opens at `http://localhost:8501`. Pick a strategy, type a query, see the context + LLM response.

---

## What `uv run` does

`uv run <command>` runs a command **inside the project's virtual environment** (`.venv/`). This means:
- All 93 installed packages are available
- The `src/ctxfeed/` package is importable
- You don't need to manually activate the venv

Without `uv run`, Python won't find the project dependencies.

**Equivalent long-form:**
```powershell
# These two do the same thing:
uv run python scripts/01_fetch_data.py
.venv\Scripts\python.exe scripts\01_fetch_data.py
```

---

## Quick Tests (no Ollama needed)

Useful for verifying the codebase works without needing the LLM.

### Check all 48 ground truth functions

```
uv run python -c "from ctxfeed.evaluation.ground_truth import GROUND_TRUTH_REGISTRY; print(len(GROUND_TRUTH_REGISTRY), 'functions registered')"
```

### Verify benchmark loads

```
uv run python -c "from ctxfeed.evaluation.benchmark import load_benchmark; q = load_benchmark('benchmarks/benchmark_queries.json'); print(len(q), 'queries loaded')"
```

### Test S5 agentic context selection

```
uv run python -c "
from ctxfeed.config import Config
from ctxfeed.ingestion.fetch_data import fetch_stock_data
from ctxfeed.ingestion.replay_engine import replay_all
from ctxfeed.storage.stream_buffer import StreamBuffer
from ctxfeed.storage.historical_store import HistoricalStore
from ctxfeed.strategies.s5_agentic import AgenticStrategy
from ctxfeed.llm.ollama_client import OllamaClient

cfg = Config.from_yaml('config/config.yaml')
data = fetch_stock_data(cfg.data.tickers, cfg.data.start_date, cfg.data.end_date, cfg.data.cache_dir)
buf = StreamBuffer(max_size=cfg.buffer.max_size)
store = HistoricalStore(':memory:')
from ctxfeed.ingestion.replay_engine import replay_all; replay_all(data, buf, store)
client = OllamaClient(cfg.llm)
s5 = AgenticStrategy(client)
ctx = s5.select_context('What was AAPL closing price?', buf, store, cfg)
print('Rows:', ctx.rows_selected, '| Fallback used:', ctx.metadata.get('used_fallback'))
store.close()
"
```

### Check if Ollama is reachable

```
uv run python -c "from ctxfeed.llm.ollama_client import OllamaClient; from ctxfeed.config import Config; c = OllamaClient(Config.from_yaml('config/config.yaml').llm); print('Available:', c.is_available())"
```

---

## Ollama Commands

These are Ollama CLI commands, not project commands. Run them in any terminal.

> **On Windows**, `ollama` may not be on PATH after install. Use the full path:
> ```powershell
> & "$env:LOCALAPPDATA\Programs\Ollama\ollama.exe" pull mistral
> ```

| Command | What it does |
|---|---|
| `ollama serve` | Start the Ollama server (leave running) |
| `ollama pull phi3:mini` | Download Phi-3 Mini model (~2.3 GB) |
| `ollama pull mistral` | Download Mistral 7B model (~4.1 GB) — **current eval model** |
| `ollama list` | Show downloaded models |
| `ollama ps` | Show currently loaded models (with GPU info) |
| `ollama run mistral "test"` | Quick interactive test |
| `ollama rm phi3:mini` | Delete a model |

---

## Changing the LLM model

The current model is already set to `"mistral"` in `config/config.yaml`. To switch back to the lightweight dev model:

```yaml
llm:
  model: "phi3:mini"   # faster/lighter for quick dev tests
```

No code changes needed. The evaluation script reads the model name from config.

---

## Re-downloading stock data

Delete the cached CSVs and re-run:

```powershell
Remove-Item data\raw\*.csv
uv run python scripts/01_fetch_data.py
```

---

## Troubleshooting

| Problem | Fix |
|---|---|
| `'uv' is not recognized` | Add `C:\Users\kiron\.local\bin` to PATH (see top of this doc) |
| `ModuleNotFoundError: ctxfeed` | Run from inside `context-feed-eval/`, not the parent folder |
| `No module named 'pandas'` | Run `uv sync` first |
| Ollama connection refused | Run `ollama serve` in another terminal |
| Model not found | Run `ollama pull phi3:mini` |
| Very slow LLM responses | Check `ollama ps` — model should show GPU memory usage |
| `FileNotFoundError: config.yaml` | You're in the wrong directory — `cd context-feed-eval` |
