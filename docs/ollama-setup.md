# Ollama Setup Guide

## What this is and why you need it

This project is a **comparative evaluation of four context-feeding strategies** for LLM-based natural language querying over streaming stock market (OHLCV) data. The experiment asks: *does it matter how you select which data rows to put in the LLM's context window?*

**Ollama** is an open-source tool that runs large language models locally on your machine via a REST API. This project uses it as the **LLM inference backend** — it's how we actually run the language model that answers questions about the stock data.

### Where Ollama fits in the pipeline

The evaluation pipeline works like this:

1. **Data** — 1 year of daily stock prices for 4 tickers (AAPL, GOOG, MSFT, TSLA), fetched from Yahoo Finance
2. **Strategy** — one of four strategies selects which subset of that data to include in the LLM prompt (this is the variable under test)
3. **LLM call via Ollama** — the selected data + the user's question are sent to the model, which generates an answer
4. **Scoring** — the answer is compared against programmatic ground truth for accuracy, and optionally judged for relevance by a second LLM call

Ollama is needed for step 3. Without it, the strategies can select context (steps 1–2 work), but no LLM responses are generated and no evaluation can happen.

### Which strategies need Ollama and why

| Strategy | Needs Ollama? | Why |
|---|---|---|
| S1 (Naive Window) | Only for answering | Selects the last 50 rows — no LLM needed for context selection |
| S2 (Intent-Driven) | **Yes, for classification + answering** | Uses a zero-shot LLM call to classify the query intent (recency/trend/anomaly/comparison) before selecting context |
| S3 (Embedding Retrieval) | Only for answering | Uses a local sentence-transformer model (not Ollama) for embeddings |
| S4 (Summarisation) | **Yes, for summarisation + answering** | Uses an LLM call to generate a natural-language summary of historical data |

All four strategies need Ollama for the final query-answering step. S2 and S4 also make additional LLM calls during context selection.

### Model choices

- **Phi-3 Mini** (~2.3 GB) — small and fast. Used for local development and the Streamlit demo on a laptop GPU. Not the model used for paper results.
- **Mistral 7B** (~4.1 GB) — the model used for the actual evaluation sweep (paper results). Run on Google Colab free tier with a T4 GPU.

The model name is set in `config/config.yaml` under `llm.model`. The evaluation script checks that Ollama is running and the model is loaded before starting.

---

## 1. Install Ollama

Go to [ollama.com/download](https://ollama.com/download) and download the Windows installer. Run it.

After install, open a **new terminal** and check it works:

```
ollama --version
```

You should see a version number (e.g. `0.9.x`).

---

## 2. Pull the model

For **local development and demo** (your laptop):

```
ollama pull phi3:mini
```

This downloads ~2.3 GB. Takes a few minutes.

For the **evaluation sweep** (Colab with a T4 GPU):

```
ollama pull mistral
```

This downloads ~4.1 GB. Use this for the paper results.

---

## 3. Start the server

```
ollama serve
```

Leave this terminal open. Ollama runs on `http://localhost:11434`.

> **Note:** On Windows, Ollama may auto-start as a background service after install. If `ollama serve` says "address already in use", it's already running — skip this step.

---

## 4. Verify it works

Open a **second terminal** and test:

```
ollama run phi3:mini "Say hello"
```

You should get a response. Close with `/bye`.

---

## 5. Run the evaluation

From the project directory:

```bash
# Fast run (no judge scoring) — do this first
uv run python scripts/02_run_evaluation.py --skip-judge

# Full run (with LLM-as-judge relevance scoring)
uv run python scripts/02_run_evaluation.py
```

The fast run does **48 queries × 4 strategies = 192 LLM calls**.
The full run adds another 192 judge calls = **384 total**.

With phi3:mini on a laptop GPU, expect ~1–3 seconds per call.

---

## 6. Analyze results

```bash
uv run python scripts/03_analyze_results.py
```

Outputs tables to terminal and plots to `results/plots/`.

---

## 7. Run the demo

```bash
uv run streamlit run app/demo.py
```

Opens a browser at `http://localhost:8501`. Pick a strategy, type a query.

---

## Colab Setup (for paper results with Mistral 7B)

Paste this in a Colab notebook cell:

```python
# Install Ollama
!curl -fsSL https://ollama.com/install.sh | sh

# Start server in background
import subprocess
subprocess.Popen(["ollama", "serve"])

# Wait for startup
import time; time.sleep(5)

# Pull Mistral
!ollama pull mistral
```

Then change `config.yaml`:

```yaml
llm:
  model: "mistral"
```

And run the evaluation as normal.

---

## Troubleshooting

| Problem | Fix |
|---|---|
| `ollama` not found | Restart your terminal after install |
| "address already in use" | Ollama is already running — skip `ollama serve` |
| Model not found error | Run `ollama pull phi3:mini` (or `mistral`) |
| Very slow responses | Check GPU is being used: `ollama ps` shows loaded models |
| Connection refused in Python | Make sure `ollama serve` is running in another terminal |
