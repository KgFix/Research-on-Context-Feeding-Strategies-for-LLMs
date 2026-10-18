# A Comparative Evaluation of Context-Feeding Strategies for LLM-Based Natural Language Querying over Streaming Tabular Data

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Ollama](https://img.shields.io/badge/Ollama-Mistral%207B-orange.svg)](https://ollama.com)
[![License: Academic](https://img.shields.io/badge/License-Academic%20Research-green.svg)](LICENSE)

**UKZN Computer Science Honours Research Project (COMP700, 2026)**  
* **Author:** Kirone Gopaul (Student No. 223014456)  
* **Supervisor:** Dr. Okuthe P. Kogeda  
* **Affiliation:** College of Agriculture, Engineering & Science, University of KwaZulu-Natal, South Africa  
* **Repository:** [https://github.com/KgFix/Research-on-Context-Feeding-Strategies-for-LLMs](https://github.com/KgFix/Research-on-Context-Feeding-Strategies-for-LLMs)

---

## Table of Contents
1. [Overview & Research Motivation](#overview--research-motivation)
2. [The Six Context-Feeding Strategies](#the-six-context-feeding-strategies)
3. [Empirical Results Summary](#empirical-results-summary)
4. [Prerequisites & System Requirements](#prerequisites--system-requirements)
5. [Step-by-Step Reproduction Guide](#step-by-step-reproduction-guide)
   - [Step 1: Clone the Repository](#step-1-clone-the-repository)
   - [Step 2: Install Ollama & Pull Mistral 7B](#step-2-install-ollama--pull-mistral-7b)
   - [Step 3: Setup the Python Environment](#step-3-setup-the-python-environment)
   - [Step 4: Ingest and Cache Streaming Market Data](#step-4-ingest-and-cache-streaming-market-data)
   - [Step 5: Execute the Evaluation Sweep](#step-5-execute-the-evaluation-sweep)
   - [Step 6: Reproduce Paper Analysis, Tables, and Charts](#step-6-reproduce-paper-analysis-tables-and-charts)
   - [Step 7: Launch the Interactive Streamlit Demo (Optional)](#step-7-launch-the-interactive-streamlit-demo-optional)
6. [Evaluation Metrics & Scoring Logic](#evaluation-metrics--scoring-logic)
7. [Repository File Structure](#repository-file-structure)
8. [Troubleshooting & FAQ](#troubleshooting--faq)

---

## Overview & Research Motivation

Large Language Models (LLMs) reason effectively over tabular data via In-Context Learning (ICL). However, current tabular reasoning systems and Retrieval-Augmented Generation (RAG) frameworks assume static, pre-curated tables. When querying continuous, chronologically shifting numerical streams (such as live market feeds), choosing which subset of records to inject into the model's finite context window is a critical performance bottleneck.

This repository contains the complete, reproducible evaluation harness benchmarking **six interchangeable context-feeding strategies** over an equity market stream (AAPL, GOOG, MSFT, TSLA) powered locally by **Mistral 7B Instruct**. Across 1,001 experimental runs, the harness evaluates each query across:
* **Programmatic Ground-Truth Accuracy** (48 deterministic Pandas functions)
* **LLM-as-a-Judge Relevance** (1–5 Likert scale)
* **RAGAS Answer Relevancy** (vector semantic similarity)
* **End-to-End Latency** (wall-clock milliseconds)

---

## The Six Context-Feeding Strategies

| ID | Strategy Name | Architectural Paradigm | Context Selection Mechanism | Query-Aware? |
|---|---|---|---|---|
| **S1** | **Naive Sliding Window** | Temporal Baseline | Direct tail slice of 50 latest rows from in-memory circular buffer (`deque`) | No |
| **S2** | **Query-Intent-Driven** | Heuristic / Slicing | Zero-shot intent classification $\to$ customized row sampling (recency, trend, anomaly, comparison) | Yes |
| **S3** | **Dense Vector RAG** | Semantic Retrieval | Serializes rows to text; retrieves top-$k$ ($k=20$) via SBERT (`all-MiniLM-L6-v2`) cosine similarity | Yes |
| **S4** | **Summarisation + Buffer** | Hybrid Temporal | Caches LLM-generated statistical summary of older history + 20 latest buffer rows | Partially |
| **S5** | **Agentic Critique Loop** | Reflective Routing | Initial intent context $\to$ self-critique verification $\to$ fallback to 100-row expanded window if rejected | Yes |
| **S6** | **In-Database Text-to-SQL** | Tool Execution | Mistral 7B generates SQLite query executed directly against relational database engine | Yes |

---

## Empirical Results Summary

Key findings across 1,001 experimental runs reported in the paper:

| Strategy | Accuracy (Mean) | SD | Judge (1–5) | RAGAS (0–1) | Latency (ms) | Wilcoxon $p$-value | Finding |
|---|---|---|---|---|---|---|---|
| **S1: Naive Window** | 0.365 | 0.483 | 3.00 | 0.480 | 8,947 | *Baseline* | Standard text linearization baseline |
| **S2: Intent-Driven** | 0.333 | 0.473 | 2.91 | 0.509 | 15,161 | 0.6547 | Slicing heuristic adds latency without accuracy gain |
| **S3: Dense Embedding** | **0.154** | 0.362 | 2.48 | 0.537 | 18,672 | **0.0045\*** | Vector similarity scrambles temporal order ($p < 0.01$) |
| **S4: Summarisation** | 0.347 | 0.477 | 3.23 | 0.537 | 12,056 | 0.3173 | Summary mitigates token bloat but arithmetic fails |
| **S5: Agentic Critique** | 0.360 | 0.482 | 3.13 | 0.535 | 16,398 | 0.5637 | Multi-step reflection doubles latency with 0% gain |
| **S6: Text-to-SQL** | **0.512** | 0.502 | **3.58** | 0.502 | **8,240** | **0.0067\*** | **Pareto dominant: highest accuracy, lowest latency** |

*\*Statistically significant difference relative to S1 baseline at $\alpha = 0.05$ (paired Wilcoxon signed-rank test).*

---

## Prerequisites & System Requirements

### Hardware Requirements
* **RAM:** 8 GB minimum (16 GB recommended).
* **GPU (Recommended):** NVIDIA GPU with 4GB+ VRAM (e.g., RTX 3050, GTX 1660, or better).
* **CPU-Only Fallback:** If no dedicated GPU is available, Ollama will execute inference on CPU threads (note: latency will increase from ~8s to ~25–35s per run).

### Software Requirements
* **Operating System:** Windows 10/11, Ubuntu 20.04+, or macOS (Apple Silicon supported).
* **Python:** Version 3.10, 3.11, or 3.12 installed.
* **Git:** For cloning the repository.
* **Ollama:** Open-source local inference engine ([https://ollama.com](https://ollama.com)).
* **Package Manager:** `uv` (recommended for instant, deterministic dependency resolution) or standard `pip`.

---

## Step-by-Step Reproduction Guide

### Step 1: Clone the Repository

Open your terminal (PowerShell on Windows, or Bash on Linux/macOS) and clone the repository:

```bash
git clone https://github.com/KgFix/Research-on-Context-Feeding-Strategies-for-LLMs.git
cd Research-on-Context-Feeding-Strategies-for-LLMs
```

---

### Step 2: Install Ollama & Pull Mistral 7B

The pipeline uses **Mistral 7B Instruct** running locally and offline via Ollama.

1. **Install Ollama:**
   * **Windows:** Download and run the installer from [https://ollama.com/download/windows](https://ollama.com/download/windows).
   * **macOS:** Download the `.zip` from [https://ollama.com/download/mac](https://ollama.com/download/mac).
   * **Linux:** Run `curl -fsSL https://ollama.com/install.sh | sh`.

2. **Pull the Mistral 7B model:**
   In your terminal, execute:
   ```bash
   ollama pull mistral:7b
   ```
   *(This downloads the quantized 4-bit weights, ~4.1 GB. You only need to do this once).*

3. **Verify Ollama is running:**
   ```bash
   ollama list
   ```
   You should see `mistral:7b` (or `mistral:latest`) listed in the output. On Windows and macOS, Ollama runs automatically in your background taskbar. If running on headless Linux, launch it with `ollama serve`.

---

### Step 3: Setup the Python Environment

#### Method A: Using `uv` (Recommended — Fast & Deterministic)
If you have `uv` installed:
```bash
uv sync
```
*If you do not have `uv`, install it in seconds:*
* Windows (PowerShell): `irm https://astral.sh/uv/install.ps1 | iex`
* Linux/macOS: `curl -LsSf https://astral.sh/uv/install.sh | sh`

#### Method B: Using Standard Python Virtual Environment (`venv` + `pip`)
If you prefer standard Python tools:
```bash
# 1. Create a virtual environment
python -m venv .venv

# 2. Activate the virtual environment:
# On Windows (PowerShell):
.venv\Scripts\Activate.ps1
# On Linux/macOS:
source .venv/bin/activate

# 3. Install dependencies in editable mode
pip install -e .
```

---

### Step 4: Ingest and Cache Streaming Market Data

Download the historical equity records (AAPL, GOOG, MSFT, TSLA; 250 trading days each; 1,000 rows total) from Yahoo Finance and populate the local cache:

```bash
# Using uv:
uv run python scripts/01_fetch_data.py

# Or using standard venv:
python scripts/01_fetch_data.py
```

* **What this does:**
  * Fetches clean daily OHLCV equity data via `yfinance`.
  * Saves cached raw CSV files to `data/raw/`.
  * Builds and indexes the local SQLite database at `data/db/historical_store.sqlite` indexed on `(ticker, date)`.
  * Simulates stream replay into the in-memory circular buffer (`deque`, max length 250).

---

### Step 5: Execute the Evaluation Sweep

Run the automated evaluation benchmark across all strategies:

#### Option 1: Fast Execution (Skips LLM-as-a-judge to save time)
Recommended for quick verification of accuracy and latency:
```bash
uv run python scripts/02_run_evaluation.py --skip-judge
```

#### Option 2: Full Evaluation Sweep (Includes Judge & RAGAS Relevance)
Runs all 48 queries across all strategies, scoring accuracy, LLM-as-judge relevance (1–5), and RAGAS relevancy:
```bash
uv run python scripts/02_run_evaluation.py
```

* **Where results are stored:**
  Every inference prompt, retrieved context table, raw model completion, execution latency, and metric score is logged to `data/db/results.sqlite`.

---

### Step 6: Reproduce Paper Analysis, Tables, and Charts

Once evaluation runs are logged, run the analysis scripts to reproduce the exact statistical tables and charts printed in the research paper:

#### 1. Generate Overall Results, Wilcoxon Tests, and Category Breakdowns:
```bash
uv run python scripts/03_analyze_results.py
```
* **Outputs produced:**
  * `results/summary.csv` — Mean/SD accuracy, relevance, latency, and sample counts (reproducing Table 1).
  * Category breakdown and modality breakdown tables (reproducing Tables 2 and 3).
  * Paired Wilcoxon signed-rank hypothesis test $p$-values.
  * Publication charts saved to `results/plots/` (reproducing Figure 2).

#### 2. Generate S6 (Text-to-SQL) Failure Taxonomy:
```bash
uv run python scripts/04_analyze_s6_errors.py
```
* **Outputs produced:**
  * `results/s6_error_taxonomy.csv` — Detailed classification of all 129 S6 runs into dialect mismatches, misplaced clauses, and syntax errors (reproducing Table 5).

---

### Step 7: Launch the Interactive Streamlit Demo (Optional)

To interactively test individual financial queries in real-time, view context windows, and inspect model outputs side-by-side, launch the built-in web dashboard:

```bash
uv run streamlit run app/demo.py
```
Open your browser at `http://localhost:8501`.

---

## Evaluation Metrics & Scoring Logic

The evaluation harness uses four orthogonal metrics to evaluate context effectiveness:

1. **Programmatic Ground-Truth Accuracy ($A \in \{0, 1\}$):**
   * Implemented in `src/ctxfeed/evaluation/ground_truth.py`.
   * Evaluated across 48 queries using deterministic Pandas aggregation functions.
   * Tolerances: $\pm 1\%$ relative tolerance on continuous numerical values; exact match on categorical tickers/dates; directional agreement on binary/ternary trends.
   * Completely bypasses LLM hallucination in the scoring loop.

2. **LLM-as-a-Judge Relevance ($R_{\text{judge}} \in [1, 5]$):**
   * Implemented in `src/ctxfeed/evaluation/judge.py`.
   * Prompts Mistral 7B to evaluate answer correctness and hallucination on a 5-point Likert scale against the known ground truth.

3. **RAGAS Answer Relevancy ($R_{\text{ragas}} \in [0, 1]$):**
   * Implemented in `src/ctxfeed/evaluation/ragas_relevance.py`.
   * Generates synthetic questions from the model's answer and measures cosine similarity against the original query using sentence embeddings.

4. **End-to-End Latency ($L_{\text{ms}}$):**
   * Wall-clock execution time in milliseconds covering context retrieval, prompt assembly, and local model inference.

---

## Repository File Structure

```text
Research-on-Context-Feeding-Strategies-for-LLMs/
├── benchmarks/
│   └── benchmark_queries.json        # 48 curated queries across 4 intent categories
├── config/
│   └── config.yaml                   # Hyperparameters (buffer size, tickers, temperatures)
├── data/
│   ├── raw/                          # Cached raw market data from yfinance (AAPL, GOOG, MSFT, TSLA)
│   └── db/
│       ├── historical_store.sqlite   # SQLite store of 1,000 historical market records
│       └── results.sqlite            # Full experimental evaluation database (1,001 logged runs)
├── results/
│   ├── summary.csv                   # Aggregated empirical metrics across strategies
│   ├── s6_error_taxonomy.csv         # Detailed failure classification of Text-to-SQL runs
│   └── plots/                        # Generated figures (Pareto frontier, category accuracy)
├── scripts/
│   ├── 01_fetch_data.py              # Ingests market data and initialises historical database
│   ├── 02_run_evaluation.py          # Main evaluation execution harness
│   ├── 03_analyze_results.py         # Computes metrics, Wilcoxon tests, and plots
│   └── 04_analyze_s6_errors.py       # Computes S6 SQL error taxonomy
├── src/
│   └── ctxfeed/                      # Core Python package
│       ├── config.py                 # YAML configuration parser
│       ├── formatting/               # Markdown table prompt serializers
│       ├── ingestion/                # yfinance fetcher and chronological stream replayer
│       ├── storage/                  # Circular buffer (deque) and SQLite store
│       ├── llm/                      # Ollama HTTP client and prompt templates
│       ├── strategies/               # Implementations of S1 through S6
│       │   ├── s1_naive_window.py
│       │   ├── s2_intent_driven.py
│       │   ├── s3_embedding_retrieval.py
│       │   ├── s4_summarization.py
│       │   ├── s5_agentic.py
│       │   └── s6_sql_agent.py
│       └── evaluation/               # Ground truth functions, judge, runner, and DB store
├── app/
│   └── demo.py                       # Streamlit web GUI for interactive query testing
├── pyproject.toml                    # Python package metadata and dependencies
└── README.md                         # Detailed project and reproduction documentation
```

---

## Troubleshooting & FAQ

#### 1. `ConnectionRefusedError: Failed to connect to Ollama at http://localhost:11434`
* **Cause:** The Ollama inference daemon is not running.
* **Fix:** Open a terminal and run `ollama serve`, or open the Ollama application from your operating system's applications menu. Verify it is running by visiting `http://localhost:11434` in your browser (it should display *"Ollama is running"*).

#### 2. `sqlite3.OperationalError: database is locked`
* **Cause:** Another program (such as DB Browser for SQLite or an active Python interactive console) has opened `results.sqlite` or `historical_store.sqlite` in write mode.
* **Fix:** Close any database viewer tools before launching `02_run_evaluation.py`.

#### 3. SentenceTransformers model download fails on Strategy S3
* **Cause:** The first time Strategy S3 (Dense Embedding) is called, it downloads the lightweight `all-MiniLM-L6-v2` model weights (~80 MB) from Hugging Face.
* **Fix:** Ensure your computer has active internet access during the initial execution. Once downloaded, weights are cached locally in `~/.cache/huggingface/` and run completely offline.

#### 4. Can I run this with a different LLM (e.g., Llama 3, Phi-3)?
* **Answer:** Yes. Pull your preferred model via Ollama (e.g., `ollama pull llama3:8b` or `ollama pull phi3:mini`) and update `model_name` in `config/config.yaml`.

---

## Citation & Contact

If you find this codebase or research useful, please cite:

```bibtex
@misc{gopaul2026contextfeeding,
  author       = {Kirone Gopaul and Okuthe P. Kogeda},
  title        = {A Comparative Evaluation of Context-Feeding Strategies for LLM-Based Natural Language Querying over Streaming Tabular Data},
  year         = {2026},
  institution  = {University of KwaZulu-Natal},
  school       = {College of Agriculture, Engineering & Science},
  howpublished = {\url{https://github.com/KgFix/Research-on-Context-Feeding-Strategies-for-LLMs}}
}
```

For questions or assistance with reproduction, please contact **Kirone Gopaul** at `223014456@stu.ukzn.ac.za`.
