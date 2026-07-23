# Architecture & Implementation Reference

> Single reference document for all code, design decisions, and extension points.
> Updated: 21 Jul 2026

---

## 1. Project Overview

This codebase implements a **controlled comparative experiment** evaluating five context-feeding strategies for LLM-based natural language querying over streaming stock market data. The only variable between runs is *which data the strategy selects for the LLM's context window*.

**Strategies:** S1 (Naive Window), S2 (Intent-Driven), S3 (Embedding Retrieval), S4 (Summarisation), S5 (Agentic Routing — added Jul 2026), S6 (Text-to-SQL Agent — added Jul 2026).

**Metrics:** Binary accuracy vs programmatic ground truth, RAGAS Answer Relevancy score (cosine-similarity-based, 0–1), and optional LLM-as-judge score (1–5).

**Not a production system.** Engineering effort is biased toward result validity (fixed decoding params, logged prompts, programmatic ground truth) over production concerns (concurrency, error recovery, test coverage).

---

## 2. Directory Map

```
context-feed-eval/
├── config/config.yaml           # All tuneable parameters (§3)
├── pyproject.toml               # Dependencies + build config
├── README.md
│
├── src/ctxfeed/                 # Main package (§4–§9)
│   ├── config.py                # YAML → frozen dataclasses
│   ├── ingestion/               # Data download + replay
│   │   ├── fetch_data.py        #   yfinance download + CSV cache
│   │   └── replay_engine.py     #   Chronological row-by-row generator
│   ├── storage/                 # Data stores
│   │   ├── stream_buffer.py     #   In-memory deque (recent rows)
│   │   └── historical_store.py  #   SQLite (full history)
│   ├── formatting/
│   │   └── markdown_table.py    #   DataFrame → Markdown + row → text
│   ├── strategies/              # The five strategies (§6)
│   │   ├── base.py              #   Strategy ABC + ContextResult
│   │   ├── registry.py          #   Name → instance lookup
│   │   ├── intent_classifier.py #   Zero-shot LLM classifier (for S2)
│   │   ├── s1_naive_window.py
│   │   ├── s2_intent_driven.py
│   │   ├── s3_embedding_retrieval.py
│   │   ├── s4_summarization.py
│   │   ├── s5_agentic.py        #   Self-correcting critique+fallback loop
│   │   └── s6_sql_agent.py      #   SOTA Text-to-SQL agentic code generation
│   ├── llm/                     # LLM interaction (§7)
│   │   ├── ollama_client.py     #   REST API wrapper
│   │   └── prompts.py           #   All prompt templates
│   ├── evaluation/              # Experiment harness (§8)
│   │   ├── benchmark.py         #   JSON loader + validation
│   │   ├── ground_truth.py      #   Programmatic answer computation
│   │   ├── answer_parser.py     #   Extract "Answer: <value>" from LLM output
│   │   ├── metrics.py           #   3-type accuracy scoring
│   │   ├── judge.py             #   LLM-as-judge relevance (1–5)
│   │   ├── ragas_relevance.py   #   RAGAS answer relevancy (cosine sim, 0-1)
│   │   ├── results_store.py     #   SQLite results schema
│   │   └── runner.py            #   Full query × strategy sweep
│   └── utils/
│       └── logging_config.py
│
├── benchmarks/
│   └── benchmark_queries.json   # Query–answer benchmark set
├── scripts/
│   ├── 01_fetch_data.py         # Download + cache stock data
│   ├── 02_run_evaluation.py     # Run the full experiment
│   └── 03_analyze_results.py   # Generate tables, plots, stats
├── data/raw/                    # Cached CSV files (gitignored)
├── data/db/                     # SQLite databases (gitignored)
│   ├── results.sqlite           #   Current run results
│   └── results_run1_phi3mini.sqlite  # Run 1 backup (phi3:mini baseline)
└── scratch/                     # Throwaway scripts
```

---

## 3. Configuration — `config/config.yaml`

All tuneable values live in one YAML file, loaded by `Config.from_yaml()` into frozen dataclasses. Dataclass defaults match `strategy_specs.md` so the system works even with a minimal YAML.

### Reference table

| Key path | Default | Purpose |
|---|---|---|
| `data.tickers` | `[AAPL, GOOG, MSFT, TSLA]` | Stocks to fetch |
| `data.start_date` | `2025-07-01` | Dataset start (inclusive) |
| `data.end_date` | `2026-06-30` | Dataset end (inclusive) |
| `data.cache_dir` | `data/raw` | Where CSVs are cached |
| `buffer.max_size` | `250` | Stream buffer capacity (rows) |
| `storage.db_path` | `data/db/historical_store.sqlite` | Historical store location |
| `strategies.s1_naive.window_size` | `50` | S1: rows to select |
| `strategies.s2_intent.recency_window` | `30` | S2-RECENCY: rows per ticker |
| `strategies.s2_intent.trend_recent` | `5` | S2-TREND: recent rows |
| `strategies.s2_intent.trend_lookback_days` | `[30, 60, 90]` | S2-TREND: lookback points |
| `strategies.s2_intent.trend_samples_per_lookback` | `3` | S2-TREND: rows per lookback |
| `strategies.s2_intent.anomaly_k_before` | `10` | S2-ANOMALY: rows before |
| `strategies.s2_intent.anomaly_k_after` | `5` | S2-ANOMALY: rows after |
| `strategies.s2_intent.comparison_per_ticker` | `20` | S2-COMPARISON: rows per ticker |
| `strategies.s3_embedding.model_name` | `all-MiniLM-L6-v2` | S3: sentence transformer model |
| `strategies.s3_embedding.top_k` | `20` | S3: rows to retrieve |
| `strategies.s4_summary.recent_window` | `20` | S4: recent raw rows |
| `strategies.s4_summary.summary_max_words` | `150` | S4: target summary length |
| `llm.model` | `phi3:mini` | Ollama model name |
| `llm.base_url` | `http://localhost:11434` | Ollama API URL |
| `llm.temperature` | `0.1` | Low for reproducibility |
| `llm.top_p` | `0.9` | Nucleus sampling |
| `llm.seed` | `42` | Deterministic sampling |
| `evaluation.results_db` | `data/db/results.sqlite` | Results database |
| `evaluation.numeric_tolerance` | `0.01` | ±1% for numeric accuracy |
| `judge.model` | `null` | null = use same model as `llm.model` |
| `judge.temperature` | `0.1` | Judge decoding temperature |

**Decision: frozen dataclasses.** Config objects are immutable after loading. This prevents accidental mutation mid-experiment, which would invalidate results. If you need different settings for different runs, load separate Config instances.

**Decision: defaults in code, not just YAML.** Every dataclass field has a default matching strategy_specs.md. This means Config() with no YAML file still produces a valid configuration — useful for testing.

---

## 4. Data Layer

### 4.1 `ingestion/fetch_data.py`

**Purpose:** Downloads daily OHLCV data from Yahoo Finance, caches locally as CSV.

**Key behaviour:**
- Checks `data/raw/{TICKER}.csv` first — skips download if file exists
- Standardises column names to lowercase: `date, ticker, open, high, low, close, volume`
- Dates stored as ISO strings (`2026-06-03`), prices rounded to 2dp
- Returns a single concatenated DataFrame sorted by `(date, ticker)`

**Decision: CSV cache, not Parquet.** CSV is human-readable and debuggable. At ~1000 rows total, performance difference is negligible.

**Decision: auto_adjust=True.** yfinance's adjusted prices are used. This means `close` is the adjusted close, which accounts for splits/dividends. Consistent across all tickers.

**To change the data source:** Replace `_download_ticker()`. The rest of the pipeline only requires a DataFrame with columns `[ticker, date, open, high, low, close, volume]`.

### 4.2 `ingestion/replay_engine.py`

**Purpose:** Simulates a live stream by iterating through cached data chronologically.

**Two functions:**
- `replay()` — generator that yields one row dict at a time, pushing each into both the buffer and historical store
- `replay_all()` — convenience wrapper that exhausts the generator (used in evaluation where all data must be loaded before queries run)

**Decision: replay_all() for evaluation.** The evaluation queries assume the full dataset is available (e.g. "largest drop ever"). Row-by-row replay is available for the Streamlit demo where you want to simulate a live feed.

### 4.3 `storage/stream_buffer.py`

**Purpose:** In-memory fixed-size buffer holding the most recent rows.

**Implementation:** `collections.deque(maxlen=N)`. When full, oldest rows are automatically dropped.

**Key methods:**
- `push(row)` — append one row dict
- `get_latest(n)` — most recent n rows as DataFrame, chronologically sorted
- `get_latest_for_ticker(ticker, n)` — filtered to one ticker, then take last n

**Decision: no concurrency.** The replay engine is a single sequential loop. No threads, no locks. This is a research pipeline, not a production stream processor.

**Decision: stores raw dicts, returns DataFrames.** Dicts are lightweight for append-heavy workloads. DataFrames are returned for read operations because strategies and formatting work with pandas.

### 4.4 `storage/historical_store.py`

**Purpose:** SQLite database storing the complete replayed history for lookback queries.

**Schema:**
```sql
ohlcv(ticker TEXT, date TEXT, open REAL, high REAL, low REAL,
      close REAL, volume INTEGER, PRIMARY KEY (ticker, date))
INDEX idx_ticker_date ON ohlcv (ticker, date)
```

**Key methods:**
- `insert_row(row)` / `insert_dataframe(df)` — write, with `INSERT OR REPLACE` for idempotency
- `query(ticker, start_date, end_date, limit)` — filtered read, always sorted ascending
- `get_nearest_rows(ticker, target_date, n)` — returns the n rows closest to a date (used by S2-TREND for stratified sampling)

**Decision: SQLite, not in-memory DataFrame.** At 1000 rows a DataFrame would work fine, but SQLite gives us indexed timestamp queries and persists across script runs. The `get_nearest_rows()` query uses `JULIANDAY()` distance — this would be awkward with pure pandas.

**Decision: WAL journal mode.** Write-Ahead Logging gives better concurrent read performance during the evaluation sweep (the runner reads while logging writes).

---

## 5. Formatting — `formatting/markdown_table.py`

**Purpose:** Converts DataFrames and row dicts into text for LLM consumption.

**Two functions:**
- `df_to_markdown(df, highlight_rows)` — full Markdown table with header/separator/rows. Optional row highlighting (prefix `>>> `) for S2-ANOMALY
- `row_to_text(row)` — single-row text string for S3 embedding: `"AAPL | 2026-06-03 | Open: 195.20 | ..."`

**Decision: Markdown tables.** The literature (Sui et al., 2024 — "Table Meets LLM") shows LLMs handle Markdown tables well. It's also readable in logs.

**Decision: volume formatted with commas.** `1,234,567` is more readable for the LLM than `1234567`. Prices use 2dp.

---

## 6. Strategies

### 6.0 Core contract — `strategies/base.py`

The **architectural linchpin**. Every strategy implements one method:

```python
class Strategy(ABC):
    @property
    def name(self) -> str: ...

    def select_context(
        self, query, buffer, store, config
    ) -> ContextResult: ...
```

`ContextResult` is a dataclass:
```python
@dataclass
class ContextResult:
    context_text: str       # Formatted text for the LLM prompt
    rows_selected: int      # Row count (for logging/analysis)
    metadata: dict          # Strategy-specific info (intent, scores, etc.)
```

**Decision: strategy receives both buffer and store.** Some strategies only use the buffer (S1), some only the store (S2-ANOMALY), some use both (S2-TREND, S4). Passing both keeps the interface uniform.

**Decision: config passed per-call, not at init.** Strategies are stateless with respect to config. This means you could theoretically run the same strategy instance with different configs in the same process — useful for sensitivity analysis.

### 6.1 `s1_naive_window.py` — Naive Fixed Window

**What it does:** Takes the last 50 rows from the buffer. No query awareness. The baseline.

**Parameters:** `config.strategies.s1_naive.window_size` (default 50)

**Data source:** Buffer only.

**Why this is the baseline:** It's the simplest plausible approach. If a more complex strategy can't beat "just use the most recent data", that strategy isn't pulling its weight.

**Known limitation:** No ticker filtering. A query about AAPL gets rows from all four tickers, wasting context space.

### 6.2 `s2_intent_driven.py` — Query-Intent-Driven Windowing

**What it does:** Two-phase process:
1. **Classify** the query intent via zero-shot LLM call (→ `intent_classifier.py`)
2. **Dispatch** to one of four sub-strategies based on the classified intent

**Sub-strategies:**

| Intent | What it selects | Data source |
|---|---|---|
| RECENCY | Last 30 rows for the queried ticker | Buffer |
| TREND | 5 recent + 3 samples at each of {T-30d, T-60d, T-90d} ≈ 14 rows | Buffer + Store |
| ANOMALY | 10 before + anomaly + 5 after the largest daily move ≈ 16 rows | Store |
| COMPARISON | Last 20 rows per ticker, interleaved chronologically | Buffer (fallback: Store) |

**`intent_classifier.py`**: Separate module. Returns a `QueryIntent` enum. Fallback: if the LLM output can't be parsed, defaults to RECENCY (safest — behaves like a filtered S1).

**Ticker extraction:** `_extract_ticker(query)` uses regex word boundaries (`\bAAPL\b`) against a known set. `_extract_tickers()` also detects "all"/"which"/"every" as implying all tickers.

**Decision: separate classifier module.** Keeps classification testable independently of the dispatch logic.

**Decision: RECENCY as fallback.** If the classifier fails, you get the safest behaviour (most similar to the baseline). Failing to ANOMALY or TREND with wrong context would produce worse results.

### 6.3 `s3_embedding_retrieval.py` — Embedding-Based Retrieval

**What it does:** Encodes all historical rows as text, embeds with a sentence transformer, and retrieves the top-k rows most similar to the query by cosine similarity.

**Parameters:**
- Model: `all-MiniLM-L6-v2`
- Top-k: 20 rows

**Key implementation details:**
- **Lazy indexing:** The embedding matrix is built on the first call to `select_context()` and cached in memory for all subsequent calls
- **Row text format:** `"AAPL | 2026-06-03 | Open: 195.20 | High: 198.50 | ..."` (from `markdown_table.row_to_text()`)
- **Cosine similarity:** Implemented manually with numpy (normalise + dot product) — no external dependency needed
- **Result ordering:** Top-k rows are re-sorted chronologically, not by similarity score. The LLM sees time-ordered data.

**Known limitation (documented in strategy_specs.md):** `all-MiniLM-L6-v2` was designed for natural language text, not numerical OHLCV rows. Embeddings are dominated by textual structure (ticker names, column labels) with limited sensitivity to numerical magnitude. This is a deliberate design choice — testing whether a popular RAG technique transfers to this domain is part of the study's contribution.

**Decision: lazy build, not eager.** The embedding index is ~1000 rows × 384 dimensions — small enough to build in seconds. Building lazily means the cost is only paid when S3 is actually used.

### 6.4 `s4_summarization.py` — Summarisation-Augmented Context

**What it does:** Generates a natural language summary of a ticker's full history via LLM, then prepends it to the most recent 20 raw data rows.

**Context format:**
```
## Historical Summary for AAPL
{LLM-generated paragraph: price range, trend, avg volume, outliers}

## Recent Data (last 20 trading days)
{Markdown table of 20 rows}
```

**Parameters:**
- Recent window: 20 rows
- Summary scope: all historical data excluding the 20 recent rows
- Summary target length: ~150 words

**Caching:** Summaries are cached per ticker in a `dict[str, str]`. Generated once per ticker and reused across all queries about that ticker.

**Decision: cache per ticker, not per query.** The dataset is a fixed 1-year snapshot. Regenerating the summary per query would multiply LLM calls (one summary call per query) for no benefit, and introduce wording inconsistency between runs.

**Decision: exclude recent rows from summary input.** Avoids information duplication between the summary paragraph and the raw data table.

### 6.5 `s5_agentic.py` — Agentic Self-Correcting Routing (added Jul 2026)

**What it does:** Transforms context selection from a static rule into an iterative critique-and-fallback loop.

**Agentic loop:**
1. **Primary fetch:** Calls S2 Intent-Driven to get an initial context
2. **Critique:** Makes a fast LLM call asking: *"Does this data contain enough information to answer the query? (Yes/No)"*
3. **Decision:**
   - If **Yes** → proceeds with S2 context (no extra cost beyond the critique call)
   - If **No** → falls back to an expanded naive window (100 rows vs S1's 50)

**Test results (phi3:mini):**
- Recency query: Critique = "Yes" → 30 rows used, no fallback
- Anomaly query: Critique = "No" → fallback triggered, 11 rows → 100 rows

**Research question:** Do accuracy gains from self-correction justify the extra latency (1–2 additional LLM calls per query)?

**Decision: S2 as primary, not S1.** S2 scored highest in Run 1 (22.9%) and is query-aware. Starting with the best static strategy means fallback is only triggered when even the best context is insufficient.

**Decision: expanded S1 as fallback.** S3 scored lowest in Run 1 (6.3%) — switching to it would likely make things worse. Doubling the naive window is a safe, interpretable fallback.

**Decision: default to "sufficient" on unparseable critique.** If the LLM outputs something other than "Yes/No", `"no" not in response` defaults to True. Avoids unnecessary fallbacks on garbled output.

**Metadata logged:** `used_fallback` (bool), `critique_response` (raw text), `primary_rows` (S2 row count before fallback), `fallback_type` ("expanded_naive").

### 6.6 `s6_sql_agent.py` — Text-to-SQL Agent (SOTA Code Generation)

**What it does:** Bypasses the "linearization bottleneck" of injecting raw rows into the prompt. Instead of reading the tabular data directly, the LLM is given the `ohlcv` schema and asked to generate a SQL query. The system then deterministically executes the SQL against the SQLite `HistoricalStore`.

**Agentic flow:**
1. LLM receives the SQL schema and the user's natural language query.
2. LLM generates a SQL query.
3. The runner extracts the SQL and executes it via `pandas.read_sql_query` against the `store._conn`.
4. The result (or an empty DataFrame) is formatted and returned as the context.

**Fallback:** If the LLM produces invalid SQL syntax, or queries a non-existent column, a `sqlite3.Error` is caught. The agent logs the failure and safely falls back to the S1 Naive Window to ensure the evaluation loop does not crash.

**Why this is State-of-the-Art:** Research in 2024-2026 concluded that standard RAG techniques fail on dense numerical reasoning tasks. The SOTA approach is to use the LLM purely as a logical planner (generating code/SQL) and offload the actual math to a deterministic execution engine.

### 6.7 `strategies/registry.py` — Strategy Registry

Simple `dict[str, Strategy]` with register/lookup functions. The evaluation runner uses `get_all_strategies()` to iterate.

**Why a registry instead of a list:** Allows looking up strategies by name (useful in the Streamlit demo and for partial re-runs of specific strategies).

---

## 7. LLM Interaction

### 7.1 `llm/ollama_client.py`

**Purpose:** Single point of contact with Ollama's REST API.

**Key method:** `generate(prompt, system, model_override, temperature_override) → (response_text, latency_seconds)`

**Design decisions:**
- **Fixed decoding params.** Temperature (0.1), top_p (0.9), and seed (42) are set in config and applied to every call. This ensures the strategy, not sampling randomness, explains differences between runs.
- **model_override parameter.** Allows the judge to use a different model without creating a separate client instance.
- **Timeout: 120 seconds.** Generous enough for a 7B model on a T4 GPU. Prevents indefinite hangs.
- **No streaming.** `"stream": false` in the payload. We need the complete response for parsing and scoring.

**`is_available()`:** Checks both that Ollama is reachable and that the configured model is loaded. Used by the evaluation script to fail fast with a clear error message.

### 7.2 `llm/prompts.py`

**Four prompt template groups, all using `str.format()` with named placeholders:**

| Template | Used by | Placeholders |
|---|---|---|
| `ANALYST_SYSTEM_PROMPT` | Query-answering (all strategies) | None (static) |
| `QUERY_PROMPT` | Query-answering | `{context}`, `{query}` |
| `INTENT_CLASSIFIER_SYSTEM` + `_PROMPT` | S2 intent classification | `{query}` |
| `SUMMARISATION_SYSTEM` + `_PROMPT` | S4 summary generation | `{max_words}`, `{ticker}`, `{context}` |
| `JUDGE_SYSTEM` + `_PROMPT` | LLM-as-judge | `{query}`, `{ground_truth}`, `{response}` |
| `CRITIQUE_SYSTEM` + `_PROMPT` | S5 sufficiency check | `{query}`, `{context}` |
| `RAGAS_SYSTEM` + `_PROMPT` | RAGAS reverse-question generation | `{answer}` |

**Decision: str.format(), not f-strings.** Templates are static strings, not computed at import time. This makes them inspectable, loggable, and copyable into documentation.

**Decision: structured output enforced in the prompt.** The analyst prompt requires `Answer: <value>` on the final line. The judge prompt requires `Score: <integer>`. This makes parsing deterministic.

---

## 8. Evaluation Harness

### 8.1 `evaluation/benchmark.py`

**Purpose:** Load and validate `benchmarks/benchmark_queries.json`.

**JSON schema per query:**
```json
{
  "id": "recency_01",
  "category": "recency",
  "query": "What was AAPL's closing price on 2026-06-03?",
  "answer_type": "numeric",
  "ground_truth_fn": "aapl_close_2026_06_03",
  "tickers": ["AAPL"]
}
```

**Validation checks:**
- Required fields: `id`, `category`, `query`, `answer_type`, `ground_truth_fn`
- Valid categories: `recency`, `trend`, `anomaly`, `comparison`
- Valid answer types: `exact`, `numeric`, `directional`

**`BenchmarkQuery` dataclass:** Carries all query metadata through the pipeline. `ground_truth` field is populated at runtime by the runner.

**Current benchmark: 48 queries (12 per category)**

| Category | Count | Answer types |
|---|---|---|
| recency | 12 | 10 numeric, 2 directional |
| trend | 12 | 4 numeric, 8 directional |
| anomaly | 12 | 6 exact (dates), 6 numeric |
| comparison | 12 | 10 exact (ticker names), 2 directional |
| **Total** | **48** | **21 numeric, 11 directional, 16 exact** |

### 8.2 `evaluation/ground_truth.py`

**Purpose:** Compute correct answers programmatically from the actual data at evaluation time.

**Pattern:** Registry-based. Each function is decorated with `@register("name")` and takes the full DataFrame.

**Helper functions** (reduce repetition across 48 functions):
```python
_ticker(data, t)    # Filter + sort ascending by date
_latest(data, t)    # Most recent row for a ticker
_tail(data, t, n)   # Last n rows for a ticker
```

**Example:**
```python
@register("aapl_close_latest")
def _(data: pd.DataFrame) -> str:
    return f"{_latest(data, 'AAPL')['close']:.2f}"
```

The `ground_truth_fn` field in the benchmark JSON must match a registered name.

**Decision: computed at eval time, not hardcoded.** If you re-download data (e.g. wider date range), ground truth automatically updates. Hardcoded answers would silently become wrong.

**Decision: relative references, not absolute dates.** Queries use "most recent", "past month", "over the period" rather than specific calendar dates. This keeps the benchmark valid even if the download window shifts.

**To add a new query:** Write a `@register("name")` function in `ground_truth.py`, then add the matching JSON entry in `benchmark_queries.json`.

### 8.3 `evaluation/answer_parser.py`

**Purpose:** Extract the structured value from LLM free-text responses.

**Two functions:**
- `parse_answer(response) → str | None` — finds the last `Answer: <value>` line
- `parse_judge_score(response) → int | None` — finds the last `Score: <integer>` line (validated 1–5)

**Decision: search for the LAST match.** Models sometimes echo the format instruction. Taking the last occurrence gets the actual answer.

### 8.4 `evaluation/metrics.py`

**Purpose:** Score predicted answers against ground truth.

**Three scoring types:**

| `answer_type` | Scoring | Details |
|---|---|---|
| `exact` | String match | Case-insensitive, whitespace-trimmed |
| `numeric` | ±1% tolerance | Extracts first number from each string, checks `|pred - truth| / |truth| ≤ 0.01` |
| `directional` | Agreement | Classifies both into up/down using keyword sets, checks match |

**Returns:** `1.0` (correct) or `0.0` (incorrect). Binary scoring keeps analysis simple.

**Decision: ±1% not ±0.5%.** LLMs round numbers in their responses. 1% is forgiving enough to not penalise rounding while still catching genuinely wrong answers.

### 8.5 `evaluation/judge.py`

**Purpose:** Rate response quality on a 1–5 scale via a separate LLM call.

**Input:** The query, the LLM's response, and the ground truth answer.
**Output:** `(score, justification_text)`.

**Rubric (from strategy_specs.md):**

| Score | Meaning |
|---|---|
| 1 | Completely irrelevant or factually wrong |
| 2 | Partially relevant, missing key info or significant errors |
| 3 | Relevant, mostly correct, lacks depth or precision |
| 4 | Accurate, relevant, well-supported by data |
| 5 | Accurate, insightful, directly addresses query with context |

**Decision: same model as judge.** Using Mistral 7B for both answering and judging. This is standard practice (Zheng et al., 2023). Self-judging bias is acknowledged as a limitation in the paper.

### 8.6 `evaluation/results_store.py`

**Purpose:** Persist every evaluation run to SQLite for analysis.

**Schema:**
```
results(id, query_id, strategy_name, query_text, query_category,
        answer_type, ground_truth, full_prompt, llm_response,
        parsed_answer, accuracy, relevance_score, judge_response,
        ragas_score, latency_ms, context_rows, strategy_metadata, created_at)
```

**Key columns for analysis:**
- `accuracy` — 0.0 or 1.0
- `relevance_score` — 1–5 integer (LLM-as-judge, optional)
- `ragas_score` — 0.0–1.0 float (RAGAS answer relevancy, optional)
- `latency_ms` — end-to-end query time
- `full_prompt` — the exact prompt sent (for auditability)
- `strategy_metadata` — JSON blob with strategy-specific info (e.g. S5 `used_fallback`, S2 intent)

**`load_all()`** returns a DataFrame — plugs directly into the analysis notebook.

### 8.7 `evaluation/runner.py`

**Purpose:** The core experiment loop. Runs every query through every strategy.

**Flow per (query, strategy) pair:**
1. `strategy.select_context()` → context text
2. Format prompt with `QUERY_PROMPT.format(context=..., query=...)`
3. `client.generate()` → response + latency
4. `parse_answer()` → extracted answer
5. `score_accuracy()` → 0.0 or 1.0
6. `judge_response()` → 1–5 score (optional, `--skip-judge`)
7. `compute_ragas_score()` → 0–1 float (optional, `--skip-ragas`)
8. Save `ResultRow` to results database

**Decision: per-run error handling.** If one (query, strategy) pair fails, it logs the error and continues. A single failure doesn't abort the full sweep.

**Decision: ground truth computed per query, not pre-computed.** `compute_ground_truth()` is called once per query (not per strategy), so it's efficient but always uses the actual data.

---

## 9. Entry Scripts

### `scripts/01_fetch_data.py`

Downloads and caches stock data. Run once. Subsequent runs load from cache.

```bash
uv run python scripts/01_fetch_data.py
```

### `scripts/02_run_evaluation.py`

Runs the full evaluation sweep. Requires Ollama to be running.

```bash
uv run python scripts/02_run_evaluation.py              # Full run (judge + RAGAS)
uv run python scripts/02_run_evaluation.py --skip-judge  # No judge scoring
uv run python scripts/02_run_evaluation.py --skip-judge --skip-ragas  # Fastest
```

**Startup sequence:**
1. Fetch data (from cache)
2. Create buffer + store, replay all data
3. Create Ollama client, check availability
4. Register all 5 strategies (S1–S5)
5. Load benchmark queries
6. Run sweep, save results
7. Print summary statistics

### `scripts/03_analyze_results.py`

Generates paper-ready outputs from `data/db/results.sqlite`.

```bash
uv run python scripts/03_analyze_results.py
```

**Outputs:**
- **Table 1:** Overall mean accuracy, RAGAS relevancy, judge relevance, latency per strategy
- **Table 2:** Accuracy breakdown by (strategy × query category)
- **Table 3:** Accuracy breakdown by (strategy × answer type)
- **Figure 1:** Grouped bar chart — accuracy by category
- **Figure 2:** Box plot — LLM-as-judge relevance score distributions
- **Figure 3:** Box plot — RAGAS answer relevancy score distributions
- **Figure 4:** Bar chart — mean latency with error bars
- **Statistical tests:** Pairwise Wilcoxon signed-rank tests vs S1 baseline (accuracy, relevance, RAGAS)
- **CSV export:** `results/summary.csv`

All plots saved to `results/plots/`.

### `app/demo.py` — Streamlit Demo

Interactive UI for querying with any strategy. Works partially without Ollama (S1/S3 show context, S2/S4 need LLM).

```bash
uv run streamlit run app/demo.py
```

**Features:**
- Strategy selector (sidebar dropdown)
- Data overview (sidebar metrics)
- Sample query buttons
- Side-by-side view: selected context + LLM response
- Strategy metadata display
- Full prompt preview

---

## 10. Data Flow Diagram

```
yfinance API
    │
    ▼
fetch_data.py ──► CSV cache (data/raw/*.csv)
    │
    ▼
replay_engine.py
    │
    ├──► StreamBuffer (in-memory deque, last 250 rows)
    │
    └──► HistoricalStore (SQLite, full ~1000 rows)
              │
    ┌─────────┼─────────────────────────┬──────────────────┐
    │         │                         │                  │
    ▼         ▼                         ▼                  ▼
 Strategy  Strategy                  Strategy          Strategy
 reads     reads                     reads             (S5: runs S2
 buffer    store                     both              + critique
  (S1)     (S2-ANOMALY, S3)          (S2-TREND, S4)    + fallback)
    │         │                         │                  │
    └─────────┼─────────────────────────┴──────────────────┘
              │
              ▼
        ContextResult.context_text
              │
              ▼
        QUERY_PROMPT.format(context=..., query=...)
              │
              ▼
        OllamaClient.generate()
              │
              ▼
        LLM response text
              │
    ┌─────────┼──────────┬─────────────────┐
    ▼         ▼          ▼                 ▼
 parse     score      judge            RAGAS
 answer    accuracy   relevance        answer
            (0/1)     (1-5, opt)       relevancy
    │         │          │             (0-1, opt)
    └─────────┼──────────┴─────────────────┘
              ▼
        ResultRow → results.sqlite
```

---

## 11. Key Design Decisions (Summary)

| Decision | Rationale |
|---|---|
| Frozen dataclasses for config | Prevent accidental mutation mid-experiment |
| CSV cache for yfinance data | Human-readable, debuggable at ~1000 rows |
| deque buffer + SQLite store | Buffer for recency, SQLite for indexed lookback |
| Single `Strategy.select_context()` interface | Makes all 5 strategies interchangeable in the evaluation loop |
| Strategies receive both buffer and store | Uniform interface; each strategy uses what it needs |
| Config passed per-call, not at init | Supports sensitivity analysis with different configs |
| Fixed low temperature (0.1) | Strategy, not randomness, should explain differences |
| `Answer: <value>` enforced in prompt | Enables deterministic answer parsing |
| Ground truth computed at eval time | Stays correct if data changes |
| Same model for answering and judging | Standard practice; bias acknowledged as limitation |
| Per-run error handling in the sweep | One failure doesn't abort 240 runs |
| Summaries cached per ticker (S4) | Avoids redundant LLM calls; consistent wording across queries |
| Embedding index built lazily (S3) | Cost only paid when S3 is actually used |
| No concurrency in buffer | Sequential replay; research pipeline, not production |
| S5 defaults to "sufficient" on bad critique output | Avoids unnecessary fallback on unparseable responses |
| S2 as S5 primary, expanded-S1 as fallback | S2 is best static strategy; S3 scored worst so not used as fallback |
| RAGAS alongside 1–5 judge (not replacing it) | Allows paper to compare objective vs subjective metrics |

---

## 12. How to Extend

### Add a new benchmark query

1. Write a `@register("name")` function in `ground_truth.py`
2. Add a JSON entry in `benchmark_queries.json` with `ground_truth_fn` matching the name
3. Valid categories: `recency`, `trend`, `anomaly`, `comparison`
4. Valid answer types: `exact`, `numeric`, `directional`

### Change the LLM model

Edit `config.yaml`:
```yaml
llm:
  model: "mistral"   # or "llama3.2", "phi3:mini", etc.
```
Then pull it: `ollama pull mistral`

### Add a new ticker

1. Add to `config.yaml → data.tickers`
2. Add to `_KNOWN_TICKERS` in `s2_intent_driven.py` and `s4_summarization.py`
3. Delete cached CSVs in `data/raw/` to trigger re-download
4. Add benchmark queries for the new ticker

### Add a fifth strategy

S5 (`s5_agentic.py`) already exists. To add a sixth:

1. Create `strategies/s6_your_strategy.py` implementing `Strategy` ABC
2. Register in `scripts/02_run_evaluation.py`: `register_strategy(YourStrategy())`
3. No other code changes needed — the runner iterates all registered strategies

### Run on Colab/Kaggle

```python
!curl -fsSL https://ollama.com/install.sh | sh
!nohup ollama serve &
!ollama pull mistral
```
Then run the same scripts. Change `config.yaml → llm.model` to `"mistral"`.

---

## 13. Dependencies

| Package | Version | Purpose |
|---|---|---|
| pandas | ≥2.0 | DataFrames everywhere |
| numpy | ≥1.24 | S3 cosine similarity |
| yfinance | ≥0.2.31 | Stock data download |
| sentence-transformers | ≥2.2 | S3 embeddings (loads torch) |
| streamlit | ≥1.28 | Demo UI (not yet built) |
| requests | ≥2.31 | Ollama REST calls |
| scipy | ≥1.11 | Wilcoxon tests in analysis |
| matplotlib | ≥3.7 | Result plots |
| pyyaml | ≥6.0 | Config loading |
| pytest | ≥7.0 | Dev-only, smoke tests |

**Total installed packages:** 93 (including transitive dependencies like torch, transformers, etc.)

**Zero-cost stack:** Everything is open-source and runs locally or on free Colab/Kaggle.

---

## 14. Empirical Findings

### Run 1: `phi3:mini` Baseline (Early Jul 2026)
A baseline run was executed using the `phi3:mini` model (a highly capable but small 3.8B parameter model). This run established the baseline performance of the four initial strategies before Mistral 7B was evaluated.

**Key results:**
- Overall accuracy was remarkably low across all strategies (12% to 22%), largely due to the small model struggling with numerical reasoning and strictly formatted output.
- **S2 Intent-Driven** was the most accurate (22.9%), significantly outperforming the S1 baseline (12.5%).
- **S3 Embedding Retrieval** performed worse than random/naive (6.3%), confirming the hypothesis that sentence-transformers tuned on natural language struggle to differentiate numeric tabular OHLCV data.
- **S4 Summarisation** provided no accuracy benefit (12.5%, identical to S1).

### Run 2: Mistral 7B & Agentic Improvements (Jul 21, 2026)
The pipeline was upgraded to **Mistral 7B** and evaluated with **RAGAS Answer Relevancy** and a new **S5 Agentic Routing** strategy.

**Key results:**
- Mistral 7B achieved a much higher floor, with all text-injection strategies (S1, S2, S4, S5) hitting a performance ceiling of roughly **33.3% to 35.4%** accuracy.
- **S5 Agentic Routing** tied with S1 in strict binary accuracy (33.3%), but its RAGAS Answer Relevancy score (0.430) was a statistically significant improvement over S1 (0.355), proving the critique loop retrieves mathematically *better* data, even if Mistral still failed the final numerical calculation step.
- The 35% accuracy ceiling across all text-injection strategies provided strong empirical evidence that injecting raw tabular data into an LLM context window is fundamentally flawed for math-heavy financial reasoning.

**Evolution to SOTA:** The realization that text-injection RAG hits a hard accuracy ceiling directly motivated the creation of **S6 (Text-to-SQL Agent)**. By shifting the LLM's role from doing math on text tables to writing executable SQL code, the architecture mirrors 2026 SOTA RAG methodologies.
