# Strategy Specifications

**Purpose:** Pin the exact definitions of all four context-feeding strategies *before* writing code. This document is the single source of truth for what each strategy does, its parameters, and how its output is formatted. Everything downstream — prompts, metrics, the methodology section of the paper — refers back to this.

**Last updated:** 14 Jul 2026

---

## Shared Conventions

| Parameter | Value | Source |
|---|---|---|
| **Buffer size (N)** | 250 rows | `config.yaml → buffer.max_size` |
| **Context row ordering** | Ascending (chronological, oldest → newest) | All strategies |
| **Context format** | Markdown table with columns: `Date | Ticker | Open | High | Low | Close | Volume` | `formatting/markdown_table.py` |
| **LLM decoding** | `temperature: 0.1`, `top_p: 0.9`, `seed: 42` (if supported) | Fixed across all strategies for reproducibility |
| **Answer format** | Prompt requires model to end with `Answer: <value>` on a final line | `prompts.py` |

---

## S1 — Naive Fixed Window (Baseline)

### Description
Selects the most recent N rows from the stream buffer with no awareness of the query content. This is the control condition — the simplest possible approach.

### Parameters
| Parameter | Value |
|---|---|
| Window size | N = **50 rows** (most recent from buffer) |
| Query awareness | None |
| Data source | Stream buffer only |

### Selection Logic
1. Read the last 50 rows from the stream buffer (or all rows if buffer contains fewer than 50)
2. Sort ascending by timestamp
3. Format as Markdown table
4. No filtering by ticker — all tickers in the window are included

### Known Limitation
For queries about specific tickers, the context may contain irrelevant rows from other tickers, wasting token budget.

---

## S2 — Query-Intent-Driven Windowing

### Description
First classifies the query into one of four intent categories via a zero-shot LLM call, then dispatches to a specialised sub-strategy that selects contextually appropriate data.

### Intent Classification
- **Classifier:** Zero-shot LLM prompt (same model) that returns exactly one of: `RECENCY`, `TREND`, `ANOMALY`, `COMPARISON`
- **Prompt template:** Defined in `prompts.py → INTENT_CLASSIFIER_PROMPT`
- **Fallback:** If the classifier output doesn't match any category, default to `RECENCY`

### Sub-Strategy Parameters

#### RECENCY
| Parameter | Value |
|---|---|
| Window size | **30 rows** for the queried ticker |
| Data source | Stream buffer, filtered by ticker |

**Logic:** Filter buffer for the queried ticker(s), take the last 30 rows, sort ascending.

#### TREND
| Parameter | Value |
|---|---|
| Recent window | Last **5 rows** for the queried ticker |
| Historical samples | **3 rows** at each of 30, 60, and 90 calendar days back |
| Total rows | ~14 rows |
| Data source | Stream buffer (recent) + historical store (lookback samples) |

**Logic:**
1. Last 5 rows from buffer for the queried ticker
2. Query historical store for 3 rows closest to each of {T-30d, T-60d, T-90d}
3. Combine and sort ascending by timestamp
4. Format as Markdown table with a `---` separator between the recent block and historical samples

#### ANOMALY
| Parameter | Value |
|---|---|
| Detection method | Row with the maximum absolute daily return for the queried ticker in the historical store |
| Context window | **k_before = 10** rows before the anomaly, **k_after = 5** rows after |
| Total rows | ~16 rows (10 + 1 anomaly + 5) |
| Data source | Historical store |

**Logic:**
1. Compute daily returns for the queried ticker across the full historical store
2. Identify the row with the largest absolute return
3. Retrieve k_before=10 rows before and k_after=5 rows after that row
4. Sort ascending, mark the anomaly row with a `>>> ` prefix in the Markdown table

#### COMPARISON
| Parameter | Value |
|---|---|
| Per-ticker window | Last **20 rows** per queried ticker |
| Data source | Stream buffer, filtered per ticker |

**Logic:**
1. Identify all tickers mentioned in the query (or all 4 tickers if query says "all" / "which ticker")
2. For each ticker, take the last 20 rows from the buffer
3. Interleave chronologically (sort all rows by timestamp ascending)
4. Format as single Markdown table

### Known Limitation
The intent classification step adds one extra LLM call per query, increasing latency. Misclassification routes the query to the wrong sub-strategy.

---

## S3 — Embedding-Based Retrieval

### Description
Encodes each data row as a text string and embeds it using a sentence transformer. At query time, embeds the query and retrieves the top-k most similar rows by cosine similarity.

### Parameters
| Parameter | Value |
|---|---|
| Embedding model | `all-MiniLM-L6-v2` (via `sentence-transformers`) |
| Top-k | **20 rows** |
| Similarity metric | Cosine similarity |
| Data source | Full historical store (all rows, all tickers) |
| Row text template | `"{ticker} | {date} | Open: {open} | High: {high} | Low: {low} | Close: {close} | Volume: {volume}"` |

### Selection Logic
1. At evaluation startup, encode all rows in the historical store using the row text template → embedding matrix (computed once, cached in memory)
2. At query time, encode the query string → query embedding
3. Compute cosine similarity between query embedding and all row embeddings
4. Select top-20 rows by similarity score
5. Sort selected rows ascending by timestamp (not by similarity score)
6. Format as Markdown table

### Known Limitation
`all-MiniLM-L6-v2` is designed for semantic similarity over natural language text, not numerical OHLCV rows. The row-to-text conversion produces embeddings dominated by textual structure (ticker names, column labels) with limited sensitivity to numerical magnitude or temporal ordering. **This is a known design-scope limitation** — testing whether a popular RAG technique transfers to numerical streaming data is part of the study's contribution.

---

## S4 — Summarisation-Augmented Context

### Description
Uses the LLM to generate a natural language summary of older historical data for a ticker, then prepends that summary to a window of recent rows. Combines breadth (summary of long history) with recency (recent raw data).

### Parameters
| Parameter | Value |
|---|---|
| Recent window (M) | Last **20 rows** for the queried ticker |
| Summary scope | Full historical data for the queried ticker (excluding the M recent rows) |
| Summary generation | One LLM call per ticker, **cached for the entire evaluation** |
| Summary prompt | Defined in `prompts.py → SUMMARIZATION_PROMPT` |
| Data source | Stream buffer (recent) + historical store (summary source) |

### Selection Logic
1. Check if a cached summary exists for the queried ticker
   - If not: query historical store for all rows of that ticker (excluding last M), format as Markdown table, send to LLM with summarisation prompt, cache the returned summary
2. Take the last M=20 rows for the queried ticker from the buffer
3. Assemble context as:
   ```
   ## Historical Summary for {TICKER}
   {cached_summary}

   ## Recent Data (last 20 trading days)
   {markdown_table_of_M_rows}
   ```
4. This assembled text becomes the context block in the final prompt

### Summary Prompt Guidelines
The summarisation prompt should instruct the LLM to produce a concise paragraph covering:
- Overall price range (min/max close) over the period
- General trend direction (upward, downward, sideways)
- Average daily volume
- Any notable outlier days (if obvious)

Target summary length: ~100–150 words.

### Known Limitation
Summary quality depends on the LLM's ability to extract statistical patterns from a large Markdown table. Summary is generated once and reused — it cannot adapt to query-specific time ranges within the historical period.

---

## S5 — Agentic Routing (Self-Correcting)

### Description
Uses S2 as a primary router, but wraps it in an orchestrator. After the S2 context is selected, the agent explicitly "critiques" the retrieved data to ensure it is sufficient to answer the query. If it is not (e.g., query asks for a specific date that isn't in the S2 window), it falls back to an expanded naive window.

### Parameters
| Parameter | Value |
|---|---|
| Primary Router | S2 Intent-Driven Strategy |
| Fallback Strategy | Expanded S1 Naive Window (N=100) |
| Critique Model | Zero-shot LLM call (same model) |
| Data source | S2 context + Historical store (for fallback) |

### Selection Logic
1. Call S2 to get initial context text.
2. Prompt LLM to critique whether the context contains the data needed for the query.
3. If LLM replies "SUFFICIENT", return S2 context.
4. If LLM replies "INSUFFICIENT", trigger fallback.
5. Fallback retrieves the last 100 rows from the historical store for the queried ticker (or all if no specific ticker).
6. Format fallback as a Markdown table and return.

### Known Limitation
Adds an extra LLM call (critique) per query, doubling latency over S2.

---

## S6 — Text-to-SQL Agent (State-of-the-Art)

### Description
Bypasses the "linearization bottleneck" entirely. Instead of injecting raw data rows into the LLM prompt, the LLM is given the database schema and asked to write a native SQL query. The system executes the SQL and returns the deterministic result.

### Parameters
| Parameter | Value |
|---|---|
| Generation Model | Zero-shot LLM call (Text-to-SQL prompt) |
| Fallback Strategy | S1 Naive Window (N=50) |
| Data source | Direct SQLite execution on Historical store |

### Selection Logic
1. Provide the LLM with the exact `ohlcv` SQLite schema and the natural language query.
2. Extract the generated SQL query from the LLM's response.
3. Execute the SQL against the local SQLite database.
4. If successful, return the formatted SQL query output.
5. If invalid SQL or execution fails, silently catch the exception, log the failure, and fall back to S1 Naive Window to ensure stability.

### Known Limitation
Requires high logical/coding capability from the LLM to generate valid SQL syntax. Struggles slightly if the schema doesn't match the LLM's pre-training assumptions.

---

## LLM-as-Judge Relevance Rubric

The judge is a separate LLM call (same model: Mistral 7B on Colab) that rates each response on a 1–5 scale. The judge prompt includes the original query, the LLM's response, and the ground-truth answer.

| Score | Meaning |
|---|---|
| **1** | Completely irrelevant or factually wrong |
| **2** | Partially relevant but missing key information or contains significant errors |
| **3** | Relevant and mostly correct, but lacks depth or precision |
| **4** | Accurate, relevant, and well-supported by the data |
| **5** | Accurate, insightful, and directly addresses the query with appropriate context |

**Judge prompt template:** Defined in `prompts.py → JUDGE_PROMPT`. The prompt instructs the judge to return only a single integer (1–5) on its final line, preceded by a brief justification.

**Known limitation:** Using the same model as both answerer and judge introduces potential self-judging bias. This is acknowledged in the paper's limitations section. Mitigated partially by low temperature and structured output format.

---

## Accuracy Scoring Types

Defined in `evaluation/metrics.py`:

| Query Type | Scoring Method | Details |
|---|---|---|
| **Categorical** (ticker, date, yes/no) | Exact string match | Case-insensitive, whitespace-trimmed |
| **Continuous** (price, volume, percentage) | Tolerance-based | ±1% relative tolerance: `abs(predicted - actual) / abs(actual) <= 0.01` |
| **Directional** (trend up/down) | Agreement | Both predicted and actual must indicate the same direction |

---

## Context Token Budget Estimates

Rough estimates to confirm all strategies fit within Mistral 7B's 8K context window:

| Component | Est. Tokens |
|---|---|
| System prompt | ~200 |
| Context (50 rows Markdown table, worst case) | ~1,500 |
| User query | ~30–50 |
| Reserved for response | ~500 |
| **Total** | **~2,250 / 8,192** |

All strategies are well within budget. S4 (summary + 20 rows) may be slightly larger due to the summary paragraph, but still comfortably under 3,000 tokens total.
