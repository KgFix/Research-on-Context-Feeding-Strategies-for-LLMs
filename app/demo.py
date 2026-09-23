"""Streamlit demo: interactive querying with strategy selection.

Run:
    uv run streamlit run app/demo.py

Lets you type a natural language query, choose a strategy, and see:
- The context selected by that strategy
- The LLM response
- The accuracy score (if a matching benchmark query exists)
"""

import sys
sys.path.insert(0, "src")

import streamlit as st
import pandas as pd

from ctxfeed.config import Config
from ctxfeed.ingestion.fetch_data import fetch_stock_data
from ctxfeed.ingestion.replay_engine import replay_all
from ctxfeed.storage.stream_buffer import StreamBuffer
from ctxfeed.storage.historical_store import HistoricalStore
from ctxfeed.strategies.s1_naive_window import NaiveWindowStrategy
from ctxfeed.strategies.s2_intent_driven import IntentDrivenStrategy
from ctxfeed.strategies.s3_embedding_retrieval import EmbeddingRetrievalStrategy
from ctxfeed.strategies.s4_summarization import SummarisationStrategy
from ctxfeed.strategies.s5_agentic import AgenticStrategy
from ctxfeed.strategies.s6_sql_agent import SqlAgentStrategy
from ctxfeed.llm.ollama_client import OllamaClient
from ctxfeed.llm.prompts import ANALYST_SYSTEM_PROMPT, QUERY_PROMPT
from ctxfeed.evaluation.answer_parser import parse_answer


# ── Session state setup ──────────────────────────────────────


@st.cache_resource
def init_pipeline():
    """Load config, fetch data, build buffer and store. Cached across reruns."""
    config = Config.from_yaml("config/config.yaml")
    data = fetch_stock_data(
        config.data.tickers, config.data.start_date,
        config.data.end_date, config.data.cache_dir,
    )
    buffer = StreamBuffer(max_size=config.buffer.max_size)
    store = HistoricalStore(config.storage.db_path)
    replay_all(data, buffer, store)
    return config, data, buffer, store


@st.cache_resource
def init_strategies(_config, _buffer, _store):
    """Instantiate all strategies. Cached across reruns."""
    client = OllamaClient(_config.llm)
    strategies = {
        "S1 — Naive Window": NaiveWindowStrategy(),
        "S2 — Intent-Driven": IntentDrivenStrategy(llm_client=client),
        "S3 — Embedding Retrieval": EmbeddingRetrievalStrategy(),
        "S4 — Summarisation": SummarisationStrategy(llm_client=client),
        "S5 — Agentic Routing": AgenticStrategy(client=client),
        "S6 — Text-to-SQL Agent (SOTA)": SqlAgentStrategy(llm_client=client),
    }
    return strategies, client


# ── Page config ──────────────────────────────────────────────

st.set_page_config(
    page_title="Context-Feed Evaluation Demo",
    page_icon="📊",
    layout="wide",
)

st.title("📊 Context-Feed Strategy Demo")
st.caption(
    "Compare how different context-feeding strategies select data "
    "for LLM-based querying over streaming stock data."
)

# ── Custom CSS to prevent table column overflow & overlap ─────
st.markdown(
    """
    <style>
    /* Prevent flex column items from overflowing horizontally into adjacent columns */
    [data-testid="column"] {
        min-width: 0 !important;
    }
    
    /* Make Markdown tables horizontally scrollable when wider than container */
    .stMarkdown table {
        display: block !important;
        overflow-x: auto !important;
        max-width: 100% !important;
        white-space: nowrap !important;
        border-collapse: collapse !important;
    }
    
    .stMarkdown th, .stMarkdown td {
        padding: 6px 12px !important;
    }
    
    /* Sleek scrollbar for the table */
    .stMarkdown table::-webkit-scrollbar {
        height: 6px;
    }
    .stMarkdown table::-webkit-scrollbar-thumb {
        background: #4a5568;
        border-radius: 3px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ── Initialise ───────────────────────────────────────────────

config, data, buffer, store = init_pipeline()

# Check Ollama availability
client = OllamaClient(config.llm)
ollama_available = client.is_available()

if not ollama_available:
    st.warning(
        f"⚠️ Ollama not available at {config.llm.base_url} "
        f"with model '{config.llm.model}'. "
        f"S2, S4, S5, and S6 require Ollama. S1 and S3 work without it."
    )

strategies, client = init_strategies(config, buffer, store)

# ── Sidebar: data overview ───────────────────────────────────

with st.sidebar:
    st.header("Data Overview")
    st.metric("Total rows", len(data))
    st.metric("Tickers", ", ".join(config.data.tickers))
    st.metric("Date range", f"{data['date'].min()} — {data['date'].max()}")
    st.metric("Buffer size", f"{buffer.size}/{buffer.max_size}")

    st.divider()
    st.header("Strategy Selection")
    selected_strategy = st.selectbox(
        "Choose a strategy:",
        list(strategies.keys()),
    )

    st.divider()
    st.header("Sample Queries")
    sample_queries = [
        "What was AAPL's most recent closing price?",
        "Which ticker had the highest average volume?",
        "Has MSFT been trending upward over the past month?",
        "On which day did TSLA have its largest single-day drop?",
        "What was the highest closing price across all stocks?",
        "Compare the average volume of AAPL and MSFT.",
    ]
    for sq in sample_queries:
        if st.button(sq, key=sq):
            st.session_state["query_input"] = sq

# ── Main area: query input ───────────────────────────────────

query = st.text_input(
    "Enter your query:",
    value=st.session_state.get("query_input", ""),
    placeholder="e.g. What was AAPL's most recent closing price?",
)

if query:
    strategy = strategies[selected_strategy]

    # Show context selection with comfortable column spacing
    col1, col2 = st.columns([1, 1], gap="large")

    with col1:
        st.subheader("📋 Selected Context")
        with st.spinner("Selecting context..."):
            ctx = strategy.select_context(query, buffer, store, config)

        st.metric("Rows selected", ctx.rows_selected)

        # Strategy-specific UI enhancements
        if strategy.name == "s6_sql_agent":
            gen_sql = ctx.metadata.get("generated_sql")
            if gen_sql:
                st.markdown("##### ⚡ Generated SQL Query")
                st.code(gen_sql, language="sql")
            if ctx.metadata.get("used_fallback"):
                st.warning(f"⚠️ SQL execution failed ({ctx.metadata.get('sql_error')}). Fell back to S1 Naive Window.")
            else:
                st.success("✅ Deterministic SQL executed directly against SQLite.")
        elif strategy.name == "s5_agentic":
            if ctx.metadata.get("used_fallback"):
                st.warning("⚠️ Critique evaluated context as INSUFFICIENT — triggered expanded 100-row window.")
            else:
                st.success("✅ Critique evaluated S2 context as SUFFICIENT.")

        if ctx.metadata:
            with st.expander("Strategy metadata"):
                st.json(ctx.metadata)

        st.markdown(ctx.context_text)

    with col2:
        st.subheader("🤖 LLM Response")

        llm_required_strategies = ("s2_intent", "s4_summary", "s5_agentic", "s6_sql_agent")
        if not ollama_available and strategy.name in llm_required_strategies:
            st.error("This strategy requires Ollama to be running.")
        elif not ollama_available:
            # S1 and S3 can show context without LLM
            st.info(
                "Ollama not available — showing context only. "
                "Start Ollama to get LLM responses."
            )
            # Show what the prompt would look like
            full_prompt = QUERY_PROMPT.format(context=ctx.context_text, query=query)
            with st.expander("Full prompt (would be sent to LLM)"):
                st.code(full_prompt, language="markdown")
        else:
            with st.spinner("Querying LLM..."):
                full_prompt = QUERY_PROMPT.format(
                    context=ctx.context_text, query=query
                )
                response, latency = client.generate(
                    full_prompt, system=ANALYST_SYSTEM_PROMPT
                )

            st.metric("Latency", f"{latency*1000:.0f} ms")

            parsed = parse_answer(response)
            if parsed:
                st.success(f"**Answer:** {parsed}")

            with st.expander("Full response"):
                st.markdown(response)

            with st.expander("Full prompt"):
                st.code(full_prompt, language="markdown")

# ── Footer ───────────────────────────────────────────────────

st.divider()
st.caption(
    "Context-Feed Evaluation Pipeline — "
    "UKZN CS Honours 2026 — Kirone Gopaul"
)
