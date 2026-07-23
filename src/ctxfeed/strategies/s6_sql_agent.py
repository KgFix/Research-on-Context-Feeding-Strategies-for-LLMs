import logging
import re
import pandas as pd
from typing import Any

from ctxfeed.config import Config
from ctxfeed.storage.historical_store import HistoricalStore
from ctxfeed.storage.stream_buffer import StreamBuffer
from ctxfeed.strategies.base import Strategy, ContextResult
from ctxfeed.llm.ollama_client import OllamaClient
from ctxfeed.strategies.s1_naive_window import NaiveWindowStrategy

logger = logging.getLogger(__name__)

SQL_AGENT_PROMPT = """You are a highly skilled SQL data analyst.
Your task is to translate a natural language question into a precise SQL query.

The database contains a single table named `ohlcv` with the following schema:
- ticker (TEXT)
- date (TEXT, YYYY-MM-DD)
- open (REAL)
- high (REAL)
- low (REAL)
- close (REAL)
- volume (INTEGER)

Respond ONLY with the SQL query. Do NOT include markdown blocks, explanations, or any other text. 
Ensure the query answers the user's question precisely.

Example question: "Which ticker had the highest closing price on 2023-01-05?"
Example response: SELECT ticker FROM ohlcv WHERE date = '2023-01-05' ORDER BY close DESC LIMIT 1;

Question: "{query}"
"""

class SqlAgentStrategy(Strategy):
    """S6: State-of-the-art Agentic Text-to-SQL.
    
    Translates the query into a SQL query, executes it against the HistoricalStore,
    and returns the SQL execution results as the context.
    If SQL generation or execution fails, it falls back to S1 Naive Window.
    """

    def __init__(self, llm_client: OllamaClient) -> None:
        self.llm_client = llm_client
        self.fallback_strategy = NaiveWindowStrategy()

    @property
    def name(self) -> str:
        return "s6_sql_agent"

    def _extract_sql(self, llm_response: str) -> str:
        """Extract the SQL query from the LLM response, stripping markdown/text."""
        # Remove ```sql ... ``` if present
        match = re.search(r"```(?:sql)?\s*(.*?)\s*```", llm_response, re.DOTALL | re.IGNORECASE)
        if match:
            return match.group(1).strip()
        
        # Otherwise, clean up typical unwanted prefixes
        cleaned = llm_response.strip()
        if cleaned.lower().startswith("sql"):
            cleaned = cleaned[3:].strip()
        
        # Try to find the first SELECT and stop at semicolon
        select_idx = cleaned.upper().find("SELECT")
        if select_idx != -1:
            cleaned = cleaned[select_idx:]
            semi_idx = cleaned.find(";")
            if semi_idx != -1:
                cleaned = cleaned[:semi_idx+1]
        
        return cleaned.strip()

    def select_context(
        self,
        query: str,
        buffer: StreamBuffer,
        store: HistoricalStore,
        config: Config,
    ) -> ContextResult:
        prompt = SQL_AGENT_PROMPT.format(query=query)
        
        metadata: dict[str, Any] = {"used_fallback": False, "generated_sql": None, "sql_error": None}
        
        try:
            # 1. Generate SQL
            llm_response, _ = self.llm_client.generate(prompt)
            sql_query = self._extract_sql(llm_response)
            metadata["generated_sql"] = sql_query
            
            logger.info(f"S6 generated SQL: {sql_query}")
            
            if not sql_query.upper().startswith("SELECT"):
                raise ValueError(f"Invalid SQL generated: {sql_query}")

            # 2. Execute SQL
            df = pd.read_sql_query(sql_query, store._conn)
            
            # If the dataframe is empty, the LLM might have queried non-existent dates.
            # But technically it's a valid SQL execution. We'll pass the empty df.
            
            # 3. Format result
            context_text = f"--- DATABASE SQL QUERY RESULT ---\n"
            context_text += f"Query Executed: {sql_query}\n\n"
            if df.empty:
                context_text += "Result: [Empty - No data matched the query]"
            else:
                context_text += df.to_markdown(index=False)
                
            logger.info(f"S6 executed SQL successfully, returning {len(df)} rows.")
            
            return ContextResult(
                context_text=context_text,
                rows_selected=len(df),
                metadata=metadata
            )
            
        except Exception as e:
            # Catch LLM failure, SQL parsing failure, or execution failure
            logger.warning(f"S6 SQL failed ({str(e)}) - falling back to S1 Naive Window.")
            metadata["used_fallback"] = True
            metadata["sql_error"] = str(e)
            
            fallback_result = self.fallback_strategy.select_context(query, buffer, store, config)
            # Merge metadata
            fallback_result.metadata.update(metadata)
            return fallback_result
