# s3: embedding-based retrieval strategy
# turns each ohlcv row into text, embeds with sentence-transformers, takes top-k by cosine similarity
# note: all-MiniLM was trained on text not financial numbers, so similarity matches tickers/labels better than numbers

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd

from ctxfeed.config import Config
from ctxfeed.formatting.markdown_table import df_to_markdown, row_to_text
from ctxfeed.storage.historical_store import HistoricalStore
from ctxfeed.storage.stream_buffer import StreamBuffer
from ctxfeed.strategies.base import ContextResult, Strategy

logger = logging.getLogger(__name__)


class EmbeddingRetrievalStrategy(Strategy):
    def __init__(self) -> None:
        self._model = None
        self._embeddings: Optional[np.ndarray] = None
        self._rows: Optional[pd.DataFrame] = None

    @property
    def name(self) -> str:
        return "s3_embedding"

    def select_context(
        self,
        query: str,
        buffer: StreamBuffer,
        store: HistoricalStore,
        config: Config,
    ) -> ContextResult:
        # embed query and pull top-k nearest rows
        cfg = config.strategies.s3_embedding


        # Build index on first call
        if self._embeddings is None:
            self._build_index(store, cfg.model_name)

        # Embed the query
        query_embedding = self._model.encode([query])[0]

        # Cosine similarity
        similarities = _cosine_similarity(query_embedding, self._embeddings)

        # Select top-k indices
        top_k = min(cfg.top_k, len(similarities))
        top_indices = np.argsort(similarities)[-top_k:][::-1]

        # Get selected rows, sorted chronologically
        selected = self._rows.iloc[top_indices].copy()
        selected = selected.sort_values("date").reset_index(drop=True)

        # Collect similarity scores for metadata
        top_scores = similarities[top_indices].tolist()

        return ContextResult(
            context_text=df_to_markdown(selected),
            rows_selected=len(selected),
            metadata={
                "top_k": top_k,
                "similarity_scores": top_scores,
                "model_name": cfg.model_name,
            },
        )

    def _build_index(self, store: HistoricalStore, model_name: str) -> None:
        # build embedding index from sqlite history (cached in memory)
        from sentence_transformers import SentenceTransformer

        logger.info("Building embedding index with %s...", model_name)

        self._model = SentenceTransformer(model_name)
        self._rows = store.query()  # All rows

        if self._rows.empty:
            self._embeddings = np.array([])
            logger.warning("Historical store is empty: no embeddings built")
            return

        # Convert each row to text
        texts = [row_to_text(row) for _, row in self._rows.iterrows()]

        # Encode all texts
        self._embeddings = self._model.encode(texts, show_progress_bar=True)

        logger.info(
            "Embedding index built: %d rows, shape %s",
            len(texts),
            self._embeddings.shape,
        )


def _cosine_similarity(query_vec: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    # dot product of l2-normalized embeddings
    if matrix.size == 0:
        return np.array([])

    query_norm = query_vec / (np.linalg.norm(query_vec) + 1e-10)
    matrix_norms = np.linalg.norm(matrix, axis=1, keepdims=True) + 1e-10
    matrix_normed = matrix / matrix_norms

    return matrix_normed @ query_norm

