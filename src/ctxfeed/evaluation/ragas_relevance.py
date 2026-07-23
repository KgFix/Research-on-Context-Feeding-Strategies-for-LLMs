"""RAGAS-style Answer Relevancy scoring.

Computes an objective relevance metric by reverse-engineering the
generated answer: if the answer is relevant to the query, then
questions derived from the answer should semantically align with
the original query.

Algorithm (per RAGAS framework):
    1. LLM generates 3 questions the answer appears to address
    2. Embed the original query and the 3 generated questions
    3. Score = mean cosine similarity between original and generated

Output: a float in [0, 1] where 1 = perfectly relevant.

This replaces subjective LLM self-judging with a deterministic,
mathematically defensible metric.

Reference: Es et al., "RAGAS: Automated Evaluation of Retrieval
Augmented Generation", arXiv:2309.15217, 2023.
"""

from __future__ import annotations

import logging
import re
from typing import Optional

import numpy as np
from sentence_transformers import SentenceTransformer

from ctxfeed.llm.ollama_client import OllamaClient
from ctxfeed.llm.prompts import RAGAS_PROMPT, RAGAS_SYSTEM

logger = logging.getLogger(__name__)

# Shared embedding model (same as S3 uses)
_model: Optional[SentenceTransformer] = None


def _get_embedding_model() -> SentenceTransformer:
    """Lazy-load the sentence transformer model (shared singleton)."""
    global _model
    if _model is None:
        logger.info("Loading embedding model for RAGAS scoring...")
        _model = SentenceTransformer("all-MiniLM-L6-v2")
    return _model


def _parse_questions(response: str) -> list[str]:
    """Extract numbered questions from the LLM response.

    Handles formats like:
        1. What was the closing price?
        1) What was the closing price?
        1: What was the closing price?
    """
    # Match lines starting with a number followed by . or ) or :
    pattern = r"^\s*\d+[\.\)\:]\s*(.+)$"
    matches = re.findall(pattern, response, re.MULTILINE)

    # Clean up and filter empty
    questions = [q.strip() for q in matches if q.strip()]
    return questions[:3]  # Cap at 3


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two vectors."""
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


def compute_ragas_score(
    query: str,
    answer: str,
    client: OllamaClient,
) -> tuple[Optional[float], list[str]]:
    """Compute RAGAS Answer Relevancy score.

    Args:
        query: The original user query.
        answer: The LLM's generated answer text.
        client: Ollama client for generating reverse questions.

    Returns:
        Tuple of (score, generated_questions).
        Score is 0-1 float, or None if computation failed.
        generated_questions is the list of reverse-engineered questions.
    """
    if not answer or not answer.strip():
        logger.warning("Empty answer — RAGAS score is None")
        return None, []

    # Step 1: Generate reverse questions from the answer
    prompt = RAGAS_PROMPT.format(answer=answer)
    response, _ = client.generate(prompt, system=RAGAS_SYSTEM)
    questions = _parse_questions(response)

    if len(questions) == 0:
        logger.warning(
            "Failed to parse reverse questions from: '%s'",
            response[:100],
        )
        return None, []

    logger.debug("RAGAS reverse questions: %s", questions)

    # Step 2: Embed original query and generated questions
    model = _get_embedding_model()
    texts = [query] + questions
    embeddings = model.encode(texts, normalize_embeddings=True)

    query_emb = embeddings[0]
    question_embs = embeddings[1:]

    # Step 3: Mean cosine similarity
    similarities = [
        _cosine_similarity(query_emb, q_emb)
        for q_emb in question_embs
    ]

    score = float(np.mean(similarities))

    logger.info(
        "RAGAS score: %.3f (from %d questions, sims=%s)",
        score,
        len(questions),
        [round(s, 3) for s in similarities],
    )

    return score, questions
