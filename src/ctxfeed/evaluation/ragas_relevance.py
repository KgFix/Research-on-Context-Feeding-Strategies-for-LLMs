# ragas answer relevancy score (es et al. 2023)
# asks llm to generate questions that match the answer, then checks cosine similarity with original query

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
    # ragas answer relevancy: generates 3 reverse questions from answer and computes cosine sim to query
    if not answer or not answer.strip():
        logger.warning("Empty answer: RAGAS score is None")
        return None, []

    # generate reverse questions from answer text
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

    # embed original query and generated questions
    model = _get_embedding_model()
    texts = [query] + questions
    embeddings = model.encode(texts, normalize_embeddings=True)

    query_emb = embeddings[0]
    question_embs = embeddings[1:]

    # compute mean cosine similarity
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
