"""Extract the structured answer line from LLM free-text responses.

The query-answering prompt requires the model to end with:
    Answer: <value>

This module extracts that value so metrics.py can compare it
to ground truth without ambiguity.
"""

from __future__ import annotations

import re
from typing import Optional


def parse_answer(response: str) -> Optional[str]:
    """Extract the 'Answer: <value>' from an LLM response.

    Searches from the end of the response backward for a line
    matching 'Answer: ...' (case-insensitive).

    Args:
        response: The full LLM response text.

    Returns:
        The extracted answer value (stripped), or None if no
        answer line is found.
    """
    if not response:
        return None

    # Primary: 'Answer: <value>' at start of a line (case-insensitive)
    matches = re.findall(
        r"^[Aa]nswer:\s*(.+)$", response, re.MULTILINE
    )

    # Fallback: 'Answer: <value>' anywhere in the text (models sometimes
    # write it inline, e.g. "The closing price is Answer: 281.74")
    if not matches:
        matches = re.findall(
            r"[Aa]nswer:\s*(.+?)(?:\.\s|$)", response
        )

    if matches:
        answer = matches[-1].strip()
        # Strip common decorations: $, %, leading/trailing quotes
        answer = answer.strip("$%\"'")
        return answer

    return None


def parse_judge_score(response: str) -> Optional[int]:
    """Extract the 'Score: <integer>' from a judge LLM response.

    Args:
        response: The full judge response text.

    Returns:
        The score as an integer (1-5), or None if not found
        or out of range.
    """
    if not response:
        return None

    matches = re.findall(
        r"^[Ss]core:\s*(\d+)$", response, re.MULTILINE
    )

    if matches:
        score = int(matches[-1])
        if 1 <= score <= 5:
            return score

    return None
