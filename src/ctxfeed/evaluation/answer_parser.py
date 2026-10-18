# extracts the 'Answer: <value>' line from the llm response
# handles edge cases where model puts answer inline or with dollar signs

from __future__ import annotations

import re
from typing import Optional


def parse_answer(response: str) -> Optional[str]:
    # extract answer line via regex, check line start first then fallback to inline
    if not response:
        return None

    # check for 'Answer: <value>' at start of line
    matches = re.findall(
        r"^[Aa]nswer:\s*(.+)$", response, re.MULTILINE
    )

    # fallback: search anywhere in response text if model placed it inline
    if not matches:
        matches = re.findall(
            r"[Aa]nswer:\s*(.+?)(?:\.\s|$)", response
        )

    if matches:
        answer = matches[-1].strip()
        # strip symbols like $, %, and quotation marks
        answer = answer.strip("$%\"'")
        return answer

    return None


def parse_judge_score(response: str) -> Optional[int]:
    # extracts 1-5 score from judge response

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
