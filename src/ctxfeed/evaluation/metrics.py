# accuracy scoring across benchmark query types
# handles exact match, relative tolerance (+-1%), and directional trend agreement

from __future__ import annotations

import re
from typing import Optional


def score_accuracy(
    predicted: Optional[str],
    ground_truth: str,
    answer_type: str,
    tolerance: float = 0.01,
) -> float:
    # returns 1.0 if correct, 0.0 otherwise

    if predicted is None:
        return 0.0

    if answer_type == "exact":
        return _score_exact(predicted, ground_truth)
    elif answer_type == "numeric":
        return _score_numeric(predicted, ground_truth, tolerance)
    elif answer_type == "directional":
        return _score_directional(predicted, ground_truth)
    else:
        # unknown type, fall back to exact match
        return _score_exact(predicted, ground_truth)


def _score_exact(predicted: str, ground_truth: str) -> float:
    # case-insensitive whitespace-trimmed exact match
    return 1.0 if predicted.strip().upper() == ground_truth.strip().upper() else 0.0


def _score_numeric(predicted: str, ground_truth: str, tolerance: float) -> float:
    # numeric comparison within relative tolerance
    pred_num = _extract_number(predicted)
    truth_num = _extract_number(ground_truth)

    if pred_num is None or truth_num is None:
        return 0.0

    if truth_num == 0:
        return 1.0 if pred_num == 0 else 0.0

    relative_error = abs(pred_num - truth_num) / abs(truth_num)
    return 1.0 if relative_error <= tolerance else 0.0


def _score_directional(predicted: str, ground_truth: str) -> float:
    # check direction agreement (up/down, yes/no, increase/decrease)
    pred_dir = _classify_direction(predicted)
    truth_dir = _classify_direction(ground_truth)

    if pred_dir is None or truth_dir is None:
        return 0.0

    return 1.0 if pred_dir == truth_dir else 0.0


def _extract_number(text: str) -> Optional[float]:
    """Extract the first number from a text string.

    Handles integers, decimals, negative numbers, and comma separators.
    """
    # Remove commas in numbers (e.g. "1,234,567")
    cleaned = text.replace(",", "")
    match = re.search(r"-?\d+\.?\d*", cleaned)
    if match:
        return float(match.group())
    return None


def _classify_direction(text: str) -> Optional[bool]:
    """Classify text as positive (True) or negative (False) direction.

    Returns None if direction can't be determined.
    """
    upper = text.strip().upper()

    positive = {"YES", "UP", "UPWARD", "INCREASE", "INCREASED", "RISING", "POSITIVE", "TRUE", "HIGHER"}
    negative = {"NO", "DOWN", "DOWNWARD", "DECREASE", "DECREASED", "FALLING", "NEGATIVE", "FALSE", "LOWER"}

    for word in positive:
        if word in upper:
            return True
    for word in negative:
        if word in upper:
            return False

    return None
