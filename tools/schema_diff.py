"""
Schema drift diagnosis
----------------------
Shared by both tool registries. Given the fields a pipeline requires and the
fields an upstream file actually delivered, work out which missing field was
probably renamed to which new one.

The original implementation only accepted substring matches, which meant the
two most important demo cases both diagnosed poorly:

  complaint_type       -> complaint_category   no suggestion at all
  model_b_quality_score-> model_b_score        suggested "model_b" instead

A rename is a near-miss, not a substring, so this scores candidates on string
similarity and on shared underscore-separated tokens, and takes the best.

Python 3.9 compatible.
"""

from difflib import SequenceMatcher
from typing import Dict, List

# Below this score the fields are unrelated and no guess is better than a wrong
# guess — an incorrect diagnosis is worse than "missing field, cause unknown".
MIN_CONFIDENCE = 0.45


def _token_overlap(a: str, b: str) -> float:
    ta, tb = set(a.split("_")), set(b.split("_"))
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / max(len(ta), len(tb))


def _score(missing: str, candidate: str) -> float:
    ratio   = SequenceMatcher(None, missing, candidate).ratio()
    overlap = _token_overlap(missing, candidate)
    # Substring containment is still a strong signal when it happens.
    contains = 0.9 if (missing in candidate or candidate in missing) else 0.0
    return max(ratio, overlap, contains)


def suggest_renames(
    required_fields: List[str],
    actual_fields:   List[str],
    missing:         List[str],
) -> Dict[str, str]:
    """
    Map each missing field to its most likely new name.

    Only fields the schema does NOT already expect are considered — a field
    that is required and present cannot also be somebody's rename. Each
    candidate is claimed at most once, so two missing fields never collapse
    onto the same guess.
    """
    unclaimed = [f for f in actual_fields if f not in required_fields]
    suggestions: Dict[str, str] = {}

    # Rank every (missing, candidate) pair and assign greedily by confidence,
    # so the strongest match wins its candidate first.
    pairs = sorted(
        ((_score(m, c), m, c) for m in missing for c in unclaimed),
        key=lambda t: (-t[0], t[1], t[2]),
    )

    taken = set()
    for score, m, candidate in pairs:
        if score < MIN_CONFIDENCE or m in suggestions or candidate in taken:
            continue
        suggestions[m] = candidate
        taken.add(candidate)

    return suggestions
