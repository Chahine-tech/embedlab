"""Reranking by how much of the query a candidate actually covers.

A dependency-free reranker, so the stage can be exercised and regression-tested
without model weights. It is not a placeholder: coverage asks a different
question from BM25, which rewards a document for repeating one rare query term
and is indifferent to the terms it never mentions. A candidate matching four of
five query terms once each outranks one matching a single term five times.

That difference is the point. A reranker that agreed with the retriever would
reorder nothing, and the stage would have nothing to measure.
"""

from __future__ import annotations

import math
import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from embedlab.domain.ids import UnitId

IMPL_VERSION = 1

_TOKEN = re.compile(r"\w+", re.UNICODE)


def _terms(text: str, *, lower: bool) -> set[str]:
    prepared = text.lower() if lower else text
    return {match.group() for match in _TOKEN.finditer(prepared)}


class CoverageReranker:
    """Score a candidate by the share of query terms it contains."""

    def __init__(
        self,
        *,
        name: str = "coverage",
        lower: bool = True,
        length_penalty: float = 0.0,
    ) -> None:
        self._name = name
        self._lower = lower
        self._length_penalty = length_penalty

    @property
    def name(self) -> str:
        return self._name

    @property
    def descriptor(self) -> Mapping[str, object]:
        return {
            "kind": "coverage",
            "impl": "embedlab.adapters.coverage",
            "impl_version": IMPL_VERSION,
            "lower": self._lower,
            # Long candidates cover more terms by accident; this discounts them.
            "length_penalty": self._length_penalty,
        }

    def rerank(self, query: str, candidates: Sequence[tuple[UnitId, str]]) -> dict[UnitId, float]:
        wanted = _terms(query, lower=self._lower)
        if not wanted:
            # Nothing to cover: leave the order to the stage's tie-break rather
            # than inventing a preference.
            return {unit: 0.0 for unit, _ in candidates}

        scored: dict[UnitId, float] = {}
        for unit, text in candidates:
            found = _terms(text, lower=self._lower)
            coverage = len(wanted & found) / len(wanted)
            penalty = self._length_penalty * math.log1p(len(found))
            scored[unit] = coverage - penalty
        return scored
