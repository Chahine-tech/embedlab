"""What a retrieval system returned, and what the labels say it should have.

Pure vocabulary: no numpy, no I/O. The shapes here are the contract between
adapters, stages and artifacts.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from embedlab.domain.ids import DocId, QueryId, UnitId

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence


@dataclass(frozen=True, slots=True)
class Ranked:
    """One retrieved item at one position.

    `rank` is 1-based, matching how every report and TREC run file reads.
    """

    doc_id: UnitId
    """Whichever unit was retrieved.

    A chunk-level run holds ChunkIds here. Those never reach an artifact: a run
    is folded back to documents before it is published, so the `doc_id` column
    on disk really does hold documents.
    """

    rank: int
    score: float


type Run = Mapping[QueryId, Sequence[Ranked]]
"""What one retrieval system returned for every query, best first."""

type Qrels = Mapping[QueryId, Mapping[DocId, int]]
"""Graded relevance labels. 0 means judged irrelevant, absent means unjudged.
a distinction that matters: an unjudged document is not evidence of a failure."""


def rank_of_best_relevant(ranked: Sequence[Ranked], relevant: Mapping[DocId, int]) -> int | None:
    """Best rank achieved by any positively-judged document, or None if absent.

    Returns the best rank rather than assuming a single correct answer: a query
    may legitimately have several acceptable documents, and scoring against only
    one of them would invent failures.
    """
    for item in ranked:
        # Only meaningful on a document-level run. A chunk-level one is folded
        # before it reaches here, so the unit really is a document.
        if relevant.get(DocId(item.doc_id), 0) > 0:
            return item.rank
    return None


def order_deterministically(scored: Mapping[UnitId, float], k: int) -> list[Ranked]:
    """Take the top-k, breaking score ties by doc_id ascending.

    The tie-break is not cosmetic. Two documents with identical scores would
    otherwise be ordered by dict insertion or by a sort that varies with the
    backend, so re-running the same config could produce a different run, and
    every diff against it would report movements that never happened. This is
    the single cheapest guard against a diff tool that lies.
    """
    ordered = sorted(scored.items(), key=lambda pair: (-pair[1], pair[0]))
    return [
        Ranked(doc_id=unit_id, rank=position, score=float(score))
        for position, (unit_id, score) in enumerate(ordered[:k], start=1)
    ]
