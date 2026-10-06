"""Applying a reranker, and keeping what it moved.

The stage records the ranking it was handed as well as the one it produced.
Without the before, a demotion is invisible: the final ranking alone cannot say
whether a gold document at rank four arrived there or was pushed there, and
telling those apart is the only reason to measure a reranker at all.

A reranker only ever sees candidates, so recall is fixed before it starts. Any
reading of a rerank diff that credits one with finding a document is measuring
something that cannot have happened.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from embedlab.domain.ids import DocId, QueryId
from embedlab.domain.retrieval import order_deterministically, rank_of_best_relevant

if TYPE_CHECKING:
    from collections.abc import Mapping

    from embedlab.adapters.base import Reranker
    from embedlab.domain.retrieval import Qrels, Ranked, Run

IMPL_VERSION = 1


@dataclass(frozen=True, slots=True)
class Reranked:
    """What a reranker produced, and what it was given."""

    before: Run
    after: dict[QueryId, list[Ranked]]

    def moved(self, qrels: Qrels) -> dict[QueryId, tuple[int | None, int | None]]:
        """Where the gold sat before and after, per judged query."""
        movement: dict[QueryId, tuple[int | None, int | None]] = {}
        for query_id, labels in qrels.items():
            if not any(grade > 0 for grade in labels.values()):
                continue
            was, now = self.before.get(query_id), self.after.get(query_id)
            if was is None or now is None:
                continue
            movement[QueryId(query_id)] = (
                rank_of_best_relevant(was, labels),
                rank_of_best_relevant(now, labels),
            )
        return movement


def rerank(
    run: Run,
    reranker: Reranker,
    queries: Mapping[QueryId, str],
    texts: Mapping[DocId, str],
    *,
    k: int,
) -> Reranked:
    """Reorder each query's candidates, keeping the ranking they arrived in."""
    after: dict[QueryId, list[Ranked]] = {}
    for query_id, ranked in run.items():
        # The run is document-level by the time it gets here: a chunked
        # retrieval is folded before reranking, so a reranker never has to
        # reason about pieces of a document it cannot see whole.
        candidates = [(item.doc_id, texts[DocId(item.doc_id)]) for item in ranked]
        scored = reranker.rerank(queries[QueryId(query_id)], candidates)
        after[QueryId(query_id)] = order_deterministically(scored, k)
    return Reranked(before=run, after=after)
