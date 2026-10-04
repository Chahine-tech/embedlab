"""Per-query metrics, delegated to `ir_measures`.

We do not implement nDCG, MRR or recall ourselves. The formulas are easy; the
*conventions* are not — unjudged documents, graded gain, truncation, and above
all tie-breaking all have established trec_eval semantics that published
numbers depend on. Getting one of them subtly wrong in a tool whose whole claim
is "trust my diff" would be fatal, so the reference implementation is the
engine and our job is to feed it correctly.

Per query, never only aggregate: the product's subject is the individual
failure, and an aggregate cannot be clicked.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from embedlab.domain.ids import QueryId

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from embedlab.domain.retrieval import Qrels, Run

IMPL_VERSION = 1

DEFAULT_MEASURES: tuple[str, ...] = ("nDCG@10", "RR", "R@10")


class MeasuresUnavailableError(RuntimeError):
    """The `measures` extra is not installed."""


@dataclass(frozen=True, slots=True)
class Evaluation:
    """Metric values per query, plus the aggregate over judged queries."""

    measures: tuple[str, ...]
    per_query: dict[QueryId, dict[str, float]]
    aggregate: dict[str, float]


def _as_scored(ranked: Sequence) -> dict[str, float]:
    """Convert our ordered run into the score map ir_measures expects.

    Scores are synthesised from *rank*, not copied from the retriever. This is
    deliberate: ir_measures re-sorts by score and applies trec_eval's own
    tie-break (document id descending), which is the opposite of ours. Passing
    raw scores would therefore evaluate a slightly different ordering than the
    one we reported and will show the user. Synthetic strictly-decreasing scores
    make the evaluated order exactly the reported order, with no ties to break.
    """
    total = len(ranked)
    return {str(item.doc_id): float(total - item.rank + 1) for item in ranked}


def evaluate(
    run: Run,
    qrels: Qrels,
    *,
    measures: Sequence[str] = DEFAULT_MEASURES,
) -> Evaluation:
    """Score a run against labels, per query and in aggregate.

    Only queries with at least one positive label are scored: a query with no
    known relevant document cannot distinguish a retrieval failure from a gap
    in the labels.
    """
    try:
        import ir_measures
    except ImportError as error:  # pragma: no cover - exercised by the extra being absent
        msg = (
            "evaluation needs the 'measures' extra: uv sync --extra measures "
            "(installs ir-measures and pytrec-eval)"
        )
        raise MeasuresUnavailableError(msg) from error

    parsed = [ir_measures.parse_measure(measure) for measure in measures]

    judged = {
        str(query_id): {str(doc_id): int(grade) for doc_id, grade in labels.items()}
        for query_id, labels in qrels.items()
        if any(grade > 0 for grade in labels.values())
    }
    scored_run = {
        str(query_id): _as_scored(ranked)
        for query_id, ranked in run.items()
        if str(query_id) in judged
    }

    per_query: dict[QueryId, dict[str, float]] = {
        QueryId(query_id): {} for query_id in sorted(judged)
    }
    for metric in ir_measures.iter_calc(parsed, judged, scored_run):
        per_query[QueryId(metric.query_id)][str(metric.measure)] = float(metric.value)

    aggregate = {
        str(measure): float(value)
        for measure, value in ir_measures.calc_aggregate(parsed, judged, scored_run).items()
    }

    return Evaluation(
        measures=tuple(str(measure) for measure in parsed),
        per_query=per_query,
        aggregate=aggregate,
    )


def summarise(evaluation: Evaluation) -> Mapping[str, float]:
    """Aggregate values keyed by measure name, for reports."""
    return dict(evaluation.aggregate)
