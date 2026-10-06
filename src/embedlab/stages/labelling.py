"""Putting failures in front of a person, and measuring the rules against them.

The diagnosis rules decide what a failure is called, and nothing has ever
checked whether a person would agree. Dataset-relative thresholds fixed the
scale problem; they did not make the labels true.

Two deliberate choices shape this.

**The sample is blind.** An export carries the query, the gold document, what
was retrieved and the deterministic evidence, and never the rule's own guess. A
labeller shown the guess agrees with it, and the agreement measured afterwards
would be the rules marking their own homework.

**The sample is seeded.** Which failures a person spent an hour on is part of
the result, so a rerun has to produce the same ones.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import TYPE_CHECKING

from embedlab.domain.ids import DocId, QueryId
from embedlab.domain.taxonomy import FailureKind, Symptom

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from embedlab.domain.retrieval import Run
    from embedlab.stages.diagnose import Diagnosis

IMPL_VERSION = 1

LABELLABLE: tuple[str, ...] = (
    FailureKind.HARD_NEGATIVE.value,
    FailureKind.LEXICAL_MISMATCH.value,
    FailureKind.CHUNKING.value,
    FailureKind.DUPLICATE_COLLISION.value,
    FailureKind.CROSS_LINGUAL.value,
    FailureKind.OUTLIER.value,
    FailureKind.UNEXPLAINED.value,
)
"""What a person may write in the `cause` column.

`unexplained` is offered on purpose: a labeller who cannot name a cause must be
able to say so, otherwise the exercise forces agreement with whichever category
is nearest and measures nothing.
"""


@dataclass(frozen=True, slots=True)
class Case:
    """One failure, as a person needs to see it."""

    query_id: QueryId
    query: str
    symptom: str
    gold_rank: int | None
    gold_id: str
    gold_text: str
    retrieved: tuple[tuple[int, str, str, bool], ...]
    """rank, doc id, text, whether it is relevant."""

    query_gold_overlap: float
    competitor_gold_overlap: float


def sample_failures(
    diagnosis: Diagnosis,
    run: Run,
    queries: Mapping[QueryId, str],
    corpus: Mapping[DocId, str],
    qrels: Mapping[QueryId, Mapping[DocId, int]],
    *,
    size: int = 50,
    seed: int = 20261006,
    top: int = 5,
    snippet: int = 400,
) -> list[Case]:
    """Draw failures to label, reproducibly."""
    failing = sorted(diagnosis.failing())
    chosen = failing if size >= len(failing) else random.Random(seed).sample(failing, size)

    cases: list[Case] = []
    for query_id in sorted(chosen):
        evidence = diagnosis.evidence[query_id]
        labels = qrels[query_id]
        gold = evidence.best_relevant
        if gold is None:
            positives = [(grade, doc) for doc, grade in labels.items() if grade > 0]
            gold = max(positives, key=lambda pair: (pair[0], pair[1]))[1]

        cases.append(
            Case(
                query_id=query_id,
                query=queries[query_id],
                symptom=evidence.symptom.value,
                gold_rank=evidence.gold_rank,
                gold_id=str(gold),
                gold_text=corpus[gold][:snippet],
                retrieved=tuple(
                    (
                        item.rank,
                        str(item.doc_id),
                        corpus[DocId(item.doc_id)][:snippet],
                        labels.get(DocId(item.doc_id), 0) > 0,
                    )
                    for item in run[query_id][:top]
                ),
                query_gold_overlap=evidence.query_gold_overlap,
                competitor_gold_overlap=evidence.competitor_gold_overlap,
            )
        )
    return cases


@dataclass(frozen=True, slots=True)
class Agreement:
    """How often the rules and a person named the same cause."""

    labelled: int
    compared: int
    agreed: int
    per_cause: dict[str, tuple[int, int]]
    """cause -> (times the rules said it, times a person also did)."""

    missed: dict[str, int]
    """cause -> times a person said it and no rule did."""

    @property
    def accuracy(self) -> float:
        return self.agreed / self.compared if self.compared else 0.0

    def precision(self, cause: str) -> float:
        said, confirmed = self.per_cause.get(cause, (0, 0))
        return confirmed / said if said else 0.0


def score_labels(
    symptoms: Mapping[QueryId, str],
    hypothesised: Mapping[QueryId, frozenset[str]],
    labels: Mapping[QueryId, str],
) -> Agreement:
    """Compare human labels against what the rules produced.

    Takes plain mappings rather than a diagnosis object, so it reads the
    published artifacts the same way any other consumer would.

    A query counts as agreement when the person's cause appears among the ones
    the rules hypothesised. Rules may propose several, so this is generous to
    them by design: a measure that flattered them less would need the labeller
    to rank causes, which is a harder thing to ask for.
    """
    compared = agreed = 0
    per_cause: dict[str, tuple[int, int]] = {}
    missed: dict[str, int] = {}

    for query_id, human in labels.items():
        symptom = symptoms.get(query_id)
        if symptom is None or symptom == Symptom.OK.value:
            continue
        compared += 1

        proposed = hypothesised.get(query_id, frozenset())
        hit = human in proposed
        agreed += int(hit)

        for cause in proposed:
            said, confirmed = per_cause.get(cause, (0, 0))
            per_cause[cause] = (said + 1, confirmed + int(cause == human))
        if not hit:
            missed[human] = missed.get(human, 0) + 1

    return Agreement(
        labelled=len(labels),
        compared=compared,
        agreed=agreed,
        per_cause=per_cause,
        missed=missed,
    )


def validate_label(cause: str) -> str:
    """Accept only a cause the taxonomy knows, naming the alternatives."""
    cleaned = cause.strip().lower().replace(" ", "_").replace("-", "_")
    if cleaned not in LABELLABLE:
        msg = f"unknown cause {cause!r}; expected one of {', '.join(LABELLABLE)}"
        raise ValueError(msg)
    return cleaned


def unlabelled(cases: Sequence[Case], labels: Mapping[QueryId, str]) -> list[QueryId]:
    """Cases drawn but never labelled, so a partial sheet is visible as partial."""
    return [case.query_id for case in cases if case.query_id not in labels]
