"""Git diff, but for retrieval quality.

The central artifact of the product. Two rules shape it:

First, the unit of comparison is the *query*, not the average. "+184 improved,
-96 regressed" is the headline; the aggregate delta is a footnote. An aggregate
cannot be clicked, and averaging hides the individual queries a developer
actually has to fix. A model can raise mean recall while breaking the handful
of queries that matter most.

Second, a diff is reported together with how much of it to believe. The trust
verdict from `embedlab.artifacts.comparability` travels with the numbers rather
than being checked and discarded, and a verdict of INCOMPARABLE refuses instead
of rendering.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import TYPE_CHECKING

from embedlab.domain.ids import QueryId
from embedlab.domain.retrieval import rank_of_best_relevant
from embedlab.domain.taxonomy import Symptom

if TYPE_CHECKING:
    from collections.abc import Mapping

    from embedlab.artifacts.comparability import Verdict
    from embedlab.domain.retrieval import Qrels, Run
    from embedlab.stages.diagnose import Diagnosis
    from embedlab.stages.significance import Significance

IMPL_VERSION = 1

_MISSING_RANK = 1 << 30
"""Sort weight for "no relevant document retrieved". Worse than any real rank,
without pretending to be a rank."""


class Direction(StrEnum):
    IMPROVED = "improved"
    REGRESSED = "regressed"
    UNCHANGED = "unchanged"


@dataclass(frozen=True, slots=True)
class QueryDelta:
    """What changed for one query between two runs."""

    query_id: QueryId
    left_rank: int | None
    right_rank: int | None
    direction: Direction
    arbitrary: bool = False
    """The movement is explained by a score tie on one side, not by the models.

    Kept on the delta itself so a caller cannot report the headline without
    having the means to qualify it. The first fixture run produced a diff whose
    only "improvement" was of this kind."""

    @property
    def became_failure(self) -> bool:
        """Was first on the left and no longer is: the regressions that sting."""
        return self.left_rank == 1 and self.right_rank != 1

    @property
    def became_success(self) -> bool:
        return self.right_rank == 1 and self.left_rank != 1


@dataclass(frozen=True, slots=True)
class RetrievalDiff:
    left_name: str
    right_name: str
    trust: Verdict
    deltas: tuple[QueryDelta, ...]
    significance: dict[str, Significance]
    """Per-measure difference with its confidence interval and p-value.

    There is deliberately no bare `measure_deltas` field. A delta without its
    interval is the single most misleading number this tool could print, so the
    only way to obtain one is to go through the evidence for it.
    """

    def by_direction(self, direction: Direction) -> tuple[QueryDelta, ...]:
        return tuple(delta for delta in self.deltas if delta.direction is direction)

    @property
    def improved(self) -> tuple[QueryDelta, ...]:
        return self.by_direction(Direction.IMPROVED)

    @property
    def regressed(self) -> tuple[QueryDelta, ...]:
        return self.by_direction(Direction.REGRESSED)

    @property
    def unchanged(self) -> tuple[QueryDelta, ...]:
        return self.by_direction(Direction.UNCHANGED)

    @property
    def arbitrary(self) -> tuple[QueryDelta, ...]:
        """Movements caused by a score tie. Never report these as quality."""
        return tuple(delta for delta in self.deltas if delta.arbitrary)

    @property
    def headline(self) -> str:
        line = (
            f"+{len(self.improved)} improved  "
            f"-{len(self.regressed)} regressed  "
            f"={len(self.unchanged)} unchanged"
        )
        if self.arbitrary:
            line += f"   ({len(self.arbitrary)} of them arbitrary: score ties)"
        return line

    @property
    def real_differences(self) -> tuple[Significance, ...]:
        """Measures whose change survives the paired test. Often empty, honestly."""
        return tuple(result for _, result in sorted(self.significance.items()) if result.is_real)


def _weight(rank: int | None) -> int:
    return _MISSING_RANK if rank is None else rank


def compare_runs(
    left: Run,
    right: Run,
    qrels: Qrels,
    *,
    left_name: str,
    right_name: str,
    trust: Verdict,
    significance: Mapping[str, Significance] | None = None,
    left_diagnosis: Diagnosis | None = None,
    right_diagnosis: Diagnosis | None = None,
) -> RetrievalDiff:
    """Per-query comparison of where the gold document landed.

    Only judged queries take part: without a positive label there is nothing to
    have improved or regressed.
    """
    if trust.refuses:
        reasons = "; ".join(trust.reasons)
        msg = f"refusing to diff {left_name} against {right_name}: {reasons}"
        raise ValueError(msg)

    deltas: list[QueryDelta] = []
    for query_id in sorted(qrels):
        labels = qrels[query_id]
        if not any(grade > 0 for grade in labels.values()):
            continue
        left_ranked, right_ranked = left.get(query_id), right.get(query_id)
        if left_ranked is None or right_ranked is None:
            continue

        left_rank = rank_of_best_relevant(left_ranked, labels)
        right_rank = rank_of_best_relevant(right_ranked, labels)

        if _weight(right_rank) < _weight(left_rank):
            direction = Direction.IMPROVED
        elif _weight(right_rank) > _weight(left_rank):
            direction = Direction.REGRESSED
        else:
            direction = Direction.UNCHANGED

        # A movement is arbitrary when either side placed the gold behind an
        # identically-scored document: the tie-break decided it, not the model.
        arbitrary = direction is not Direction.UNCHANGED and any(
            diagnosis is not None
            and (evidence := diagnosis.evidence.get(QueryId(query_id))) is not None
            and evidence.tied_with_top1
            for diagnosis in (left_diagnosis, right_diagnosis)
        )

        deltas.append(
            QueryDelta(
                query_id=QueryId(query_id),
                left_rank=left_rank,
                right_rank=right_rank,
                direction=direction,
                arbitrary=arbitrary,
            )
        )

    return RetrievalDiff(
        left_name=left_name,
        right_name=right_name,
        trust=trust,
        deltas=tuple(deltas),
        significance=dict(significance or {}),
    )


def failure_sets(diagnosis: Diagnosis) -> dict[str, tuple[QueryId, ...]]:
    """Group failing queries by hypothesised cause.

    A query appears in every set whose cause was hypothesised for it, so the
    counts intentionally sum to more than the number of failures. A query can
    be both a lexical mismatch and a hard negative, and forcing a single label
    would be a fiction. Reports must say "queries in set", never "% of failures".
    """
    sets: dict[str, list[QueryId]] = {}
    for query_id, hypotheses in diagnosis.hypotheses.items():
        if diagnosis.evidence[query_id].symptom is Symptom.OK:
            continue
        for hypothesis in hypotheses:
            sets.setdefault(str(hypothesis.kind), []).append(query_id)
    return {name: tuple(sorted(members)) for name, members in sorted(sets.items())}
