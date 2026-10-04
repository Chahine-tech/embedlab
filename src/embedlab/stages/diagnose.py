"""Why a query failed — facts first, interpretations second.

The separation is structural, not a convention: `Evidence` is arithmetic over
the ranking and the corpus, reproducible to the bit. `Hypothesis` is an
interpretation carrying a producer and a confidence. A judge may later add
hypotheses; it may never write evidence. That is what keeps the judge
replaceable, measurable against human labels, and unable to quietly turn a
guess into a fact.

Every threshold below is a guess until it is calibrated against human labels.
They are named, gathered in one place, and reported with the hypothesis so a
reader can see what produced a label.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from embedlab.domain.ids import DocId, QueryId
from embedlab.domain.retrieval import rank_of_best_relevant
from embedlab.domain.taxonomy import FailureKind, Producer, Symptom

if TYPE_CHECKING:
    from collections.abc import Mapping

    from embedlab.domain.retrieval import Qrels, Run

IMPL_VERSION = 2

_TOKEN = re.compile(r"\w+", re.UNICODE)

MIN_QUERIES_FOR_CALIBRATION = 30
"""Below this, no cause is hypothesised beyond the deterministic ones.

Quantiles over a handful of queries describe the handful, not the dataset.
Emitting a confident-looking label from them would be worse than admitting the
dataset is too small to support one.
"""

LOW_OVERLAP_QUANTILE = 0.25
"""A query/gold overlap in the bottom quartile *of this dataset* counts as low.

Absolute thresholds were tried first and measurably failed. An IDF-weighted
Jaccard has no dataset-independent scale, and a hand-picked 0.08 fired on
135 of 137 SciFact failures — a rule that is almost always true discriminates
nothing. Relative thresholds at least normalise the scale.
"""

HIGH_COMPETITOR_QUANTILE = 0.75
"""A competitor/gold overlap in the top quartile of the observed rank failures.

The absolute threshold tried first (0.18) exceeded the maximum value present in
SciFact (0.168) and therefore never fired at all.
"""


def _tokens(text: str) -> set[str]:
    return {match.group().lower() for match in _TOKEN.finditer(text)}


def _inverse_document_frequency(corpus: Mapping[DocId, str]) -> dict[str, float]:
    total = len(corpus)
    document_frequency: dict[str, int] = {}
    for text in corpus.values():
        for token in _tokens(text):
            document_frequency[token] = document_frequency.get(token, 0) + 1
    return {
        token: math.log((1.0 + total) / (1.0 + frequency)) + 1.0
        for token, frequency in document_frequency.items()
    }


def _weighted_overlap(left: set[str], right: set[str], idf: Mapping[str, float]) -> float:
    """IDF-weighted Jaccard.

    Weighted rather than plain so that sharing "the" and "a" does not read as
    sharing meaning, which is precisely the mistake a lexical-mismatch detector
    must not make.
    """
    if not left or not right:
        return 0.0
    union = left | right
    union_weight = sum(idf.get(token, 1.0) for token in union)
    if union_weight == 0.0:
        return 0.0
    shared_weight = sum(idf.get(token, 1.0) for token in left & right)
    return shared_weight / union_weight


@dataclass(frozen=True, slots=True)
class Calibration:
    """Dataset-relative thresholds for the cause rules.

    Derived from the dataset's own distributions, which fixes the scale problem
    absolute constants had. It does **not** validate the causal claims: calling
    the bottom quartile of vocabulary overlap a "lexical mismatch" is still an
    interpretation, and only agreement with human labels can test it.
    """

    n: int
    low_query_gold_overlap: float
    high_competitor_gold_overlap: float

    @property
    def sufficient(self) -> bool:
        return self.n >= MIN_QUERIES_FOR_CALIBRATION


def _quantile(values: list[float], fraction: float) -> float:
    """Nearest-rank quantile, without pulling in numpy for four numbers."""
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, round(fraction * (len(ordered) - 1))))
    return ordered[index]


def calibrate(evidence: Mapping[QueryId, Evidence]) -> Calibration:
    """Derive thresholds from the evidence of one run over one dataset.

    The query/gold overlap quantile is taken over *every* judged query, not only
    the failures: it describes what this corpus's vocabulary overlap normally
    looks like. The competitor/gold quantile can only come from rank failures,
    since it is zero by construction wherever the top result is relevant.
    """
    all_overlaps = [found.query_gold_overlap for found in evidence.values()]
    rank_failures = [
        found.competitor_gold_overlap
        for found in evidence.values()
        if found.symptom is Symptom.RANK
    ]
    return Calibration(
        n=len(all_overlaps),
        low_query_gold_overlap=_quantile(all_overlaps, LOW_OVERLAP_QUANTILE),
        high_competitor_gold_overlap=_quantile(rank_failures, HIGH_COMPETITOR_QUANTILE),
    )


@dataclass(frozen=True, slots=True)
class Evidence:
    """Deterministic observations. No model, no judge, no threshold."""

    query_id: QueryId
    symptom: Symptom
    gold_rank: int | None
    best_relevant: DocId | None
    top1: DocId
    top1_is_relevant: bool
    score_margin: float | None
    """Score gap between rank 1 and the best relevant document, when retrieved."""

    query_gold_overlap: float
    """IDF-weighted overlap between the query and the best relevant document."""

    competitor_gold_overlap: float
    """IDF-weighted overlap between the rank-1 document and the gold document.

    Zero by construction whenever the rank-1 document *is* relevant, so this
    feature is degenerate on successful queries and can only ever discriminate
    *among* failures. Using it to predict failure would look like a perfect
    signal and measure nothing but its own definition.
    """

    query_token_count: int

    tied_with_top1: bool
    """The gold document scored *exactly* the same as the rank-1 document.

    Then its lower rank is decided by the tie-break, not by the model. Any diff
    that moves such a query is reporting an arbitrary ordering as a quality
    change — observed on the very first fixture run, where BM25 scored the gold
    and a hard negative identically to the last float bit.
    """

    gold_score_is_zero: bool
    """The gold was only in the scorer's zero-score tail: padding, not a hit."""

    retrieved_with_zero_score: int
    """How many of the returned documents the scorer gave zero. A top-K mostly
    filled with these makes recall@K look better than retrieval actually is."""


@dataclass(frozen=True, slots=True)
class Hypothesis:
    """An interpretation of evidence. Never asserted as fact."""

    query_id: QueryId
    kind: FailureKind
    confidence: float
    producer: Producer
    because: tuple[str, ...] = field(default_factory=tuple)
    """Human-readable evidence trail, so a label can be argued with."""


@dataclass(frozen=True, slots=True)
class Diagnosis:
    evidence: dict[QueryId, Evidence]
    hypotheses: dict[QueryId, tuple[Hypothesis, ...]]
    calibration: Calibration
    """The thresholds the hypotheses were drawn against. Kept on the result so a
    label can be re-read later against the numbers that produced it."""

    def failing(self) -> tuple[QueryId, ...]:
        return tuple(
            query_id
            for query_id, evidence in self.evidence.items()
            if evidence.symptom is not Symptom.OK
        )


def collect_evidence(
    run: Run,
    qrels: Qrels,
    corpus: Mapping[DocId, str],
    *,
    idf: Mapping[str, float] | None = None,
    queries: Mapping[QueryId, str],
) -> dict[QueryId, Evidence]:
    """Compute the deterministic facts for every judged query."""
    weights = idf if idf is not None else _inverse_document_frequency(corpus)
    evidence: dict[QueryId, Evidence] = {}

    for query_id in sorted(qrels):
        labels = qrels[query_id]
        if not any(grade > 0 for grade in labels.values()):
            continue  # unjudged: cannot distinguish a failure from a labelling gap
        ranked = run.get(query_id)
        if not ranked:
            continue

        gold_rank = rank_of_best_relevant(ranked, labels)
        best_relevant = next(
            (item.doc_id for item in ranked if labels.get(item.doc_id, 0) > 0), None
        )
        top = ranked[0]
        top1_is_relevant = labels.get(top.doc_id, 0) > 0

        gold_score = (
            next(item.score for item in ranked if item.doc_id == best_relevant)
            if best_relevant is not None
            else None
        )
        gold_score_is_zero = gold_score == 0.0

        if gold_rank is None or gold_score_is_zero:
            # A gold document the scorer gave zero was not retrieved; it is
            # padding that happened to land inside top-K. Calling that a ranking
            # problem would understate the failure.
            symptom = Symptom.MISS
        elif gold_rank == 1:
            symptom = Symptom.OK
        else:
            symptom = Symptom.RANK

        # Which gold document to reason about when none was retrieved: the
        # highest-graded label, tie-broken by id so the choice is reproducible.
        reference_gold = best_relevant
        if reference_gold is None:
            positives = [(grade, doc_id) for doc_id, grade in labels.items() if grade > 0]
            reference_gold = max(positives, key=lambda pair: (pair[0], pair[1]))[1]

        gold_tokens = _tokens(corpus[reference_gold])
        query_tokens = _tokens(queries[query_id])
        margin = None if gold_score is None else top.score - gold_score

        evidence[query_id] = Evidence(
            query_id=query_id,
            symptom=symptom,
            gold_rank=gold_rank,
            best_relevant=best_relevant,
            top1=top.doc_id,
            top1_is_relevant=top1_is_relevant,
            score_margin=margin,
            query_gold_overlap=_weighted_overlap(query_tokens, gold_tokens, weights),
            competitor_gold_overlap=(
                0.0
                if top1_is_relevant
                else _weighted_overlap(_tokens(corpus[top.doc_id]), gold_tokens, weights)
            ),
            query_token_count=len(query_tokens),
            tied_with_top1=(not top1_is_relevant and margin == 0.0),
            gold_score_is_zero=gold_score_is_zero,
            retrieved_with_zero_score=sum(1 for item in ranked if item.score == 0.0),
        )
    return evidence


def interpret(evidence: Evidence, calibration: Calibration) -> tuple[Hypothesis, ...]:
    """Rule-based causes for one query's failure.

    Returns causes only. The symptom is already a fact on the evidence and is
    never restated as a hypothesis.
    """
    if evidence.symptom is Symptom.OK:
        return ()

    hypotheses: list[Hypothesis] = []

    if evidence.tied_with_top1:
        # Deterministic, and needs no calibration: not a model failure at all
        # but a measurement artifact. Stated first so nobody reads the ranking
        # as evidence of a quality difference.
        hypotheses.append(
            Hypothesis(
                query_id=evidence.query_id,
                kind=FailureKind.SCORE_TIE,
                confidence=0.9,
                producer=Producer.RULE,
                because=(
                    "the gold document and the rank-1 document received identical "
                    "scores, so their order is decided by the tie-break rather "
                    "than by the retriever",
                ),
            )
        )

    if not calibration.sufficient:
        hypotheses.append(
            Hypothesis(
                query_id=evidence.query_id,
                kind=FailureKind.UNEXPLAINED,
                confidence=0.0,
                producer=Producer.RULE,
                because=(
                    f"only {calibration.n} judged queries: too few to establish what "
                    "normal vocabulary overlap looks like in this dataset, so no "
                    "cause is claimed",
                ),
            )
        )
        return tuple(hypotheses)

    if (
        evidence.symptom is Symptom.RANK
        and evidence.competitor_gold_overlap >= calibration.high_competitor_gold_overlap
    ):
        hypotheses.append(
            Hypothesis(
                query_id=evidence.query_id,
                kind=FailureKind.HARD_NEGATIVE,
                confidence=0.5,
                producer=Producer.RULE,
                because=(
                    f"rank-1 document {evidence.top1!r} shares "
                    f"{evidence.competitor_gold_overlap:.1%} of its distinctive "
                    f"vocabulary with the gold document, in the top quartile of the "
                    f"rank failures in this dataset "
                    f"(>= {calibration.high_competitor_gold_overlap:.1%})",
                ),
            )
        )

    if evidence.query_gold_overlap <= calibration.low_query_gold_overlap:
        hypotheses.append(
            Hypothesis(
                query_id=evidence.query_id,
                kind=FailureKind.LEXICAL_MISMATCH,
                confidence=0.4,
                producer=Producer.RULE,
                because=(
                    f"query and gold document share {evidence.query_gold_overlap:.1%} "
                    f"of their distinctive vocabulary, in the bottom quartile for this "
                    f"dataset (<= {calibration.low_query_gold_overlap:.1%})",
                ),
            )
        )

    if not hypotheses:
        hypotheses.append(
            Hypothesis(
                query_id=evidence.query_id,
                kind=FailureKind.UNEXPLAINED,
                confidence=0.0,
                producer=Producer.RULE,
                because=(
                    f"symptom is {evidence.symptom.value}, but no rule matched the "
                    "deterministic evidence",
                ),
            )
        )

    return tuple(hypotheses)


def diagnose(
    run: Run,
    qrels: Qrels,
    corpus: Mapping[DocId, str],
    queries: Mapping[QueryId, str],
) -> Diagnosis:
    evidence = collect_evidence(run, qrels, corpus, queries=queries)
    calibration = calibrate(evidence)
    return Diagnosis(
        evidence=evidence,
        hypotheses={query_id: interpret(item, calibration) for query_id, item in evidence.items()},
        calibration=calibration,
    )
