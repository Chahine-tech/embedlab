"""Evidence is arithmetic; hypotheses are interpretation. Keep them apart."""

import pytest

from embedlab.domain.ids import DocId, QueryId
from embedlab.domain.retrieval import Ranked
from embedlab.domain.taxonomy import FailureKind, Producer, Symptom
from embedlab.stages.diagnose import (
    MIN_QUERIES_FOR_CALIBRATION,
    Calibration,
    calibrate,
    collect_evidence,
    diagnose,
    interpret,
)


def calibration(
    *,
    n: int = MIN_QUERIES_FOR_CALIBRATION,
    low: float = 0.05,
    high: float = 0.10,
) -> Calibration:
    """Thresholds stated outright, so a rule test does not depend on quantiles."""
    return Calibration(n=n, low_query_gold_overlap=low, high_competitor_gold_overlap=high)


CORPUS = {
    DocId("gold"): "production token rotation procedure rotate revoke grace period",
    DocId("near"): "api token creation procedure create scope expiry",
    DocId("far"): "invoices billing cycles metered hourly refunds prorated",
}
QUERIES = {QueryId("q"): "how do I rotate the production token"}


def run_with(*pairs):
    return {
        QueryId("q"): [
            Ranked(DocId(doc_id), position, score)
            for position, (doc_id, score) in enumerate(pairs, start=1)
        ]
    }


def interpret_with(evidence, **kwargs):
    return interpret(evidence, calibration(**kwargs))


def evidence_for(*pairs, qrels=None):
    labels = qrels or {QueryId("q"): {DocId("gold"): 2}}
    return collect_evidence(run_with(*pairs), labels, CORPUS, queries=QUERIES)[QueryId("q")]


def test_gold_first_is_ok():
    assert evidence_for(("gold", 0.9), ("near", 0.5)).symptom is Symptom.OK


def test_gold_second_is_a_rank_problem():
    found = evidence_for(("near", 0.9), ("gold", 0.5))
    assert found.symptom is Symptom.RANK
    assert found.gold_rank == 2


def test_gold_absent_is_a_miss():
    found = evidence_for(("near", 0.9), ("far", 0.5))
    assert found.symptom is Symptom.MISS
    assert found.gold_rank is None
    assert found.best_relevant is None


def test_gold_only_in_the_zero_score_tail_is_a_miss_not_a_rank_problem():
    """Padding is not retrieval. Calling it a ranking problem understates it."""
    found = evidence_for(("near", 0.9), ("gold", 0.0))
    assert found.gold_score_is_zero is True
    assert found.symptom is Symptom.MISS
    assert found.gold_rank == 2  # the rank is still reported truthfully


def test_identical_scores_are_flagged_as_a_tie():
    """The case the first fixture run actually hit."""
    found = evidence_for(("near", 0.871), ("gold", 0.871))
    assert found.tied_with_top1 is True
    assert found.score_margin == 0.0


def test_a_relevant_rank_one_is_never_a_tie():
    assert evidence_for(("gold", 0.5), ("near", 0.5)).tied_with_top1 is False


def test_zero_score_returns_are_counted():
    found = evidence_for(("near", 0.9), ("gold", 0.4), ("far", 0.0))
    assert found.retrieved_with_zero_score == 1


def test_score_margin_is_measured_against_the_best_relevant():
    found = evidence_for(("near", 0.9), ("gold", 0.4))
    assert found.score_margin == pytest.approx(0.5)


def test_competitor_overlap_is_zero_when_rank_one_is_relevant():
    assert evidence_for(("gold", 0.9), ("near", 0.5)).competitor_gold_overlap == 0.0


def test_a_near_miss_competitor_overlaps_the_gold_more_than_an_unrelated_one():
    near = evidence_for(("near", 0.9), ("gold", 0.5))
    far = evidence_for(("far", 0.9), ("gold", 0.5))
    assert near.competitor_gold_overlap > far.competitor_gold_overlap


def test_unjudged_queries_produce_no_evidence():
    labels = {QueryId("q"): {DocId("gold"): 0}}
    assert collect_evidence(run_with(("gold", 0.9)), labels, CORPUS, queries=QUERIES) == {}


def test_a_tie_is_hypothesised_as_a_measurement_artifact():
    hypotheses = interpret_with(evidence_for(("near", 0.871), ("gold", 0.871)))
    kinds = [hypothesis.kind for hypothesis in hypotheses]
    assert FailureKind.SCORE_TIE in kinds
    assert kinds[0] is FailureKind.SCORE_TIE, "the artifact must be stated before any cause"


def test_hypotheses_carry_their_producer_and_an_evidence_trail():
    for hypothesis in interpret_with(evidence_for(("near", 0.9), ("gold", 0.5))):
        assert hypothesis.producer is Producer.RULE
        assert hypothesis.because, "a label that cannot be argued with is not a hypothesis"


def test_a_successful_query_yields_no_hypothesis():
    assert interpret_with(evidence_for(("gold", 0.9), ("near", 0.5))) == ()


def test_a_failure_always_yields_at_least_one_hypothesis():
    assert interpret_with(evidence_for(("far", 0.9), ("near", 0.5))) != ()


def test_diagnose_lists_failing_queries(mini):
    pytest.importorskip("bm25s", reason="needs the 'lexical' extra")
    from embedlab.adapters.lexical import BM25Retriever

    retriever = BM25Retriever()
    retriever.index(mini.corpus)
    run = retriever.search(mini.queries, k=10)

    diagnosis = diagnose(run, mini.qrels, mini.corpus, mini.queries)
    failing = diagnosis.failing()

    assert set(failing) <= set(mini.judged_queries)
    for query_id in failing:
        assert diagnosis.evidence[query_id].symptom is not Symptom.OK
        assert diagnosis.hypotheses[query_id]


def test_an_unmatched_failure_is_named_unexplained_not_restated():
    """The fallback must not dress the symptom up as a cause.

    `far` shares no distinctive vocabulary with the gold, so neither the
    hard-negative nor the lexical-mismatch rule applies to the competitor; the
    query/gold overlap is high enough that no cause fits at all.
    """
    found = evidence_for(("far", 0.9), ("gold", 0.5))
    hypotheses = interpret_with(found)

    assert [h.kind for h in hypotheses] == [FailureKind.UNEXPLAINED]
    assert hypotheses[0].confidence == 0.0
    assert found.symptom.value in hypotheses[0].because[0]


def test_unexplained_carries_no_confidence():
    """A label that explains nothing must not look like a finding."""
    for hypothesis in interpret_with(evidence_for(("far", 0.9), ("gold", 0.5))):
        if hypothesis.kind is FailureKind.UNEXPLAINED:
            assert hypothesis.confidence == 0.0


def test_symptoms_are_exclusive_and_causes_are_not(mini):
    """The two axes, asserted rather than documented.

    Every judged query has exactly one symptom; a query may carry several
    causes, which is why failure-set counts overlap on purpose.
    """
    pytest.importorskip("bm25s", reason="needs the 'lexical' extra")
    from embedlab.adapters.lexical import BM25Retriever

    retriever = BM25Retriever()
    retriever.index(mini.corpus)
    diagnosis = diagnose(
        retriever.search(mini.queries, k=10), mini.qrels, mini.corpus, mini.queries
    )

    for query_id, found in diagnosis.evidence.items():
        assert isinstance(found.symptom, Symptom)
        causes = [h.kind for h in diagnosis.hypotheses[query_id]]
        assert len(set(causes)) == len(causes), "a cause must not be hypothesised twice"
        assert not any(cause.value in {"ok", "rank", "miss"} for cause in causes), (
            "a symptom must never appear as a cause"
        )


def test_a_dataset_too_small_to_calibrate_claims_no_cause():
    """Quantiles over a handful of queries describe the handful, not the dataset."""
    found = evidence_for(("near", 0.9), ("gold", 0.5))
    hypotheses = interpret(found, calibration(n=5))

    assert [h.kind for h in hypotheses] == [FailureKind.UNEXPLAINED]
    assert "too few" in hypotheses[0].because[0]


def test_a_tie_is_still_reported_without_calibration():
    """The deterministic artifact does not depend on any threshold."""
    found = evidence_for(("near", 0.871), ("gold", 0.871))
    kinds = [h.kind for h in interpret(found, calibration(n=3))]
    assert FailureKind.SCORE_TIE in kinds


def test_the_lexical_rule_fires_at_or_below_the_threshold():
    found = evidence_for(("far", 0.9), ("gold", 0.5))
    at_threshold = calibration(low=found.query_gold_overlap, high=1.0)
    just_below = calibration(low=found.query_gold_overlap - 1e-9, high=1.0)

    assert FailureKind.LEXICAL_MISMATCH in [h.kind for h in interpret(found, at_threshold)]
    assert FailureKind.LEXICAL_MISMATCH not in [h.kind for h in interpret(found, just_below)]


def test_the_hard_negative_rule_needs_a_rank_symptom():
    """A miss has no competitor that outranked the gold."""
    missed = evidence_for(("near", 0.9), ("far", 0.5))
    permissive = calibration(low=0.0, high=0.0)
    assert FailureKind.HARD_NEGATIVE not in [h.kind for h in interpret(missed, permissive)]

    ranked = evidence_for(("near", 0.9), ("gold", 0.5))
    assert FailureKind.HARD_NEGATIVE in [h.kind for h in interpret(ranked, permissive)]


def test_calibration_quantiles_come_from_the_dataset(mini):
    pytest.importorskip("bm25s", reason="needs the 'lexical' extra")
    from embedlab.adapters.lexical import BM25Retriever

    retriever = BM25Retriever()
    retriever.index(mini.corpus)
    evidence = collect_evidence(
        retriever.search(mini.queries, k=10), mini.qrels, mini.corpus, queries=mini.queries
    )
    derived = calibrate(evidence)

    assert derived.n == len(evidence)
    assert derived.sufficient is False, "the 7-query fixture must not pretend to be calibrated"
    assert 0.0 <= derived.low_query_gold_overlap <= 1.0


def test_calibration_is_insufficient_below_the_floor():
    assert calibration(n=MIN_QUERIES_FOR_CALIBRATION - 1).sufficient is False
    assert calibration(n=MIN_QUERIES_FOR_CALIBRATION).sufficient is True
