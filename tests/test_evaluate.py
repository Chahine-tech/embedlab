"""Metrics: delegated to ir_measures, but fed the ordering we actually reported."""

import pytest

from embedlab.domain.ids import DocId, QueryId
from embedlab.domain.retrieval import Ranked

pytest.importorskip("ir_measures", reason="needs the 'measures' extra")

from embedlab.stages.evaluate import evaluate


def ranked(*doc_ids, scores=None):
    values = scores if scores is not None else [1.0 / (index + 1) for index in range(len(doc_ids))]
    return [
        Ranked(DocId(doc_id), position, score)
        for position, (doc_id, score) in enumerate(zip(doc_ids, values, strict=True), start=1)
    ]


def test_gold_first_is_a_perfect_reciprocal_rank():
    run = {QueryId("q1"): ranked("gold", "other")}
    qrels = {QueryId("q1"): {DocId("gold"): 1}}
    assert evaluate(run, qrels, measures=["RR"]).per_query[QueryId("q1")]["RR"] == 1.0


def test_gold_second_halves_the_reciprocal_rank():
    run = {QueryId("q1"): ranked("other", "gold")}
    qrels = {QueryId("q1"): {DocId("gold"): 1}}
    assert evaluate(run, qrels, measures=["RR"]).per_query[QueryId("q1")]["RR"] == 0.5


def test_reported_order_is_the_evaluated_order_even_under_a_tie():
    """Why scores are synthesised from rank rather than copied.

    Both documents carry the same raw score. ir_measures would re-sort by score
    and apply trec_eval's own tie-break (document id descending), which here
    would lift "gold" above "alpha" and report RR=1.0 — a different ranking from
    the one the user is shown. Synthesising from rank keeps the two in step.
    """
    run = {QueryId("q1"): ranked("alpha", "gold", scores=[0.5, 0.5])}
    qrels = {QueryId("q1"): {DocId("gold"): 1}}

    assert evaluate(run, qrels, measures=["RR"]).per_query[QueryId("q1")]["RR"] == 0.5

    import ir_measures

    raw = {"q1": {"alpha": 0.5, "gold": 0.5}}
    naive = ir_measures.calc_aggregate([ir_measures.parse_measure("RR")], {"q1": {"gold": 1}}, raw)
    assert naive[ir_measures.parse_measure("RR")] == 1.0, (
        "if this ever equals 0.5, trec_eval's tie-break changed and the comment above is stale"
    )


def test_unjudged_queries_are_excluded():
    run = {QueryId("q1"): ranked("gold"), QueryId("q2"): ranked("whatever")}
    qrels = {QueryId("q1"): {DocId("gold"): 1}, QueryId("q2"): {DocId("x"): 0}}
    evaluation = evaluate(run, qrels, measures=["RR"])
    assert set(evaluation.per_query) == {QueryId("q1")}


def test_graded_relevance_reaches_ndcg():
    """A grade-2 document first must beat a grade-1 document first."""
    qrels = {QueryId("q1"): {DocId("high"): 2, DocId("low"): 1}}
    better = evaluate({QueryId("q1"): ranked("high", "low")}, qrels, measures=["nDCG@10"])
    worse = evaluate({QueryId("q1"): ranked("low", "high")}, qrels, measures=["nDCG@10"])
    assert better.aggregate["nDCG@10"] > worse.aggregate["nDCG@10"]


def test_aggregate_is_the_mean_over_judged_queries():
    qrels = {QueryId("q1"): {DocId("gold"): 1}, QueryId("q2"): {DocId("gold"): 1}}
    run = {QueryId("q1"): ranked("gold", "x"), QueryId("q2"): ranked("x", "gold")}
    evaluation = evaluate(run, qrels, measures=["RR"])
    assert evaluation.aggregate["RR"] == pytest.approx(0.75)


def test_measure_names_round_trip():
    qrels = {QueryId("q1"): {DocId("gold"): 1}}
    evaluation = evaluate({QueryId("q1"): ranked("gold")}, qrels, measures=["nDCG@10", "R@10"])
    assert set(evaluation.aggregate) == {"nDCG@10", "R@10"}
    assert set(evaluation.per_query[QueryId("q1")]) == {"nDCG@10", "R@10"}
