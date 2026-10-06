"""Reordering candidates, and seeing what the reordering cost."""

import pytest

from embedlab.adapters.base import Reranker
from embedlab.adapters.coverage import CoverageReranker
from embedlab.cache.fingerprint import fingerprint
from embedlab.domain.ids import DocId, QueryId
from embedlab.domain.retrieval import Ranked
from embedlab.stages.rerank import rerank

TEXTS = {
    DocId("covers"): "rotate the production token safely",
    DocId("repeats"): "token token token token token",
    DocId("unrelated"): "billing invoices and refunds",
}
QUERIES = {QueryId("q"): "rotate production token"}


def run_of(*pairs):
    return {
        QueryId("q"): [
            Ranked(DocId(doc), position, score)
            for position, (doc, score) in enumerate(pairs, start=1)
        ]
    }


def test_it_satisfies_the_protocol_structurally():
    assert isinstance(CoverageReranker(), Reranker)


def test_coverage_prefers_breadth_over_repetition():
    """The reason this reranker is not a placeholder.

    BM25 rewards repeating one rare term; coverage asks how much of the query
    is present at all. A reranker that agreed with the retriever would reorder
    nothing and the stage would have nothing to measure.
    """
    scores = CoverageReranker().rerank(
        "rotate production token",
        [(DocId("covers"), TEXTS[DocId("covers")]), (DocId("repeats"), TEXTS[DocId("repeats")])],
    )
    assert scores[DocId("covers")] > scores[DocId("repeats")]


def test_reranking_can_reorder_the_candidates():
    reordered = rerank(
        run_of(("repeats", 0.9), ("covers", 0.4)), CoverageReranker(), QUERIES, TEXTS, k=10
    )
    assert [item.doc_id for item in reordered.after[QueryId("q")]] == ["covers", "repeats"]


def test_the_order_it_was_given_is_kept():
    """Without the before, a demotion is invisible."""
    before = run_of(("repeats", 0.9), ("covers", 0.4))
    reordered = rerank(before, CoverageReranker(), QUERIES, TEXTS, k=10)
    assert [item.doc_id for item in reordered.before[QueryId("q")]] == ["repeats", "covers"]


def test_a_reranker_cannot_add_a_document():
    """It sees candidates, never the corpus.

    The ceiling is the recall of the list it was handed. Given a deeper list
    than the final k it may well raise recall@k, by promoting what retrieval
    buried; what it can never do is return something that was not a candidate.
    """
    before = run_of(("repeats", 0.9), ("covers", 0.4))
    reordered = rerank(before, CoverageReranker(), QUERIES, TEXTS, k=10)

    assert {item.doc_id for item in reordered.after[QueryId("q")]} == {"repeats", "covers"}
    assert "unrelated" not in {item.doc_id for item in reordered.after[QueryId("q")]}


def test_it_cannot_return_more_than_k():
    reordered = rerank(
        run_of(("covers", 0.9), ("repeats", 0.4)), CoverageReranker(), QUERIES, TEXTS, k=1
    )
    assert len(reordered.after[QueryId("q")]) == 1


def test_a_query_with_no_terms_leaves_the_order_to_the_tie_break():
    """Nothing to cover: inventing a preference would be worse than a tie."""
    scores = CoverageReranker().rerank("", [(DocId("covers"), "anything")])
    assert set(scores.values()) == {0.0}


def test_moved_reports_where_the_gold_was_and_where_it_went():
    qrels = {QueryId("q"): {DocId("covers"): 1}}
    reordered = rerank(
        run_of(("repeats", 0.9), ("covers", 0.4)), CoverageReranker(), QUERIES, TEXTS, k=10
    )
    assert reordered.moved(qrels)[QueryId("q")] == (2, 1)


def test_moved_skips_a_query_with_no_positive_label():
    qrels = {QueryId("q"): {DocId("covers"): 0}}
    reordered = rerank(
        run_of(
            ("covers", 0.9),
        ),
        CoverageReranker(),
        QUERIES,
        TEXTS,
        k=10,
    )
    assert reordered.moved(qrels) == {}


def test_an_empty_candidate_list_is_handled():
    assert CoverageReranker().rerank("anything", []) == {}


@pytest.mark.parametrize(
    "changed",
    [{"lower": False}, {"length_penalty": 0.2}],
    ids=lambda kw: "-".join(kw),
)
def test_every_behavioural_parameter_reaches_the_descriptor(changed):
    baseline = fingerprint(dict(CoverageReranker().descriptor))
    assert fingerprint(dict(CoverageReranker(**changed).descriptor)) != baseline


def test_the_length_penalty_discounts_long_candidates():
    """Long candidates cover more terms by accident."""
    plain = CoverageReranker(length_penalty=0.0)
    penalised = CoverageReranker(length_penalty=0.5)
    long_text = "rotate production token " + " ".join(f"w{i}" for i in range(200))

    assert (
        penalised.rerank("rotate production token", [(DocId("d"), long_text)])[DocId("d")]
        < plain.rerank("rotate production token", [(DocId("d"), long_text)])[DocId("d")]
    )
