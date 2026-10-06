"""Ordering and gold-rank semantics: the arithmetic every diff sits on."""

from embedlab.domain.ids import DocId, UnitId
from embedlab.domain.retrieval import Ranked, order_deterministically, rank_of_best_relevant


def scored(**by_doc: float) -> dict[UnitId, float]:
    """Scores keyed by DocId.

    The NewType is the point: it is what stops a ChunkId being passed where a
    DocId belongs once chunking exists, so tests go through the constructor
    rather than weakening the signature to plain `str`.
    """
    result: dict[UnitId, float] = {}
    for doc_id, score in by_doc.items():
        result[DocId(doc_id)] = score
    return result


def test_orders_by_descending_score():
    ranked = order_deterministically(scored(a=0.1, b=0.9, c=0.5), 3)
    assert [item.doc_id for item in ranked] == ["b", "c", "a"]
    assert [item.rank for item in ranked] == [1, 2, 3]


def test_ties_break_by_doc_id_ascending():
    """Any total order is arbitrary; what matters is that it never varies."""
    ranked = order_deterministically(scored(zebra=0.5, apple=0.5), 2)
    assert [item.doc_id for item in ranked] == ["apple", "zebra"]


def test_tie_break_is_independent_of_insertion_order():
    """The failure this prevents: a rerun shuffling equal scores and faking a diff."""
    first = order_deterministically(scored(a=0.5, b=0.5, c=0.5), 3)
    second = order_deterministically(scored(c=0.5, b=0.5, a=0.5), 3)
    assert [item.doc_id for item in first] == [item.doc_id for item in second]


def test_truncates_to_k():
    assert len(order_deterministically(scored(a=0.3, b=0.2, c=0.1), 2)) == 2


def test_k_larger_than_corpus_is_safe():
    assert len(order_deterministically(scored(a=0.3), 10)) == 1


def test_ranks_are_one_based():
    assert order_deterministically(scored(a=1.0), 1)[0].rank == 1


def test_best_relevant_rank_takes_the_first_positive():
    ranked = [
        Ranked(DocId("x"), 1, 0.9),
        Ranked(DocId("gold-b"), 2, 0.8),
        Ranked(DocId("gold-a"), 3, 0.7),
    ]
    relevant = {DocId("gold-a"): 1, DocId("gold-b"): 2}
    assert rank_of_best_relevant(ranked, relevant) == 2


def test_best_relevant_rank_is_none_when_absent():
    ranked = [Ranked(DocId("x"), 1, 0.9)]
    assert rank_of_best_relevant(ranked, {DocId("gold"): 1}) is None


def test_zero_graded_labels_are_not_relevant():
    """relevance 0 means judged-and-irrelevant, not relevant."""
    ranked = [Ranked(DocId("judged-bad"), 1, 0.9)]
    assert rank_of_best_relevant(ranked, {DocId("judged-bad"): 0}) is None
