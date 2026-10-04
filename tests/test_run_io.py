"""Artifact round-trips. A score off by one bit would fabricate a diff."""

import math

import pytest

from embedlab.artifacts.run_io import read_run, write_run
from embedlab.domain.ids import DocId, QueryId
from embedlab.domain.retrieval import Ranked


def run_of(pairs_by_query):
    return {
        QueryId(query_id): [
            Ranked(DocId(doc_id), position, score)
            for position, (doc_id, score) in enumerate(pairs, start=1)
        ]
        for query_id, pairs in pairs_by_query.items()
    }


def test_round_trip_preserves_order_and_ranks(tmp_path):
    original = run_of({"q1": [("b", 0.9), ("a", 0.5)], "q2": [("c", 0.1)]})
    write_run(tmp_path, original)
    restored = read_run(tmp_path)

    assert restored.keys() == original.keys()
    for query_id in original:
        assert [(i.doc_id, i.rank) for i in restored[query_id]] == [
            (i.doc_id, i.rank) for i in original[query_id]
        ]


@pytest.mark.parametrize(
    "score",
    [0.0, 1.0, 0.8709996938705444, 1e-300, 1 - 2**-52, math.pi],
    ids=["zero", "one", "the-observed-tie", "tiny", "near-one", "pi"],
)
def test_scores_survive_bit_exactly(tmp_path, score):
    write_run(tmp_path, run_of({"q": [("d", score)]}))
    assert read_run(tmp_path)[QueryId("q")][0].score == score


def test_a_stored_tie_keeps_the_order_it_was_written_in(tmp_path):
    """Re-sorting on read would re-break the tie, possibly differently."""
    original = run_of({"q": [("zebra", 0.5), ("apple", 0.5)]})
    write_run(tmp_path, original)
    assert [i.doc_id for i in read_run(tmp_path)[QueryId("q")]] == ["zebra", "apple"]


def test_an_empty_run_round_trips(tmp_path):
    write_run(tmp_path, {})
    assert read_run(tmp_path) == {}


def test_query_without_results_is_simply_absent(tmp_path):
    write_run(
        tmp_path, {QueryId("q1"): [], QueryId("q2"): run_of({"x": [("d", 1.0)]})[QueryId("x")]}
    )
    restored = read_run(tmp_path)
    assert set(restored) == {QueryId("q2")}


def test_unicode_identifiers_survive(tmp_path):
    original = run_of({"requête-1": [("docs/sécurité/jetons.md", 0.5)]})
    write_run(tmp_path, original)
    restored = read_run(tmp_path)
    assert restored[QueryId("requête-1")][0].doc_id == "docs/sécurité/jetons.md"
