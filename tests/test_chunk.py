"""Cutting documents up, and putting the results back together."""

import pytest

from embedlab.domain.ids import ChunkId, DocId, QueryId
from embedlab.domain.retrieval import Ranked
from embedlab.stages.chunk import fixed_words, fold_to_documents, whole_documents

CORPUS = {
    DocId("short"): "one two three",
    DocId("long"): " ".join(f"w{i}" for i in range(25)),
    DocId("empty"): "   ",
}


def test_whole_documents_is_the_identity_cut():
    chunks = whole_documents({DocId("a"): "text", DocId("b"): "other"})
    assert set(chunks.texts) == {ChunkId("a"), ChunkId("b")}
    assert chunks.source[ChunkId("a")] == DocId("a")
    assert chunks.strategy == {"kind": "whole"}


def test_a_short_document_becomes_one_chunk():
    chunks = fixed_words(CORPUS, size=10, overlap=2)
    assert chunks.of(DocId("short")) == [ChunkId("short#0")]
    assert chunks.texts[ChunkId("short#0")] == "one two three"


def test_a_long_document_becomes_several_overlapping_chunks():
    chunks = fixed_words(CORPUS, size=10, overlap=2)
    pieces = chunks.of(DocId("long"))
    assert len(pieces) > 1

    first = chunks.texts[ChunkId("long#0")].split()
    second = chunks.texts[ChunkId("long#1")].split()
    assert len(first) == 10
    assert first[-2:] == second[:2], "the overlap must actually overlap"


def test_every_word_survives_the_cut():
    """A window boundary may split an idea; it must not drop a word."""
    chunks = fixed_words(CORPUS, size=7, overlap=0)
    seen: list[str] = []
    for index in range(len(chunks.of(DocId("long")))):
        seen.extend(chunks.texts[ChunkId(f"long#{index}")].split())
    assert seen == CORPUS[DocId("long")].split()


def test_an_empty_document_produces_nothing():
    chunks = fixed_words(CORPUS, size=10, overlap=0)
    assert chunks.of(DocId("empty")) == []


def test_chunk_ids_are_derived_from_the_document():
    """A chunk has to say where it came from, or folding is guesswork."""
    chunks = fixed_words(CORPUS, size=5, overlap=0)
    for chunk_id, doc_id in chunks.source.items():
        assert str(chunk_id).startswith(str(doc_id))


def test_the_strategy_travels_with_the_chunks():
    """A different cut is a different corpus, so it belongs in a cache key."""
    chunks = fixed_words(CORPUS, size=12, overlap=3)
    assert chunks.strategy == {"kind": "fixed_words", "size": 12, "overlap": 3}
    assert fixed_words(CORPUS, size=12, overlap=4).strategy != chunks.strategy


@pytest.mark.parametrize(
    ("size", "overlap"),
    [(0, 0), (-1, 0), (10, 10), (10, 11), (10, -1)],
    ids=["zero-size", "negative-size", "overlap-equals-size", "overlap-exceeds", "negative"],
)
def test_an_impossible_cut_is_refused(size, overlap):
    with pytest.raises(ValueError, match=r"size|overlap"):
        fixed_words(CORPUS, size=size, overlap=overlap)


def test_the_chunk_count_per_document_is_available():
    chunks = fixed_words(CORPUS, size=10, overlap=2)
    counts = chunks.per_document
    assert counts[DocId("short")] == 1
    assert counts[DocId("long")] > 1
    assert DocId("empty") not in counts


# --- folding ----------------------------------------------------------------


def ranked(*pairs):
    return [
        Ranked(DocId(chunk), position, score)
        for position, (chunk, score) in enumerate(pairs, start=1)
    ]


def test_a_document_keeps_its_best_chunk():
    chunks = fixed_words({DocId("a"): " ".join(f"w{i}" for i in range(20))}, size=10, overlap=0)
    run = {QueryId("q"): ranked(("a#1", 0.9), ("a#0", 0.4))}

    folded = fold_to_documents(run, chunks, k=10)
    assert [(i.doc_id, i.score) for i in folded[QueryId("q")]] == [("a", 0.9)]


def test_two_documents_are_ordered_by_their_best_chunks():
    corpus = {DocId("a"): "x " * 30, DocId("b"): "y " * 30}
    chunks = fixed_words(corpus, size=10, overlap=0)
    run = {QueryId("q"): ranked(("a#0", 0.3), ("b#2", 0.8), ("a#1", 0.7), ("b#0", 0.1))}

    folded = fold_to_documents(run, chunks, k=10)
    assert [i.doc_id for i in folded[QueryId("q")]] == ["b", "a"]
    assert [i.rank for i in folded[QueryId("q")]] == [1, 2]


def test_folding_cannot_reorder_an_exact_tie():
    """Folding must not become a second, invisible tie-break."""
    corpus = {DocId("zebra"): "x " * 12, DocId("apple"): "y " * 12}
    chunks = fixed_words(corpus, size=10, overlap=0)
    run = {QueryId("q"): ranked(("zebra#0", 0.5), ("apple#0", 0.5))}

    folded = fold_to_documents(run, chunks, k=10)
    assert [i.doc_id for i in folded[QueryId("q")]] == ["apple", "zebra"]


def test_folding_truncates_to_k():
    corpus = {DocId(f"d{i}"): "word " * 5 for i in range(6)}
    chunks = whole_documents(corpus)
    run = {QueryId("q"): ranked(*((f"d{i}", 1.0 - i / 10) for i in range(6)))}
    assert len(fold_to_documents(run, chunks, k=3)[QueryId("q")]) == 3


def test_a_chunk_from_outside_the_cut_is_ignored():
    """A stale run must not invent a document that this cut does not contain."""
    chunks = whole_documents({DocId("a"): "text"})
    run = {QueryId("q"): ranked(("a", 0.9), ("ghost", 0.8))}
    assert [i.doc_id for i in fold_to_documents(run, chunks, k=10)[QueryId("q")]] == ["a"]
