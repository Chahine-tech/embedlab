"""Adapter guarantees: reproducible output, and descriptors that hide nothing."""

import pytest

from embedlab.adapters.base import Retriever
from embedlab.cache.fingerprint import fingerprint
from embedlab.domain.ids import QueryId

bm25s = pytest.importorskip("bm25s", reason="needs the 'lexical' extra")

from embedlab.adapters.lexical import BM25Retriever  # noqa: E402
from embedlab.adapters.tfidf import TfidfRetriever  # noqa: E402


def build(factory, dataset):
    retriever = factory()
    retriever.index(dataset.corpus)
    return retriever


@pytest.fixture(params=[BM25Retriever, TfidfRetriever], ids=["bm25", "tfidf"])
def factory(request):
    return request.param


def test_satisfies_the_protocol_structurally(factory):
    """No base class: shape is the contract."""
    assert isinstance(factory(), Retriever)


def test_rerunning_gives_a_bit_identical_run(factory, mini):
    """If this ever fails, every diff the tool produces is untrustworthy."""
    first = build(factory, mini).search(mini.queries, k=10)
    second = build(factory, mini).search(mini.queries, k=10)

    assert first.keys() == second.keys()
    for query_id in first:
        assert [(i.doc_id, i.rank, i.score) for i in first[query_id]] == [
            (i.doc_id, i.rank, i.score) for i in second[query_id]
        ]


def test_index_order_does_not_depend_on_dict_order(factory, mini):
    shuffled = dict(reversed(list(mini.corpus.items())))
    straight = build(factory, mini).search(mini.queries, k=10)

    retriever = factory()
    retriever.index(shuffled)
    reversed_run = retriever.search(mini.queries, k=10)

    for query_id in straight:
        assert [i.doc_id for i in straight[query_id]] == [i.doc_id for i in reversed_run[query_id]]


def test_ranks_are_contiguous_and_one_based(factory, mini):
    run = build(factory, mini).search(mini.queries, k=5)
    for ranked in run.values():
        assert [item.rank for item in ranked] == list(range(1, len(ranked) + 1))


def test_k_is_clamped_to_the_corpus(factory, mini):
    run = build(factory, mini).search(mini.queries, k=500)
    for ranked in run.values():
        assert len(ranked) == len(mini.corpus)


def test_empty_queries_are_handled(factory, mini):
    assert build(factory, mini).search({}, k=10) == {}


def test_searching_before_indexing_raises(factory, mini):
    with pytest.raises(RuntimeError, match="index\\(\\) must be called"):
        factory().search(mini.queries, k=10)


def test_descriptor_is_fingerprintable(factory):
    assert fingerprint(dict(factory().descriptor)).startswith("fp_")


@pytest.mark.parametrize(
    ("factory_fn", "changed"),
    [
        (BM25Retriever, {"k1": 1.2}),
        (BM25Retriever, {"b": 0.4}),
        (BM25Retriever, {"method": "robertson"}),
        (BM25Retriever, {"stopwords": None}),
        (BM25Retriever, {"dtype": "float64"}),
        (TfidfRetriever, {"ngram_range": (2, 4)}),
        (TfidfRetriever, {"sublinear_tf": False}),
        (TfidfRetriever, {"smooth_idf": False}),
        (TfidfRetriever, {"dtype": "float32"}),
    ],
)
def test_every_behavioural_parameter_reaches_the_descriptor(factory_fn, changed):
    """A parameter missing from the descriptor is an invisible source of fake
    diffs: the cache would reuse an artifact produced under different settings."""
    baseline = fingerprint(dict(factory_fn().descriptor))
    altered = fingerprint(dict(factory_fn(**changed).descriptor))
    assert baseline != altered, f"{changed} does not change the descriptor"


def test_bm25_parameters_actually_change_the_ranking(mini):
    """Guards against a descriptor that advertises a knob the adapter ignores."""
    default = build(BM25Retriever, mini).search(mini.queries, k=10)

    aggressive = BM25Retriever(b=0.0, k1=0.3)
    aggressive.index(mini.corpus)
    altered = aggressive.search(mini.queries, k=10)

    scores_default = [i.score for i in default[QueryId("q-rotate")]]
    scores_altered = [i.score for i in altered[QueryId("q-rotate")]]
    assert scores_default != scores_altered


def test_bm25_rejects_an_unknown_stemmer(mini):
    with pytest.raises(ValueError, match="unknown stemmer"):
        BM25Retriever(stemmer="nonexistent").index(mini.corpus)


def test_tfidf_handles_a_query_with_no_known_ngrams(mini):
    """An all-zero query vector must not divide by zero."""
    retriever = build(TfidfRetriever, mini)
    run = retriever.search({QueryId("q"): "中文テスト"}, k=5)
    assert len(run[QueryId("q")]) == 5
    assert all(item.score == 0.0 for item in run[QueryId("q")])


def test_disabling_stopwords_changes_the_scores(mini):
    """Pins a path that only the descriptor test covered before.

    bm25s annotates `stopwords` as str | list[str] but accepts None to disable
    removal entirely. The adapter passes None through, so the behaviour needs a
    runtime test rather than a type-level assurance.
    """
    with_stopwords = build(BM25Retriever, mini).search(mini.queries, k=5)

    without = BM25Retriever(stopwords=None)
    without.index(mini.corpus)
    plain = without.search(mini.queries, k=5)

    assert [i.score for i in with_stopwords[QueryId("q-rotate")]] != [
        i.score for i in plain[QueryId("q-rotate")]
    ]
