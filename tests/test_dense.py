"""The dense adapter, and the descriptor that keeps its artifacts honest.

Uses the 20-document fixture rather than a real corpus: these assertions are
about the adapter's contract, not about retrieval quality, and a small corpus
keeps them fast enough to run on every change.
"""

import pytest

from embedlab.adapters.base import Retriever
from embedlab.adapters.dense import DenseRetriever
from embedlab.cache.fingerprint import fingerprint
from embedlab.domain.ids import QueryId

MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def test_an_unknown_dtype_is_refused_before_anything_is_downloaded():
    """Fails on construction, not three minutes into an encode."""
    with pytest.raises(ValueError, match="unknown dtype"):
        DenseRetriever(model_id=MODEL, dtype="bfloat16")


def test_it_satisfies_the_protocol_structurally():
    assert isinstance(DenseRetriever(model_id=MODEL), Retriever)


def test_the_name_defaults_to_the_model_without_its_namespace():
    assert DenseRetriever(model_id=MODEL).name == "all-MiniLM-L6-v2"
    assert DenseRetriever(model_id=MODEL, name="jina-v5").name == "jina-v5"


def test_searching_before_indexing_raises():
    with pytest.raises(RuntimeError, match=r"index\(\) must be called"):
        DenseRetriever(model_id=MODEL).search({QueryId("q"): "text"}, k=5)


@pytest.mark.parametrize(
    "changed",
    [
        {"dtype": "float16"},
        {"device": "mps"},
        {"normalize": False},
        {"batch_size": 8},
        {"query_prompt": "query: "},
        {"document_prompt": "passage: "},
        {"max_seq_length": 128},
        {"revision": "deadbeef"},
    ],
    ids=lambda kw: "-".join(kw),
)
def test_everything_that_can_move_an_embedding_reaches_the_descriptor(changed):
    """An omission here is an artifact reused across a change that moved the
    numbers — the exact failure the fingerprint exists to prevent."""
    baseline = fingerprint(dict(DenseRetriever(model_id=MODEL).descriptor))
    altered = fingerprint(dict(DenseRetriever(model_id=MODEL, **changed).descriptor))
    assert baseline != altered, f"{changed} does not change the descriptor"


def test_a_different_model_is_a_different_descriptor():
    other = "sentence-transformers/all-MiniLM-L12-v2"
    assert fingerprint(dict(DenseRetriever(model_id=MODEL).descriptor)) != fingerprint(
        dict(DenseRetriever(model_id=other).descriptor)
    )


def test_the_descriptor_names_the_weights_not_only_the_label():
    """A model republished under the same name is a different model."""
    pinned = DenseRetriever(model_id=MODEL, revision="abc123").descriptor
    assert pinned["revision"] == "abc123"
    assert pinned["model_id"] == MODEL


def test_an_unresolved_revision_is_none_rather_than_a_guess():
    """A wrong revision is worse than an absent one: it would claim two
    different sets of weights are the same."""
    assert DenseRetriever(model_id=MODEL).descriptor["revision"] is None


def test_the_descriptor_is_fingerprintable():
    assert fingerprint(dict(DenseRetriever(model_id=MODEL).descriptor)).startswith("fp_")


# --- tests below need the model weights on disk -----------------------------

pytest.importorskip("sentence_transformers", reason="needs the 'local' extra")


@pytest.fixture(scope="module")
def indexed(mini_corpus):
    retriever = DenseRetriever(model_id=MODEL, device="cpu", dtype="float32")
    retriever.index(mini_corpus)
    return retriever


@pytest.fixture(scope="module")
def mini_corpus():
    from pathlib import Path

    from embedlab.artifacts.dataset import load_dataset

    return load_dataset(Path(__file__).parent / "fixtures" / "mini").corpus


@pytest.fixture(scope="module")
def mini_queries():
    from pathlib import Path

    from embedlab.artifacts.dataset import load_dataset

    return load_dataset(Path(__file__).parent / "fixtures" / "mini").queries


@pytest.mark.slow
def test_it_returns_contiguous_one_based_ranks(indexed, mini_queries):
    for ranked in indexed.search(mini_queries, k=5).values():
        assert [item.rank for item in ranked] == [1, 2, 3, 4, 5]


@pytest.mark.slow
def test_k_is_clamped_to_the_corpus(indexed, mini_queries, mini_corpus):
    for ranked in indexed.search(mini_queries, k=500).values():
        assert len(ranked) == len(mini_corpus)


@pytest.mark.slow
def test_rerunning_the_same_configuration_is_bit_identical(mini_corpus, mini_queries):
    """Without this the cache would be returning a different run each time."""
    first = DenseRetriever(model_id=MODEL, device="cpu", dtype="float32")
    first.index(mini_corpus)
    second = DenseRetriever(model_id=MODEL, device="cpu", dtype="float32")
    second.index(mini_corpus)

    a, b = first.search(mini_queries, k=10), second.search(mini_queries, k=10)
    for query_id in a:
        assert [(i.doc_id, i.rank, i.score) for i in a[query_id]] == [
            (i.doc_id, i.rank, i.score) for i in b[query_id]
        ]


@pytest.mark.slow
def test_normalisation_changes_the_scores(mini_corpus, mini_queries):
    """Guards against a descriptor advertising a knob the adapter ignores."""
    plain = DenseRetriever(model_id=MODEL, normalize=False)
    plain.index(mini_corpus)
    normalised = DenseRetriever(model_id=MODEL, normalize=True)
    normalised.index(mini_corpus)

    a = [i.score for i in plain.search(mini_queries, k=5)[QueryId("q-rotate")]]
    b = [i.score for i in normalised.search(mini_queries, k=5)[QueryId("q-rotate")]]
    assert a != b


@pytest.mark.slow
def test_a_query_prompt_changes_the_embedding(mini_corpus, mini_queries):
    """Jina and Qwen require distinct query and document prefixes, so a prompt
    that silently did nothing would be a quiet correctness bug."""
    plain = DenseRetriever(model_id=MODEL)
    plain.index(mini_corpus)
    prompted = DenseRetriever(model_id=MODEL, query_prompt="Represent this query: ")
    prompted.index(mini_corpus)

    a = [i.score for i in plain.search(mini_queries, k=5)[QueryId("q-rotate")]]
    b = [i.score for i in prompted.search(mini_queries, k=5)[QueryId("q-rotate")]]
    assert a != b


# --- the measurements behind the fingerprint's contents --------------------
#
# These pin findings from a full SciFact run (5,183 documents, 300 queries):
# float16 reordered the top-10 of 22 queries and moved 2 gold ranks, while MPS
# and a different batch size changed nothing at all. They are repeated here on
# the fixture, at the level of the embeddings rather than the rankings, so a
# torch upgrade that changes any of it fails a test instead of silently
# changing every cached artifact's meaning.


def embeddings(corpus, **kwargs):
    """The indexed matrix itself: these assertions are about the artifact the
    cache stores, not about the rankings derived from it."""
    retriever = DenseRetriever(model_id=MODEL, **kwargs)
    retriever.index(corpus)
    matrix = retriever._matrix
    assert matrix is not None, "index() left no matrix behind"
    return matrix


@pytest.mark.slow
def test_dtype_changes_the_embeddings(mini_corpus):
    """Why dtype is in the cache key, and the one precaution that was needed."""
    import numpy as np

    at32 = embeddings(mini_corpus, device="cpu", dtype="float32")
    at16 = embeddings(mini_corpus, device="cpu", dtype="float16")

    assert not np.allclose(at32, at16.astype(np.float32), atol=0, rtol=0), (
        "float16 produced bit-identical embeddings; if torch gained an exact "
        "path, the cost of keeping dtype in the key should be revisited"
    )


FLOAT32_EPSILON = 1.2e-7


@pytest.mark.slow
def test_batch_size_changes_the_embeddings_but_only_in_the_last_bits(mini_corpus):
    """Padding differs between batch shapes and float addition is not
    associative, so the vectors are not identical — measured at ~1e-7, which is
    float32 epsilon. Small enough to have left every SciFact ranking untouched,
    and still enough that a cached matrix produced at another batch size is not
    the matrix this configuration asks for."""
    import numpy as np

    small = embeddings(mini_corpus, device="cpu", dtype="float32", batch_size=4)
    large = embeddings(mini_corpus, device="cpu", dtype="float32", batch_size=64)

    assert not np.array_equal(small, large)
    assert np.abs(small - large).max() < 1e-5, "reassociation noise should stay near epsilon"


@pytest.mark.slow
def test_the_device_changes_the_embeddings_but_only_in_the_last_bits(mini_corpus):
    """Same shape of result as batch size, measured at ~2e-7 for MPS against
    CPU. ROCm and CUDA are untested and have no reason to be kinder."""
    import numpy as np
    import torch

    if not torch.backends.mps.is_available():
        pytest.skip("no MPS backend on this machine")

    on_cpu = embeddings(mini_corpus, device="cpu", dtype="float32")
    on_gpu = embeddings(mini_corpus, device="mps", dtype="float32")

    assert not np.array_equal(on_cpu, on_gpu)
    assert np.abs(on_cpu - on_gpu).max() < 1e-5


@pytest.mark.slow
def test_dtype_moves_the_embeddings_far_more_than_device_or_batching(mini_corpus):
    """The ordering that justifies treating dtype as the active risk.

    float16 moved the vectors by ~4e-4 against ~1e-7 for the other two: three
    orders of magnitude, and the only one of the three that reordered any
    SciFact ranking.
    """
    import numpy as np

    base = embeddings(mini_corpus, device="cpu", dtype="float32", batch_size=32)
    half = embeddings(mini_corpus, device="cpu", dtype="float16", batch_size=32)
    batched = embeddings(mini_corpus, device="cpu", dtype="float32", batch_size=4)

    assert np.abs(base - half.astype(np.float32)).max() > 100 * np.abs(base - batched).max()
