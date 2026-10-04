"""Caching must make a rerun cheaper without making it different."""

import json

import pytest

from embedlab.artifacts.run_io import STAGE_FILE, read_stage_provenance
from embedlab.cache.store import CacheStore
from embedlab.pipeline.run import execute, retrieve_key

pytest.importorskip("bm25s", reason="needs the 'lexical' extra")
pytest.importorskip("ir_measures", reason="needs the 'measures' extra")

from embedlab.adapters.lexical import BM25Retriever
from embedlab.adapters.tfidf import TfidfRetriever


def stage_of(manifest, name):
    """A stage that must exist; fails the test loudly rather than on None."""
    provenance = manifest.stage(name)
    assert provenance is not None, f"manifest is missing the {name!r} stage"
    return provenance


@pytest.fixture
def store(tmp_path):
    return CacheStore(tmp_path / "cache")


def test_a_cold_run_computes_and_a_warm_run_reuses(mini, store):
    cold = execute(mini, BM25Retriever(), k=10, store=store)
    warm = execute(mini, BM25Retriever(), k=10, store=store)

    assert cold.from_cache is False
    assert warm.from_cache is True


def test_a_reused_run_is_bit_identical(mini, store):
    """A cache that returns something slightly different is worse than none."""
    cold = execute(mini, BM25Retriever(), k=10, store=store)
    warm = execute(mini, BM25Retriever(), k=10, store=store)

    assert cold.run.keys() == warm.run.keys()
    for query_id in cold.run:
        assert [(i.doc_id, i.rank, i.score) for i in cold.run[query_id]] == [
            (i.doc_id, i.rank, i.score) for i in warm.run[query_id]
        ]
    assert cold.evaluation.aggregate == warm.evaluation.aggregate


def test_a_reused_stage_reports_the_environment_it_was_produced_in(mini, store):
    """The reason provenance lives on the entry.

    After a hit, the retrieve stage must still describe the moment it was
    computed. If it reported today's environment instead, the gate would believe
    a month-old artifact was produced with today's libraries.
    """
    cold = execute(mini, BM25Retriever(), k=10, store=store)
    stored = stage_of(cold.manifest, "retrieve")

    entry = store.lookup("retrieve", stored.key)
    tampered = json.loads((entry / STAGE_FILE).read_text())
    tampered["environment"]["packages"]["numpy"] = "0.0.1-from-the-past"
    tampered["created_at"] = "2020-01-01T00:00:00Z"
    (entry / STAGE_FILE).write_text(json.dumps(tampered))

    warm = execute(mini, BM25Retriever(), k=10, store=store)
    retrieve = stage_of(warm.manifest, "retrieve")

    assert retrieve.environment.packages["numpy"] == "0.0.1-from-the-past"
    assert retrieve.created_at.year == 2020
    # The freshly computed stage is unaffected and describes today.
    assert (
        stage_of(warm.manifest, "evaluate").environment.packages["numpy"] != "0.0.1-from-the-past"
    )


def test_changing_a_retriever_parameter_misses_the_cache(mini, store):
    execute(mini, BM25Retriever(), k=10, store=store)
    altered = execute(mini, BM25Retriever(k1=0.9), k=10, store=store)
    assert altered.from_cache is False


def test_changing_k_misses_the_cache(mini, store):
    execute(mini, BM25Retriever(), k=10, store=store)
    assert execute(mini, BM25Retriever(), k=5, store=store).from_cache is False


def test_different_retrievers_do_not_share_an_entry(mini, store):
    execute(mini, BM25Retriever(), k=10, store=store)
    assert execute(mini, TfidfRetriever(), k=10, store=store).from_cache is False


def test_relabelling_the_dataset_does_not_invalidate_retrieval(mini, store):
    """Retrieval does not depend on the labels, so re-judging must stay cheap."""
    execute(mini, BM25Retriever(), k=10, store=store)

    relabelled = type(mini)(
        name=mini.name,
        corpus=mini.corpus,
        queries=mini.queries,
        qrels={next(iter(mini.judged_queries)): {next(iter(mini.corpus)): 1}},
    )
    reused = execute(relabelled, BM25Retriever(), k=10, store=store)
    assert reused.from_cache is True


def test_running_without_a_store_caches_nothing(mini, store):
    assert execute(mini, BM25Retriever(), k=10).from_cache is False
    assert store.lookup("retrieve", retrieve_key(mini, BM25Retriever(), k=10)) is None


def test_an_entry_whose_recorded_key_contradicts_its_location_is_refused(mini, store):
    """Corruption must surface, not be read as a valid artifact."""
    cold = execute(mini, BM25Retriever(), k=10, store=store)
    entry = store.lookup("retrieve", stage_of(cold.manifest, "retrieve").key)

    tampered = json.loads((entry / STAGE_FILE).read_text())
    tampered["key"] = "fp_" + "9" * 32
    (entry / STAGE_FILE).write_text(json.dumps(tampered))

    with pytest.raises(ValueError, match="corrupt cache entry"):
        execute(mini, BM25Retriever(), k=10, store=store)


def test_reading_provenance_without_an_expected_key_is_permitted(mini, store):
    cold = execute(mini, BM25Retriever(), k=10, store=store)
    entry = store.lookup("retrieve", stage_of(cold.manifest, "retrieve").key)
    assert read_stage_provenance(entry).stage == "retrieve"


def test_the_evaluate_stage_is_not_cached(mini, store):
    """Metrics are cheap, so a metric change never needs cache invalidation."""
    cold = execute(mini, BM25Retriever(), k=10, store=store)
    assert store.lookup("evaluate", stage_of(cold.manifest, "evaluate").key) is None
