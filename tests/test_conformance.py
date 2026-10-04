"""Does this harness agree with published numbers?

The most valuable test here, and the only one that can catch a whole-pipeline
error. Every other test checks a component against our own expectations; this
one checks the assembled thing — dataset loading, title concatenation, BM25
parameters, rank ordering, tie-breaking and metric computation — against a
figure somebody else published.

Requires the dataset, which is not in the repository:

    uv run --extra datasets python -m embedlab.cli.fetch beir/scifact/test

Skipped when it is absent, so the suite stays runnable offline.
"""

from pathlib import Path

import pytest

from embedlab.artifacts.dataset import load_dataset

SCIFACT = Path(__file__).parent.parent / "datasets" / "beir-scifact-test"

pytest.importorskip("bm25s", reason="needs the 'lexical' extra")
pytest.importorskip("ir_measures", reason="needs the 'measures' extra")

if not SCIFACT.exists():  # pragma: no cover - environment dependent
    pytest.skip(
        "SciFact not fetched; see this module's docstring",
        allow_module_level=True,
    )

from embedlab.adapters.lexical import BM25Retriever  # noqa: E402
from embedlab.stages.evaluate import evaluate  # noqa: E402

PUBLISHED_BM25_NDCG10 = 0.665
"""BEIR's reported BM25 nDCG@10 on SciFact (Elasticsearch, title + text).

Source: Thakur et al., "BEIR: A Heterogenous Benchmark for Zero-shot Evaluation
of Information Retrieval Models" (2021), table 2.
"""

TOLERANCE = 0.02
"""Tight enough to catch a broken pipeline, loose enough to absorb the
difference between two BM25 implementations.

An exact match is not the goal and would be suspicious: BEIR's number comes
from Elasticsearch's analyzer, ours from bm25s with Lucene-variant scoring and
a different tokeniser. What the test asserts is that we are measuring the same
quantity, not that we reproduce someone else's tokenisation.
"""


@pytest.fixture(scope="module")
def scifact():
    return load_dataset(SCIFACT)


@pytest.fixture(scope="module")
def bm25_run(scifact):
    retriever = BM25Retriever()
    retriever.index(scifact.corpus)
    return retriever.search(scifact.queries, k=10)


def test_the_dataset_loads_at_the_documented_size(scifact):
    assert len(scifact.corpus) == 5183
    assert len(scifact.queries) == 300
    assert len(scifact.judged_queries) == 300


def test_bm25_reproduces_the_published_ndcg(scifact, bm25_run):
    """If this drifts, suspect the harness before suspecting the retriever."""
    measured = evaluate(bm25_run, scifact.qrels, measures=["nDCG@10"]).aggregate["nDCG@10"]
    assert measured == pytest.approx(PUBLISHED_BM25_NDCG10, abs=TOLERANCE), (
        f"measured nDCG@10={measured:.4f} against published "
        f"{PUBLISHED_BM25_NDCG10} (tolerance {TOLERANCE})"
    )


def test_the_corpus_carries_titles(scifact):
    """BEIR's published numbers assume title + body; without it they are not
    comparable, and the difference is large enough to mistake for a bug."""
    first = next(iter(scifact.corpus.values()))
    assert "." in first
    assert len(first) > 200


def test_a_real_dataset_can_be_calibrated(scifact, bm25_run):
    """The 7-query fixture cannot; 300 queries can. This is what the floor is for."""
    from embedlab.stages.diagnose import diagnose

    diagnosis = diagnose(bm25_run, scifact.qrels, scifact.corpus, scifact.queries)
    assert diagnosis.calibration.sufficient is True
    assert diagnosis.calibration.n == 300


def test_the_rules_explain_some_but_not_all_failures(scifact, bm25_run):
    """Pins the honest state of the taxonomy rather than a hoped-for one.

    Both earlier absolute thresholds were measurably broken here: one fired on
    135 of 137 failures, the other on none. The bounds below are deliberately
    wide — they exist to catch a rule going degenerate again, not to freeze a
    particular distribution.
    """
    from embedlab.stages.diagnose import diagnose
    from embedlab.stages.diff import failure_sets

    diagnosis = diagnose(bm25_run, scifact.qrels, scifact.corpus, scifact.queries)
    failures = diagnosis.failing()
    sets = failure_sets(diagnosis)

    assert 80 < len(failures) < 200, f"{len(failures)} failures is outside the expected region"

    unexplained = len(sets.get("unexplained", ()))
    assert unexplained < len(failures), "no rule fires at all"
    assert unexplained > 0, "every failure explained is a sign the rules are too permissive"

    for name in ("lexical_mismatch", "hard_negative"):
        count = len(sets.get(name, ()))
        assert 0 < count < len(failures), f"{name} fires on {count} of {len(failures)} failures"
