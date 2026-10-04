"""The whole slice: dataset in, diagnosed and diffed runs out."""

import pytest

from embedlab.artifacts.comparability import Trust, assess
from embedlab.domain.taxonomy import Symptom
from embedlab.pipeline.run import STAGES_FOR_RETRIEVAL_DIFF, execute
from embedlab.stages.diff import compare_runs

pytest.importorskip("bm25s", reason="needs the 'lexical' extra")
pytest.importorskip("ir_measures", reason="needs the 'measures' extra")

from embedlab.adapters.lexical import BM25Retriever
from embedlab.adapters.tfidf import TfidfRetriever

UNCHANGED: dict[str, bool | None] = dict.fromkeys(STAGES_FOR_RETRIEVAL_DIFF, False)
"""No engine source changed between the two runs, stated per stage.

Passing a bare `False` here used to collapse to `{}` — "could not be
determined" — so these assertions were weaker than they read. The type checker
found it; the tests passed either way.
"""


def stage_of(manifest, name):
    """A stage that must exist; fails loudly instead of raising on None."""
    provenance = manifest.stage(name)
    assert provenance is not None, f"manifest is missing the {name!r} stage"
    return provenance


@pytest.fixture
def outcomes(mini):
    return execute(mini, BM25Retriever(), k=10), execute(mini, TfidfRetriever(), k=10)


def test_a_run_carries_its_own_provenance(outcomes, mini):
    left, _ = outcomes
    manifest = left.manifest

    assert manifest.corpus == mini.corpus_fingerprint
    assert manifest.labels == mini.labels_fingerprint
    assert {stage.stage for stage in manifest.stages} == {"retrieve", "evaluate"}
    assert stage_of(manifest, "retrieve").inputs == (
        mini.corpus_fingerprint,
        mini.labels_fingerprint,
    )


def test_two_configurations_get_different_run_identities(outcomes):
    left, right = outcomes
    assert left.manifest.run_id != right.manifest.run_id
    assert stage_of(left.manifest, "retrieve").key != stage_of(right.manifest, "retrieve").key


def test_the_same_configuration_gets_a_stable_stage_key(mini):
    """If this drifts, caching reuses nothing and every run is a cold start."""
    first = execute(mini, BM25Retriever(), k=10)
    second = execute(mini, BM25Retriever(), k=10)
    assert stage_of(first.manifest, "retrieve").key == stage_of(second.manifest, "retrieve").key
    assert first.manifest.config == second.manifest.config


def test_changing_k_changes_the_stage_key(mini):
    narrow = execute(mini, BM25Retriever(), k=3)
    wide = execute(mini, BM25Retriever(), k=10)
    assert stage_of(narrow.manifest, "retrieve").key != stage_of(wide.manifest, "retrieve").key


def test_every_judged_query_is_evaluated_and_diagnosed(outcomes, mini):
    for outcome in outcomes:
        assert set(outcome.evaluation.per_query) == set(mini.judged_queries)
        assert set(outcome.diagnosis.evidence) == set(mini.judged_queries)


def test_evidence_and_metrics_agree_about_success(outcomes):
    """A query whose gold is first must have RR == 1, and vice versa."""
    for outcome in outcomes:
        for query_id, evidence in outcome.diagnosis.evidence.items():
            reciprocal = outcome.evaluation.per_query[query_id]["RR"]
            if evidence.symptom is Symptom.OK:
                assert reciprocal == 1.0
            else:
                assert reciprocal < 1.0


def test_the_two_systems_are_comparable_and_produce_a_diff(outcomes, mini):
    left, right = outcomes
    trust = assess(
        left.manifest, right.manifest, stages=STAGES_FOR_RETRIEVAL_DIFF, code_changed=UNCHANGED
    )
    # Same machine, same packages, same data: nothing should refuse.
    assert trust.trust is not Trust.INCOMPARABLE

    diff = compare_runs(
        left.run,
        right.run,
        mini.qrels,
        left_name=left.name,
        right_name=right.name,
        trust=trust,
        left_diagnosis=left.diagnosis,
        right_diagnosis=right.diagnosis,
    )
    assert len(diff.deltas) == len(mini.judged_queries)
    assert len(diff.improved) + len(diff.regressed) + len(diff.unchanged) == len(diff.deltas)


def test_the_fixture_diff_is_entirely_arbitrary(outcomes, mini):
    """Locks in the finding from the first run.

    On this fixture the only movement between BM25 and TF-IDF comes from BM25
    scoring the gold and a hard negative identically, so the tie-break decides
    it. If a future change makes this diff look like a real improvement without
    the tie being resolved, this test should fail and be read carefully.
    """
    left, right = outcomes
    trust = assess(
        left.manifest, right.manifest, stages=STAGES_FOR_RETRIEVAL_DIFF, code_changed=UNCHANGED
    )
    diff = compare_runs(
        left.run,
        right.run,
        mini.qrels,
        left_name=left.name,
        right_name=right.name,
        trust=trust,
        left_diagnosis=left.diagnosis,
        right_diagnosis=right.diagnosis,
    )
    assert diff.improved
    assert diff.arbitrary == diff.improved
    assert diff.regressed == ()


def test_diffing_a_run_against_itself_moves_nothing(outcomes, mini):
    left, _ = outcomes
    trust = assess(
        left.manifest, left.manifest, stages=STAGES_FOR_RETRIEVAL_DIFF, code_changed=UNCHANGED
    )
    diff = compare_runs(
        left.run,
        left.run,
        mini.qrels,
        left_name="a",
        right_name="b",
        trust=trust,
        left_diagnosis=left.diagnosis,
        right_diagnosis=left.diagnosis,
    )
    assert diff.improved == ()
    assert diff.regressed == ()
    assert len(diff.unchanged) == len(mini.judged_queries)


def test_the_demo_runs(capsys, mini):
    from embedlab.cli.demo import main

    assert main([str(mini_directory())]) == 0
    output = capsys.readouterr().out
    assert "WHAT CHANGED?" in output
    assert "WHY DID THIS FAIL?" in output
    assert "trust:" in output


def mini_directory():
    from pathlib import Path

    return Path(__file__).parent / "fixtures" / "mini"
