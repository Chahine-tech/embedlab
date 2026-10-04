"""The gate that keeps a cheap cache key from producing an untrustworthy diff.

Everything is judged from the environment recorded on each *stage*, because a
cached stage may be far older than the run that reuses it.
"""

from datetime import UTC, datetime

import pytest

from embedlab.artifacts.comparability import Trust, assess
from embedlab.artifacts.manifest import Environment, RunManifest, StageProvenance
from embedlab.domain.ids import Fingerprint, RunId

CLEAN_PACKAGES = {"numpy": "2.5.3", "bm25s": "0.3.12", "ranx": "0.3.21"}


def env(**overrides) -> Environment:
    base = {
        "python": "3.13.13",
        "platform": "darwin-arm64",
        "engine_commit": "a" * 40,
        "engine_dirty": False,
        "packages": dict(CLEAN_PACKAGES),
    }
    return Environment(**(base | overrides))


DEFAULT_PACKAGES = {
    "retrieve": ("bm25s", "numpy", "scipy"),
    "evaluate": ("ir-measures", "numpy", "ranx"),
}


def stage(
    name: str,
    *,
    impl_version: int = 1,
    environment: Environment | None = None,
    packages: tuple[str, ...] | None = None,
):
    return StageProvenance(
        stage=name,
        impl_version=impl_version,
        key=Fingerprint("fp_" + "0" * 32),
        params=Fingerprint("fp_" + "1" * 32),
        packages=packages if packages is not None else DEFAULT_PACKAGES.get(name, ()),
        environment=environment or env(),
        created_at=datetime(2026, 10, 4, tzinfo=UTC),
    )


def manifest(*, stages=None, schema: int = 1, corpus: str = "c" * 32, labels: str = "l" * 32):
    return RunManifest(
        schema_version=schema,
        run_id=RunId("run-1"),
        config=Fingerprint("fp_" + "2" * 32),
        corpus=Fingerprint("fp_" + corpus),
        labels=Fingerprint("fp_" + labels),
        environment=env(),
        stages=tuple(stages or [stage("retrieve"), stage("evaluate")]),
    )


BOTH = ["retrieve", "evaluate"]


def test_identical_runs_are_fully_trusted():
    verdict = assess(manifest(), manifest(), stages=BOTH)
    assert verdict.trust is Trust.IDENTICAL
    assert verdict.reasons == ()


def test_differing_commits_with_unchanged_source_stay_comparable():
    """Committing must not forbid diffing, or the gate is unusable."""
    other = manifest(stages=[stage("retrieve", environment=env(engine_commit="b" * 40))])
    verdict = assess(manifest(), other, stages=["retrieve"], code_changed={"retrieve": False})
    assert verdict.trust is Trust.COMPARABLE


def test_differing_commits_with_changed_source_are_suspect():
    other = manifest(stages=[stage("retrieve", environment=env(engine_commit="b" * 40))])
    verdict = assess(manifest(), other, stages=["retrieve"], code_changed={"retrieve": True})
    assert verdict.trust is Trust.SUSPECT
    assert any("source affecting stage" in reason for reason in verdict.reasons)


def test_forgotten_impl_version_bump_is_still_caught():
    """The core claim: human discipline is not on the critical path.

    impl_version matches on both sides (the bump was forgotten), yet the
    verdict is suspect because the source changed between the two commits.
    """
    other = manifest(stages=[stage("evaluate", environment=env(engine_commit="b" * 40))])
    verdict = assess(manifest(), other, stages=["evaluate"], code_changed={"evaluate": True})
    assert verdict.trust is Trust.SUSPECT


def test_declared_semantic_change_is_suspect():
    other = manifest(stages=[stage("retrieve", impl_version=2)])
    verdict = assess(manifest(), other, stages=["retrieve"], code_changed={"retrieve": False})
    assert verdict.trust is Trust.SUSPECT
    assert any("impl_version" in reason for reason in verdict.reasons)


def test_an_undeterminable_source_comparison_is_suspect_not_fine():
    other = manifest(stages=[stage("retrieve", environment=env(engine_commit="b" * 40))])
    assert assess(manifest(), other, stages=["retrieve"]).trust is Trust.SUSPECT
    assert (
        assess(manifest(), other, stages=["retrieve"], code_changed={"retrieve": None}).trust
        is Trust.SUSPECT
    )


def test_a_missing_code_changed_entry_does_not_count_as_clean():
    other = manifest(stages=[stage("retrieve", environment=env(engine_commit="b" * 40))])
    verdict = assess(manifest(), other, stages=["retrieve"], code_changed={"evaluate": False})
    assert verdict.trust is Trust.SUSPECT


def test_dirty_tree_is_suspect():
    other = manifest(stages=[stage("retrieve", environment=env(engine_dirty=True))])
    verdict = assess(manifest(), other, stages=["retrieve"], code_changed={"retrieve": False})
    assert verdict.trust is Trust.SUSPECT
    assert any("dirty" in reason for reason in verdict.reasons)


def test_unknown_provenance_is_suspect():
    other = manifest(stages=[stage("retrieve", environment=env(engine_commit=None))])
    verdict = assess(manifest(), other, stages=["retrieve"])
    assert verdict.trust is Trust.SUSPECT
    assert any("unknown provenance" in reason for reason in verdict.reasons)


def test_a_stale_cached_stage_is_judged_on_its_own_environment():
    """The reason environments live on stages.

    The run was assembled today, but its retrieve artifact came from the cache
    and was produced with an older numpy. The gate must see the artifact's
    environment, not the assembling process's.
    """
    stale = env(packages={**CLEAN_PACKAGES, "numpy": "2.0.0"}, engine_commit="b" * 40)
    cached = manifest(stages=[stage("retrieve", environment=stale), stage("evaluate")])

    verdict = assess(manifest(), cached, stages=["retrieve"], code_changed={"retrieve": False})
    assert verdict.trust is Trust.SUSPECT
    assert any("numpy differs" in reason for reason in verdict.reasons)

    # ...and the fresh evaluate stage of the same run is untouched by it.
    assert (
        assess(manifest(), cached, stages=["evaluate"], code_changed={"evaluate": False}).trust
        is Trust.IDENTICAL
    )


def test_schema_mismatch_refuses():
    verdict = assess(manifest(), manifest(schema=2), stages=["retrieve"])
    assert verdict.trust is Trust.INCOMPARABLE
    assert verdict.refuses


def test_missing_stage_refuses():
    other = manifest(stages=[stage("evaluate")])
    verdict = assess(manifest(), other, stages=["retrieve"])
    assert verdict.trust is Trust.INCOMPARABLE


def test_different_labels_refuse():
    """Two runs scored on different queries answer different questions."""
    verdict = assess(manifest(), manifest(labels="z" * 32), stages=["retrieve"])
    assert verdict.trust is Trust.INCOMPARABLE
    assert any("different queries or labels" in reason for reason in verdict.reasons)


def test_different_corpora_refuse():
    verdict = assess(manifest(), manifest(corpus="z" * 32), stages=["retrieve"])
    assert verdict.trust is Trust.INCOMPARABLE


def test_a_torch_upgrade_does_not_block_a_lexical_diff():
    """Relevance comes from the adapter that ran, not from the stage name."""
    with_torch = env(packages={**CLEAN_PACKAGES, "torch": "2.6.0"})
    other = manifest(stages=[stage("retrieve", environment=with_torch)])

    assert (
        assess(manifest(), other, stages=["retrieve"], code_changed={"retrieve": False}).trust
        is Trust.IDENTICAL
    )


def test_a_torch_upgrade_does_block_a_dense_diff():
    """The same stage, a different adapter, the opposite verdict.

    A static per-stage package map could satisfy this test or the one above,
    never both.
    """
    dense = ("numpy", "sentence-transformers", "torch")
    base = manifest(stages=[stage("retrieve", packages=dense)])
    upgraded = manifest(
        stages=[
            stage(
                "retrieve",
                packages=dense,
                environment=env(packages={**CLEAN_PACKAGES, "torch": "2.6.0"}),
            )
        ]
    )

    verdict = assess(base, upgraded, stages=["retrieve"], code_changed={"retrieve": False})
    assert verdict.trust is Trust.SUSPECT
    assert any("torch differs" in reason for reason in verdict.reasons)


def test_the_relevant_packages_are_the_union_of_both_sides():
    """Comparing a dense run with a lexical one: torch is decisive even though
    only one of the two artifacts names it."""
    lexical = manifest(stages=[stage("retrieve", packages=("bm25s", "numpy"))])
    dense = manifest(
        stages=[
            stage(
                "retrieve",
                packages=("numpy", "torch"),
                environment=env(packages={**CLEAN_PACKAGES, "torch": "2.6.0"}),
            )
        ]
    )

    verdict = assess(lexical, dense, stages=["retrieve"], code_changed={"retrieve": False})
    assert any("torch differs" in reason for reason in verdict.reasons)


def test_relevant_package_change_is_suspect():
    changed = env(packages={**CLEAN_PACKAGES, "bm25s": "0.4.0"})
    other = manifest(stages=[stage("retrieve", environment=changed)])
    verdict = assess(manifest(), other, stages=["retrieve"], code_changed={"retrieve": False})
    assert verdict.trust is Trust.SUSPECT
    assert any("bm25s differs" in reason for reason in verdict.reasons)


def test_irrelevant_package_change_is_ignored_for_that_stage():
    """ranx cannot change what retrieval returned."""
    changed = env(packages={**CLEAN_PACKAGES, "ranx": "0.4.0"})
    other = manifest(
        stages=[stage("retrieve", environment=changed), stage("evaluate", environment=changed)]
    )

    assert (
        assess(manifest(), other, stages=["retrieve"], code_changed={"retrieve": False}).trust
        is Trust.IDENTICAL
    )
    assert (
        assess(manifest(), other, stages=["evaluate"], code_changed={"evaluate": False}).trust
        is Trust.SUSPECT
    )


@pytest.mark.parametrize("field", ["python", "platform"])
def test_interpreter_and_platform_changes_are_suspect(field):
    other = manifest(stages=[stage("retrieve", environment=env(**{field: "changed"}))])
    verdict = assess(manifest(), other, stages=["retrieve"], code_changed={"retrieve": False})
    assert verdict.trust is Trust.SUSPECT


def test_worst_finding_wins():
    dirty = manifest(stages=[stage("retrieve", environment=env(engine_dirty=True))], schema=2)
    verdict = assess(manifest(), dirty, stages=["retrieve"])
    assert verdict.trust is Trust.INCOMPARABLE
    assert len(verdict.reasons) > 1
