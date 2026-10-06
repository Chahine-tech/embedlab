"""Comparing two runs, and publishing the result.

The demo used to assemble this by hand: ask git what changed, assess trust,
run the significance tests, diff the rankings. Doing it in one place means a
second reader cannot assemble it slightly differently, which for a tool whose
subject is trust would be its own kind of bug.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from embedlab.artifacts.comparability import assess
from embedlab.artifacts.provenance import relevant_source_changed
from embedlab.artifacts.workspace import write_comparison_bundle
from embedlab.cache.fingerprint import fingerprint
from embedlab.pipeline.run import STAGES_FOR_RETRIEVAL_DIFF
from embedlab.stages.diff import compare_runs
from embedlab.stages.significance import DEFAULT_RESAMPLES, DEFAULT_SEED, compare_measures

if TYPE_CHECKING:
    from pathlib import Path

    from embedlab.artifacts.dataset import Dataset
    from embedlab.artifacts.manifest import RunManifest
    from embedlab.artifacts.workspace import Workspace
    from embedlab.pipeline.run import RunOutcome
    from embedlab.stages.diff import RetrievalDiff


@dataclass(frozen=True, slots=True)
class Comparison:
    comparison_id: str
    diff: RetrievalDiff
    left_run_id: str
    right_run_id: str
    directory: Path | None = None
    """Where it was published, when a workspace was given."""


def _commit_for(manifest: RunManifest, stage: str) -> str | None:
    """The commit that produced one stage, or None when the stage is absent.

    None degrades correctly: the source comparison then reports "unknown",
    which the gate treats as suspect. A missing stage is separately refused
    outright, but reaching that refusal must not require surviving an
    AttributeError first.
    """
    provenance = manifest.stage(stage)
    return None if provenance is None else provenance.environment.engine_commit


def source_changed_per_stage(
    left: RunManifest,
    right: RunManifest,
    *,
    stages: tuple[str, ...] = STAGES_FOR_RETRIEVAL_DIFF,
    repo: Path | None = None,
) -> dict[str, bool | None]:
    """Ask git, per stage, against the commits each artifact recorded.

    Per stage rather than once for the pair, because a cached artifact may be
    far older than the run reusing it.
    """
    return {
        stage: relevant_source_changed(
            _commit_for(left, stage),
            _commit_for(right, stage),
            stages=[stage],
            repo=repo,
        )
        for stage in stages
    }


def compare(
    left: RunOutcome,
    right: RunOutcome,
    dataset: Dataset,
    *,
    workspace: Workspace | None = None,
    seed: int = DEFAULT_SEED,
    resamples: int = DEFAULT_RESAMPLES,
    repo: Path | None = None,
) -> Comparison:
    """Diff two runs, qualified by how far the diff can be trusted."""
    trust = assess(
        left.manifest,
        right.manifest,
        stages=STAGES_FOR_RETRIEVAL_DIFF,
        code_changed=source_changed_per_stage(left.manifest, right.manifest, repo=repo),
    )

    shared = tuple(
        measure for measure in left.evaluation.measures if measure in right.evaluation.measures
    )
    significance = compare_measures(
        left.evaluation.per_query,
        right.evaluation.per_query,
        measures=shared,
        seed=seed,
        resamples=resamples,
    )

    diff = compare_runs(
        left.run,
        right.run,
        dataset.qrels,
        left_name=left.name,
        right_name=right.name,
        trust=trust,
        significance=significance,
        left_diagnosis=left.diagnosis,
        right_diagnosis=right.diagnosis,
    )

    comparison_id = "{}-vs-{}-{}".format(
        left.name,
        right.name,
        fingerprint(
            {
                "left": left.manifest.config,
                "right": right.manifest.config,
                "seed": seed,
                "resamples": resamples,
            }
        )[3:11],
    )

    directory = None
    if workspace is not None:
        directory = write_comparison_bundle(
            workspace.comparison_dir(comparison_id),
            diff=diff,
            left_run_id=str(left.manifest.run_id),
            right_run_id=str(right.manifest.run_id),
        )
        workspace.write_index()

    return Comparison(
        comparison_id=comparison_id,
        diff=diff,
        left_run_id=str(left.manifest.run_id),
        right_run_id=str(right.manifest.run_id),
        directory=directory,
    )
