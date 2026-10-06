"""Executing one retrieval configuration end to end.

The imperative shell: it does the I/O, calls the pure stages in order, consults
the cache, and records provenance. All of the logic lives below it.

Only the expensive stage is cached. Retrieval indexes a corpus and runs every
query; evaluation is arithmetic over a few thousand rows and is recomputed
every time, which keeps a metric change from needing cache invalidation at all.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from typing import TYPE_CHECKING

from pydantic import ValidationError

from embedlab.artifacts.manifest import (
    ADAPTER_PACKAGES,
    STAGE_PACKAGES,
    RunManifest,
    StageProvenance,
)
from embedlab.artifacts.provenance import capture_environment
from embedlab.artifacts.run_io import (
    read_run,
    read_stage_provenance,
    write_run,
    write_stage_provenance,
)
from embedlab.artifacts.workspace import write_run_bundle
from embedlab.cache.fingerprint import fingerprint
from embedlab.domain.ids import Fingerprint, RunId
from embedlab.stages.diagnose import diagnose
from embedlab.stages.evaluate import DEFAULT_MEASURES, evaluate

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from embedlab.adapters.base import Retriever
    from embedlab.artifacts.dataset import Dataset
    from embedlab.artifacts.workspace import Workspace
    from embedlab.cache.store import CacheStore
    from embedlab.domain.retrieval import Run
    from embedlab.stages.diagnose import Diagnosis
    from embedlab.stages.evaluate import Evaluation

EVALUATE_IMPL_VERSION = 1

STAGES_FOR_RETRIEVAL_DIFF = ("retrieve", "evaluate")
"""The stages a ranking comparison actually depends on. Passed to the gate so a
change to the diagnosis code does not block a retrieval diff."""


@dataclass(frozen=True, slots=True)
class RunOutcome:
    """Everything one configuration produced, with its provenance attached."""

    name: str
    run: Run
    evaluation: Evaluation
    diagnosis: Diagnosis
    manifest: RunManifest
    from_cache: bool = False
    """Whether retrieval was reused rather than recomputed. Worth reporting: a
    cached run's numbers were produced in the environment recorded on its stage,
    not in today's."""


def retrieve_key(dataset: Dataset, retriever: Retriever, *, k: int) -> Fingerprint:
    """Cache key for a retrieval artifact.

    Everything that can change the output is in here: the corpus content, the
    queries, and the adapter's full descriptor (which includes its library
    version, parameters and dtype). Anything the descriptor omits becomes an
    invisible source of fake diffs.
    """
    return fingerprint(
        {
            "stage": "retrieve",
            "corpus": dataset.corpus_fingerprint,
            "queries": fingerprint(dict(dataset.queries)),
            "params": fingerprint({"retriever": dict(retriever.descriptor), "k": k}),
        }
    )


def _impl_version(retriever: Retriever) -> int:
    """Read the adapter's declared semantic version out of its descriptor.

    Validated rather than coerced: a descriptor carrying a non-integer here
    would silently become `int(...)` of something unintended, and this value
    decides whether two runs may be compared.
    """
    declared = dict(retriever.descriptor).get("impl_version", 1)
    if not isinstance(declared, int) or isinstance(declared, bool):
        msg = f"{retriever.name!r} declares impl_version={declared!r}; it must be an integer"
        raise TypeError(msg)
    return declared


def _relevant_packages(stage: str, descriptor: Mapping[str, object]) -> tuple[str, ...]:
    """Packages that can move this artifact, from the stage and the adapter."""
    kind = descriptor.get("kind")
    adapter = ADAPTER_PACKAGES.get(str(kind), ()) if isinstance(kind, str) else ()
    return tuple(sorted({*STAGE_PACKAGES.get(stage, ()), *adapter}))


def _retrieve(
    dataset: Dataset,
    retriever: Retriever,
    *,
    k: int,
    store: CacheStore | None,
) -> tuple[Run, StageProvenance, bool]:
    key = retrieve_key(dataset, retriever, k=k)
    params = fingerprint({"retriever": dict(retriever.descriptor), "k": k})

    if store is not None and (entry := store.lookup("retrieve", key)) is not None:
        try:
            # The stored provenance is returned unchanged: this artifact really
            # was produced then, under that commit and those library versions,
            # and the gate must see that rather than today's environment.
            provenance = read_stage_provenance(entry, expect_key=key)
        except ValidationError:
            # Written by an older engine whose provenance lacked a field this
            # one relies on. Recompute rather than reading it loosely: a
            # half-understood provenance would disarm the gate silently.
            store.forget("retrieve", key)
        else:
            return read_run(entry), provenance, True

    retriever.index(dataset.corpus)
    run = retriever.search(dataset.queries, k=k)

    provenance = StageProvenance(
        stage="retrieve",
        # Taken from the adapter: the adapter is what can change semantics here.
        impl_version=_impl_version(retriever),
        key=key,
        inputs=(dataset.corpus_fingerprint, dataset.labels_fingerprint),
        params=params,
        packages=_relevant_packages("retrieve", retriever.descriptor),
        environment=capture_environment(),
        created_at=datetime.now(UTC),
    )

    if store is not None:
        with store.reserve("retrieve", key) as scratch:
            write_run(scratch, run)
            write_stage_provenance(scratch, provenance)

    return run, provenance, False


def execute(
    dataset: Dataset,
    retriever: Retriever,
    *,
    k: int = 10,
    measures: Sequence[str] = DEFAULT_MEASURES,
    store: CacheStore | None = None,
    workspace: Workspace | None = None,
) -> RunOutcome:
    """Index, search, score and diagnose one retrieval configuration.

    With a workspace, the result is also published as files: the run, its
    per-query metrics, its evidence and its hypotheses. That bundle is the
    contract with every reader downstream, so nothing has to import this engine
    to use what it produced.
    """
    run, retrieval, from_cache = _retrieve(dataset, retriever, k=k, store=store)

    evaluation = evaluate(run, dataset.qrels, measures=measures)
    diagnosis = diagnose(run, dataset.qrels, dataset.corpus, dataset.queries)

    evaluate_params = fingerprint({"measures": tuple(measures)})
    config = fingerprint(
        {
            "dataset": dataset.name,
            "retriever": dict(retriever.descriptor),
            "k": k,
            "measures": tuple(measures),
        }
    )

    manifest = RunManifest(
        run_id=RunId(f"{retriever.name}-{config[3:11]}"),
        config=config,
        corpus=dataset.corpus_fingerprint,
        labels=dataset.labels_fingerprint,
        environment=capture_environment(),
        stages=(
            retrieval,
            StageProvenance(
                stage="evaluate",
                impl_version=EVALUATE_IMPL_VERSION,
                key=fingerprint(
                    {
                        "stage": "evaluate",
                        "run": retrieval.key,
                        "labels": dataset.labels_fingerprint,
                        "params": evaluate_params,
                    }
                ),
                inputs=(retrieval.key, dataset.labels_fingerprint),
                params=evaluate_params,
                packages=STAGE_PACKAGES.get("evaluate", ()),
                environment=capture_environment(),
                created_at=datetime.now(UTC),
            ),
        ),
    )

    outcome = RunOutcome(
        name=retriever.name,
        run=run,
        evaluation=evaluation,
        diagnosis=diagnosis,
        manifest=manifest,
        from_cache=from_cache,
    )

    if workspace is not None:
        write_run_bundle(
            workspace.run_dir(str(manifest.run_id)),
            manifest_json=manifest.model_dump_json(),
            run=run,
            evaluation=evaluation,
            diagnosis=diagnosis,
        )

    return outcome
