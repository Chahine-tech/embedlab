"""Run provenance: what produced an artifact, and under what conditions.

The manifest exists to answer one question correctly: *may these two runs be
compared?* Cache keys answer a different, cheaper question (may this artifact
be reused?) and are allowed to be imperfect. Keeping the two apart is what lets
the expensive stages have stable keys without making the diff untrustworthy.

Nothing here is ever fed to `fingerprint()` wholesale: `created_at` is
deliberately recorded and deliberately excluded from identity.
"""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field

from embedlab.domain.ids import Fingerprint, RunId

SCHEMA_VERSION = 1
"""Bumped on any breaking change to the artifact layout.

Versioned from the first commit on purpose: the alternative is discovering in
three weeks that last month's runs are unreadable.
"""

NUMERIC_PACKAGES: tuple[str, ...] = (
    "numpy",
    "polars",
    "pyarrow",
    "bm25s",
    "scipy",
    "ranx",
    "ir-measures",
    "torch",
    "sentence-transformers",
    "transformers",
)
"""Packages whose version can change the numbers we report.

An allowlist rather than "everything installed": recording ruff's version would
make the comparability gate fire on changes that cannot possibly move a score,
and a gate that cries wolf gets overridden out of habit.
"""


class Environment(BaseModel):
    """The part of the world that can change a result without the config changing."""

    model_config = ConfigDict(extra="forbid")

    python: str
    platform: str
    engine_commit: str | None = None
    """Git SHA of the engine. None outside a repository."""

    engine_dirty: bool = False
    """True if the working tree had uncommitted changes. A dirty run is never
    reproducible and the gate treats it as incomparable to anything else."""

    packages: dict[str, str] = Field(default_factory=dict)
    """Resolved versions, restricted to NUMERIC_PACKAGES."""

    def identity(self) -> dict[str, object]:
        """The subset that participates in comparability decisions."""
        return {
            "python": self.python,
            "platform": self.platform,
            "engine_commit": self.engine_commit,
            "engine_dirty": self.engine_dirty,
            "packages": dict(self.packages),
        }


class StageProvenance(BaseModel):
    """How one artifact came to exist."""

    model_config = ConfigDict(extra="forbid")

    stage: str
    impl_version: int
    """Manually bumped when this stage's *semantics* change. Controls recompute
    only, never trust, which the environment gate handles."""

    key: Fingerprint
    """The cache key this artifact is stored under."""

    inputs: tuple[Fingerprint, ...] = ()
    """Upstream stage keys. Makes the DAG explicit in the artifact itself, so a
    diff can tell which stages two runs share."""

    params: Fingerprint
    """Fingerprint of the stage's resolved parameters."""

    packages: tuple[str, ...]
    """Which package versions can move *this* artifact's numbers.

    Recorded when the artifact is produced, from the adapter that produced it,
    rather than inferred later from the stage name. Required rather than
    defaulted: an entry written before this existed would otherwise read as
    "no package matters here" and quietly disarm the gate.
    """

    environment: Environment
    """The environment *this artifact* was produced in.

    Per stage rather than per run, because a cached stage may have been computed
    weeks ago under a different commit and a different numpy. A single run-level
    environment would claim a reused artifact was produced with today's
    libraries, which would quietly defeat the comparability gate: the exact
    failure the gate exists to prevent.
    """

    created_at: datetime
    """Provenance only. Never part of any identity."""


class RunManifest(BaseModel):
    """Everything needed to decide whether a run can be trusted and compared."""

    model_config = ConfigDict(extra="forbid")

    schema_version: int = SCHEMA_VERSION
    run_id: RunId
    config: Fingerprint
    """Fingerprint of the canonicalised experiment config."""

    corpus: Fingerprint
    """Identity of the retrievable content."""

    labels: Fingerprint
    """Identity of the queries and qrels this run was scored against.

    Recorded so the gate can refuse outright when two runs were not evaluated
    on the same data, the one case that is meaningless rather than merely
    suspect.
    """

    environment: Environment
    """The environment of the process that assembled this run.

    Context, not the gate's input: comparability is decided from each stage's
    own environment, since stages can be produced at different times.
    """

    stages: tuple[StageProvenance, ...] = ()

    def stage(self, name: str) -> StageProvenance | None:
        for provenance in self.stages:
            if provenance.stage == name:
                return provenance
        return None


ADAPTER_PACKAGES: dict[str, tuple[str, ...]] = {
    "bm25": ("bm25s",),
    "tfidf": ("scipy",),
    "dense": ("torch", "sentence-transformers", "transformers"),
}
"""Packages that can move a given adapter's numbers.

Keyed by the adapter's own `kind` rather than by the stage, because the stage
name does not say what ran inside it. A dense retriever and a BM25 retriever
both execute in `retrieve`, and a torch upgrade is decisive for one and
irrelevant to the other. Resolving this from the stage alone forces a choice
between missing the first case and crying wolf on the second.
"""

STAGE_PACKAGES: dict[str, tuple[str, ...]] = {
    "chunk": (),
    "embed": ("numpy", "torch", "sentence-transformers", "transformers"),
    "retrieve": ("numpy", "bm25s", "scipy"),
    "rerank": ("numpy", "torch", "sentence-transformers"),
    "evaluate": ("numpy", "ranx", "ir-measures"),
    "diagnose": ("numpy", "scipy"),
}
"""Which package versions can move each stage's numbers.

Per-stage rather than global so that a torch upgrade does not block a diff of
two BM25 runs. A gate that fires on irrelevant changes gets overridden by
reflex, which is worse than no gate.
"""
