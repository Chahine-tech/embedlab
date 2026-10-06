"""Reading and writing run artifacts.

Parquet for the tabular output, JSON for the manifest. Neutral formats on
purpose: they are the contract between the engine and anything that reads it
later (a notebook, a UI, or this engine three months from now), so neither
side has to import the other.

Round-trip exactness matters more than compactness here. A score that comes
back off by one bit would reorder near-ties and fabricate a diff, so scores are
stored as float64 and the ordering is stored explicitly as `rank` rather than
being re-derived from the scores on read.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import polars as pl

from embedlab.artifacts.manifest import RunManifest, StageProvenance
from embedlab.domain.ids import DocId, Fingerprint, QueryId
from embedlab.domain.retrieval import Ranked

if TYPE_CHECKING:
    from pathlib import Path

    from embedlab.domain.retrieval import Run

RUN_FILE = "run.parquet"
MANIFEST_FILE = "manifest.json"

_SCHEMA = {
    "query_id": pl.String,
    "doc_id": pl.String,
    "rank": pl.UInt32,
    "score": pl.Float64,
}


def write_run(directory: Path, run: Run) -> Path:
    """Write a run as one row per retrieved document."""
    rows: dict[str, list[object]] = {
        "query_id": [],
        "doc_id": [],
        "rank": [],
        "score": [],
    }
    for query_id in sorted(run):
        for item in run[query_id]:
            rows["query_id"].append(str(query_id))
            rows["doc_id"].append(str(item.doc_id))
            rows["rank"].append(item.rank)
            rows["score"].append(item.score)

    path = directory / RUN_FILE
    # Snappy, for the same reason the published artifacts use it: a reader
    # should not need an extra codec to open what this engine wrote.
    pl.DataFrame(rows, schema=_SCHEMA).write_parquet(path, compression="snappy")
    return path


def read_run(directory: Path) -> dict[QueryId, list[Ranked]]:
    """Read a run back, preserving the stored order rather than re-sorting."""
    frame = pl.read_parquet(directory / RUN_FILE)
    run: dict[QueryId, list[Ranked]] = {}
    for row in frame.sort(["query_id", "rank"]).iter_rows(named=True):
        run.setdefault(QueryId(row["query_id"]), []).append(
            Ranked(
                doc_id=DocId(row["doc_id"]),
                rank=int(row["rank"]),
                score=float(row["score"]),
            )
        )
    return run


def write_manifest(directory: Path, manifest: RunManifest) -> Path:
    path = directory / MANIFEST_FILE
    path.write_text(manifest.model_dump_json(indent=2), encoding="utf-8")
    return path


def read_manifest(directory: Path) -> RunManifest:
    """Read a manifest, refusing anything this version cannot interpret.

    `extra="forbid"` on the models means an artifact written by a newer engine
    fails here rather than silently losing the fields this version does not
    know about.
    """
    raw = json.loads((directory / MANIFEST_FILE).read_text(encoding="utf-8"))
    return RunManifest.model_validate(raw)


STAGE_FILE = "stage.json"


def write_stage_provenance(directory: Path, provenance: StageProvenance) -> Path:
    """Store how this cache entry was produced, next to the artifact itself.

    Per entry rather than per run: an entry is reused by many runs, and the only
    truthful answer to "what produced this" is the one recorded when it was
    written.
    """
    path = directory / STAGE_FILE
    path.write_text(provenance.model_dump_json(indent=2), encoding="utf-8")
    return path


def read_stage_provenance(
    directory: Path, *, expect_key: Fingerprint | None = None
) -> StageProvenance:
    """Read an entry's provenance, refusing one that contradicts its location."""
    raw = json.loads((directory / STAGE_FILE).read_text(encoding="utf-8"))
    provenance = StageProvenance.model_validate(raw)
    if expect_key is not None and provenance.key != expect_key:
        msg = (
            f"corrupt cache entry {directory}: it records key {provenance.key} but is "
            f"stored under {expect_key}"
        )
        raise ValueError(msg)
    return provenance
