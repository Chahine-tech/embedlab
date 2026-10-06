"""Where finished artifacts live, and the shape they take on disk.

The engine's contract with everything downstream. A notebook, a browser or this
code in three months reads these files; none of them should have to import the
engine to do it, which is why the tabular parts are Parquet and the rest is
JSON.

Two kinds of bundle, because they have different owners. An evaluation and a
diagnosis belong to one run, so they live with it. A diff belongs to a pair, so
it references two runs rather than copying them, and re-comparing two existing
runs never recomputes their diagnosis.

Schemas are long rather than wide wherever the columns would otherwise depend on
configuration: one row per (query, measure) survives adding a measure, one row
per (query, cause) survives a query carrying two causes.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import TYPE_CHECKING, cast

import polars as pl

if TYPE_CHECKING:
    from pathlib import Path

    from embedlab.domain.retrieval import Run
    from embedlab.stages.diagnose import Diagnosis
    from embedlab.stages.diff import RetrievalDiff
    from embedlab.stages.evaluate import Evaluation

type JsonObject = dict[str, object]
"""A published manifest, read back. Values stay `object` on purpose: the file
is a contract with readers outside this engine, so its shape is validated at
the point of use rather than asserted here."""

LAYOUT_VERSION = 1
"""Bumped when a file moves or a column changes meaning.

Separate from the manifest's schema version: a reader can understand the layout
and still refuse a manifest it was not written for.
"""

MANIFEST = "manifest.json"
RUN_FILE = "run.parquet"
METRICS = "metrics.parquet"
EVIDENCE = "evidence.parquet"
HYPOTHESES = "hypotheses.parquet"
DELTAS = "deltas.parquet"

_METRICS_SCHEMA = {"query_id": pl.String, "measure": pl.String, "value": pl.Float64}

_EVIDENCE_SCHEMA = {
    "query_id": pl.String,
    "symptom": pl.String,
    "gold_rank": pl.Int32,
    "best_relevant": pl.String,
    "top1": pl.String,
    "top1_is_relevant": pl.Boolean,
    "score_margin": pl.Float64,
    "query_gold_overlap": pl.Float64,
    "competitor_gold_overlap": pl.Float64,
    "query_token_count": pl.Int32,
    "tied_with_top1": pl.Boolean,
    "gold_score_is_zero": pl.Boolean,
    "retrieved_with_zero_score": pl.Int32,
}

_HYPOTHESES_SCHEMA = {
    "query_id": pl.String,
    "kind": pl.String,
    "confidence": pl.Float64,
    "producer": pl.String,
    "because": pl.String,
}

_DELTAS_SCHEMA = {
    "query_id": pl.String,
    "left_rank": pl.Int32,
    "right_rank": pl.Int32,
    "direction": pl.String,
    "arbitrary": pl.Boolean,
}


@dataclass(frozen=True, slots=True)
class Workspace:
    """A directory of published runs and comparisons."""

    root: Path

    def run_dir(self, run_id: str) -> Path:
        return self.root / "run" / _safe(run_id)

    def comparison_dir(self, comparison_id: str) -> Path:
        return self.root / "comparison" / _safe(comparison_id)

    def runs(self) -> list[str]:
        return _listing(self.root / "run")

    def comparisons(self) -> list[str]:
        return _listing(self.root / "comparison")


def _safe(identifier: str) -> str:
    if not identifier or "/" in identifier or identifier.startswith("."):
        msg = f"invalid identifier {identifier!r}"
        raise ValueError(msg)
    return identifier


def _listing(directory: Path) -> list[str]:
    if not directory.is_dir():
        return []
    return sorted(child.name for child in directory.iterdir() if child.is_dir())


def _write(frame: pl.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.write_parquet(path)


def write_run_bundle(
    directory: Path,
    *,
    manifest_json: str,
    run: Run,
    evaluation: Evaluation,
    diagnosis: Diagnosis,
) -> Path:
    """Everything one configuration produced, as files."""
    directory.mkdir(parents=True, exist_ok=True)

    (directory / MANIFEST).write_text(
        json.dumps(
            {
                "layout_version": LAYOUT_VERSION,
                "manifest": json.loads(manifest_json),
                "measures": list(evaluation.measures),
                "calibration": {
                    "n": diagnosis.calibration.n,
                    "sufficient": diagnosis.calibration.sufficient,
                    "low_query_gold_overlap": diagnosis.calibration.low_query_gold_overlap,
                    "high_competitor_gold_overlap": (
                        diagnosis.calibration.high_competitor_gold_overlap
                    ),
                },
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    rows: dict[str, list[object]] = {"query_id": [], "doc_id": [], "rank": [], "score": []}
    for query_id in sorted(run):
        for item in run[query_id]:
            rows["query_id"].append(str(query_id))
            rows["doc_id"].append(str(item.doc_id))
            rows["rank"].append(item.rank)
            rows["score"].append(item.score)
    _write(
        pl.DataFrame(
            rows,
            schema={
                "query_id": pl.String,
                "doc_id": pl.String,
                "rank": pl.UInt32,
                "score": pl.Float64,
            },
        ),
        directory / RUN_FILE,
    )

    metrics: dict[str, list[object]] = {"query_id": [], "measure": [], "value": []}
    for query_id, values in sorted(evaluation.per_query.items()):
        for measure, value in sorted(values.items()):
            metrics["query_id"].append(str(query_id))
            metrics["measure"].append(measure)
            metrics["value"].append(value)
    _write(pl.DataFrame(metrics, schema=_METRICS_SCHEMA), directory / METRICS)

    evidence: dict[str, list[object]] = {name: [] for name in _EVIDENCE_SCHEMA}
    for query_id, found in sorted(diagnosis.evidence.items()):
        evidence["query_id"].append(str(query_id))
        evidence["symptom"].append(found.symptom.value)
        evidence["gold_rank"].append(found.gold_rank)
        evidence["best_relevant"].append(
            None if found.best_relevant is None else str(found.best_relevant)
        )
        evidence["top1"].append(str(found.top1))
        evidence["top1_is_relevant"].append(found.top1_is_relevant)
        evidence["score_margin"].append(found.score_margin)
        evidence["query_gold_overlap"].append(found.query_gold_overlap)
        evidence["competitor_gold_overlap"].append(found.competitor_gold_overlap)
        evidence["query_token_count"].append(found.query_token_count)
        evidence["tied_with_top1"].append(found.tied_with_top1)
        evidence["gold_score_is_zero"].append(found.gold_score_is_zero)
        evidence["retrieved_with_zero_score"].append(found.retrieved_with_zero_score)
    _write(pl.DataFrame(evidence, schema=_EVIDENCE_SCHEMA), directory / EVIDENCE)

    hypotheses: dict[str, list[object]] = {name: [] for name in _HYPOTHESES_SCHEMA}
    for query_id, found in sorted(diagnosis.hypotheses.items()):
        for hypothesis in found:
            hypotheses["query_id"].append(str(query_id))
            hypotheses["kind"].append(hypothesis.kind.value)
            hypotheses["confidence"].append(hypothesis.confidence)
            hypotheses["producer"].append(hypothesis.producer.value)
            # One string rather than a list: the trail is read, not queried, and
            # a nested column would cost every reader a schema it does not need.
            hypotheses["because"].append(" ".join(hypothesis.because))
    _write(pl.DataFrame(hypotheses, schema=_HYPOTHESES_SCHEMA), directory / HYPOTHESES)

    return directory


def read_run_manifest(directory: Path) -> JsonObject:
    """The published manifest, as plain data.

    Returned as `object`-valued rather than typed: it is a published contract
    read by things that are not this engine, and pretending here that its shape
    is guaranteed would push an unchecked assumption onto every caller.
    """
    loaded: object = json.loads((directory / MANIFEST).read_text(encoding="utf-8"))
    if not isinstance(loaded, dict):
        msg = f"{directory / MANIFEST} does not contain a JSON object"
        raise ValueError(msg)
    # json.loads gives an untyped dict; the isinstance check above is all the
    # structure this layer promises, and callers validate the rest at use.
    return cast("JsonObject", loaded)


def write_comparison_bundle(
    directory: Path,
    *,
    diff: RetrievalDiff,
    left_run_id: str,
    right_run_id: str,
) -> Path:
    """One comparison, referencing the two runs rather than copying them.

    A reader that wants the rankings behind a movement loads the two run
    bundles, which is why this file carries their ids and nothing else of
    theirs: duplicating them here would let the copy drift from the original.
    """
    directory.mkdir(parents=True, exist_ok=True)

    (directory / MANIFEST).write_text(
        json.dumps(
            {
                "layout_version": LAYOUT_VERSION,
                "left": {"run_id": left_run_id, "name": diff.left_name},
                "right": {"run_id": right_run_id, "name": diff.right_name},
                "trust": {
                    "level": diff.trust.trust.value,
                    "reasons": list(diff.trust.reasons),
                },
                "significance": {
                    measure: {
                        "delta": result.delta,
                        "ci_low": result.ci_low,
                        "ci_high": result.ci_high,
                        "p_value": result.p_value,
                        "n": result.n,
                        "resamples": result.resamples,
                        "seed": result.seed,
                        "alpha": result.alpha,
                        "confidence": result.confidence,
                        # Stored rather than left to the reader: whether a delta
                        # may be called a difference is the engine's judgement,
                        # and three readers would otherwise derive it three ways.
                        "is_real": result.is_real,
                        "underpowered": result.underpowered,
                    }
                    for measure, result in sorted(diff.significance.items())
                },
                "counts": {
                    "improved": len(diff.improved),
                    "regressed": len(diff.regressed),
                    "unchanged": len(diff.unchanged),
                    "arbitrary": len(diff.arbitrary),
                },
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    deltas: dict[str, list[object]] = {name: [] for name in _DELTAS_SCHEMA}
    for delta in diff.deltas:
        deltas["query_id"].append(str(delta.query_id))
        deltas["left_rank"].append(delta.left_rank)
        deltas["right_rank"].append(delta.right_rank)
        deltas["direction"].append(delta.direction.value)
        deltas["arbitrary"].append(delta.arbitrary)
    _write(pl.DataFrame(deltas, schema=_DELTAS_SCHEMA), directory / DELTAS)

    return directory


def read_comparison_manifest(directory: Path) -> JsonObject:
    """The published comparison manifest, as plain data."""
    return read_run_manifest(directory)
