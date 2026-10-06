"""The command line.

Five verbs: run an experiment, list what a workspace holds, describe one
run, draw failures for a person to name, and measure the rules against what
they decided. Everything
prints the identifiers it produced, so the next command can be typed without
opening a file.

There is deliberately no `diff` verb for two already published runs. `run`
compares them as part of executing the experiment, and a fourth verb that
always printed "not built yet" would advertise a capability in `--help` that
does not exist.

argparse rather than a CLI library: the surface is three subcommands and the
dependency would carry no weight.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import TYPE_CHECKING

from embedlab.adapters.registry import build, build_reranker
from embedlab.artifacts.dataset import DatasetError, load_dataset
from embedlab.artifacts.workspace import Workspace, read_comparison_manifest, read_run_manifest
from embedlab.cache.store import CacheStore
from embedlab.cli.config import Chunking, ConfigError, load_experiment
from embedlab.domain.ids import DocId, QueryId
from embedlab.domain.taxonomy import Symptom
from embedlab.pipeline.compare import compare
from embedlab.pipeline.run import execute
from embedlab.stages.chunk import Chunks, fixed_words, whole_documents

if TYPE_CHECKING:
    import polars as pl

    from embedlab.artifacts.dataset import Dataset
    from embedlab.stages.diagnose import Diagnosis

DEFAULT_ROOT = Path("runs")


def _workspace(root: Path) -> tuple[Workspace, CacheStore]:
    return Workspace(root=root), CacheStore(root / "cache")


def run(arguments: argparse.Namespace) -> int:
    experiment = load_experiment(arguments.experiment)
    dataset = load_dataset(experiment.dataset)
    workspace, store = _workspace(arguments.root)

    print(f"{experiment.name}")
    print(
        f"  {dataset.name}: {len(dataset.corpus):,} docs, "
        f"{len(dataset.judged_queries)} judged queries, k={experiment.k}"
    )

    outcomes = {}
    for system in experiment.systems:
        options = system.model_dump(exclude={"kind", "chunking", "reranker", "candidates"})
        retriever = build(system.kind, **options)
        reranker = (
            None
            if system.reranker is None
            else build_reranker(
                system.reranker.kind, **system.reranker.model_dump(exclude={"kind"})
            )
        )
        chunks = _cut(dataset, system.chunking)
        if chunks is not None:
            per_document = chunks.per_document
            average = sum(per_document.values()) / max(1, len(per_document))
            print(f"\n  {system.name}: {len(chunks.texts):,} chunks ({average:.1f} per document)")
        outcome = execute(
            dataset,
            retriever,
            k=experiment.k,
            measures=experiment.measures,
            store=store,
            workspace=workspace,
            chunks=chunks,
            reranker=reranker,
            candidates=system.candidates,
        )
        outcomes[system.name] = outcome
        origin = " (reused)" if outcome.from_cache else ""
        scores = "  ".join(
            f"{measure}={value:.4f}"
            for measure, value in sorted(outcome.evaluation.aggregate.items())
        )
        print(f"\n  {outcome.manifest.run_id}{origin}\n    {scores}")

    baseline = outcomes[experiment.resolved_baseline()]
    for system in experiment.others():
        comparison = compare(baseline, outcomes[system.name], dataset, workspace=workspace)
        diff = comparison.diff
        print(f"\n  {comparison.comparison_id}")
        print(f"    {diff.headline}")
        for _, result in sorted(diff.significance.items()):
            print(f"    {result.describe()}")
        print(f"    trust: {diff.trust.trust.value}")
        for reason in diff.trust.reasons:
            print(f"      {reason}")

    return 0


def listing(arguments: argparse.Namespace) -> int:
    workspace, _ = _workspace(arguments.root)
    runs, comparisons = workspace.runs(), workspace.comparisons()
    if not runs and not comparisons:
        print(f"{arguments.root}: nothing published yet")
        return 0
    for identifier in runs:
        manifest = read_run_manifest(workspace.run_dir(identifier))
        calibration = manifest.get("calibration")
        judged = calibration.get("n") if isinstance(calibration, dict) else "?"
        print(f"run         {identifier}  ({judged} judged queries)")
    for identifier in comparisons:
        manifest = read_comparison_manifest(workspace.comparison_dir(identifier))
        trust = manifest.get("trust")
        level = trust.get("level") if isinstance(trust, dict) else "?"
        print(f"comparison  {identifier}  (trust {level})")
    return 0


def show(arguments: argparse.Namespace) -> int:
    import polars as pl

    workspace, _ = _workspace(arguments.root)
    directory = workspace.run_dir(arguments.run_id)
    if not directory.is_dir():
        print(f"no run {arguments.run_id!r} under {arguments.root}", file=sys.stderr)
        return 1

    manifest = read_run_manifest(directory)
    measures = manifest.get("measures")
    metrics = pl.read_parquet(directory / "metrics.parquet")
    evidence = pl.read_parquet(directory / "evidence.parquet")
    hypotheses = pl.read_parquet(directory / "hypotheses.parquet")

    print(arguments.run_id)
    if isinstance(measures, list):
        for measure in measures:
            mean = metrics.filter(pl.col("measure") == measure)["value"].mean()
            print(f"  {measure:<10} {mean:.4f}")

    failures = evidence.filter(pl.col("symptom") != Symptom.OK.value).height
    print(f"\n  {failures} of {evidence.height} judged queries failed")
    for row in (
        evidence["symptom"].value_counts().sort("count", descending=True).iter_rows(named=True)
    ):
        print(f"    symptom  {row['symptom']:<14} {row['count']:>4}")
    for row in (
        hypotheses["kind"].value_counts().sort("count", descending=True).iter_rows(named=True)
    ):
        print(f"    cause    {row['kind']:<14} {row['count']:>4}")
    print("\n  Symptoms are exclusive; causes overlap, so they do not sum to the failures.")
    return 0


def _cut(dataset: Dataset, chunking: Chunking | None) -> Chunks | None:
    """Build the cut a system asked for, or none at all.

    `None` and `whole` are not the same thing: `None` indexes the documents as
    they are, `whole` goes through the chunking stage with one chunk each. They
    produce the same ranking and deliberately different cache keys, so a run
    that declared a strategy is never confused with one that declared nothing.
    """
    if chunking is None:
        return None
    if chunking.kind == "whole":
        return whole_documents(dataset.corpus)
    return fixed_words(dataset.corpus, size=chunking.size, overlap=chunking.overlap)


def _published(workspace: Workspace, run_id: str) -> tuple[Path, dict]:
    directory = workspace.run_dir(run_id)
    if not directory.is_dir():
        msg = f"no run {run_id!r} under {workspace.root}"
        raise ValueError(msg)
    return directory, read_run_manifest(directory)


def label(arguments: argparse.Namespace) -> int:
    """Draw failures for a person to name, without showing them the guess."""
    import polars as pl

    from embedlab.artifacts.run_io import read_run
    from embedlab.stages.labelling import LABELLABLE, render_sheet, sample_failures

    workspace, _ = _workspace(arguments.root)
    directory, manifest = _published(workspace, arguments.run_id)

    recorded = manifest.get("dataset")
    if not isinstance(recorded, dict) or "path" not in recorded:
        msg = f"{arguments.run_id} was published before run bundles recorded their dataset"
        raise ValueError(msg)
    dataset = load_dataset(Path(str(recorded["path"])))

    evidence = pl.read_parquet(directory / "evidence.parquet")
    hypotheses = pl.read_parquet(directory / "hypotheses.parquet")
    diagnosis = _rebuild_diagnosis(evidence, hypotheses)

    cases = sample_failures(
        diagnosis,
        read_run(directory),
        dataset.queries,
        dataset.corpus,
        dataset.qrels,
        size=arguments.sample,
        seed=arguments.seed,
    )

    arguments.out.write_text(
        render_sheet(cases, run_id=arguments.run_id, seed=arguments.seed), encoding="utf-8"
    )

    print(f"{len(cases)} failures written to {arguments.out}")
    print(f"  seed {arguments.seed}, so the same draw comes back on a rerun")
    print(f"  fill the 'cause' column with one of: {', '.join(LABELLABLE)}")
    print(f"  then: embedlab score {arguments.run_id} {arguments.out}")
    print("\n  The rules' own guess is deliberately absent: shown it, a labeller")
    print("  agrees with it, and the agreement measured afterwards means nothing.")
    return 0


def score(arguments: argparse.Namespace) -> int:
    """Measure the rules against what a person decided."""
    import polars as pl

    from embedlab.stages.labelling import parse_sheet, score_labels

    workspace, _ = _workspace(arguments.root)
    directory, _ = _published(workspace, arguments.run_id)

    try:
        labels = parse_sheet(arguments.sheet.read_text(encoding="utf-8"))
    except ValueError as error:
        msg = f"{arguments.sheet}: {error}"
        raise ValueError(msg) from error

    if not labels:
        print(f"{arguments.sheet}: no query has a cause yet", file=sys.stderr)
        return 1

    evidence = pl.read_parquet(directory / "evidence.parquet")
    hypotheses = pl.read_parquet(directory / "hypotheses.parquet")
    symptoms = dict(zip(evidence["query_id"], evidence["symptom"], strict=True))
    proposed: dict[str, set[str]] = {}
    for row in hypotheses.iter_rows(named=True):
        proposed.setdefault(row["query_id"], set()).add(row["kind"])

    agreement = score_labels(
        symptoms,  # pyright: ignore[reportArgumentType]
        {key: frozenset(value) for key, value in proposed.items()},  # pyright: ignore[reportArgumentType]
        labels,  # pyright: ignore[reportArgumentType]
    )

    print(f"{arguments.run_id} against {arguments.sheet}")
    print(f"  {agreement.compared} labelled failures compared")
    print(
        f"  the rules named a cause the labeller agreed with {agreement.agreed} times"
        f" ({agreement.accuracy:.0%})"
    )
    if agreement.per_cause:
        print("\n  per cause the rules proposed:")
        for cause, (said, confirmed) in sorted(agreement.per_cause.items()):
            print(
                f"    {cause:<18} proposed {said:>3}, confirmed {confirmed:>3}"
                f"  ({confirmed / said:.0%})"
                if said
                else f"    {cause}"
            )
    if agreement.missed:
        print("\n  causes a person named that no rule proposed:")
        for cause, count in sorted(agreement.missed.items(), key=lambda pair: -pair[1]):
            print(f"    {cause:<18} {count:>3}")
    return 0


def _rebuild_diagnosis(evidence: pl.DataFrame, hypotheses: pl.DataFrame) -> Diagnosis:
    """Reconstruct just enough of a diagnosis from the published tables."""
    from embedlab.domain.taxonomy import FailureKind, Producer
    from embedlab.domain.taxonomy import Symptom as SymptomEnum
    from embedlab.stages.diagnose import Calibration, Diagnosis, Evidence, Hypothesis

    found: dict[QueryId, Evidence] = {}
    for row in evidence.iter_rows(named=True):
        found[QueryId(row["query_id"])] = Evidence(
            query_id=QueryId(row["query_id"]),
            symptom=SymptomEnum(row["symptom"]),
            gold_rank=row["gold_rank"],
            best_relevant=None if row["best_relevant"] is None else DocId(row["best_relevant"]),
            top1=DocId(row["top1"]),
            top1_is_relevant=row["top1_is_relevant"],
            score_margin=row["score_margin"],
            query_gold_overlap=row["query_gold_overlap"],
            competitor_gold_overlap=row["competitor_gold_overlap"],
            query_token_count=row["query_token_count"],
            tied_with_top1=row["tied_with_top1"],
            gold_score_is_zero=row["gold_score_is_zero"],
            retrieved_with_zero_score=row["retrieved_with_zero_score"],
            rank_before_rerank=row["rank_before_rerank"],
        )

    proposed: dict[QueryId, list[Hypothesis]] = {}
    for row in hypotheses.iter_rows(named=True):
        proposed.setdefault(QueryId(row["query_id"]), []).append(
            Hypothesis(
                query_id=QueryId(row["query_id"]),
                kind=FailureKind(row["kind"]),
                confidence=row["confidence"],
                producer=Producer(row["producer"]),
                because=(row["because"],),
            )
        )

    return Diagnosis(
        evidence=found,
        hypotheses={key: tuple(value) for key, value in proposed.items()},
        calibration=Calibration(
            n=len(found), low_query_gold_overlap=0.0, high_competitor_gold_overlap=0.0
        ),
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="embedlab", description=__doc__)
    parser.add_argument(
        "--root", type=Path, default=DEFAULT_ROOT, help="workspace directory (default: runs)"
    )
    commands = parser.add_subparsers(dest="command", required=True)

    runner = commands.add_parser("run", help="execute an experiment file")
    runner.add_argument("experiment", type=Path)
    runner.set_defaults(handler=run)

    commands.add_parser("list", help="what the workspace holds").set_defaults(handler=listing)

    shower = commands.add_parser("show", help="one run's health")
    shower.add_argument("run_id")
    shower.set_defaults(handler=show)

    labeller = commands.add_parser("label", help="draw failures for a person to name")
    labeller.add_argument("run_id")
    labeller.add_argument("--out", type=Path, default=Path("labels.md"))
    labeller.add_argument("--sample", type=int, default=50)
    labeller.add_argument("--seed", type=int, default=20261006)
    labeller.set_defaults(handler=label)

    scorer = commands.add_parser("score", help="measure the rules against those labels")
    scorer.add_argument("run_id")
    scorer.add_argument("sheet", type=Path)
    scorer.set_defaults(handler=score)

    arguments = parser.parse_args(sys.argv[1:] if argv is None else argv)
    try:
        return int(arguments.handler(arguments))
    except (ConfigError, DatasetError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
