"""The command line.

Three verbs, because the engine has three things worth asking it: run an
experiment, list what a workspace holds, and describe one run. Everything
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

from embedlab.adapters.registry import build
from embedlab.artifacts.dataset import DatasetError, load_dataset
from embedlab.artifacts.workspace import Workspace, read_comparison_manifest, read_run_manifest
from embedlab.cache.store import CacheStore
from embedlab.cli.config import ConfigError, load_experiment
from embedlab.domain.taxonomy import Symptom
from embedlab.pipeline.compare import compare
from embedlab.pipeline.run import execute

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
        options = system.model_dump(exclude={"kind"})
        retriever = build(system.kind, **options)
        outcome = execute(
            dataset,
            retriever,
            k=experiment.k,
            measures=experiment.measures,
            store=store,
            workspace=workspace,
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

    arguments = parser.parse_args(sys.argv[1:] if argv is None else argv)
    try:
        return int(arguments.handler(arguments))
    except (ConfigError, DatasetError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
