"""Failure-first terminal report.

Deliberately opens on failures rather than on an aggregate score: the headline
is "N queries failed", and every number is followed by the evidence it rests on.

A scratch front-end for the spike, not the product's interface. It exists to
make the artifacts readable while the engine is being built, and to be the
thing a reader runs first.
"""

from __future__ import annotations

import sys
from pathlib import Path

from embedlab.adapters.lexical import BM25Retriever
from embedlab.adapters.tfidf import TfidfRetriever
from embedlab.artifacts.dataset import load_dataset
from embedlab.artifacts.manifest import RunManifest
from embedlab.artifacts.workspace import Workspace
from embedlab.cache.store import CacheStore
from embedlab.domain.ids import QueryId
from embedlab.domain.taxonomy import Symptom
from embedlab.pipeline.compare import compare
from embedlab.pipeline.run import RunOutcome, execute
from embedlab.stages.diagnose import Diagnosis
from embedlab.stages.diff import failure_sets

RULE = "─" * 72


def _rank(value: int | None) -> str:
    return "not retrieved" if value is None else f"#{value}"


def _health(outcome: RunOutcome) -> None:
    diagnosis = outcome.diagnosis
    failing = diagnosis.failing()
    judged = len(diagnosis.evidence)

    origin = "  [retrieval reused from cache]" if outcome.from_cache else ""
    print(f"\n{outcome.name}  ({outcome.manifest.run_id}){origin}")
    print(f"  {len(failing)} of {judged} judged queries failed to put a relevant doc first")
    for measure, value in sorted(outcome.evaluation.aggregate.items()):
        print(f"  {measure:<10} {value:.3f}")

    sets = failure_sets(diagnosis)
    if sets:
        print("  failure sets (a query may appear in several):")
        width = max(len(name) for name in sets)
        for name, members in sorted(sets.items(), key=lambda item: (-len(item[1]), item[0])):
            bar = "█" * len(members)
            print(f"    {name:<{width}}  {len(members):>2}  {bar}")


def _failure_detail(outcome: RunOutcome, query_id: QueryId, queries: dict[QueryId, str]) -> None:
    evidence = outcome.diagnosis.evidence[query_id]
    ranked = outcome.run[query_id]

    print(f"\n{RULE}\nWHY DID THIS FAIL?   {query_id}\n{RULE}")
    print(f'\nQUERY\n  "{queries[query_id]}"')
    print(f"\nSYMPTOM\n  {evidence.symptom.value}: gold at {_rank(evidence.gold_rank)}")
    print(f"\n{outcome.name.upper()} TOP RESULTS")
    for item in ranked[:5]:
        mark = "✓" if item.doc_id == evidence.best_relevant else " "
        print(f"  {mark} {item.rank}. {item.doc_id:<22} {item.score:.3f}")

    print("\nEVIDENCE (deterministic)")
    print(f"  query/gold vocabulary overlap      {evidence.query_gold_overlap:.1%}")
    print(f"  rank-1/gold vocabulary overlap     {evidence.competitor_gold_overlap:.1%}")
    if evidence.score_margin is not None:
        print(f"  score margin above gold            {evidence.score_margin:.3f}")
    print(f"  tied with rank-1                   {evidence.tied_with_top1}")
    print(
        f"  returned with zero score           {evidence.retrieved_with_zero_score}"
        f" of {len(ranked)}"
    )

    print("\nHYPOTHESES (interpretation)")
    for hypothesis in outcome.diagnosis.hypotheses[query_id]:
        print(
            f"  {hypothesis.kind.value}  (confidence {hypothesis.confidence:.1f},"
            f" by {hypothesis.producer.value})"
        )
        for reason in hypothesis.because:
            print(f"    because {reason}")


def _commit_for(manifest: RunManifest, stage: str) -> str | None:
    """The commit that produced one stage, or None when the stage is absent.

    None degrades correctly: `relevant_source_changed` reports "unknown", which
    the gate treats as suspect. A missing stage is separately refused outright
    by `assess`, so nothing is swallowed, but reaching that refusal must not
    require surviving an AttributeError first.
    """
    provenance = manifest.stage(stage)
    return None if provenance is None else provenance.environment.engine_commit


def main(argv: list[str] | None = None) -> int:
    arguments = sys.argv[1:] if argv is None else argv
    directory = Path(arguments[0]) if arguments else Path("tests/fixtures/mini")

    dataset = load_dataset(directory)
    print(
        f"{RULE}\nDataset {dataset.name}: {len(dataset.corpus)} docs, "
        f"{len(dataset.judged_queries)} judged queries\n{RULE}"
    )

    root = Path("runs")
    store = CacheStore(root / "cache")
    workspace = Workspace(root=root)
    left = execute(dataset, BM25Retriever(), k=10, store=store, workspace=workspace)
    right = execute(dataset, TfidfRetriever(), k=10, store=store, workspace=workspace)

    for outcome in (left, right):
        _health(outcome)

    comparison = compare(left, right, dataset, workspace=workspace)
    diff, trust = comparison.diff, comparison.diff.trust

    print(f"\n{RULE}\nWHAT CHANGED?   {diff.left_name} → {diff.right_name}\n{RULE}")
    print(f"\n  {diff.headline}")
    for _, result in sorted(diff.significance.items()):
        print(f"  {result.describe()}")

    print(f"\n  trust: {trust.trust.value}")
    for reason in trust.reasons:
        print(f"    - {reason}")

    for direction, deltas in (("IMPROVED", diff.improved), ("REGRESSED", diff.regressed)):
        if deltas:
            print(f"\n  {direction}")
            for delta in deltas:
                flag = "   ⚠ arbitrary (score tie)" if delta.arbitrary else ""
                print(
                    f"    {delta.query_id:<16} {_rank(delta.left_rank)} → "
                    f"{_rank(delta.right_rank)}{flag}"
                )

    # Pick the most instructive single case: a query the two systems disagree on.
    interesting = next(
        (d for d in diff.improved if d.became_success),
        next(iter(diff.improved), None) or next(iter(diff.regressed), None),
    )
    if interesting is not None:
        _failure_detail(left, interesting.query_id, dataset.queries)
        _failure_detail(right, interesting.query_id, dataset.queries)

    shared = [
        query_id
        for query_id in left.diagnosis.evidence
        if left.diagnosis.evidence[query_id].symptom is not Symptom.OK
        and right.diagnosis.evidence[query_id].symptom is not Symptom.OK
    ]
    if shared:
        print(f"\n{RULE}\nFAILS FOR BOTH SYSTEMS\n{RULE}")
        for query_id in shared:
            print(f"  {query_id:<16} {queries_hint(left.diagnosis, query_id)}")

    print(f"\n{RULE}\nPUBLISHED\n{RULE}")
    print(f"  run         {workspace.run_dir(left.manifest.run_id)}")
    print(f"  run         {workspace.run_dir(right.manifest.run_id)}")
    print(f"  comparison  {workspace.comparison_dir(comparison.comparison_id)}")

    return 0


def queries_hint(diagnosis: Diagnosis, query_id: QueryId) -> str:
    kinds = ", ".join(h.kind.value for h in diagnosis.hypotheses[query_id])
    return kinds or "no hypothesis"


if __name__ == "__main__":
    raise SystemExit(main())
