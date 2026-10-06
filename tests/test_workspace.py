"""The on-disk contract: what a reader gets without importing this engine."""

import json

import pytest

from embedlab.artifacts.workspace import (
    DELTAS,
    EVIDENCE,
    HYPOTHESES,
    LAYOUT_VERSION,
    MANIFEST,
    METRICS,
    RUN_FILE,
    Workspace,
    read_comparison_manifest,
    read_run_manifest,
)

pl = pytest.importorskip("polars")
pytest.importorskip("bm25s", reason="needs the 'lexical' extra")
pytest.importorskip("ir_measures", reason="needs the 'measures' extra")

from embedlab.adapters.lexical import BM25Retriever  # noqa: E402
from embedlab.pipeline.compare import compare  # noqa: E402
from embedlab.pipeline.run import execute  # noqa: E402


def at(manifest, *keys):
    """Walk a published manifest, asserting its shape on the way down.

    The manifest's values are typed `object` on purpose: it is a contract read
    by things that are not this engine, so every reader validates. These tests
    are readers, and this is them doing it.
    """
    node: object = manifest
    for key in keys:
        assert isinstance(node, dict), f"expected an object at {key!r}, found {type(node).__name__}"
        assert key in node, f"missing key {key!r}"
        node = node[key]
    return node


def number(value) -> float:
    """A manifest value that must be numeric, checked rather than assumed."""
    assert isinstance(value, (int, float)), f"expected a number, found {type(value).__name__}"
    return float(value)


@pytest.fixture
def workspace(tmp_path):
    return Workspace(root=tmp_path / "runs")


@pytest.fixture
def published(mini, workspace):
    left = execute(mini, BM25Retriever(name="plain"), k=10, workspace=workspace)
    right = execute(mini, BM25Retriever(name="loose", k1=0.8, b=0.4), k=10, workspace=workspace)
    comparison = compare(left, right, mini, workspace=workspace, resamples=200)
    return left, right, comparison


def test_identifiers_cannot_escape_the_workspace(workspace):
    for bad in ("../elsewhere", "a/b", ".hidden", ""):
        with pytest.raises(ValueError, match="invalid identifier"):
            workspace.run_dir(bad)


def test_a_run_publishes_every_file_a_reader_needs(published, workspace, mini):
    left, _, _ = published
    directory = workspace.run_dir(str(left.manifest.run_id))

    for name in (MANIFEST, RUN_FILE, METRICS, EVIDENCE, HYPOTHESES):
        assert (directory / name).exists(), f"{name} missing"

    manifest = read_run_manifest(directory)
    assert at(manifest, "layout_version") == LAYOUT_VERSION
    assert at(manifest, "measures") == list(left.evaluation.measures)
    assert at(manifest, "manifest", "run_id") == str(left.manifest.run_id)
    assert at(manifest, "calibration", "n") == len(mini.judged_queries)


def test_the_published_metrics_match_the_run(published, workspace):
    left, _, _ = published
    frame = pl.read_parquet(workspace.run_dir(str(left.manifest.run_id)) / METRICS)

    assert set(frame["measure"].unique()) == set(left.evaluation.measures)
    for row in frame.iter_rows(named=True):
        assert row["value"] == left.evaluation.per_query[row["query_id"]][row["measure"]]


def test_the_published_evidence_matches_the_diagnosis(published, workspace):
    left, _, _ = published
    frame = pl.read_parquet(workspace.run_dir(str(left.manifest.run_id)) / EVIDENCE)

    assert frame.height == len(left.diagnosis.evidence)
    for row in frame.iter_rows(named=True):
        found = left.diagnosis.evidence[row["query_id"]]
        assert row["symptom"] == found.symptom.value
        assert row["gold_rank"] == found.gold_rank
        assert row["tied_with_top1"] == found.tied_with_top1


def test_a_query_with_several_causes_gets_several_rows(published, workspace):
    """Long format, so a second cause does not need a schema change."""
    left, _, _ = published
    frame = pl.read_parquet(workspace.run_dir(str(left.manifest.run_id)) / HYPOTHESES)
    expected = sum(len(h) for h in left.diagnosis.hypotheses.values())
    assert frame.height == expected
    assert set(frame.columns) == {"query_id", "kind", "confidence", "producer", "because"}


def test_every_hypothesis_carries_its_producer_and_trail(published, workspace):
    left, _, _ = published
    frame = pl.read_parquet(workspace.run_dir(str(left.manifest.run_id)) / HYPOTHESES)
    for row in frame.iter_rows(named=True):
        assert row["producer"] in {"rule", "judge", "human"}
        assert row["because"], "a label that cannot be argued with is not a hypothesis"


def test_a_comparison_references_its_runs_rather_than_copying_them(published, workspace):
    left, right, comparison = published
    directory = workspace.comparison_dir(comparison.comparison_id)

    manifest = read_comparison_manifest(directory)
    assert at(manifest, "left", "run_id") == str(left.manifest.run_id)
    assert at(manifest, "right", "run_id") == str(right.manifest.run_id)
    assert "run" not in manifest, "the rankings belong to the run bundles"


def test_the_comparison_stores_the_verdict_rather_than_leaving_it_derivable(published, workspace):
    """Three readers would otherwise derive `is_real` three ways."""
    _, _, comparison = published
    manifest = read_comparison_manifest(workspace.comparison_dir(comparison.comparison_id))

    assert at(manifest, "trust", "level") in {"identical", "comparable", "suspect", "incomparable"}
    significance = at(manifest, "significance")
    assert isinstance(significance, dict)
    for measure in significance:
        assert measure in comparison.diff.significance
        assert at(significance, measure, "is_real") == (
            comparison.diff.significance[measure].is_real
        )
        low, delta, high = (
            number(at(significance, measure, key)) for key in ("ci_low", "delta", "ci_high")
        )
        assert low <= delta <= high
        assert at(significance, measure, "seed")
        assert at(significance, measure, "resamples")


def test_the_deltas_round_trip(published, workspace):
    _, _, comparison = published
    frame = pl.read_parquet(workspace.comparison_dir(comparison.comparison_id) / DELTAS)

    assert frame.height == len(comparison.diff.deltas)
    by_id = {d.query_id: d for d in comparison.diff.deltas}
    for row in frame.iter_rows(named=True):
        delta = by_id[row["query_id"]]
        assert row["left_rank"] == delta.left_rank
        assert row["right_rank"] == delta.right_rank
        assert row["direction"] == delta.direction.value
        assert row["arbitrary"] == delta.arbitrary


def test_the_counts_in_the_manifest_match_the_deltas(published, workspace):
    _, _, comparison = published
    directory = workspace.comparison_dir(comparison.comparison_id)
    manifest = read_comparison_manifest(directory)
    frame = pl.read_parquet(directory / DELTAS)

    counts = at(manifest, "counts")
    assert isinstance(counts, dict)
    for direction, count in counts.items():
        if direction == "arbitrary":
            assert count == frame.filter(pl.col("arbitrary")).height
        else:
            assert count == frame.filter(pl.col("direction") == direction).height


def test_the_workspace_lists_what_it_holds(published, workspace):
    left, right, comparison = published
    assert sorted(workspace.runs()) == sorted(
        {str(left.manifest.run_id), str(right.manifest.run_id)}
    )
    assert workspace.comparisons() == [comparison.comparison_id]


def test_an_empty_workspace_lists_nothing(workspace):
    assert workspace.runs() == []
    assert workspace.comparisons() == []


def test_running_without_a_workspace_publishes_nothing(mini, workspace):
    execute(mini, BM25Retriever(), k=10)
    assert workspace.runs() == []


def test_the_manifest_is_plain_json(published, workspace):
    """A reader must not need this engine to open it."""
    left, _, _ = published
    raw = (workspace.run_dir(str(left.manifest.run_id)) / MANIFEST).read_text()
    assert isinstance(json.loads(raw), dict)


def test_the_workspace_publishes_an_index(published, workspace):
    """A reader should not have to scrape a directory listing: whether one is
    served at all depends on the web server, and its markup is not a contract."""
    left, right, comparison = published
    index = json.loads((workspace.root / "index.json").read_text())

    assert index["layout_version"] == LAYOUT_VERSION
    assert sorted(index["runs"]) == sorted({str(left.manifest.run_id), str(right.manifest.run_id)})
    assert index["comparisons"] == [comparison.comparison_id]


def test_the_index_is_refreshed_by_publishing(mini, workspace):
    from embedlab.adapters.lexical import BM25Retriever as Retriever

    execute(mini, Retriever(name="first"), k=5, workspace=workspace)
    before = json.loads((workspace.root / "index.json").read_text())["runs"]

    execute(mini, Retriever(name="second"), k=5, workspace=workspace)
    after = json.loads((workspace.root / "index.json").read_text())["runs"]

    assert len(after) == len(before) + 1
