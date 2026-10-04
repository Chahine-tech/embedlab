"""Environment capture and git-derived source change detection."""

import subprocess

import pytest

from embedlab.artifacts.manifest import NUMERIC_PACKAGES
from embedlab.artifacts.provenance import (
    capture_environment,
    relevant_source_changed,
    stage_source_paths,
)


def git(repo, *args):
    subprocess.run(["git", *args], cwd=repo, check=True, capture_output=True)


def head(repo) -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=repo, check=True, capture_output=True, text=True
    )
    return result.stdout.strip()


@pytest.fixture
def repo(tmp_path):
    """A repo shaped like the engine, with one commit."""
    git(tmp_path, "init", "-q")
    git(tmp_path, "config", "user.email", "test@example.invalid")
    git(tmp_path, "config", "user.name", "test")

    (tmp_path / "src" / "embedlab" / "stages").mkdir(parents=True)
    (tmp_path / "src" / "embedlab" / "stages" / "retrieve.py").write_text("K = 10\n")
    (tmp_path / "src" / "embedlab" / "stages" / "evaluate.py").write_text("M = 'ndcg'\n")
    (tmp_path / "docs").mkdir()
    (tmp_path / "docs" / "notes.md").write_text("hello\n")

    git(tmp_path, "add", "-A")
    git(tmp_path, "commit", "-qm", "initial")
    return tmp_path


def test_captures_a_usable_environment():
    environment = capture_environment()
    assert environment.python.startswith("3.13")
    assert environment.platform
    assert set(environment.packages) <= set(NUMERIC_PACKAGES)
    # numpy ships in the core dependencies, so it must be recorded.
    assert "numpy" in environment.packages


def test_no_repository_yields_unknown_provenance(tmp_path):
    environment = capture_environment(repo=tmp_path)
    assert environment.engine_commit is None
    # Dirtiness is not claimed either way when there is no repo.
    assert environment.engine_dirty is False


def test_clean_repository_is_not_dirty(repo):
    environment = capture_environment(repo=repo)
    assert environment.engine_commit == head(repo)
    assert environment.engine_dirty is False


def test_uncommitted_change_is_dirty(repo):
    (repo / "src" / "embedlab" / "stages" / "retrieve.py").write_text("K = 20\n")
    assert capture_environment(repo=repo).engine_dirty is True


def test_untracked_file_is_dirty(repo):
    (repo / "scratch.py").write_text("x = 1\n")
    assert capture_environment(repo=repo).engine_dirty is True


def test_same_commit_means_no_change(repo):
    sha = head(repo)
    assert relevant_source_changed(sha, sha, stages=["retrieve"], repo=repo) is False


def test_change_to_a_stage_is_detected(repo):
    before = head(repo)
    (repo / "src" / "embedlab" / "stages" / "retrieve.py").write_text("K = 20\n")
    git(repo, "commit", "-aqm", "change retrieve")
    after = head(repo)

    assert relevant_source_changed(before, after, stages=["retrieve"], repo=repo) is True


def test_change_to_another_stage_is_not_relevant(repo):
    """Editing evaluate must not block a diff of retrieval results."""
    before = head(repo)
    (repo / "src" / "embedlab" / "stages" / "evaluate.py").write_text("M = 'map'\n")
    git(repo, "commit", "-aqm", "change evaluate")
    after = head(repo)

    assert relevant_source_changed(before, after, stages=["retrieve"], repo=repo) is False
    assert relevant_source_changed(before, after, stages=["evaluate"], repo=repo) is True


def test_change_outside_the_engine_is_not_relevant(repo):
    before = head(repo)
    (repo / "docs" / "notes.md").write_text("rewritten\n")
    git(repo, "commit", "-aqm", "docs")
    after = head(repo)

    assert relevant_source_changed(before, after, stages=["retrieve"], repo=repo) is False


def test_shared_foundation_change_is_relevant_to_every_stage(repo):
    before = head(repo)
    cache = repo / "src" / "embedlab" / "cache"
    cache.mkdir()
    (cache / "fingerprint.py").write_text("DIGEST = 16\n")
    git(repo, "add", "-A")
    git(repo, "commit", "-qm", "cache")
    after = head(repo)

    for stage in ("retrieve", "evaluate", "embed"):
        assert relevant_source_changed(before, after, stages=[stage], repo=repo) is True


def test_unknown_commit_is_unknown_not_clean(repo):
    """The dangerous default would be False. It must be None."""
    assert relevant_source_changed("deadbeef" * 5, head(repo), stages=["retrieve"], repo=repo) is (
        None
    )


def test_missing_commits_are_unknown(repo):
    assert relevant_source_changed(None, head(repo), stages=["retrieve"], repo=repo) is None
    assert relevant_source_changed(head(repo), None, stages=["retrieve"], repo=repo) is None


def test_outside_a_repository_the_answer_is_unknown(tmp_path):
    assert relevant_source_changed("a" * 40, "b" * 40, stages=["retrieve"], repo=tmp_path) is None


def test_stage_paths_include_shared_foundations_and_the_named_stage():
    paths = stage_source_paths(["retrieve"])
    assert "src/embedlab/stages/retrieve.py" in paths
    assert "src/embedlab/domain" in paths
    assert "src/embedlab/stages/evaluate.py" not in paths
