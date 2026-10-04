"""Capture the environment a run actually executed in."""

from __future__ import annotations

import platform
import subprocess
import sys
from collections.abc import Iterable
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

from embedlab.artifacts.manifest import NUMERIC_PACKAGES, Environment

_GIT_TIMEOUT_S = 5


def _git(args: list[str], *, cwd: Path) -> str | None:
    """Run a read-only git command, returning None if git or the repo is absent."""
    try:
        result = subprocess.run(
            ["git", *args],
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=_GIT_TIMEOUT_S,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if result.returncode != 0:
        return None
    return result.stdout.strip()


def _installed_versions() -> dict[str, str]:
    """Resolved versions of the packages that can move a score.

    Absent packages are simply omitted, so a run with torch and a run without
    differ in the record — which is correct: they cannot be compared blindly.
    """
    found: dict[str, str] = {}
    for name in NUMERIC_PACKAGES:
        try:
            found[name] = version(name)
        except PackageNotFoundError:
            continue
    return found


def capture_environment(*, repo: Path | None = None) -> Environment:
    """Snapshot the current environment.

    `repo` defaults to the engine's own source tree, so the recorded commit is
    the engine's, not that of whatever directory the user happened to run from.
    """
    root = repo if repo is not None else Path(__file__).resolve().parent.parent.parent.parent
    commit = _git(["rev-parse", "HEAD"], cwd=root)
    status = _git(["status", "--porcelain"], cwd=root)
    if commit is None:
        # No repository (or no commit yet): provenance is unknown. The gate keys
        # off `engine_commit is None` for that, so we do not overload `dirty`.
        dirty = False
    elif status is None:
        # Repo exists but status failed: assume the worst rather than claim clean.
        dirty = True
    else:
        dirty = bool(status)

    return Environment(
        python=platform.python_version(),
        platform=f"{sys.platform}-{platform.machine()}",
        engine_commit=commit,
        engine_dirty=dirty,
        packages=_installed_versions(),
    )


_SHARED_SOURCE: tuple[str, ...] = (
    "src/embedlab/domain",
    "src/embedlab/cache",
    "src/embedlab/artifacts",
)
"""Foundations every stage sits on. A change here is relevant to all of them."""


def stage_source_paths(stages: Iterable[str]) -> tuple[str, ...]:
    """Source paths whose change could alter the given stages' output.

    Deliberately path-based rather than call-graph based. A static call graph is
    not computable in Python once Protocol adapters and a YAML registry do the
    dispatch, and an 80%-correct graph fails silently in exactly the cases it is
    bought for. Paths over-approximate instead — the safe direction.
    """
    paths = {*_SHARED_SOURCE, "src/embedlab/adapters", "src/embedlab/pipeline"}
    paths.update(f"src/embedlab/stages/{stage}.py" for stage in stages)
    return tuple(sorted(paths))


def relevant_source_changed(
    commit_a: str | None,
    commit_b: str | None,
    *,
    stages: Iterable[str],
    repo: Path | None = None,
) -> bool | None:
    """Did engine source relevant to `stages` change between two commits?

    Returns None when the question cannot be answered (missing commits, shallow
    clone, no git). The caller must treat None as "unknown", never as "no".
    """
    if commit_a is None or commit_b is None:
        return None
    if commit_a == commit_b:
        return False

    root = repo if repo is not None else Path(__file__).resolve().parent.parent.parent.parent
    try:
        result = subprocess.run(
            ["git", "diff", "--quiet", commit_a, commit_b, "--", *stage_source_paths(stages)],
            cwd=root,
            capture_output=True,
            text=True,
            timeout=_GIT_TIMEOUT_S,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None

    # 0 = no difference, 1 = difference. Anything else (unknown revision,
    # shallow clone) is an error and must surface as unknown, not as "clean".
    if result.returncode == 0:
        return False
    if result.returncode == 1:
        return True
    return None
