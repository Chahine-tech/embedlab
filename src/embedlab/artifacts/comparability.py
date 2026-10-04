"""May these two runs be compared, and how far can the answer be trusted?

A binary allow/refuse gate does not work here. Requiring an identical engine
commit would reject every diff taken after any commit — as unusable as putting
the commit in every cache key. Requiring nothing silently compares runs
computed with different semantics, which is the failure this module exists to
prevent.

So the verdict is a *trust level*, surfaced alongside the diff. That is the same
stance the product takes everywhere else: show the number, and show how much of
it to believe.

Everything is judged per stage, using the environment recorded on that stage's
provenance rather than on the run. A cached artifact may be weeks old, and
asking "what produced *this* artifact" is the only question whose answer is
reliable. Relevance is per stage too, so editing the diagnosis code does not
block a retrieval diff and a torch upgrade does not block a BM25 diff — a gate
that fires on irrelevant changes gets overridden by reflex.

This module is pure. Whether source changed between two commits needs git, so
the caller determines it and passes it in, which keeps the policy testable
without a repository.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import TYPE_CHECKING

from embedlab.artifacts.manifest import Environment, RunManifest

if TYPE_CHECKING:
    from collections.abc import Collection, Mapping


class Trust(StrEnum):
    """How much of a diff between two runs to believe."""

    IDENTICAL = "identical"
    """Same environment, same stage semantics, both clean. Fully trustworthy."""

    COMPARABLE = "comparable"
    """Provenance differs but nothing that can move the numbers does."""

    SUSPECT = "suspect"
    """Something that can move the numbers differs. The diff still renders, with
    every difference named. This is where a forgotten impl_version bump lands —
    reached mechanically, without relying on anyone having remembered."""

    INCOMPARABLE = "incomparable"
    """The comparison is meaningless. Refuse."""


_SEVERITY = {Trust.IDENTICAL: 0, Trust.COMPARABLE: 1, Trust.SUSPECT: 2, Trust.INCOMPARABLE: 3}


@dataclass(frozen=True)
class Verdict:
    trust: Trust
    reasons: tuple[str, ...] = ()

    @property
    def refuses(self) -> bool:
        return self.trust is Trust.INCOMPARABLE


def _data_findings(left: RunManifest, right: RunManifest) -> list[tuple[Trust, str]]:
    """Checks that make a comparison meaningless rather than merely doubtful."""
    findings: list[tuple[Trust, str]] = []

    if left.schema_version != right.schema_version:
        findings.append(
            (
                Trust.INCOMPARABLE,
                f"artifact schema differs: v{left.schema_version} vs v{right.schema_version}",
            )
        )
    if left.labels != right.labels:
        findings.append(
            (
                Trust.INCOMPARABLE,
                "runs were scored against different queries or labels "
                f"({left.labels} vs {right.labels})",
            )
        )
    if left.corpus != right.corpus:
        findings.append(
            (
                Trust.INCOMPARABLE,
                f"runs were executed over different corpora ({left.corpus} vs {right.corpus})",
            )
        )
    return findings


def _environment_findings(
    stage: str,
    left: Environment,
    right: Environment,
    *,
    code_changed: bool | None,
    packages: tuple[str, ...],
) -> list[tuple[Trust, str]]:
    findings: list[tuple[Trust, str]] = []

    for side, environment in (("left", left), ("right", right)):
        if environment.engine_commit is None:
            findings.append(
                (
                    Trust.SUSPECT,
                    f"{side} {stage!r} artifact has unknown provenance "
                    "(not produced inside a git repository)",
                )
            )
        elif environment.engine_dirty:
            findings.append(
                (
                    Trust.SUSPECT,
                    f"{side} {stage!r} artifact was produced from a dirty working tree "
                    "and is not reproducible",
                )
            )

    if left.python != right.python:
        findings.append(
            (
                Trust.SUSPECT,
                f"python differs for stage {stage!r}: {left.python} vs {right.python}",
            )
        )
    if left.platform != right.platform:
        findings.append(
            (
                Trust.SUSPECT,
                f"platform differs for stage {stage!r}: {left.platform} vs {right.platform}",
            )
        )

    for package in packages:
        version_left = left.packages.get(package)
        version_right = right.packages.get(package)
        if version_left != version_right:
            findings.append(
                (
                    Trust.SUSPECT,
                    f"{package} differs and affects stage {stage!r}: "
                    f"{version_left or 'absent'} vs {version_right or 'absent'}",
                )
            )

    if left.engine_commit != right.engine_commit:
        if code_changed is None:
            findings.append(
                (
                    Trust.SUSPECT,
                    f"stage {stage!r} was produced at different engine commits and the "
                    "relevant source could not be compared",
                )
            )
        elif code_changed:
            findings.append(
                (
                    Trust.SUSPECT,
                    f"engine source affecting stage {stage!r} changed between the two commits",
                )
            )
        else:
            findings.append(
                (
                    Trust.COMPARABLE,
                    f"stage {stage!r} was produced at different engine commits, but no "
                    "source affecting it changed",
                )
            )

    return findings


def assess(
    left: RunManifest,
    right: RunManifest,
    *,
    stages: Collection[str],
    code_changed: Mapping[str, bool | None] | None = None,
) -> Verdict:
    """Assess two runs over the stages a comparison actually depends on.

    `stages` comes from the dependency graph: comparing retrieval results does
    not care that the diagnosis code changed. `code_changed` maps a stage name
    to the caller's git-derived answer for that stage's source paths; a missing
    entry or None means "could not be determined" and is treated as suspect,
    never as clean.
    """
    findings = _data_findings(left, right)
    resolved = code_changed or {}

    for stage in sorted(set(stages)):
        provenance_left, provenance_right = left.stage(stage), right.stage(stage)
        if provenance_left is None or provenance_right is None:
            missing = "left" if provenance_left is None else "right"
            findings.append(
                (
                    Trust.INCOMPARABLE,
                    f"stage {stage!r} is absent from the {missing} run",
                )
            )
            continue

        if provenance_left.impl_version != provenance_right.impl_version:
            findings.append(
                (
                    Trust.SUSPECT,
                    f"stage {stage!r} semantics were declared changed "
                    f"(impl_version {provenance_left.impl_version} vs "
                    f"{provenance_right.impl_version})",
                )
            )

        # The union of both sides: if one run used a dense retriever and the
        # other did not, torch is decisive for the comparison even though only
        # one artifact names it.
        packages = tuple(sorted(set(provenance_left.packages) | set(provenance_right.packages)))
        findings.extend(
            _environment_findings(
                stage,
                provenance_left.environment,
                provenance_right.environment,
                code_changed=resolved.get(stage),
                packages=packages,
            )
        )

    if not findings:
        return Verdict(Trust.IDENTICAL)

    worst = max(findings, key=lambda finding: _SEVERITY[finding[0]])[0]
    return Verdict(worst, tuple(reason for _, reason in findings))
