"""How much of a comparison to believe.

Vocabulary rather than machinery. The rules that reach a verdict need run
manifests and so live with the artifacts; the verdict itself is a word the
whole system speaks, and placing it here is what lets a pure stage carry one
without depending on the I/O layer.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class Trust(StrEnum):
    """How much of a diff between two runs to believe."""

    IDENTICAL = "identical"
    """Same environment, same stage semantics, both clean. Fully trustworthy."""

    COMPARABLE = "comparable"
    """Provenance differs but nothing that can move the numbers does."""

    SUSPECT = "suspect"
    """Something that can move the numbers differs. The diff still renders, with
    every difference named. This is where a forgotten impl_version bump lands,
    reached mechanically, without relying on anyone having remembered."""

    INCOMPARABLE = "incomparable"
    """The comparison is meaningless. Refuse."""


SEVERITY: dict[Trust, int] = {
    Trust.IDENTICAL: 0,
    Trust.COMPARABLE: 1,
    Trust.SUSPECT: 2,
    Trust.INCOMPARABLE: 3,
}
"""Ordering used to let the worst finding decide a verdict."""


@dataclass(frozen=True)
class Verdict:
    trust: Trust
    reasons: tuple[str, ...] = ()

    @property
    def refuses(self) -> bool:
        return self.trust is Trust.INCOMPARABLE
