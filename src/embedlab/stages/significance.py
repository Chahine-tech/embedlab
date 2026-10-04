"""Is a difference between two runs real, or is it noise?

Without this, the headline lies by omission. At a few hundred queries a
1.5-point nDCG gap is routinely indistinguishable from resampling noise, and
most comparison tools print the bare number. The very first run of this engine
produced "+1 improved" that turned out to rest entirely on a score tie: the
same class of mistake one level up.

Two standard paired procedures, both computed on *our* per-query values rather
than re-derived from the runs:

* a **paired bootstrap** for the confidence interval on the mean difference;
* a **paired randomisation test** (sign-flipping) for the p-value, which is the
  conventional choice in IR evaluation.

Why not `ranx.compare`, which offers the same tests: it recomputes the metrics
itself from runs and qrels, applying its own tie-breaking. That would quietly
undo the guarantee that the evaluated ordering is the ordering we reported, so
the tests are run over the per-query values the evaluate stage already produced.

Everything is seeded. A p-value that drifted between invocations would be its
own source of phantom differences, so the seed is an explicit argument and is
recorded in the result.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Mapping

    from embedlab.domain.ids import QueryId

IMPL_VERSION = 1

DEFAULT_RESAMPLES = 10_000
DEFAULT_SEED = 20261004
DEFAULT_CONFIDENCE = 0.95
DEFAULT_ALPHA = 0.05

ADVISORY_MINIMUM_QUERIES = 50
"""Below this, treat any aggregate difference as undecided regardless of the
p-value. A judgement call, not a theorem: with a handful of queries a single
label flip moves the mean, and the interval is wide enough to be useless even
when the test happens to come out small."""


class NotPairedError(ValueError):
    """The two runs do not cover the same queries, so nothing can be paired."""


@dataclass(frozen=True, slots=True)
class Significance:
    """The evidence behind one measure's delta."""

    measure: str
    n: int
    left_mean: float
    right_mean: float
    delta: float
    ci_low: float
    ci_high: float
    p_value: float
    resamples: int
    seed: int
    alpha: float
    confidence: float

    @property
    def interval_excludes_zero(self) -> bool:
        return (self.ci_low > 0.0) or (self.ci_high < 0.0)

    @property
    def underpowered(self) -> bool:
        return self.n < ADVISORY_MINIMUM_QUERIES

    @property
    def is_real(self) -> bool:
        """Whether this difference may be reported as a difference at all.

        Deliberately conservative: the test must reject, the interval must
        exclude zero, and there must be enough queries to be worth testing.
        """
        return self.p_value < self.alpha and self.interval_excludes_zero and not self.underpowered

    def describe(self) -> str:
        if self.delta == 0.0:
            return f"{self.measure}: no change"
        direction = "better" if self.delta > 0 else "worse"
        core = (
            f"{self.measure}: {self.delta:+.4f} ({direction}), "
            f"{self.confidence:.0%} CI [{self.ci_low:+.4f}, {self.ci_high:+.4f}], "
            f"p={self.p_value:.3f}, n={self.n}"
        )
        if self.underpowered:
            return f"{core}; undecided, {self.n} queries is too few to tell"
        if not self.is_real:
            return f"{core}; indistinguishable from noise"
        return f"{core}; real"


def _paired(
    left: Mapping[QueryId, float],
    right: Mapping[QueryId, float],
) -> tuple[np.ndarray, np.ndarray, int]:
    shared = sorted(set(left) & set(right))
    if not shared:
        msg = "the two runs share no queries, so no paired comparison is possible"
        raise NotPairedError(msg)
    if set(left) != set(right):
        # Silently intersecting would compare different question sets.
        missing = sorted(set(left) ^ set(right))
        msg = (
            f"the two runs cover different queries ({len(missing)} differ, "
            f"e.g. {missing[:3]}); score both on the same query set first"
        )
        raise NotPairedError(msg)
    return (
        np.array([left[query_id] for query_id in shared], dtype=np.float64),
        np.array([right[query_id] for query_id in shared], dtype=np.float64),
        len(shared),
    )


def compare_measure(
    left: Mapping[QueryId, float],
    right: Mapping[QueryId, float],
    *,
    measure: str,
    seed: int = DEFAULT_SEED,
    resamples: int = DEFAULT_RESAMPLES,
    confidence: float = DEFAULT_CONFIDENCE,
    alpha: float = DEFAULT_ALPHA,
) -> Significance:
    """Confidence interval and p-value for one measure's paired difference."""
    left_values, right_values, n = _paired(left, right)
    differences = right_values - left_values
    observed = float(differences.mean())

    generator = np.random.default_rng(seed)

    # Bootstrap: resample queries with replacement, take the mean difference.
    indices = generator.integers(0, n, size=(resamples, n))
    bootstrap_means = differences[indices].mean(axis=1)
    tail = (1.0 - confidence) / 2.0
    ci_low, ci_high = np.quantile(bootstrap_means, [tail, 1.0 - tail])

    # Randomisation: under the null, the sign of each paired difference is
    # arbitrary, so flip each one independently and see how often the mean
    # difference gets at least as extreme as what we observed.
    signs = generator.choice(np.array([-1.0, 1.0]), size=(resamples, n))
    flipped_means = (differences * signs).mean(axis=1)
    extreme = int(np.sum(np.abs(flipped_means) >= abs(observed)))
    # +1 on both sides: an unobserved event is not evidence of impossibility,
    # and it keeps p strictly positive.
    p_value = (extreme + 1) / (resamples + 1)

    return Significance(
        measure=measure,
        n=n,
        left_mean=float(left_values.mean()),
        right_mean=float(right_values.mean()),
        delta=observed,
        ci_low=float(ci_low),
        ci_high=float(ci_high),
        p_value=float(p_value),
        resamples=resamples,
        seed=seed,
        alpha=alpha,
        confidence=confidence,
    )


def compare_measures(
    left: Mapping[QueryId, Mapping[str, float]],
    right: Mapping[QueryId, Mapping[str, float]],
    *,
    measures: tuple[str, ...],
    seed: int = DEFAULT_SEED,
    resamples: int = DEFAULT_RESAMPLES,
) -> dict[str, Significance]:
    """Run the comparison for every measure.

    No multiple-comparison correction is applied, because the measures here are
    alternative views of one ranking rather than independent hypotheses.
    Reporting several of them is not a family of tests; slicing one measure
    across many failure sets would be, and that is where a correction will be
    needed.
    """
    return {
        measure: compare_measure(
            {query_id: values[measure] for query_id, values in left.items()},
            {query_id: values[measure] for query_id, values in right.items()},
            measure=measure,
            seed=seed,
            resamples=resamples,
        )
        for measure in measures
    }
