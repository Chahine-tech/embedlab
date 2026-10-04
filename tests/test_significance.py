"""Whether a difference may be called a difference."""

import pytest

from embedlab.domain.ids import QueryId
from embedlab.stages.significance import (
    ADVISORY_MINIMUM_QUERIES,
    NotPairedError,
    compare_measure,
    compare_measures,
)


def values(numbers):
    return {QueryId(f"q{index}"): float(value) for index, value in enumerate(numbers)}


def compare(left, right, **kwargs):
    kwargs.setdefault("resamples", 2000)
    return compare_measure(values(left), values(right), measure="nDCG@10", **kwargs)


def test_identical_runs_show_no_difference():
    result = compare([0.5] * 80, [0.5] * 80)
    assert result.delta == 0.0
    assert result.ci_low == 0.0
    assert result.ci_high == 0.0
    assert result.is_real is False
    assert result.describe().endswith("no change")


def test_a_large_consistent_improvement_is_real():
    result = compare([0.4] * 80, [0.6] * 80)
    assert result.delta == pytest.approx(0.2)
    assert result.p_value < 0.05
    assert result.interval_excludes_zero
    assert result.is_real is True
    assert result.describe().endswith("real")


def test_pure_noise_is_not_real():
    """Alternating gains and losses of equal size: the mean is zero."""
    pattern = [0.5, 0.1] * 40
    other = [0.1, 0.5] * 40
    result = compare(pattern, other)
    assert result.p_value > 0.05
    assert result.ci_low < 0.0 < result.ci_high
    assert result.is_real is False


def test_a_small_sample_is_undecided_whatever_the_p_value():
    """A big effect on nine queries still may not be reported as a difference."""
    result = compare([0.0] * 9, [1.0] * 9)
    assert result.n < ADVISORY_MINIMUM_QUERIES
    assert result.underpowered is True
    assert result.is_real is False
    assert "too few to tell" in result.describe()


def test_a_single_moved_query_can_never_be_significant():
    """Observed on the fixture, and mathematically necessary.

    With exactly one non-zero paired difference, flipping signs leaves the mean
    equally extreme every time, so the randomisation test cannot reject.
    """
    left = [0.5] * 60
    right = list(left)
    right[0] = 1.0
    result = compare(left, right)
    assert result.p_value == 1.0
    assert result.is_real is False


def test_the_interval_brackets_the_observed_delta():
    result = compare([0.3, 0.4, 0.5] * 30, [0.5, 0.4, 0.7] * 30)
    assert result.ci_low <= result.delta <= result.ci_high


def test_reversing_the_comparison_flips_the_sign_but_not_the_verdict():
    forward = compare([0.4] * 80, [0.6] * 80)
    backward = compare([0.6] * 80, [0.4] * 80)
    assert backward.delta == pytest.approx(-forward.delta)
    assert backward.is_real is forward.is_real
    assert "worse" in backward.describe()


def test_the_same_seed_gives_the_same_numbers():
    """A p-value that drifted between runs would be its own phantom difference."""
    first = compare([0.3, 0.6] * 40, [0.5, 0.4] * 40, seed=7)
    second = compare([0.3, 0.6] * 40, [0.5, 0.4] * 40, seed=7)
    assert (first.p_value, first.ci_low, first.ci_high) == (
        second.p_value,
        second.ci_low,
        second.ci_high,
    )


def test_a_different_seed_moves_the_estimate_only_slightly():
    first = compare([0.3, 0.6] * 40, [0.5, 0.4] * 40, seed=1)
    second = compare([0.3, 0.6] * 40, [0.5, 0.4] * 40, seed=2)
    assert first.delta == second.delta  # the point estimate is not resampled
    assert abs(first.ci_low - second.ci_low) < 0.05


def test_the_point_estimate_is_the_mean_paired_difference():
    result = compare([0.0, 0.0], [0.5, 0.1])
    assert result.delta == pytest.approx(0.3)
    assert result.left_mean == pytest.approx(0.0)
    assert result.right_mean == pytest.approx(0.3)


def test_p_is_never_zero():
    """An unobserved event is not evidence of impossibility."""
    result = compare([0.0] * 200, [1.0] * 200, resamples=500)
    assert result.p_value > 0.0


def test_different_query_sets_are_refused():
    left = {QueryId("a"): 0.5, QueryId("b"): 0.5}
    right = {QueryId("a"): 0.5, QueryId("c"): 0.5}
    with pytest.raises(NotPairedError, match="different queries"):
        compare_measure(left, right, measure="RR")


def test_disjoint_query_sets_are_refused():
    left = {QueryId("a"): 0.5}
    right = {QueryId("z"): 0.5}
    with pytest.raises(NotPairedError, match="share no queries"):
        compare_measure(left, right, measure="RR")


def test_every_requested_measure_is_compared():
    left = {QueryId("q1"): {"RR": 0.5, "nDCG@10": 0.4}}
    right = {QueryId("q1"): {"RR": 1.0, "nDCG@10": 0.9}}
    results = compare_measures(left, right, measures=("RR", "nDCG@10"), resamples=200)
    assert set(results) == {"RR", "nDCG@10"}
    assert results["RR"].measure == "RR"


def test_the_seed_and_resample_count_are_recorded():
    result = compare([0.1] * 60, [0.2] * 60, seed=99, resamples=1500)
    assert result.seed == 99
    assert result.resamples == 1500
