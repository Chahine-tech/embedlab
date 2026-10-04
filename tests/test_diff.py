"""Git diff for retrieval: per-query movements, carrying their own trust."""

import pytest

from embedlab.artifacts.comparability import Trust, Verdict
from embedlab.domain.ids import DocId, QueryId
from embedlab.domain.retrieval import Ranked
from embedlab.stages.diagnose import diagnose
from embedlab.stages.diff import compare_runs, failure_sets

CORPUS = {
    DocId("gold"): "production token rotation procedure rotate revoke",
    DocId("near"): "api token creation procedure create scope",
    DocId("far"): "invoices billing cycles metered hourly",
}
QUERIES = {QueryId("q"): "how do I rotate the production token"}
QRELS = {QueryId("q"): {DocId("gold"): 2}}
TRUSTED = Verdict(Trust.IDENTICAL)


def run_with(*pairs):
    return {
        QueryId("q"): [
            Ranked(DocId(doc_id), position, score)
            for position, (doc_id, score) in enumerate(pairs, start=1)
        ]
    }


def diff_of(left, right, **kwargs):
    return compare_runs(
        left, right, QRELS, left_name="left", right_name="right", trust=TRUSTED, **kwargs
    )


def test_promotion_is_an_improvement():
    result = diff_of(run_with(("near", 0.9), ("gold", 0.5)), run_with(("gold", 0.9)))
    assert len(result.improved) == 1
    assert result.improved[0].left_rank == 2
    assert result.improved[0].right_rank == 1
    assert result.improved[0].became_success


def test_demotion_is_a_regression():
    result = diff_of(run_with(("gold", 0.9)), run_with(("near", 0.9), ("gold", 0.5)))
    assert len(result.regressed) == 1
    assert result.regressed[0].became_failure


def test_losing_the_gold_entirely_is_a_regression():
    result = diff_of(run_with(("gold", 0.9)), run_with(("near", 0.9), ("far", 0.5)))
    assert len(result.regressed) == 1
    assert result.regressed[0].right_rank is None


def test_finding_a_previously_missing_gold_is_an_improvement():
    result = diff_of(run_with(("near", 0.9)), run_with(("gold", 0.9)))
    assert len(result.improved) == 1
    assert result.improved[0].left_rank is None


def test_both_missing_is_unchanged():
    """None vs None must not read as a change."""
    result = diff_of(run_with(("near", 0.9)), run_with(("far", 0.9)))
    assert len(result.unchanged) == 1


def test_same_rank_is_unchanged():
    result = diff_of(run_with(("gold", 0.9)), run_with(("gold", 0.4)))
    assert len(result.unchanged) == 1
    assert result.significance == {}


def test_a_tie_driven_movement_is_marked_arbitrary():
    """The product's own weakest claim, made visible instead of hidden."""
    left = run_with(("near", 0.871), ("gold", 0.871))
    right = run_with(("gold", 0.5), ("near", 0.4))
    result = diff_of(
        left,
        right,
        left_diagnosis=diagnose(left, QRELS, CORPUS, QUERIES),
        right_diagnosis=diagnose(right, QRELS, CORPUS, QUERIES),
    )
    assert len(result.improved) == 1
    assert result.improved[0].arbitrary is True
    assert result.arbitrary == result.improved
    assert "arbitrary" in result.headline


def test_a_genuine_movement_is_not_marked_arbitrary():
    left = run_with(("near", 0.9), ("gold", 0.3))
    right = run_with(("gold", 0.9), ("near", 0.3))
    result = diff_of(
        left,
        right,
        left_diagnosis=diagnose(left, QRELS, CORPUS, QUERIES),
        right_diagnosis=diagnose(right, QRELS, CORPUS, QUERIES),
    )
    assert result.improved[0].arbitrary is False
    assert result.arbitrary == ()


def test_without_diagnoses_nothing_is_claimed_to_be_arbitrary():
    result = diff_of(run_with(("near", 0.871), ("gold", 0.871)), run_with(("gold", 0.5)))
    assert result.improved[0].arbitrary is False


def test_unchanged_queries_are_never_arbitrary():
    left = run_with(("near", 0.871), ("gold", 0.871))
    result = diff_of(
        left,
        left,
        left_diagnosis=diagnose(left, QRELS, CORPUS, QUERIES),
        right_diagnosis=diagnose(left, QRELS, CORPUS, QUERIES),
    )
    assert result.arbitrary == ()


def test_an_incomparable_verdict_refuses_rather_than_rendering():
    verdict = Verdict(Trust.INCOMPARABLE, ("different labels",))
    with pytest.raises(ValueError, match="refusing to diff"):
        compare_runs(
            run_with(("gold", 0.9)),
            run_with(("gold", 0.9)),
            QRELS,
            left_name="a",
            right_name="b",
            trust=verdict,
        )


def test_a_suspect_verdict_still_renders_but_carries_its_reasons():
    verdict = Verdict(Trust.SUSPECT, ("bm25s differs",))
    result = compare_runs(
        run_with(("gold", 0.9)),
        run_with(("gold", 0.9)),
        QRELS,
        left_name="a",
        right_name="b",
        trust=verdict,
    )
    assert result.trust.trust is Trust.SUSPECT
    assert result.trust.reasons == ("bm25s differs",)


def test_unjudged_queries_take_no_part():
    qrels = {QueryId("q"): {DocId("gold"): 0}}
    result = compare_runs(
        run_with(("gold", 0.9)),
        run_with(("near", 0.9)),
        qrels,
        left_name="a",
        right_name="b",
        trust=TRUSTED,
    )
    assert result.deltas == ()


def test_a_delta_cannot_be_reported_without_its_evidence():
    """There is no bare `measure_deltas`: the interval comes with the number."""
    assert not hasattr(diff_of(run_with(("gold", 0.9)), run_with(("gold", 0.9))), "measure_deltas")


def test_significance_passes_through_and_real_differences_are_surfaced():
    from embedlab.stages.significance import compare_measure

    left = {QueryId(f"q{i}"): 0.4 for i in range(80)}
    right = {QueryId(f"q{i}"): 0.6 for i in range(80)}
    result = compare_measure(left, right, measure="nDCG@10", resamples=2000)

    diff = diff_of(
        run_with(("gold", 0.9)), run_with(("gold", 0.9)), significance={"nDCG@10": result}
    )
    assert diff.significance["nDCG@10"].delta == pytest.approx(0.2)
    assert diff.real_differences == (result,)


def test_an_undecided_measure_is_not_a_real_difference():
    from embedlab.stages.significance import compare_measure

    left = {QueryId(f"q{i}"): 0.4 for i in range(7)}
    right = {QueryId(f"q{i}"): 0.6 for i in range(7)}
    result = compare_measure(left, right, measure="nDCG@10", resamples=2000)

    diff = diff_of(
        run_with(("gold", 0.9)), run_with(("gold", 0.9)), significance={"nDCG@10": result}
    )
    assert diff.real_differences == ()


def test_failure_sets_group_by_hypothesised_cause():
    """With one query the dataset cannot be calibrated, so the honest answer is
    `unexplained`. Grouping still works, it just has nothing confident to say."""
    run = run_with(("near", 0.9), ("gold", 0.3))
    sets = failure_sets(diagnose(run, QRELS, CORPUS, QUERIES))
    assert set(sets) == {"unexplained"}
    assert sets["unexplained"] == (QueryId("q"),)


def test_failure_sets_exclude_successful_queries():
    assert failure_sets(diagnose(run_with(("gold", 0.9)), QRELS, CORPUS, QUERIES)) == {}
