"""Drawing failures for a person, and scoring the rules against their answers."""

import pytest

from embedlab.domain.ids import QueryId
from embedlab.stages.labelling import (
    LABELLABLE,
    score_labels,
    validate_label,
)

pytest.importorskip("bm25s", reason="needs the 'lexical' extra")
pytest.importorskip("ir_measures", reason="needs the 'measures' extra")

from embedlab.adapters.lexical import BM25Retriever
from embedlab.stages.diagnose import diagnose
from embedlab.stages.labelling import sample_failures


@pytest.fixture
def diagnosed(mini):
    retriever = BM25Retriever()
    retriever.index(mini.corpus)
    run = retriever.search(mini.queries, k=10)
    return diagnose(run, mini.qrels, mini.corpus, mini.queries), run


def draw(mini, diagnosed, **kwargs):
    diagnosis, run = diagnosed
    return sample_failures(diagnosis, run, mini.queries, mini.corpus, mini.qrels, **kwargs)


def test_only_failures_are_drawn(mini, diagnosed):
    diagnosis, _ = diagnosed
    cases = draw(mini, diagnosed, size=50)
    assert {case.query_id for case in cases} == set(diagnosis.failing())


def test_the_same_seed_draws_the_same_cases(mini, diagnosed):
    """Which failures a person spent an hour on is part of the result."""
    first = draw(mini, diagnosed, size=2, seed=7)
    second = draw(mini, diagnosed, size=2, seed=7)
    assert [c.query_id for c in first] == [c.query_id for c in second]


def test_a_different_seed_can_draw_different_cases(mini, diagnosed):
    drawn = {
        tuple(c.query_id for c in draw(mini, diagnosed, size=1, seed=seed)) for seed in range(8)
    }
    assert len(drawn) > 1


def test_asking_for_more_than_exists_returns_everything(mini, diagnosed):
    diagnosis, _ = diagnosed
    assert len(draw(mini, diagnosed, size=999)) == len(diagnosis.failing())


def test_a_case_carries_what_a_person_needs_and_nothing_else(mini, diagnosed):
    """The rule's own guess is absent on purpose.

    Shown it, a labeller agrees with it, and the agreement measured afterwards
    is the rules marking their own homework.
    """
    case = draw(mini, diagnosed, size=1)[0]

    assert case.query
    assert case.gold_text
    assert case.retrieved
    assert case.symptom in {"rank", "miss"}

    fields = set(vars(case)) if hasattr(case, "__dict__") else set(type(case).__slots__)
    assert "hypothesis" not in fields
    assert not any("cause" in name or "kind" in name for name in fields)


def test_a_missing_gold_still_gets_a_document_to_judge(mini, diagnosed):
    """A query whose gold was never retrieved still needs the gold shown."""
    for case in draw(mini, diagnosed, size=50):
        if case.symptom == "miss":
            assert case.gold_text
            assert case.gold_rank is None or case.gold_rank >= 1
            return
    pytest.skip("no miss in this fixture")


def test_the_relevant_document_is_marked_among_the_retrieved(mini, diagnosed):
    for case in draw(mini, diagnosed, size=50):
        if case.gold_rank is not None and case.gold_rank <= len(case.retrieved):
            assert any(relevant for _, _, _, relevant in case.retrieved)
            return
    pytest.skip("no gold inside the shown window")


# --- scoring ---------------------------------------------------------------

SYMPTOMS = {QueryId("a"): "rank", QueryId("b"): "miss", QueryId("c"): "ok"}


def test_agreement_counts_a_cause_the_rules_proposed():
    agreement = score_labels(
        SYMPTOMS,
        {QueryId("a"): frozenset({"lexical_mismatch"})},
        {QueryId("a"): "lexical_mismatch"},
    )
    assert agreement.compared == 1
    assert agreement.agreed == 1
    assert agreement.accuracy == 1.0


def test_several_proposals_count_as_agreement_if_one_matches():
    """Generous to the rules by design: ranking causes is a harder ask."""
    agreement = score_labels(
        SYMPTOMS,
        {QueryId("a"): frozenset({"hard_negative", "lexical_mismatch"})},
        {QueryId("a"): "lexical_mismatch"},
    )
    assert agreement.agreed == 1
    assert agreement.precision("hard_negative") == 0.0
    assert agreement.precision("lexical_mismatch") == 1.0


def test_a_cause_no_rule_proposed_is_reported_as_missed():
    agreement = score_labels(
        SYMPTOMS,
        {QueryId("a"): frozenset({"unexplained"})},
        {QueryId("a"): "chunking"},
    )
    assert agreement.agreed == 0
    assert agreement.missed == {"chunking": 1}


def test_a_successful_query_is_not_scored():
    """Labelling a query that did not fail says nothing about the rules."""
    agreement = score_labels(SYMPTOMS, {}, {QueryId("c"): "chunking"})
    assert agreement.compared == 0
    assert agreement.accuracy == 0.0


def test_an_unknown_query_is_not_scored():
    agreement = score_labels(SYMPTOMS, {}, {QueryId("ghost"): "chunking"})
    assert agreement.compared == 0


@pytest.mark.parametrize("written", ["Hard Negative", "hard-negative", " HARD_NEGATIVE "])
def test_labels_are_accepted_however_they_were_typed(written):
    assert validate_label(written) == "hard_negative"


def test_an_unknown_label_names_the_alternatives():
    with pytest.raises(ValueError, match="unknown cause") as caught:
        validate_label("its just bad")
    for cause in LABELLABLE:
        assert cause in str(caught.value)


def test_unexplained_is_offered_to_the_labeller():
    """A person who cannot name a cause must be able to say so, or the
    exercise forces agreement with whichever category is nearest."""
    assert "unexplained" in LABELLABLE
    assert validate_label("unexplained") == "unexplained"
