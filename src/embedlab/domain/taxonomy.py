"""The ubiquitous language of retrieval failure.

One vocabulary, used verbatim by the engine, the artifacts, the UI and the
docs. If the UI says "hard negative" while the artifact says HARD_NEG, the
product has already lost coherence.

The hard rule enforced by the type system below: *evidence* is deterministic
and reproducible, *diagnosis* is an interpretation carrying a producer and a
confidence. A judge may never write evidence.
"""

from enum import StrEnum


class FailureKind(StrEnum):
    """*Why* a query failed: hypotheses, never asserted facts.

    Purely causal. What happened is a `Symptom`, and the two were deliberately
    separated: a query has exactly one symptom and any number of causes, so a
    single flat list would make a failure histogram double-count and would let
    an observation masquerade as an explanation.
    """

    HARD_NEGATIVE = "hard_negative"
    """A semantically similar but incorrect item outranked the gold."""

    LEXICAL_MISMATCH = "lexical_mismatch"
    """Query and relevant document use different terminology."""

    CROSS_LINGUAL = "cross_lingual"
    CROSS_MODAL = "cross_modal"

    CHUNKING = "chunking"
    """Evidence exists in the corpus but chunk boundaries prevent retrieval."""

    DUPLICATE_COLLISION = "duplicate_collision"
    """Near-duplicates consume top-K slots."""

    OUTLIER = "outlier"
    RERANK_REGRESSION = "rerank_regression"
    GENERATION = "generation"

    UNEXPLAINED = "unexplained"
    """No hypothesis matched the deterministic evidence.

    An honest fallback. The earlier design fell back to RETRIEVAL_RANK, which
    restated the symptom as though it were a cause: it reads like an
    explanation while saying nothing, and it inflated the only category the
    rules could always fill.
    """

    SCORE_TIE = "score_tie"
    """The gold and the document above it scored *identically*, so their order
    comes from the tie-break rather than from the retriever.

    A measurement artifact, not a model failure. Labelling it HARD_NEGATIVE or
    DUPLICATE_COLLISION would be exactly the quiet semantic error this
    vocabulary exists to prevent, so it gets its own name.
    """


class EvidenceKind(StrEnum):
    """Deterministic, reproducible observations. No LLM may produce these."""

    GOLD_RANK = "gold_rank"
    """Best rank achieved by any relevant document (absent if none retrieved)."""

    GOLD_IN_TOPK = "gold_in_topk"
    LEXICAL_OVERLAP = "lexical_overlap"
    """Jaccard / IDF-weighted overlap between query and gold terms."""

    TOP_HIT_IS_NEAR_DUPLICATE = "top_hit_is_near_duplicate"
    GOLD_SPANS_CHUNK_BOUNDARY = "gold_spans_chunk_boundary"
    SCORE_MARGIN = "score_margin"
    """Score gap between rank 1 and the best relevant document."""

    QUERY_TOKEN_COUNT = "query_token_count"
    GOLD_LANGUAGE_DIFFERS = "gold_language_differs"


class Producer(StrEnum):
    """Who asserted a diagnosis. Recorded so the judge stays replaceable and
    measurable against human labels."""

    RULE = "rule"
    JUDGE = "judge"
    HUMAN = "human"


class Symptom(StrEnum):
    """*What* happened: deterministic, read directly off the ranking.

    Exclusive and exhaustive: every judged query has exactly one, so symptom
    counts sum to the number of failures. Cause counts (`FailureKind`) overlap
    and do not, which is why a report must say "queries in set" rather than
    "% of failures".
    """

    OK = "ok"
    """A relevant document is at rank 1."""

    RANK = "rank"
    """A relevant document was retrieved but not first."""

    MISS = "miss"
    """No relevant document entered top-K."""
