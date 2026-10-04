"""BM25, with its variant and parameters stated rather than assumed.

A lexical baseline is only worth having if it is reproducible, and "BM25" alone
does not identify one. Lucene, Robertson, ATIRE and BM25+ are different
formulas, and tokenisation, stopwords and stemming move the numbers more than
the choice of formula does. Claiming that embeddings beat BM25 without stating
which BM25 is not a measurement. So every one of those choices is an explicit
argument here, and every one lands in the descriptor.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

from embedlab.domain.ids import DocId, QueryId
from embedlab.domain.retrieval import order_deterministically

if TYPE_CHECKING:
    from collections.abc import Mapping

    from embedlab.domain.retrieval import Ranked

IMPL_VERSION = 1
"""Bump when this adapter's semantics change.

Controls artifact reuse only, never whether two runs may be compared. The second
question is decided by the comparability gate, which reads the environment
recorded on each stage's provenance — so forgetting to bump this weakens caching
but cannot quietly make an untrustworthy diff look sound.
"""


class BM25Retriever:
    """BM25 over word tokens, backed by `bm25s`."""

    def __init__(
        self,
        *,
        name: str = "bm25",
        k1: float = 1.5,
        b: float = 0.75,
        method: str = "lucene",
        stopwords: str | None = "en",
        stemmer: str | None = None,
        token_pattern: str = r"(?u)\b\w\w+\b",
        lower: bool = True,
        dtype: str = "float32",
    ) -> None:
        self._name = name
        self._k1 = k1
        self._b = b
        self._method = method
        self._stopwords = stopwords
        self._stemmer = stemmer
        self._token_pattern = token_pattern
        self._lower = lower
        self._dtype = dtype

        self._doc_ids: list[DocId] = []
        self._index: Any | None = None

    @property
    def name(self) -> str:
        return self._name

    @property
    def descriptor(self) -> Mapping[str, object]:
        import bm25s

        return {
            "kind": "bm25",
            "impl": "bm25s",
            "impl_version": IMPL_VERSION,
            # The library version belongs here: a scoring bugfix upstream must
            # invalidate the cache, and no amount of local discipline would.
            "library_version": bm25s.__version__,
            "method": self._method,
            "k1": self._k1,
            "b": self._b,
            "stopwords": self._stopwords,
            "stemmer": self._stemmer,
            "token_pattern": self._token_pattern,
            "lower": self._lower,
            # float32 vs float64 can reorder near-tied documents.
            "dtype": self._dtype,
        }

    def _tokenize(self, texts: list[str]) -> Any:
        import bm25s

        return bm25s.tokenize(
            texts,
            lower=self._lower,
            token_pattern=self._token_pattern,
            # bm25s accepts None to disable stopword removal but annotates the
            # parameter as str | list[str]. Verified at runtime, and pinned by
            # test_disabling_stopwords_changes_the_scores.
            stopwords=cast("str", self._stopwords),
            stemmer=self._stemmer_fn(),
            show_progress=False,
        )

    def _stemmer_fn(self) -> Any:
        if self._stemmer is None:
            return None
        if self._stemmer == "snowball-en":
            import Stemmer

            return Stemmer.Stemmer("english")
        msg = f"unknown stemmer {self._stemmer!r}"
        raise ValueError(msg)

    def index(self, corpus: Mapping[DocId, str]) -> None:
        import bm25s

        # Sorted so the internal document order — and therefore any tie-break
        # the backend happens to apply — does not depend on dict ordering.
        self._doc_ids = sorted(corpus)
        tokens = self._tokenize([corpus[doc_id] for doc_id in self._doc_ids])
        index = bm25s.BM25(k1=self._k1, b=self._b, method=self._method, dtype=self._dtype)
        index.index(tokens, show_progress=False)
        self._index = index

    def search(self, queries: Mapping[QueryId, str], *, k: int) -> dict[QueryId, list[Ranked]]:
        if self._index is None:
            msg = "index() must be called before search()"
            raise RuntimeError(msg)
        if not queries:
            return {}

        query_ids = list(queries)
        tokens = self._tokenize([queries[query_id] for query_id in query_ids])
        # bm25s cannot return more documents than it holds.
        effective_k = min(k, len(self._doc_ids))
        indices, scores = self._index.retrieve(tokens, k=effective_k, show_progress=False)

        results: dict[QueryId, list[Ranked]] = {}
        for position, query_id in enumerate(query_ids):
            scored = {
                self._doc_ids[int(doc_index)]: float(score)
                for doc_index, score in zip(indices[position], scores[position], strict=True)
            }
            # Re-ordered through our own tie-break so a rerun cannot shuffle
            # equally-scored documents and fake a diff.
            results[QueryId(query_id)] = order_deterministically(scored, effective_k)
        return results
