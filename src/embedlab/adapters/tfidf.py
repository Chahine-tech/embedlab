"""Character n-gram TF-IDF with cosine similarity.

Present as a second *non-neural* system, so the pipeline can be exercised and
regression-tested end to end without downloading a model. It is not a toy
stand-in: character n-grams match morphological variation that word BM25 misses
("rotate" / "rotation"), so the two disagree in interesting ways and surface
real LEXICAL_MISMATCH and HARD_NEGATIVE cases.

Implemented directly rather than via scikit-learn to keep the dependency out
and the conventions explicit: the smoothing and normalisation choices below
all move the scores and are therefore all in the descriptor.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from scipy import sparse

from embedlab.domain.ids import QueryId, UnitId
from embedlab.domain.retrieval import order_deterministically

if TYPE_CHECKING:
    from collections.abc import Mapping

    from embedlab.domain.retrieval import Ranked

IMPL_VERSION = 1


def _ngrams(text: str, low: int, high: int, *, lower: bool) -> list[str]:
    prepared = text.lower() if lower else text
    grams: list[str] = []
    for size in range(low, high + 1):
        grams.extend(prepared[start : start + size] for start in range(len(prepared) - size + 1))
    return grams


class TfidfRetriever:
    """Cosine similarity over L2-normalised TF-IDF character n-gram vectors."""

    def __init__(
        self,
        *,
        name: str = "tfidf-char",
        ngram_range: tuple[int, int] = (3, 5),
        lower: bool = True,
        sublinear_tf: bool = True,
        smooth_idf: bool = True,
        dtype: str = "float64",
    ) -> None:
        self._name = name
        self._ngram_range = ngram_range
        self._lower = lower
        self._sublinear_tf = sublinear_tf
        self._smooth_idf = smooth_idf
        self._dtype = dtype

        self._unit_ids: list[UnitId] = []
        self._vocabulary: dict[str, int] = {}
        self._idf: np.ndarray | None = None
        self._matrix: sparse.csr_matrix | None = None

    @property
    def name(self) -> str:
        return self._name

    @property
    def descriptor(self) -> Mapping[str, object]:
        return {
            "kind": "tfidf",
            "impl": "embedlab.adapters.tfidf",
            "impl_version": IMPL_VERSION,
            "analyzer": "char",
            "ngram_range": self._ngram_range,
            "lower": self._lower,
            "sublinear_tf": self._sublinear_tf,
            "smooth_idf": self._smooth_idf,
            # float32 would reorder near-ties; recorded so it cannot change
            # silently between runs.
            "dtype": self._dtype,
        }

    def _vectorize(self, texts: list[str], *, fit: bool) -> sparse.csr_matrix:
        low, high = self._ngram_range
        rows: list[int] = []
        columns: list[int] = []
        values: list[float] = []

        for row, text in enumerate(texts):
            counts: dict[int, int] = {}
            for gram in _ngrams(text, low, high, lower=self._lower):
                if fit:
                    column = self._vocabulary.setdefault(gram, len(self._vocabulary))
                else:
                    found = self._vocabulary.get(gram)
                    if found is None:
                        continue  # unseen n-gram carries no weight
                    column = found
                counts[column] = counts.get(column, 0) + 1
            for column, count in counts.items():
                rows.append(row)
                columns.append(column)
                values.append(1.0 + np.log(count) if self._sublinear_tf else float(count))

        return sparse.csr_matrix(
            (values, (rows, columns)),
            shape=(len(texts), max(len(self._vocabulary), 1)),
            dtype=self._dtype,
        )

    def _l2_normalize(self, matrix: sparse.csr_matrix) -> sparse.csr_matrix:
        # np.asarray rather than the legacy `.A`: scipy still returns np.matrix
        # from spmatrix.sum today, and this breaks silently the day it stops.
        norms = np.sqrt(np.asarray(matrix.multiply(matrix).sum(axis=1))).ravel()
        # An all-zero row (no known n-gram) stays zero rather than dividing by 0.
        norms[norms == 0.0] = 1.0
        return sparse.diags(1.0 / norms) @ matrix

    def index(self, units: Mapping[UnitId, str]) -> None:
        self._unit_ids = sorted(units)
        self._vocabulary = {}
        counts = self._vectorize([units[unit_id] for unit_id in self._unit_ids], fit=True)

        n_docs = len(self._unit_ids)
        # getnnz counts stored entries per column directly. The vectoriser only
        # stores non-zero term frequencies, so that is the document frequency,
        # and it avoids materialising a boolean matrix the size of the corpus.
        document_frequency = counts.getnnz(axis=0)
        if self._smooth_idf:
            # sklearn's convention, stated rather than inherited silently.
            self._idf = np.log((1.0 + n_docs) / (1.0 + document_frequency)) + 1.0
        else:
            self._idf = np.log(n_docs / np.maximum(document_frequency, 1)) + 1.0

        self._matrix = self._l2_normalize(counts @ sparse.diags(self._idf))

    def search(self, queries: Mapping[QueryId, str], *, k: int) -> dict[QueryId, list[Ranked]]:
        matrix, idf = self._matrix, self._idf
        if matrix is None or idf is None:
            msg = "index() must be called before search()"
            raise RuntimeError(msg)
        if not queries:
            return {}

        query_ids = list(queries)
        counts = self._vectorize([queries[query_id] for query_id in query_ids], fit=False)
        vectors = self._l2_normalize(counts @ sparse.diags(idf))
        similarities = (vectors @ matrix.T).toarray()

        effective_k = min(k, len(self._unit_ids))
        results: dict[QueryId, list[Ranked]] = {}
        for position, query_id in enumerate(query_ids):
            row = similarities[position]
            # argpartition then our deterministic tie-break; argsort alone would
            # leave equal scores in an order that depends on the backend.
            candidates = np.argpartition(-row, effective_k - 1)[:effective_k]
            scored: dict[UnitId, float] = {
                self._unit_ids[int(index)]: float(row[index]) for index in candidates
            }
            results[QueryId(query_id)] = order_deterministically(scored, effective_k)
        return results
