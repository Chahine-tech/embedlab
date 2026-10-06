"""Cutting documents into retrievable units.

Chunking is where the two identity types stop being pedantry. Labels are per
document, retrieval becomes per chunk, and the two are not interchangeable: a
chunk that ranks first is not a document that ranks first until something says
which document it came from. `DocId` and `ChunkId` have been distinct since the
first commit precisely so this cannot be confused silently.

Folding chunk results back to documents keeps the best chunk per document and
discards the rest. That is the usual choice and it is a choice: a document
whose evidence is spread thinly over many chunks scores as its single best
chunk, which is exactly the failure mode chunking is suspected of causing.

Measuring that dilution needs the spread of a gold document's chunks recorded
on the evidence, and no rule reads it yet. It will arrive with the rule, rather
than as a column nobody consumes.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import TYPE_CHECKING

from embedlab.domain.ids import ChunkId, DocId, QueryId, UnitId
from embedlab.domain.retrieval import Ranked, order_deterministically

if TYPE_CHECKING:
    from collections.abc import Mapping

    from embedlab.domain.retrieval import Run

IMPL_VERSION = 1

_WORD = re.compile(r"\S+")


@dataclass(frozen=True, slots=True)
class Chunks:
    """Retrievable units, and the document each came from."""

    texts: dict[ChunkId, str]
    source: dict[ChunkId, DocId]
    strategy: Mapping[str, object]
    """What produced them. Belongs in a cache key: a different cut is a
    different corpus, not the same corpus seen differently."""

    def of(self, doc_id: DocId) -> list[ChunkId]:
        return [chunk for chunk, source in self.source.items() if source == doc_id]

    @property
    def per_document(self) -> dict[DocId, int]:
        counts: dict[DocId, int] = {}
        for doc_id in self.source.values():
            counts[doc_id] = counts.get(doc_id, 0) + 1
        return counts


def whole_documents(corpus: Mapping[DocId, str]) -> Chunks:
    """One chunk per document: the identity cut.

    Present so that not chunking is a strategy rather than a special case, and
    every run downstream has the same shape.
    """
    return Chunks(
        texts={ChunkId(str(doc_id)): text for doc_id, text in corpus.items()},
        source={ChunkId(str(doc_id)): doc_id for doc_id in corpus},
        strategy={"kind": "whole"},
    )


def fixed_words(
    corpus: Mapping[DocId, str],
    *,
    size: int = 180,
    overlap: int = 40,
) -> Chunks:
    """Fixed-width windows measured in whitespace-separated words.

    Words rather than tokens: a tokeniser would tie the cut to one model's
    vocabulary, and the point of comparing models is that the corpus they see
    is the same. Overlap exists because a window boundary that lands mid-idea
    is the mechanism chunking failures are blamed on.
    """
    if size < 1:
        msg = f"chunk size must be at least one word, got {size}"
        raise ValueError(msg)
    if not 0 <= overlap < size:
        msg = f"overlap must be at least zero and smaller than the size, got {overlap} of {size}"
        raise ValueError(msg)

    texts: dict[ChunkId, str] = {}
    source: dict[ChunkId, DocId] = {}
    step = size - overlap

    for doc_id in sorted(corpus):
        words = _WORD.findall(corpus[doc_id])
        if not words:
            continue
        starts = range(0, max(1, len(words) - overlap), step)
        for index, start in enumerate(starts):
            window = words[start : start + size]
            if not window:
                continue
            chunk_id = ChunkId(f"{doc_id}#{index}")
            texts[chunk_id] = " ".join(window)
            source[chunk_id] = doc_id

    return Chunks(
        texts=texts,
        source=source,
        strategy={"kind": "fixed_words", "size": size, "overlap": overlap},
    )


def fold_to_documents(run: Run, chunks: Chunks, *, k: int) -> dict[QueryId, list[Ranked]]:
    """Collapse a chunk-level ranking to a document-level one.

    Each document keeps the score of its best chunk. Ties break the same way
    they do everywhere else, so folding cannot reorder two documents that the
    retriever scored identically.
    """
    folded: dict[QueryId, list[Ranked]] = {}
    for query_id, ranked in run.items():
        best: dict[UnitId, float] = {}
        for item in ranked:
            doc_id = chunks.source.get(ChunkId(item.doc_id))
            if doc_id is None:
                continue
            current = best.get(doc_id)
            if current is None or item.score > current:
                best[doc_id] = item.score
        folded[QueryId(query_id)] = order_deterministically(best, k)
    return folded
