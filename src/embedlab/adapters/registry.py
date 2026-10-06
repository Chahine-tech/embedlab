"""Building a retriever from a name and some keywords.

The point of the adapter boundary is that adding a system is an entry in a
file, not a branch in the code. This is the one place that maps a `kind` to a
constructor, and it imports the heavy ones lazily so a lexical-only run never
pays for torch.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from embedlab.adapters.base import Retriever

KINDS = ("bm25", "tfidf", "dense")


def build(kind: str, **options: object) -> Retriever:
    """Construct a retriever, refusing an unknown kind by name."""
    if kind == "bm25":
        from embedlab.adapters.lexical import BM25Retriever

        return BM25Retriever(**options)  # pyright: ignore[reportArgumentType]
    if kind == "tfidf":
        from embedlab.adapters.tfidf import TfidfRetriever

        return TfidfRetriever(**options)  # pyright: ignore[reportArgumentType]
    if kind == "dense":
        from embedlab.adapters.dense import DenseRetriever

        return DenseRetriever(**options)  # pyright: ignore[reportArgumentType]

    msg = f"unknown retriever kind {kind!r}; expected one of {', '.join(KINDS)}"
    raise ValueError(msg)
