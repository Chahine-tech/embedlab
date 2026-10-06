"""Identity types.

These are `NewType`s, not aliases: the type checker refuses to pass a `ChunkId`
where a `DocId` is expected. Once chunking exists that confusion is otherwise
guaranteed, and it is the kind of bug that produces plausible-looking wrong
numbers rather than a crash.
"""

from typing import NewType

DocId = NewType("DocId", str)
"""A document as the user supplied it."""

ChunkId = NewType("ChunkId", str)
"""A retrievable unit produced from a document. Never interchangeable with DocId."""

QueryId = NewType("QueryId", str)

ModelId = NewType("ModelId", str)
"""Registry key, e.g. `jina-v5`. Not the provider's own model string."""

Fingerprint = NewType("Fingerprint", str)
"""Content-addressed identity of a value or a stage output. See embedlab.cache."""

RunId = NewType("RunId", str)
"""Identity of one (config, engine) execution. Derived from fingerprints."""

type UnitId = DocId | ChunkId
"""Whatever a retriever was given to index.

A retriever indexes retrievable units. Without chunking those are documents;
with it they are chunks, and the two are not interchangeable. Naming the union
is what lets a retriever be honest about taking either without the two
collapsing back into one type.
"""
