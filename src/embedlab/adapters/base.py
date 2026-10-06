"""The adapter boundary: the one place ports-and-adapters earns its keep.

Structural typing on purpose: a retriever is anything with the right shape, so
adding a system is a registry entry rather than a class in an inheritance tree.
Nothing here is a base class to subclass.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from collections.abc import Mapping

    from embedlab.domain.ids import QueryId, UnitId
    from embedlab.domain.retrieval import Ranked


@runtime_checkable
class Retriever(Protocol):
    """A system that returns ranked documents for queries.

    `descriptor` must contain *everything* that can change the output: the
    implementation, its version, and every parameter: BM25's variant and
    `k1`/`b`, a dense model's revision, dtype, device, normalisation and prompt
    templates. It is fingerprinted into the cache key, so anything omitted here
    becomes an invisible source of fake diffs.
    """

    @property
    def name(self) -> str:
        """Short label used in reports and run identifiers."""
        ...

    @property
    def descriptor(self) -> Mapping[str, object]:
        """Fully determines behaviour. Fingerprinted; omissions cause fake diffs."""
        ...

    def index(self, units: Mapping[UnitId, str]) -> None:
        """Index whatever units it is given: documents, or chunks of them."""
        ...

    def search(self, queries: Mapping[QueryId, str], *, k: int) -> dict[QueryId, list[Ranked]]:
        """Ranked results per query, best first, ties broken deterministically."""
        ...
