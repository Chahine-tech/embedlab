"""Loading an evaluation dataset, and pinning its identity.

The dataset fingerprint is what lets the diff stage refuse to compare two runs
that were not even evaluated on the same data — the one case that must end as
INCOMPARABLE rather than merely suspect.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from embedlab.cache.fingerprint import fingerprint
from embedlab.domain.ids import DocId, Fingerprint, QueryId

if TYPE_CHECKING:
    from collections.abc import Iterator

type JsonObject = dict[str, object]
"""One decoded JSONL line. `object` rather than `Any`: every field is validated
before use, so nothing should type-check merely because it was read from disk."""


class DatasetError(ValueError):
    """The dataset on disk is not usable, with the line number that proves it."""


@dataclass(frozen=True, slots=True)
class Dataset:
    """A corpus, a set of queries, and graded labels linking them."""

    name: str
    corpus: dict[DocId, str]
    queries: dict[QueryId, str]
    qrels: dict[QueryId, dict[DocId, int]]

    @property
    def corpus_fingerprint(self) -> Fingerprint:
        """Identity of the retrievable content. Ignores queries and labels, so
        re-labelling a dataset does not invalidate cached embeddings."""
        return fingerprint({"corpus": self.corpus})

    @property
    def labels_fingerprint(self) -> Fingerprint:
        """Identity of the evaluation target: queries plus labels."""
        return fingerprint({"queries": self.queries, "qrels": self.qrels})

    @property
    def judged_queries(self) -> tuple[QueryId, ...]:
        """Queries carrying at least one positive label.

        Only these can be evaluated: a query with no known relevant document
        cannot distinguish a retrieval failure from a labelling gap.
        """
        return tuple(
            query_id
            for query_id in self.queries
            if any(grade > 0 for grade in self.qrels.get(query_id, {}).values())
        )


def _read_jsonl(path: Path) -> Iterator[tuple[int, JsonObject]]:
    if not path.exists():
        msg = f"missing dataset file: {path}"
        raise DatasetError(msg)
    with path.open(encoding="utf-8") as handle:
        for number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                msg = f"{path.name}:{number}: not valid JSON ({error.msg})"
                raise DatasetError(msg) from error
            if not isinstance(record, dict):
                msg = f"{path.name}:{number}: expected an object, found {type(record).__name__}"
                raise DatasetError(msg)
            yield number, record


def _require(record: JsonObject, field: str, *, where: str) -> str:
    value = record.get(field)
    if not isinstance(value, str) or not value:
        msg = f"{where}: field {field!r} must be a non-empty string"
        raise DatasetError(msg)
    return value


def load_dataset(directory: Path, *, name: str | None = None) -> Dataset:
    """Load `corpus.jsonl`, `queries.jsonl` and `qrels.jsonl` from a directory.

    Validation is strict and reports the offending line. A dataset that loads
    quietly but wrongly produces failure counts that look plausible, which is
    far more expensive to discover than a refusal at load time.
    """
    corpus: dict[DocId, str] = {}
    for number, record in _read_jsonl(directory / "corpus.jsonl"):
        where = f"corpus.jsonl:{number}"
        doc_id = DocId(_require(record, "doc_id", where=where))
        if doc_id in corpus:
            msg = f"{where}: duplicate doc_id {doc_id!r}"
            raise DatasetError(msg)
        corpus[doc_id] = _require(record, "text", where=where)

    queries: dict[QueryId, str] = {}
    for number, record in _read_jsonl(directory / "queries.jsonl"):
        where = f"queries.jsonl:{number}"
        query_id = QueryId(_require(record, "query_id", where=where))
        if query_id in queries:
            msg = f"{where}: duplicate query_id {query_id!r}"
            raise DatasetError(msg)
        queries[query_id] = _require(record, "text", where=where)

    qrels: dict[QueryId, dict[DocId, int]] = {}
    for number, record in _read_jsonl(directory / "qrels.jsonl"):
        where = f"qrels.jsonl:{number}"
        query_id = QueryId(_require(record, "query_id", where=where))
        doc_id = DocId(_require(record, "doc_id", where=where))
        grade = record.get("relevance", 1)
        if not isinstance(grade, int) or isinstance(grade, bool):
            msg = f"{where}: 'relevance' must be an integer, found {grade!r}"
            raise DatasetError(msg)
        if query_id not in queries:
            msg = f"{where}: label refers to unknown query_id {query_id!r}"
            raise DatasetError(msg)
        if doc_id not in corpus:
            msg = f"{where}: label refers to unknown doc_id {doc_id!r}"
            raise DatasetError(msg)
        qrels.setdefault(query_id, {})[doc_id] = grade

    if not corpus:
        msg = f"{directory}: corpus is empty"
        raise DatasetError(msg)

    return Dataset(
        name=name if name is not None else directory.name,
        corpus=corpus,
        queries=queries,
        qrels=qrels,
    )
