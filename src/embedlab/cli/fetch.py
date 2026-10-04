"""Export a public IR benchmark into this project's JSONL layout.

Writing the data out instead of adapting `ir_datasets` at run time buys two
things: the engine keeps a single loading path (the same one the fixture uses,
so it is exercised constantly), and the corpus is sitting on disk as plain text
that can be read and grepped when a failure looks suspicious.

One substantive choice is recorded here rather than buried: BEIR evaluates with
the document title concatenated to its body, and every published number assumes
that. Doing otherwise would make our scores incomparable to the literature for
a reason nobody would think to look for, so it is done explicitly and named in
`--no-title` for anyone who wants to measure its effect.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterator

DEFAULT_DATASET = "beir/scifact/test"


class FetchError(RuntimeError):
    pass


def _records(dataset_id: str, *, with_title: bool) -> Iterator[tuple[str, dict]]:
    try:
        import ir_datasets
    except ImportError as error:
        msg = "fetching needs the 'datasets' extra: uv sync --extra datasets"
        raise FetchError(msg) from error

    dataset = ir_datasets.load(dataset_id)

    for doc in dataset.docs_iter():
        title = getattr(doc, "title", "") or ""
        body = getattr(doc, "text", "") or ""
        text = f"{title} {body}".strip() if with_title and title else body.strip()
        if not text:
            continue  # a document with no text cannot be retrieved or judged
        yield "corpus", {"doc_id": doc.doc_id, "text": text}

    for query in dataset.queries_iter():
        text = (getattr(query, "text", "") or "").strip()
        if not text:
            continue
        yield "queries", {"query_id": query.query_id, "text": text}

    for qrel in dataset.qrels_iter():
        yield (
            "qrels",
            {
                "query_id": qrel.query_id,
                "doc_id": qrel.doc_id,
                "relevance": int(qrel.relevance),
            },
        )


def fetch(dataset_id: str, destination: Path, *, with_title: bool = True) -> dict[str, int]:
    """Write corpus/queries/qrels JSONL into `destination`. Returns row counts."""
    destination.mkdir(parents=True, exist_ok=True)
    paths = {name: destination / f"{name}.jsonl" for name in ("corpus", "queries", "qrels")}
    counts = dict.fromkeys(paths, 0)

    handles = {name: path.open("w", encoding="utf-8") for name, path in paths.items()}
    try:
        for name, record in _records(dataset_id, with_title=with_title):
            handles[name].write(json.dumps(record, ensure_ascii=False) + "\n")
            counts[name] += 1
    finally:
        for handle in handles.values():
            handle.close()

    (destination / "source.json").write_text(
        json.dumps(
            {"dataset": dataset_id, "title_concatenated": with_title, "counts": counts}, indent=2
        ),
        encoding="utf-8",
    )
    return counts


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", nargs="?", default=DEFAULT_DATASET)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument(
        "--no-title",
        action="store_true",
        help="do not prepend the document title (diverges from published numbers)",
    )
    arguments = parser.parse_args(sys.argv[1:] if argv is None else argv)

    destination = arguments.out or Path("datasets") / arguments.dataset.replace("/", "-")
    counts = fetch(arguments.dataset, destination, with_title=not arguments.no_title)

    print(f"{arguments.dataset} -> {destination}")
    for name, count in counts.items():
        print(f"  {name:<8} {count}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
