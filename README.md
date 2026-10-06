# Embedding Lab

> A retrieval diff you can trust.

Run a dataset through competing retrieval configurations and compare them query
by query, with every reported difference qualified by three things: whether it
is statistically real, whether it is a measurement artifact, and whether the two
runs are comparable at all given what changed between them.

Status: **spike**. Nothing here is stable yet.

Metrics, significance tests and failure taxonomies all exist elsewhere and are
not what this is for: the engine consumes the first two and keeps the third
cheap. What it adds is the layer that decides how much of a diff to believe.

## Architecture in one paragraph

A build system, not a service. Each stage is a pure function from the
fingerprint of its inputs to an immutable artifact on disk
(`retrieve -> evaluate -> diagnose -> diff`; chunking and reranking are not
built yet, and embedding lives inside the dense adapter rather than as a stage
of its own). Caching is content-addressed, so if two runs share an upstream
fingerprint we
*know* the difference between them cannot come from that stage. The diff
semantics fall out of the DAG instead of being guessed after the fact.

Functional core, imperative shell, one dependency rule
(`embedlab.domain` depends on nothing), zero ceremony layers.

## Development

```bash
uv sync                  # core only: instant, no torch
uv run pytest
uv run ruff check . && uv run ruff format --check .
uv run basedpyright                            # standard, src + tests
uv run basedpyright -p typecheck-strict.json   # strict, the pure core
uv run lint-imports                            # enforces the dependency rule
```

Two type-checking passes rather than one: the adapters wrap libraries that ship
no type information, so strict mode there reports the ecosystem back at us,
while `domain`, `cache` and `artifacts` have no such excuse.

Heavy backends are opt-in extras: `--extra lexical`, `--extra measures`,
`--extra local`, `--extra datasets`.

## What it writes

Finished work is published as plain files so nothing downstream has to import
this engine. Tabular parts are Parquet, the rest is JSON.

```
runs/run/<run-id>/           manifest.json  run.parquet
                             metrics.parquet  evidence.parquet  hypotheses.parquet
runs/comparison/<id>/        manifest.json  deltas.parquet
runs/cache/retrieve/<key>/   run.parquet  stage.json
```

A run owns its evaluation and its diagnosis; a comparison references two runs
rather than copying them, so re-comparing never recomputes a diagnosis and a
copy can never drift from its original.

Schemas are long wherever a wide one would depend on configuration: one row per
(query, measure), one row per (query, cause). Adding a measure, or a query
carrying a second cause, needs no schema change.

## Try it

```bash
uv sync --extra lexical --extra measures
uv run python -m embedlab.cli.demo
```

Reports on a 20-document fixture, which is deliberately too small to support a
claim: every measure comes back `undecided, 7 queries is too few to tell`. For
real numbers, fetch a benchmark first:

```bash
uv sync --extra datasets
uv run python -m embedlab.cli.fetch beir/scifact/test
uv run python -m embedlab.cli.demo datasets/beir-scifact-test
```
