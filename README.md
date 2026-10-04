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
(`chunk -> embed -> retrieve -> rerank -> evaluate -> diagnose -> diff`).
Caching is content-addressed, so if two runs share an upstream fingerprint we
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
`--extra local`.
