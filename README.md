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
(`chunk -> retrieve -> rerank -> evaluate -> diagnose -> diff`; embedding lives
inside the dense adapter rather than as a stage of its own). Caching is
content-addressed, so if two runs share an upstream fingerprint we
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

## The viewer

`website/` is a small Vite and React app that reads the published artifacts and
nothing else: it fetches the Parquet over HTTP and joins them in memory, so it
exercises exactly the contract any other reader would.

```bash
cd website
pnpm install
pnpm dev
```

It opens on the first published comparison, or on the one named in the URL
hash. Three dependencies at runtime, no component library and no animation
library: motion is CSS transitions where a property changes and a view
transition where the detail panel is replaced wholesale, skipped while a held
arrow key is moving faster than an animation can finish.

## Running an experiment

An experiment is a file. Systems are declared, not coded, so adding a model is
an entry rather than a branch.

```yaml
# experiments/scifact.yaml
name: lexical against dense on SciFact
dataset: datasets/beir-scifact-test
k: 10
measures: [nDCG@10, R@10, RR]

baseline: bm25          # what every other system is compared against

systems:
  - name: bm25
    kind: bm25
  - name: bm25-stemmed
    kind: bm25
    stemmer: snowball-en
  - name: minilm
    kind: dense
    model_id: sentence-transformers/all-MiniLM-L6-v2
    device: cpu
    dtype: float32
```

```bash
uv run embedlab run experiments/scifact.yaml   # execute, publish, compare
uv run embedlab list                           # what the workspace holds
uv run embedlab show bm25-197cb88c             # one run's health
```

Unknown keys are refused rather than ignored. `stemer: snowball-en` silently
dropped would hand back an unstemmed run under a stemmed name, and the
comparison would be against the wrong thing with nothing to show for it:

```
error: experiments/scifact.yaml: 1 problem(s)
  systems.1.bm25.stemer: Extra inputs are not permitted
```

## Chunking

Labels are per document; chunking makes retrieval per chunk. The two are not
interchangeable, so a chunked run is folded back before anything is scored:
each document keeps the score of its best chunk.

That fold is a choice, not a convention. A document whose evidence is spread
thinly across many chunks scores as its single best chunk, which is the failure
mode chunking is suspected of causing in the first place.

```yaml
systems:
  - name: whole
    kind: bm25
    chunking: {kind: whole}
  - name: words-120
    kind: bm25
    chunking: {kind: fixed_words, size: 120, overlap: 30}
```

`experiments/chunking.yaml` runs that question on SciFact. The answer there is
worth knowing before trusting any single number: at 120-word chunks only
nDCG@10 detects the damage (p=0.032), while recall and MRR stay inside the
noise. At 60 words all three measures turn real and the damage doubles.

## Reranking

A reranker sees candidates and never the corpus, so it redistributes recall and
can never increase it. The engine records the ranking it was handed as well as
the one it produced, because the final order alone cannot say whether a gold
document at rank four arrived there or was pushed there.

```yaml
systems:
  - name: bm25
    kind: bm25
  - name: bm25-coverage
    kind: bm25
    reranker: {kind: coverage}
```

`RERANK_REGRESSION` is the one cause in the taxonomy that no threshold decides:
the gold was at a known rank and the reranker put it lower. On
`experiments/rerank.yaml` the coverage reranker fires it on 85 of 300 queries,
and `R@10` comes back exactly unchanged, which is the ceiling above showing up
in the numbers rather than only in the prose.

## Measuring the rules against a person

The diagnosis rules decide what a failure is called, and dataset-relative
thresholds fixed their scale without making their labels true. There is a path
from a published run to a measured rule:

```bash
uv run embedlab label bm25-197cb88c --sample 50 --out labels.csv
# fill the empty `cause` column, then
uv run embedlab score bm25-197cb88c labels.csv
```

The sheet carries the query, the gold document, what was retrieved and the
deterministic evidence, and **never the rule's own guess**: shown it, a
labeller agrees with it, and the agreement measured afterwards is the rules
marking their own homework. `unexplained` is an allowed answer, because a
person who cannot name a cause must be able to say so.

The draw is seeded. Which failures someone spent an hour on is part of the
result, so a rerun produces the same ones.

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
