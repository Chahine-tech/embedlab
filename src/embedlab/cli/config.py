"""The experiment file, and the rules that keep a typo from costing a day.

Every field is validated and unknown keys are refused. That strictness is the
point: `stemer: snowball-en` silently ignored would hand back an unstemmed run
under a stemmed name, and the whole comparison would be against the wrong
thing. A config that fails loudly costs a second; one that fails quietly costs
however long it takes to disbelieve a result.

Systems are a discriminated union on `kind`, so an error names the offending
system and the field rather than reporting that something somewhere is wrong.
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from embedlab.stages.evaluate import DEFAULT_MEASURES


class ConfigError(ValueError):
    """The experiment file cannot be used, with the reason."""


class WholeChunking(BaseModel):
    model_config = ConfigDict(extra="forbid")
    kind: Literal["whole"] = "whole"


class FixedWordsChunking(BaseModel):
    model_config = ConfigDict(extra="forbid")
    kind: Literal["fixed_words"]
    size: int = Field(default=180, ge=1)
    overlap: int = Field(default=40, ge=0)


Chunking = Annotated[WholeChunking | FixedWordsChunking, Field(discriminator="kind")]


class CoverageReranking(BaseModel):
    model_config = ConfigDict(extra="forbid")
    kind: Literal["coverage"]
    lower: bool = True
    length_penalty: float = 0.0


class CrossEncoderReranking(BaseModel):
    model_config = ConfigDict(extra="forbid")
    kind: Literal["cross_encoder"]
    model_id: str
    revision: str | None = None
    device: str = "cpu"
    dtype: str = "float32"
    max_length: int | None = None
    batch_size: int = 32


Reranking = Annotated[CoverageReranking | CrossEncoderReranking, Field(discriminator="kind")]


class _System(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str
    chunking: Chunking | None = None
    candidates: int = Field(default=50, ge=1)
    """How deep to retrieve before reranking.

    Ignored without a reranker. A reranker handed only the final k can shuffle
    what retrieval already chose and never promote what it buried, so measuring
    one at k would measure something nobody deploys.
    """

    reranker: Reranking | None = None
    """What reorders this system's candidates, if anything.

    A reranker only sees what retrieval found, so declaring one changes the
    order and can never change the recall.
    """
    """How this system cuts the corpus, if at all.

    Per system rather than per experiment, because comparing two cuts of the
    same corpus with the same model is the question chunking raises.
    """


class BM25System(_System):
    kind: Literal["bm25"]
    k1: float = 1.5
    b: float = 0.75
    method: str = "lucene"
    stopwords: str | None = "en"
    stemmer: str | None = None
    lower: bool = True
    dtype: str = "float32"


class TfidfSystem(_System):
    kind: Literal["tfidf"]
    ngram_range: tuple[int, int] = (3, 5)
    lower: bool = True
    sublinear_tf: bool = True
    smooth_idf: bool = True
    dtype: str = "float64"


class DenseSystem(_System):
    kind: Literal["dense"]
    model_id: str
    revision: str | None = None
    device: str = "cpu"
    dtype: str = "float32"
    normalize: bool = True
    query_prompt: str = ""
    document_prompt: str = ""
    max_seq_length: int | None = None
    batch_size: int = 32


System = Annotated[BM25System | TfidfSystem | DenseSystem, Field(discriminator="kind")]


class Experiment(BaseModel):
    """One file, one question: these systems, this data, this k."""

    model_config = ConfigDict(extra="forbid")

    name: str
    dataset: Path
    systems: list[System] = Field(min_length=1)
    k: int = Field(default=10, ge=1)
    measures: tuple[str, ...] = DEFAULT_MEASURES
    baseline: str | None = None
    """Name of the system every other one is compared against.

    Absent, the first system plays that part. Naming it explicitly is better:
    which run sits on the left of a diff decides what "improved" means.
    """

    def resolved_baseline(self) -> str:
        return self.baseline if self.baseline is not None else self.systems[0].name

    def others(self) -> list[System]:
        baseline = self.resolved_baseline()
        return [system for system in self.systems if system.name != baseline]


def load_experiment(path: Path) -> Experiment:
    """Read and validate an experiment file."""
    if not path.exists():
        msg = f"no experiment file at {path}"
        raise ConfigError(msg)

    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as error:
        msg = f"{path}: not valid YAML ({error})"
        raise ConfigError(msg) from error

    if not isinstance(raw, dict):
        found = type(raw).__name__
        msg = f"{path}: expected a mapping at the top level, found {found}"
        raise ConfigError(msg)

    try:
        experiment = Experiment.model_validate(raw)
    except ValidationError as error:
        lines = [f"{path}: {error.error_count()} problem(s)"]
        for problem in error.errors():
            where = ".".join(str(part) for part in problem["loc"]) or "(top level)"
            lines.append(f"  {where}: {problem['msg']}")
        raise ConfigError("\n".join(lines)) from error

    names = [system.name for system in experiment.systems]
    duplicates = sorted({name for name in names if names.count(name) > 1})
    if duplicates:
        msg = f"{path}: duplicate system names {duplicates}; a run is identified by its name"
        raise ConfigError(msg)

    if experiment.baseline is not None and experiment.baseline not in names:
        msg = f"{path}: baseline {experiment.baseline!r} is not one of the systems {names}"
        raise ConfigError(msg)

    return experiment
