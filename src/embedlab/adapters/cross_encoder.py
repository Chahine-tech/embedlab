"""Reranking with a cross encoder.

The standard second stage: where a bi-encoder scores a query and a document
apart and compares the results, a cross encoder reads them together and is
much better at it, which is affordable only over a handful of candidates.

The descriptor carries the same things the dense retriever's does, and for the
same reason: the weights behind a name can be republished, and dtype and device
change the numbers. A reranker that reordered differently between two runs
without the fingerprint noticing would be a fake diff one stage further down.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from embedlab.adapters.hub import resolve_revision

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from embedlab.domain.ids import UnitId

IMPL_VERSION = 1

_DTYPES = ("float16", "float32", "float64")


class CrossEncoderUnavailableError(RuntimeError):
    """The `local` extra is not installed."""


class CrossEncoderReranker:
    """Score every (query, candidate) pair with a cross encoder."""

    def __init__(
        self,
        *,
        model_id: str,
        name: str | None = None,
        revision: str | None = None,
        device: str = "cpu",
        dtype: str = "float32",
        max_length: int | None = None,
        batch_size: int = 32,
    ) -> None:
        if dtype not in _DTYPES:
            msg = f"unknown dtype {dtype!r}; expected one of {', '.join(_DTYPES)}"
            raise ValueError(msg)

        self._model_id = model_id
        self._name = name if name is not None else model_id.split("/")[-1]
        self._revision = revision
        self._device = device
        self._dtype = dtype
        self._max_length = max_length
        self._batch_size = batch_size
        self._model: Any | None = None
        self._resolved_revision: str | None = revision

    @property
    def name(self) -> str:
        return self._name

    @property
    def descriptor(self) -> Mapping[str, object]:
        try:
            import sentence_transformers  # pyright: ignore[reportMissingImports]

            library = str(sentence_transformers.__version__)
        except ImportError:
            library = "absent"

        return {
            "kind": "cross_encoder",
            "impl": "sentence-transformers",
            "impl_version": IMPL_VERSION,
            "library_version": library,
            "model_id": self._model_id,
            "revision": self._resolved_revision,
            "device": self._device,
            "dtype": self._dtype,
            "max_length": self._max_length,
            "batch_size": self._batch_size,
        }

    def _load(self) -> Any:
        if self._model is not None:
            return self._model
        try:
            # Resolved only with the 'local' extra; the type checker runs without
            # it, so the import is pinned as optional here rather than by
            # weakening the rule for the whole project.
            import torch  # pyright: ignore[reportMissingImports]
            from sentence_transformers import (  # pyright: ignore[reportMissingImports]
                CrossEncoder,
            )
        except ImportError as error:
            msg = "cross-encoder reranking needs the 'local' extra: uv sync --extra local"
            raise CrossEncoderUnavailableError(msg) from error

        torch_dtype = {
            "float16": torch.float16,
            "float32": torch.float32,
            "float64": torch.float64,
        }[self._dtype]
        model = CrossEncoder(
            self._model_id,
            revision=self._revision,
            device=self._device,
            max_length=self._max_length,
            model_kwargs={"torch_dtype": torch_dtype},
        )
        self._resolved_revision = self._revision or resolve_revision(self._model_id)
        self._model = model
        return model

    def rerank(self, query: str, candidates: Sequence[tuple[UnitId, str]]) -> dict[UnitId, float]:
        if not candidates:
            return {}
        model = self._load()
        pairs = [(query, text) for _, text in candidates]
        scores = model.predict(pairs, batch_size=self._batch_size, show_progress_bar=False)
        return {unit: float(score) for (unit, _), score in zip(candidates, scores, strict=True)}
