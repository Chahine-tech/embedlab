"""Dense retrieval over sentence-transformer embeddings.

The descriptor here is the whole point of the adapter boundary. An embedding is
a function of far more than a model name: the weights behind that name can be
republished, the same weights give different numbers at a different dtype or on
a different device, and models such as Jina and Qwen expect distinct prompt
prefixes for queries and documents. Every one of those is an explicit argument
and every one lands in the descriptor, because anything omitted becomes an
artifact silently reused across a change that moved the numbers.

Defaults are chosen for reproducibility rather than speed: float32 on CPU.
Faster settings are available and legitimate, they are simply not something to
inherit without deciding.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from embedlab.domain.ids import DocId, QueryId
from embedlab.domain.retrieval import order_deterministically

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from embedlab.domain.retrieval import Ranked

IMPL_VERSION = 1

_DTYPES = {"float32": np.float32, "float16": np.float16, "float64": np.float64}


class DenseUnavailableError(RuntimeError):
    """The `local` extra is not installed."""


class DenseRetriever:
    """Cosine (or dot-product) retrieval over a sentence-transformer model."""

    def __init__(
        self,
        *,
        model_id: str,
        name: str | None = None,
        revision: str | None = None,
        device: str = "cpu",
        dtype: str = "float32",
        normalize: bool = True,
        query_prompt: str = "",
        document_prompt: str = "",
        max_seq_length: int | None = None,
        batch_size: int = 32,
    ) -> None:
        if dtype not in _DTYPES:
            msg = f"unknown dtype {dtype!r}; expected one of {sorted(_DTYPES)}"
            raise ValueError(msg)

        self._model_id = model_id
        self._name = name if name is not None else model_id.split("/")[-1]
        self._revision = revision
        self._device = device
        self._dtype = dtype
        self._normalize = normalize
        self._query_prompt = query_prompt
        self._document_prompt = document_prompt
        self._max_seq_length = max_seq_length
        self._batch_size = batch_size

        self._model: Any | None = None
        self._resolved_revision: str | None = revision
        self._doc_ids: list[DocId] = []
        self._matrix: np.ndarray | None = None

    @property
    def name(self) -> str:
        return self._name

    @property
    def descriptor(self) -> Mapping[str, object]:
        return {
            "kind": "dense",
            "impl": "sentence-transformers",
            "impl_version": IMPL_VERSION,
            "library_version": self._library_version(),
            "model_id": self._model_id,
            # The weights, not the label. A model republished under the same
            # name is a different model, and only the revision says so.
            "revision": self._resolved_revision,
            "device": self._device,
            "dtype": self._dtype,
            "normalize": self._normalize,
            "query_prompt": self._query_prompt,
            "document_prompt": self._document_prompt,
            "max_seq_length": self._max_seq_length,
            # Padding differs between batch shapes and float addition is not
            # associative, so this moves the last bits of an embedding. Measured
            # at ~1e-7 against float32 on MiniLM, far too small to reorder any
            # of 300 SciFact queries, and still a different artifact than the
            # one this configuration asks for.
            "batch_size": self._batch_size,
        }

    @staticmethod
    def _library_version() -> str:
        try:
            import sentence_transformers  # pyright: ignore[reportMissingImports]
        except ImportError:
            return "absent"
        return str(sentence_transformers.__version__)

    def _load(self) -> Any:
        if self._model is not None:
            return self._model
        try:
            # Resolved only with the 'local' extra; the type checker runs
            # without it, which is why the import is pinned as optional here
            # rather than by weakening the rule for the whole project.
            from sentence_transformers import (  # pyright: ignore[reportMissingImports]
                SentenceTransformer,
            )
        except ImportError as error:
            msg = "dense retrieval needs the 'local' extra: uv sync --extra local"
            raise DenseUnavailableError(msg) from error

        # Unguarded on purpose: this line is only reached once
        # sentence-transformers has imported, and it depends on torch.
        import torch  # pyright: ignore[reportMissingImports]

        # The dtype governs the forward pass, not merely how the result is
        # stored. Running in float32 and rounding afterwards is a different
        # computation from running in float16, and only the former would have
        # made this knob cosmetic.
        torch_dtype = {
            "float16": torch.float16,
            "float32": torch.float32,
            "float64": torch.float64,
        }[self._dtype]
        model = SentenceTransformer(
            self._model_id,
            revision=self._revision,
            device=self._device,
            model_kwargs={"torch_dtype": torch_dtype},
        )
        if self._max_seq_length is not None:
            model.max_seq_length = self._max_seq_length

        self._resolved_revision = self._revision or _resolve_revision(model, self._model_id)
        self._model = model
        return model

    def _encode(self, texts: Sequence[str], prompt: str) -> np.ndarray:
        model = self._load()
        prepared = [prompt + text for text in texts] if prompt else list(texts)
        vectors = model.encode(
            prepared,
            batch_size=self._batch_size,
            convert_to_numpy=True,
            normalize_embeddings=False,
            show_progress_bar=False,
        )
        # Normalise at the same precision the vectors were computed in, so the
        # descriptor's dtype describes the whole pipeline rather than half of it.
        vectors = np.asarray(vectors, dtype=_DTYPES[self._dtype])  # already computed at dtype
        if self._normalize:
            norms = np.linalg.norm(vectors, axis=1, keepdims=True)
            norms[norms == 0] = 1
            vectors = vectors / norms
        return vectors

    def index(self, corpus: Mapping[DocId, str]) -> None:
        self._doc_ids = sorted(corpus)
        self._matrix = self._encode(
            [corpus[doc_id] for doc_id in self._doc_ids], self._document_prompt
        )

    def search(self, queries: Mapping[QueryId, str], *, k: int) -> dict[QueryId, list[Ranked]]:
        if self._matrix is None:
            msg = "index() must be called before search()"
            raise RuntimeError(msg)
        if not queries:
            return {}

        query_ids = list(queries)
        vectors = self._encode([queries[q] for q in query_ids], self._query_prompt)
        # float32 for the product regardless of storage dtype: a float16
        # accumulation over thousands of documents loses more than it saves.
        similarities = vectors.astype(np.float32) @ self._matrix.astype(np.float32).T

        effective_k = min(k, len(self._doc_ids))
        results: dict[QueryId, list[Ranked]] = {}
        for position, query_id in enumerate(query_ids):
            row = similarities[position]
            candidates = np.argpartition(-row, effective_k - 1)[:effective_k]
            scored = {self._doc_ids[int(i)]: float(row[i]) for i in candidates}
            results[QueryId(query_id)] = order_deterministically(scored, effective_k)
        return results


def _resolve_revision(model: Any, model_id: str) -> str | None:
    """Best effort at the commit the weights actually came from.

    Returns None rather than guessing when it cannot be determined: a wrong
    revision in the descriptor is worse than an absent one, because it would
    claim two different sets of weights are the same.
    """
    try:
        from huggingface_hub import model_info  # pyright: ignore[reportMissingImports]
    except ImportError:
        return None
    try:
        return str(model_info(model_id).sha)
    except Exception:
        # Offline, private repo, or a local path: all of them mean "unknown".
        return None
