"""Content-addressed identity.

This module is the load-bearing element of the whole engine. Every diff the
product shows is only as trustworthy as these keys: a fingerprint that is too
coarse silently reuses a stale artifact and reports a regression that does not
exist; one that is too fine re-embeds a corpus for nothing.

Design stance: **fail loudly rather than hash sloppily.** The encoder refuses
values that cannot be reproduced on another machine or another day (filesystem
paths, timestamps, NaN) instead of falling back to `str(value)`, the classic
way to end up with `<object at 0x7f3a...>` baked into a cache key.
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Mapping, Sequence, Set
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Final, cast

from embedlab.domain.ids import Fingerprint

if TYPE_CHECKING:
    from collections.abc import Buffer

_DIGEST_BYTES: Final = 16
"""128 bits. Collision risk is irrelevant at our scale; keys stay readable."""

_PREFIX: Final = "fp"


class NotFingerprintableError(TypeError):
    """A value cannot be fingerprinted reproducibly.

    Raised instead of degrading to a weaker key, because a wrong cache key is
    worse than a crash: it produces confident, wrong numbers.
    """


def _encode(value: object) -> bytes:
    """Encode to canonical, type-tagged bytes.

    Type tags keep `1`, `1.0`, `"1"` and `True` distinct. Strings and bytes are
    length-prefixed so that `["ab", "c"]` and `["a", "bc"]` cannot collide.
    """
    # bool before int: bool is a subclass of int.
    if value is None:
        return b"n;"
    if value is True:
        return b"b1;"
    if value is False:
        return b"b0;"
    if isinstance(value, Enum):
        # Qualified by class so two enums sharing a value stay distinct.
        return b"E:" + _encode(type(value).__qualname__) + _encode(value.value)
    if isinstance(value, int):
        return b"i:" + str(value).encode() + b";"
    if isinstance(value, float):
        if not math.isfinite(value):
            msg = (
                f"refusing to fingerprint non-finite float {value!r}: a parameter "
                "that is not equal to itself cannot identify a cached artifact"
            )
            raise NotFingerprintableError(msg)
        # repr() round-trips exactly for float; str() would too on py3 but repr
        # states the intent.
        return b"f:" + repr(value).encode() + b";"
    if isinstance(value, str):
        raw = value.encode()
        return b"s:" + str(len(raw)).encode() + b":" + raw + b";"
    if isinstance(value, bytes | bytearray | memoryview):
        # memoryview narrows to memoryview[Unknown]; the element type is
        # irrelevant since we hash the bytes.
        raw = bytes(cast("Buffer", value))
        return b"y:" + str(len(raw)).encode() + b":" + raw + b";"
    if isinstance(value, Mapping):
        # The element types are genuinely unknown and that is the point: any
        # object is handled recursively, or refused below.
        mapping = cast("Mapping[object, object]", value)
        keys = list(mapping.keys())
        if any(not isinstance(k, str) for k in keys):
            msg = (
                "refusing to fingerprint a mapping with non-string keys: ordering "
                "would depend on the key type's comparison semantics"
            )
            raise NotFingerprintableError(msg)
        text_keys = sorted(key for key in keys if isinstance(key, str))
        parts = [_encode(key) + _encode(mapping[key]) for key in text_keys]
        return b"d:" + b"".join(parts) + b";"
    if isinstance(value, tuple):
        # Tagged apart from list: a tuple often means "fixed arity" in a spec.
        items = cast("tuple[object, ...]", value)
        return b"t:" + b"".join(_encode(item) for item in items) + b";"
    if isinstance(value, Set):
        # Sort the encodings, not the elements: members need not be comparable.
        members = cast("Set[object]", value)
        return b"e:" + b"".join(sorted(_encode(item) for item in members)) + b";"
    if isinstance(value, Sequence):
        elements = cast("Sequence[object]", value)
        return b"l:" + b"".join(_encode(item) for item in elements) + b";"

    msg = (
        f"refusing to fingerprint {type(value).__qualname__}. Values whose "
        "identity is machine- or time-dependent (paths, datetimes, open files, "
        "arbitrary objects) must be reduced to a reproducible value first. For "
        "a file, pass fingerprint_file(path) instead of the path itself."
    )
    raise NotFingerprintableError(msg)


def fingerprint(value: object) -> Fingerprint:
    """Fingerprint any reproducibly-encodable value."""
    digest = hashlib.blake2b(_encode(value), digest_size=_DIGEST_BYTES).hexdigest()
    return Fingerprint(f"{_PREFIX}_{digest}")


def fingerprint_bytes(raw: bytes) -> Fingerprint:
    """Fingerprint opaque content already in memory."""
    digest = hashlib.blake2b(raw, digest_size=_DIGEST_BYTES).hexdigest()
    return Fingerprint(f"{_PREFIX}_{digest}")


def fingerprint_file(path: Path, *, chunk_bytes: int = 1 << 20) -> Fingerprint:
    """Fingerprint a file by its *content*, streamed.

    Deliberately ignores the path, mtime and inode: the same corpus moved or
    re-downloaded must hit the same cache entry, on this machine and on CI.
    """
    hasher = hashlib.blake2b(digest_size=_DIGEST_BYTES)
    with path.open("rb") as handle:
        while block := handle.read(chunk_bytes):
            hasher.update(block)
    return Fingerprint(f"{_PREFIX}_{hasher.hexdigest()}")
