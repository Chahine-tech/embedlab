"""Properties the whole engine's trustworthiness rests on.

Each test here corresponds to a way a diff could silently lie.
"""

import math
from enum import StrEnum

import pytest
from hypothesis import given
from hypothesis import strategies as st

from embedlab.cache.fingerprint import (
    NotFingerprintableError,
    fingerprint,
    fingerprint_bytes,
    fingerprint_file,
)
from embedlab.domain.taxonomy import FailureKind


def test_mapping_order_is_irrelevant():
    assert fingerprint({"model": "jina-v5", "k": 10}) == fingerprint({"k": 10, "model": "jina-v5"})


def test_nested_mapping_order_is_irrelevant():
    left = {"retrieval": {"k": 20, "metric": "cosine"}, "name": "a"}
    right = {"name": "a", "retrieval": {"metric": "cosine", "k": 20}}
    assert fingerprint(left) == fingerprint(right)


@pytest.mark.parametrize(
    ("left", "right"),
    [
        (10, 10.0),  # int vs float: fp16 vs fp32 configs must not collide
        (True, 1),
        (False, 0),
        (None, "None"),
        ("1", 1),
        ([1, 2], (1, 2)),  # list vs tuple
        ([1, 2], {1, 2}),  # list vs set
    ],
    ids=["int-float", "true-one", "false-zero", "none-str", "str-int", "list-tuple", "list-set"],
)
def test_types_do_not_collide(left, right):
    assert fingerprint(left) != fingerprint(right)


def test_concatenation_cannot_collide():
    """Without length prefixes, ["ab","c"] and ["a","bc"] would hash alike."""
    assert fingerprint(["ab", "c"]) != fingerprint(["a", "bc"])
    assert fingerprint({"a": "bc"}) != fingerprint({"ab": "c"})


def test_set_order_is_irrelevant():
    assert fingerprint({"b", "a", "c"}) == fingerprint({"c", "b", "a"})


def test_list_order_matters():
    assert fingerprint(["a", "b"]) != fingerprint(["b", "a"])


def test_enum_is_qualified_by_class():
    """Two enums sharing a raw value must not be interchangeable."""

    class Other(StrEnum):
        HARD_NEGATIVE = "hard_negative"

    assert fingerprint(FailureKind.HARD_NEGATIVE) != fingerprint(Other.HARD_NEGATIVE)
    assert fingerprint(FailureKind.HARD_NEGATIVE) != fingerprint("hard_negative")


def test_paths_are_refused():
    """A path would bake machine-specific state into a cache key."""
    from pathlib import Path

    with pytest.raises(NotFingerprintableError, match="fingerprint_file"):
        fingerprint(Path("/Users/someone/corpus.jsonl"))


def test_datetimes_are_refused():
    from datetime import UTC, datetime

    with pytest.raises(NotFingerprintableError):
        fingerprint(datetime.now(UTC))


def test_arbitrary_objects_are_refused():
    """The failure mode this guards: `<object at 0x7f3a...>` inside a cache key."""

    class Opaque:
        pass

    with pytest.raises(NotFingerprintableError):
        fingerprint(Opaque())


@pytest.mark.parametrize("value", [math.nan, math.inf, -math.inf])
def test_non_finite_floats_are_refused(value):
    with pytest.raises(NotFingerprintableError):
        fingerprint(value)


def test_non_string_mapping_keys_are_refused():
    with pytest.raises(NotFingerprintableError, match="non-string keys"):
        fingerprint({1: "a"})


def test_file_fingerprint_follows_content_not_location(tmp_path):
    first = tmp_path / "a" / "corpus.jsonl"
    second = tmp_path / "b" / "renamed.jsonl"
    for path in (first, second):
        path.parent.mkdir(parents=True)
        path.write_bytes(b'{"id": "doc-1"}\n')

    assert fingerprint_file(first) == fingerprint_file(second)

    second.write_bytes(b'{"id": "doc-2"}\n')
    assert fingerprint_file(first) != fingerprint_file(second)


def test_file_fingerprint_is_chunk_size_independent(tmp_path):
    path = tmp_path / "corpus.jsonl"
    path.write_bytes(b"x" * 5000)
    assert fingerprint_file(path, chunk_bytes=7) == fingerprint_file(path, chunk_bytes=4096)


def test_file_fingerprint_matches_bytes_fingerprint(tmp_path):
    path = tmp_path / "corpus.jsonl"
    payload = b"some corpus content"
    path.write_bytes(payload)
    assert fingerprint_file(path) == fingerprint_bytes(payload)


@given(st.text())
def test_strings_are_stable_across_calls(value):
    assert fingerprint(value) == fingerprint(value)


@given(
    st.dictionaries(
        st.text(min_size=1),
        st.one_of(st.integers(), st.text(), st.booleans(), st.none()),
        min_size=1,
    )
)
def test_shuffled_dicts_agree(mapping):
    shuffled = dict(reversed(list(mapping.items())))
    assert fingerprint(mapping) == fingerprint(shuffled)
