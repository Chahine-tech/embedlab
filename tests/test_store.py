"""The cache's one non-negotiable property: a reader never sees a partial entry."""

import pytest

from embedlab.cache.store import CacheStore
from embedlab.domain.ids import Fingerprint

KEY = Fingerprint("fp_" + "a" * 32)
OTHER = Fingerprint("fp_" + "b" * 32)


@pytest.fixture
def store(tmp_path):
    return CacheStore(tmp_path / "cache")


def test_an_unknown_key_is_a_miss(store):
    assert store.lookup("retrieve", KEY) is None
    assert store.has("retrieve", KEY) is False


def test_a_published_entry_is_a_hit(store):
    with store.reserve("retrieve", KEY) as scratch:
        (scratch / "run.parquet").write_bytes(b"payload")

    entry = store.lookup("retrieve", KEY)
    assert entry is not None
    assert (entry / "run.parquet").read_bytes() == b"payload"


def write_then_fail(store, *, wrote: bytes | None = b"half") -> None:
    with store.reserve("retrieve", KEY) as scratch:
        if wrote is not None:
            (scratch / "partial.parquet").write_bytes(wrote)
        msg = "boom"
        raise RuntimeError(msg)


def test_a_failed_write_publishes_nothing(store):
    """The invariant. A crash mid-write must not leave a readable entry."""
    with pytest.raises(RuntimeError, match="boom"):
        write_then_fail(store)

    assert store.lookup("retrieve", KEY) is None


def test_a_failed_write_leaves_no_scratch_directory(store):
    with pytest.raises(RuntimeError, match="boom"):
        write_then_fail(store, wrote=None)

    assert list((store.root / "retrieve").iterdir()) == []


def test_the_entry_is_invisible_until_the_block_exits(store):
    with store.reserve("retrieve", KEY) as scratch:
        (scratch / "run.parquet").write_bytes(b"payload")
        assert store.lookup("retrieve", KEY) is None, "published too early"

    assert store.lookup("retrieve", KEY) is not None


def test_a_concurrent_publication_wins_and_ours_is_discarded(store):
    """Entries are content-addressed, so the two are equivalent; first wins."""
    with store.reserve("retrieve", KEY) as scratch:
        (scratch / "run.parquet").write_bytes(b"ours")
        with store.reserve("retrieve", KEY) as other:
            (other / "run.parquet").write_bytes(b"theirs")

    entry = store.lookup("retrieve", KEY)
    assert (entry / "run.parquet").read_bytes() == b"theirs"


def test_no_scratch_directory_survives_a_race(store):
    with store.reserve("retrieve", KEY) as scratch:
        (scratch / "a").write_bytes(b"ours")
        with store.reserve("retrieve", KEY) as other:
            (other / "a").write_bytes(b"theirs")

    names = sorted(path.name for path in (store.root / "retrieve").iterdir())
    assert names == [str(KEY)]


def test_keys_and_stages_are_isolated(store):
    with store.reserve("retrieve", KEY) as scratch:
        (scratch / "x").write_bytes(b"1")
    with store.reserve("evaluate", KEY) as scratch:
        (scratch / "x").write_bytes(b"2")

    assert store.lookup("retrieve", KEY) != store.lookup("evaluate", KEY)
    assert store.lookup("retrieve", OTHER) is None


def test_forget_removes_only_the_named_entry(store):
    for key in (KEY, OTHER):
        with store.reserve("retrieve", key) as scratch:
            (scratch / "x").write_bytes(b"1")

    assert store.forget("retrieve", KEY) is True
    assert store.lookup("retrieve", KEY) is None
    assert store.lookup("retrieve", OTHER) is not None


def test_forgetting_an_absent_entry_is_not_an_error(store):
    assert store.forget("retrieve", KEY) is False


@pytest.mark.parametrize("stage", ["../escape", "a/b", ".hidden"])
def test_stage_names_cannot_escape_the_store(store, stage):
    with pytest.raises(ValueError, match="invalid stage name"):
        store.lookup(stage, KEY)
