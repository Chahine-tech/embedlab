"""Content-addressed artifact storage.

Format-agnostic on purpose: this layer owns directory layout and atomicity,
nothing else. What goes inside an entry is decided one layer up, so adding a
Parquet table or a vector file never touches the caching rules.

The invariant worth more than the speed: **a reader never sees a partial
entry.** A half-written artifact mistaken for a complete one is worse than no
cache at all: it would feed truncated results into a diff, which is the
failure this project exists to detect. So writes land in a temporary directory
and are published by a single atomic rename.
"""

from __future__ import annotations

import shutil
import uuid
from contextlib import contextmanager
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Generator
    from pathlib import Path

    from embedlab.domain.ids import Fingerprint


class CacheStore:
    """A directory of immutable, content-addressed stage outputs."""

    def __init__(self, root: Path) -> None:
        self._root = root

    @property
    def root(self) -> Path:
        return self._root

    def _entry_path(self, stage: str, key: Fingerprint) -> Path:
        if "/" in stage or stage.startswith("."):
            msg = f"invalid stage name {stage!r}"
            raise ValueError(msg)
        return self._root / stage / str(key)

    def lookup(self, stage: str, key: Fingerprint) -> Path | None:
        """The entry directory if it exists and was published, else None."""
        path = self._entry_path(stage, key)
        return path if path.is_dir() else None

    def has(self, stage: str, key: Fingerprint) -> bool:
        return self.lookup(stage, key) is not None

    @contextmanager
    def reserve(self, stage: str, key: Fingerprint) -> Generator[Path]:
        """Yield a scratch directory, then publish it atomically on success.

        On any exception the scratch directory is removed and nothing is
        published, so a crashed run leaves no half-entry behind. If another
        process published the same key while we were working, ours is discarded
        and theirs kept: entries are content-addressed, so they are equivalent
        by construction and the first one to land wins.
        """
        final = self._entry_path(stage, key)
        scratch = final.parent / f".tmp-{uuid.uuid4().hex}"
        scratch.mkdir(parents=True, exist_ok=False)
        try:
            yield scratch
        except BaseException:
            shutil.rmtree(scratch, ignore_errors=True)
            raise

        if final.exists():
            shutil.rmtree(scratch, ignore_errors=True)
            return
        try:
            scratch.replace(final)
        except OSError:
            # Lost the race between the check and the rename.
            shutil.rmtree(scratch, ignore_errors=True)
            if not final.is_dir():
                raise

    def forget(self, stage: str, key: Fingerprint) -> bool:
        """Delete one entry. Returns whether anything was there.

        Only ever used deliberately, by a `--no-cache` flag or a test. Nothing
        in the engine evicts on its own: an artifact silently disappearing would
        turn a cheap rerun into a different-looking result.
        """
        path = self.lookup(stage, key)
        if path is None:
            return False
        shutil.rmtree(path)
        return True
