"""Pinning which weights a model name actually resolved to.

Shared by every adapter that loads from the hub, because a model republished
under the same name is a different model and only the revision says so. One
copy, so the two adapters cannot drift into pinning different things.
"""

from __future__ import annotations


def resolve_revision(model_id: str) -> str | None:
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
