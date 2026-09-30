"""Typed access to decoded JSON.

Two payloads are parsed in this project — Claude's event stream and GitHub's API — and both
arrive as `object` from `json.loads`. Strict typing rejects `isinstance(x, dict)` on its own,
because it narrows to `dict[Unknown, Unknown]` and every field read after that is untyped.
That is not pedantry: reading a field that a payload does not carry, or carries under another
name, is precisely the class of bug that reached runtime in the implementation this replaces.

The casts here are sound rather than suppressions: `json.loads` only ever produces `str` keys.
"""

from __future__ import annotations

from typing import cast


def as_dict(value: object) -> dict[str, object] | None:
    return cast("dict[str, object]", value) if isinstance(value, dict) else None


def as_list(value: object) -> list[object] | None:
    return cast("list[object]", value) if isinstance(value, list) else None


def as_str(value: object) -> str | None:
    return value if isinstance(value, str) else None


def as_int(value: object) -> int:
    """Zero for anything that is not an integer, booleans included.

    `True` is an `int` in Python, so without the guard a `true` in a numeric field becomes 1 —
    which is how a boolean cost field would otherwise have been billed as one dollar.
    """
    return value if isinstance(value, int) and not isinstance(value, bool) else 0


def as_float(value: object) -> float | None:
    if isinstance(value, bool):
        return None
    return float(value) if isinstance(value, (int, float)) else None


def dicts_in(value: object) -> list[dict[str, object]]:
    """Every object in a JSON array, skipping anything that is not one.

    GitHub arrays are homogeneous in practice, so a non-object entry means the shape changed;
    skipping it degrades the result instead of ending the run.
    """
    items = as_list(value)
    if items is None:
        return []
    return [d for d in (as_dict(item) for item in items) if d is not None]


def strings_in(value: object, *, key: str) -> frozenset[str]:
    """The named field of every object in a JSON array — labels, check names, and the like."""
    found = (as_str(item.get(key)) for item in dicts_in(value))
    return frozenset(name for name in found if name)
