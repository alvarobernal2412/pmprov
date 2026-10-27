"""Plain-Python operations the AST rewriter routes non-call statements through.

``x = df[mask]``, ``df['c'] = expr`` and the like are not calls, so the
rewriter turns them into calls of these helpers and hands them to
``trace_step``. Each helper does exactly what the statement does. The
``_provtrack_frames_only`` flag tells the runtime to run the statement as
plain Python when its first argument is not a DataFrame or Series (a dict
lookup, a list slice, an attribute set on a model).
"""
from __future__ import annotations

from typing import Any

_FRAME_TYPES = ("DataFrame", "Series")


def _frames_only(func):
    func._provtrack_frames_only = True
    return func


@_frames_only
def getitem(obj, key):
    return obj[key]


@_frames_only
def loc_getitem(obj, key):
    return obj.loc[key]


@_frames_only
def iloc_getitem(obj, key):
    return obj.iloc[key]


@_frames_only
def setitem(obj, key, value):
    obj[key] = value
    return obj


@_frames_only
def loc_setitem(obj, key, value):
    obj.loc[key] = value
    return obj


@_frames_only
def iloc_setitem(obj, key, value):
    obj.iloc[key] = value
    return obj


@_frames_only
def setattr_(obj, name, value):
    setattr(obj, name, value)
    return obj


@_frames_only
def value(obj):
    """Identity: ``x = df['a'] > 3`` evaluates its right side, then records it."""
    return obj


# A value step has no "input" to compare with: its argument is its own result.
value._provtrack_no_delta = True


@_frames_only
def alias(obj):
    """Identity: ``df2 = df`` names an existing object a second time."""
    return obj


# An alias adds no data: it binds the name to the artifact the object already has.
alias._provtrack_alias = True
alias._provtrack_no_delta = True


def block_result(frames: dict) -> dict:
    """What a loop block hands to ``trace_step``: the frames it left behind, as they are."""
    return frames


def collect(namespace: dict, names: list[str]) -> dict[str, Any]:
    """The frames/series among *names* in *namespace*: what a loop block leaves behind."""
    return {
        n: namespace[n] for n in names
        if n in namespace and type(namespace[n]).__name__ in _FRAME_TYPES
    }
