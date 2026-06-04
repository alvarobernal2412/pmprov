"""Diff engine: compares two capture_snapshot() dicts into a structured delta."""

from __future__ import annotations


def compute_delta(pre: dict, post: dict) -> dict:
    """
    Compare a pre- and post-execution snapshot and return a structured delta.

    Mutation classification:
      - ``in_place``  – same object identity (id) but metadata changed
      - ``derivation`` – different object identity
    """
    if not pre or not post:
        return {"kind": "unknown"}

    mutation_type = "in_place" if pre.get("object_id") == post.get("object_id") else "derivation"
    kind = pre.get("kind")

    if kind == "dataframe" and post.get("kind") == "dataframe":
        return _dataframe_delta(pre, post, mutation_type)

    if kind == "array" and post.get("kind") == "array":
        return _array_delta(pre, post, mutation_type)

    if kind == "collection" and post.get("kind") == "collection":
        return _collection_delta(pre, post, mutation_type)

    return {
        "kind": "generic",
        "mutation_type": mutation_type,
        "pre_class": pre.get("class_name", kind),
        "post_class": post.get("class_name", post.get("kind")),
    }


# ------------------------------------------------------------------
# Type-specific delta calculators
# ------------------------------------------------------------------

def _dataframe_delta(pre: dict, post: dict, mutation_type: str) -> dict:
    pre_cols = set(pre.get("columns", []))
    post_cols = set(post.get("columns", []))
    pre_dtypes: dict = pre.get("dtypes", {})
    post_dtypes: dict = post.get("dtypes", {})
    pre_rows: int = (pre.get("shape") or [0])[0]
    post_rows: int = (post.get("shape") or [0])[0]

    dtype_changes = {
        col: {"from": pre_dtypes[col], "to": post_dtypes[col]}
        for col in pre_cols & post_cols
        if pre_dtypes.get(col) != post_dtypes.get(col)
    }
    cols_added = sorted(post_cols - pre_cols)
    cols_removed = sorted(pre_cols - post_cols)
    rows_delta = post_rows - pre_rows

    if cols_added and not cols_removed and not dtype_changes:
        modification_type = "addition"
    elif cols_removed and not cols_added and not dtype_changes:
        modification_type = "removal"
    elif dtype_changes and not cols_added and not cols_removed:
        modification_type = "casting"
    elif rows_delta > 0 and not cols_added and not cols_removed and not dtype_changes:
        modification_type = "addition"
    elif rows_delta < 0 and not cols_added and not cols_removed and not dtype_changes:
        modification_type = "removal"
    else:
        modification_type = "other"

    return {
        "kind": "dataframe",
        "mutation_type": mutation_type,
        "modification_type": modification_type,
        "columns_added": cols_added,
        "columns_removed": cols_removed,
        "dtype_changes": dtype_changes,
        "rows_before": pre_rows,
        "rows_after": post_rows,
        "rows_delta": rows_delta,
    }


def _array_delta(pre: dict, post: dict, mutation_type: str) -> dict:
    return {
        "kind": "array",
        "mutation_type": mutation_type,
        "shape_before": pre.get("shape"),
        "shape_after": post.get("shape"),
        "dtype_before": pre.get("dtype"),
        "dtype_after": post.get("dtype"),
    }


def _collection_delta(pre: dict, post: dict, mutation_type: str) -> dict:
    before = pre.get("length") or 0
    after = post.get("length") or 0
    return {
        "kind": "collection",
        "mutation_type": mutation_type,
        "length_before": before,
        "length_after": after,
        "length_delta": after - before,
    }
