"""Memory-safe state extraction: converts live Python objects into JSON metadata."""

from __future__ import annotations

from typing import Any


def capture_snapshot(obj: Any) -> dict:
    """
    Return a shallow metadata dictionary describing *obj*.

    Never performs deep copies. Detection priority:
      1. DataFrame  (Pandas / Polars)
      2. Array / Tensor  (NumPy / PyTorch)
      3. Collection  (list, dict, set – anything with __len__ except strings)
      4. Fallback
    """
    type_name = type(obj).__name__

    if type_name == "DataFrame":
        return _dataframe(obj)

    if hasattr(obj, "shape") and hasattr(obj, "dtype"):
        return _array(obj)

    if hasattr(obj, "__len__") and not isinstance(obj, (str, bytes)):
        return _collection(obj)

    return _fallback(obj)


# ------------------------------------------------------------------
# Type-specific extractors
# ------------------------------------------------------------------

def _dataframe(df) -> dict:
    try:
        return {
            "kind": "dataframe",
            "object_id": id(df),
            "shape": list(df.shape),
            "columns": [str(c) for c in df.columns],
            "dtypes": {str(c): str(t) for c, t in zip(df.columns, df.dtypes)},
        }
    except Exception:
        return _fallback(df)


def _array(arr) -> dict:
    try:
        # nbytes is standard on NumPy; Torch uses element_size() * nelement()
        nbytes: int | None = getattr(arr, "nbytes", None)
        if nbytes is None and hasattr(arr, "element_size"):
            nbytes = int(arr.element_size() * arr.nelement())
        return {
            "kind": "array",
            "object_id": id(arr),
            "shape": list(arr.shape),
            "dtype": str(arr.dtype),
            "nbytes": nbytes,
        }
    except Exception:
        return _fallback(arr)


def _collection(obj) -> dict:
    try:
        sample = list(obj.values())[:20] if isinstance(obj, dict) else list(obj)[:20]
        return {
            "kind": "collection",
            "object_id": id(obj),
            "collection_type": type(obj).__name__,
            "length": len(obj),
            "element_types": sorted({type(v).__name__ for v in sample}),
        }
    except Exception:
        return _fallback(obj)


def _fallback(obj) -> dict:
    try:
        summary = str(obj)[:150]
    except Exception:
        summary = "<unrepresentable>"
    return {
        "kind": "other",
        "object_id": id(obj),
        "class_name": type(obj).__name__,
        "summary": summary,
    }
