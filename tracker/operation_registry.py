"""
Session-independent registry that maps function qualified names to
OperationType labels.

Why this exists
---------------
OperationType (e.g., "case_filter", "attribute_derivation") is semantic
knowledge that cannot be inferred automatically from a function name or its
AST. This registry lets analysts attach that knowledge explicitly, either
up-front with the decorator form or retroactively with the direct call form.
RuntimeTracker reads the registry on every new operation encounter; if no
label has been registered the type defaults to "unknown".

Usage
-----
Decorator form (label at definition time)::

    from tracker import operation_type

    @operation_type("attribute_derivation")
    def add_case_duration(df):
        ...

Direct registration (useful for third-party functions you don't own)::

    import pm4py
    from tracker import operation_type

    operation_type("case_filter",        pm4py.filter_log_on_attribute_values)
    operation_type("process_discovery",  pm4py.discover_petri_net_inductive)

Lookup
------
The registry is queried with the ``func_name`` string captured by the AST
transformer (e.g., ``"pm4py.filter_log_on_attribute_values"``, ``"df.assign"``,
``"merge"``).  Lookup tries three keys in order:

  1. Exact ``func_name`` string.
  2. ``module.qualname`` constructed from the callable's own attributes.
  3. The last dotted segment of ``func_name`` (e.g., ``"assign"`` from
     ``"df.assign"``), which catches common pandas/polars method calls where
     the instance variable name varies.

Returns ``"unknown"`` when none of the three keys match.
"""

from __future__ import annotations

from typing import Callable

# All keys map to an OperationType name string.
_registry: dict[str, str] = {}


def operation_type(type_name: str, func: Callable | None = None):
    """
    Register *type_name* as the OperationType for *func*.

    One-argument form returns a decorator; two-argument form registers
    immediately and returns *func* unchanged.

    Parameters
    ----------
    type_name:
        Semantic category label (e.g., ``"attribute_derivation"``,
        ``"case_filter"``, ``"conformance_check"``).
    func:
        The callable to label.  If *None*, a decorator is returned.
    """
    if func is None:
        def _decorator(f: Callable) -> Callable:
            _register(f, type_name)
            return f
        return _decorator

    _register(func, type_name)
    return func


def lookup(func_name: str) -> str:
    """
    Return the registered OperationType name for *func_name*.
    Falls back to ``"unknown"`` when no label has been registered.
    """
    # 1. Exact match (covers "pm4py.filter_log_on_attribute_values", bare names, …)
    if func_name in _registry:
        return _registry[func_name]

    # 2. Last dotted segment ("df.assign" → "assign", "pd.merge" → "merge")
    tail = func_name.rsplit(".", 1)[-1]
    if tail in _registry:
        return _registry[tail]

    return "unknown"


# ------------------------------------------------------------------
# Internal helpers
# ------------------------------------------------------------------

def _register(func: Callable, type_name: str) -> None:
    """Store *type_name* under all stable keys derivable from *func*."""
    qualname: str = getattr(func, "__qualname__", None) or str(func)
    module: str = getattr(func, "__module__", None) or ""

    # Bare qualname  ("filter_log_on_attribute_values", "MyClass.method")
    _registry[qualname] = type_name

    # Module-qualified name  ("pm4py.filter_log_on_attribute_values")
    if module:
        _registry[f"{module}.{qualname}"] = type_name

    # Last segment only  ("filter_log_on_attribute_values") — already stored
    # as bare qualname above unless qualname contains dots (nested class/func).
    tail = qualname.rsplit(".", 1)[-1]
    if tail != qualname:
        _registry.setdefault(tail, type_name)  # don't clobber a more specific entry
