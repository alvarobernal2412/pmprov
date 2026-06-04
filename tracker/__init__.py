"""
ProvTrack – statement-level provenance middleware for notebook environments.

Quick start
-----------
Jupyter / IPython::

    from tracker import init_jupyter
    rt = init_jupyter()

Marimo::

    from tracker import init_marimo
    _rt = init_marimo()

Both functions accept optional ``db_path``, ``artifact_dir``, ``history_name``
and ``branch_name`` kwargs to control where provenance data is stored and how
the session is labelled.

Labelling operation types
--------------------------
OperationType cannot be inferred automatically.  Register labels explicitly::

    from tracker import operation_type
    import pm4py

    # Decorator form (functions you own):
    @operation_type("attribute_derivation")
    def add_case_duration(df): ...

    # Direct form (third-party functions):
    operation_type("case_filter", pm4py.filter_log_on_attribute_values)
    operation_type("process_discovery", pm4py.discover_petri_net_inductive)

Unregistered functions default to OperationType ``"unknown"``.
"""

from tracker.ast_rewriter import OMIT_FUNCTIONS
from tracker.kernel_hooks import init_jupyter, init_marimo
from tracker.operation_registry import operation_type, step_category
from tracker.runtime import RuntimeTracker
from tracker.storage import StorageBackend
import tracker.visualizations  # noqa: F401 — patches show_graph / show_graph_widget onto RuntimeTracker
import tracker.introspection  # noqa: F401 — patches describe_state / list_branches onto RuntimeTracker
import tracker.comparison  # noqa: F401 — patches compare_states / compare_histories onto RuntimeTracker


def omit_functions(*names: str) -> None:
    """
    Add function or method names to the provenance omit list.

    Calls whose func_name (or trailing method segment) matches any of *names*
    will not be tracked — no provenance node is created for them.

    Examples
    --------
    >>> from tracker import omit_functions
    >>> omit_functions("my_debug_helper", "tqdm")
    """
    OMIT_FUNCTIONS.update(names)


__all__ = [
    "init_jupyter",
    "init_marimo",
    "omit_functions",
    "operation_type",
    "step_category",
    "RuntimeTracker",
    "StorageBackend",
]
