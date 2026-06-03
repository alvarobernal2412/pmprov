"""
Notebook environment integration.

Provides zero-friction activation for:

  - Jupyter / IPython  – via ``ip.ast_transformers`` (official public API)
  - Marimo             – via a ``builtins.compile`` patch that intercepts
                         Marimo's cell compilation pipeline

Usage
-----
Jupyter / IPython (in a notebook cell):

    from tracker import init_jupyter
    rt = init_jupyter()           # auto-detects the running IPython shell
    # All subsequent cells are tracked automatically

Marimo (at the top of the app module):

    from tracker import init_marimo
    _rt = init_marimo()
"""

from __future__ import annotations

import ast
import builtins
import uuid
from pathlib import Path
from typing import Optional

from tracker.ast_rewriter import ProvTrackTransformer
from tracker.runtime import RuntimeTracker
from tracker.storage import StorageBackend


# ------------------------------------------------------------------
# Jupyter / IPython
# ------------------------------------------------------------------

def init_jupyter(
    ip_shell=None,
    db_path: str | Path = "provenance.db",
    artifact_dir: str | Path = "artifacts",
    agent_id: Optional[str] = None,
    history_name: Optional[str] = None,
    branch_name: str = "main",
) -> RuntimeTracker:
    """
    Register ProvTrack in the active IPython kernel.

    Parameters
    ----------
    ip_shell:
        IPython ``InteractiveShell`` instance.  If *None*, auto-detected via
        ``IPython.get_ipython()``.
    db_path:
        Path for the DuckDB / SQLite provenance database.
    artifact_dir:
        Directory where DataFrame Parquet snapshots are written.
        Defaults to ``"artifacts"`` relative to the working directory.
    agent_id:
        Override the agent identifier (defaults to the OS username).
    history_name:
        Human-readable title for the analysis session.
        Defaults to the auto-generated history UUID.
    branch_name:
        Name for the initial analysis branch.  Defaults to ``"main"``.

    Returns
    -------
    RuntimeTracker
        The active tracker instance (also stored as ``_provtrack_runtime``
        in the notebook's user namespace).
    """
    if ip_shell is None:
        try:
            from IPython import get_ipython
            ip_shell = get_ipython()
        except ImportError:
            pass
    if ip_shell is None:
        raise RuntimeError(
            "No active IPython shell found. "
            "Call init_jupyter() from inside a running Jupyter notebook."
        )

    storage = StorageBackend(db_path=db_path, artifact_dir=artifact_dir)
    runtime = RuntimeTracker(
        storage=storage,
        session_id=str(uuid.uuid4()),
        agent_id=agent_id,
        history_name=history_name,
        branch_name=branch_name,
    )

    # Make the runtime reachable inside cell code
    ip_shell.user_ns["_provtrack_runtime"] = runtime

    # Register transformer only once (guard against double-init)
    if not any(isinstance(t, ProvTrackTransformer) for t in ip_shell.ast_transformers):
        ip_shell.ast_transformers.append(ProvTrackTransformer())

    return runtime


# ------------------------------------------------------------------
# Marimo
# ------------------------------------------------------------------

def init_marimo(
    db_path: str | Path = "provenance.db",
    artifact_dir: str | Path = "artifacts",
    agent_id: Optional[str] = None,
    history_name: Optional[str] = None,
    branch_name: str = "main",
) -> RuntimeTracker:
    """
    Activate ProvTrack inside a Marimo reactive notebook.

    Patches ``builtins.compile`` so that every cell source string compiled by
    Marimo's runner is transparently routed through ``ProvTrackTransformer``
    before bytecode generation.

    The runtime is injected into ``builtins`` (rather than a local namespace)
    because Marimo cells each run in their own isolated scope.

    Parameters
    ----------
    db_path:
        Path for the DuckDB / SQLite provenance database.
    artifact_dir:
        Directory where DataFrame Parquet snapshots are written.
        Defaults to ``"artifacts"`` relative to the working directory.
    agent_id:
        Override the agent identifier (defaults to the OS username).
    history_name:
        Human-readable title for the analysis session.
        Defaults to the auto-generated history UUID.
    branch_name:
        Name for the initial analysis branch.  Defaults to ``"main"``.

    Design note on Marimo's reactive DAG
    ------------------------------------
    Marimo re-executes downstream cells automatically when upstream outputs
    change.  Each ProvTrack node stores the ``session_id`` so that cascade
    re-executions within the same session can be identified and parent-child
    links in the edges table map directly onto Marimo's execution graph.
    """
    storage = StorageBackend(db_path=db_path, artifact_dir=artifact_dir)
    session_id = str(uuid.uuid4())
    runtime = RuntimeTracker(
        storage=storage,
        session_id=session_id,
        agent_id=agent_id,
        history_name=history_name,
        branch_name=branch_name,
    )

    # Expose in builtins so every Marimo cell scope can resolve _provtrack_runtime
    builtins._provtrack_runtime = runtime  # type: ignore[attr-defined]

    _patch_compile()
    return runtime


# ------------------------------------------------------------------
# builtins.compile patch (Marimo only)
# ------------------------------------------------------------------

_ORIGINAL_COMPILE = builtins.compile
_TRANSFORMER = ProvTrackTransformer()
_PATCHED = False


def _patch_compile() -> None:
    global _PATCHED
    if _PATCHED:
        return
    _PATCHED = True

    def _compile(source, filename, mode, flags=0, dont_inherit=False, optimize=-1):
        if (
            isinstance(source, str)
            and mode in ("exec", "single")
            and _is_marimo_cell(filename)
        ):
            try:
                tree = ast.parse(source, filename, mode)
                tree = _TRANSFORMER.visit(tree)
                ast.fix_missing_locations(tree)
                # Pass the modified AST (not the source string) to the
                # original compile so location data is preserved.
                return _ORIGINAL_COMPILE(tree, filename, mode, flags, dont_inherit, optimize)
            except SyntaxError:
                pass  # fall through – let Python surface the syntax error naturally

        return _ORIGINAL_COMPILE(source, filename, mode, flags, dont_inherit, optimize)

    builtins.compile = _compile


def _is_marimo_cell(filename: str) -> bool:
    """Heuristic match for filenames Marimo assigns to cell code objects."""
    return any(
        token in filename
        for token in ("<cell", "__marimo__", "marimo_cell", "<marimo")
    )
