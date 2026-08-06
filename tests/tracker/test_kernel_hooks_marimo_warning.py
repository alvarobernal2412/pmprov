"""
Regression coverage for tracker/kernel_hooks.py::init_marimo()'s "AST patch
never activated" warning.

init_marimo() only constructs the RuntimeTracker; the actual code-rewriting
happens via patch_marimo_ast_compile(), which must run at module level before
marimo compiles any cell. If a notebook calls init_marimo() without ever having
called patch_marimo_ast_compile() first (e.g. it's called inside a cell instead
of at module level, or omitted entirely), the resulting tracker is fully
functional but silently records zero steps forever — this is exactly the
failure mode that was hardest to diagnose live (the notebook ran without error,
the sidebar just never grew past its root node). init_marimo() now warns in
this case instead of failing silently.

Note: tests/conftest.py calls patch_marimo_ast_compile() once at collection
time, so _MARIMO_PATCHED is True for the rest of the suite by default — these
tests explicitly force it back to False to simulate the "never patched" state.
"""
from __future__ import annotations

import pytest

import tracker.kernel_hooks as kernel_hooks
from tracker.storage import DuckDBSQLiteBackend as StorageBackend


@pytest.fixture
def storage(tmp_path):
    return StorageBackend(db_path=tmp_path / "prov.db", artifact_dir=tmp_path / "art")


def test_init_marimo_warns_when_ast_patch_never_ran(storage, monkeypatch, tmp_path):
    monkeypatch.setattr(kernel_hooks, "_MARIMO_PATCHED", False)

    with pytest.warns(UserWarning, match="patch_marimo_ast_compile"):
        rt = kernel_hooks.init_marimo(
            db_path=tmp_path / "prov.db",
            artifact_dir=tmp_path / "art",
            history_name="warn-test",
        )

    # The tracker itself must still be fully usable — this is a warning, not
    # a hard failure; existing single-process/script usage of init_marimo()
    # (without marimo at all) must keep working unchanged.
    assert rt is not None
    assert rt._history.history_id is not None


def test_init_marimo_silent_when_ast_patch_already_ran(storage, monkeypatch, tmp_path, recwarn):
    monkeypatch.setattr(kernel_hooks, "_MARIMO_PATCHED", True)

    kernel_hooks.init_marimo(
        db_path=tmp_path / "prov2.db",
        artifact_dir=tmp_path / "art2",
        history_name="no-warn-test",
    )

    assert not any("patch_marimo_ast_compile" in str(w.message) for w in recwarn.list)
