"""
Regression coverage for tracker/runtime.py::_capture_env().

_capture_env() probes a fixed list of libraries (pandas, polars, numpy, pm4py,
pydantic, duckdb) purely to record their versions in RuntimeEnvironment. A prior
version of this code only caught ImportError around that probe, so a library whose
*import chain* raised something else (observed live: pm4py's import triggers a
psutil.Process(parent_pid) lookup that raises psutil.NoSuchProcess in a sandboxed
session) crashed RuntimeTracker.__init__ / init_marimo() entirely, before any
tracking could happen at all. This file pins that fix: env capture must be
best-effort per-library, never fatal to tracker construction.
"""
from __future__ import annotations

import builtins
import sys

import pandas as pd
import pytest

from tracker.runtime import RuntimeTracker
from tracker.storage import DuckDBSQLiteBackend as StorageBackend


@pytest.fixture
def storage(tmp_path):
    return StorageBackend(db_path=tmp_path / "prov.db", artifact_dir=tmp_path / "art")


def _make_failing_import(failing_lib: str, exc: Exception):
    """Return a stand-in for builtins.__import__ that raises `exc` for
    `failing_lib` and delegates to the real import for everything else."""
    real_import = builtins.__import__

    def _fake_import(name, *args, **kwargs):
        if name == failing_lib:
            raise exc
        return real_import(name, *args, **kwargs)

    return _fake_import


@pytest.mark.parametrize(
    "exc",
    [
        RuntimeError("psutil.NoSuchProcess: process PID not found (pid=45504)"),
        OSError("simulated unrelated OS-level import failure"),
        ValueError("simulated arbitrary non-ImportError failure"),
    ],
    ids=["runtime-error", "os-error", "value-error"],
)
def test_capture_env_survives_non_import_error(storage, monkeypatch, exc):
    """A probed library failing with something other than ImportError must not
    prevent RuntimeTracker construction — the failure is swallowed and that
    library is simply absent from library_versions."""
    failing_lib = "duckdb"
    monkeypatch.delitem(sys.modules, failing_lib, raising=False)
    monkeypatch.setattr(builtins, "__import__", _make_failing_import(failing_lib, exc))

    rt = RuntimeTracker(storage=storage, session_id="t", history_name="test")

    assert failing_lib not in rt._env.library_versions


def test_capture_env_still_records_working_libraries(storage, monkeypatch):
    """A single failing probe must not take down the others — pandas (already
    imported by this test module) should still show up."""
    failing_lib = "polars"
    monkeypatch.delitem(sys.modules, failing_lib, raising=False)
    monkeypatch.setattr(
        builtins, "__import__", _make_failing_import(failing_lib, RuntimeError("boom"))
    )

    rt = RuntimeTracker(storage=storage, session_id="t", history_name="test")

    assert failing_lib not in rt._env.library_versions
    assert rt._env.library_versions.get("pandas") == pd.__version__


def test_capture_env_still_raises_for_genuinely_missing_library(storage, monkeypatch):
    """ImportError (the "not installed" case) must still be swallowed exactly as
    before — this isn't new behavior, just confirming the widened except clause
    didn't accidentally start propagating the common case instead."""
    failing_lib = "pydantic"
    monkeypatch.delitem(sys.modules, failing_lib, raising=False)
    monkeypatch.setattr(
        builtins, "__import__", _make_failing_import(failing_lib, ImportError("no module"))
    )

    rt = RuntimeTracker(storage=storage, session_id="t", history_name="test")

    assert failing_lib not in rt._env.library_versions
