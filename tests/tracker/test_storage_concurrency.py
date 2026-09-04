"""
Regression test for a DuckDB connection-config race: concurrent read_only=True
and read_only=False connections to the same file raise
"Connection Error: Can't open a connection to same database file with a
different configuration than existing connections" even when both connections
are always closed promptly. This is a DuckDB-level restriction, not a leak —
verified by a minimal, non-project reproduction during the fix's investigation
(see docs/superpowers/plans/2026-07-21-ci-e2e-notebook-tests.md's Task 0).

Triggered in practice by tracker/storage.py's per-call connect/close pattern:
the single-worker async write executor (RuntimeTracker.trace_step's async
saves) opens read_only=False connections on a background thread while
synchronous read paths (list_states, describe_state, show_graph, ...) open
read_only=True connections on the caller's thread — genuinely concurrent in
notebook environments (e.g. Marimo's reactive scheduler can run independent
cells in parallel), not just under artificial stress.
"""
import threading

import pandas as pd
import pytest

from tracker.storage import DuckDBSQLiteBackend as StorageBackend
from tracker.runtime import RuntimeTracker


def test_concurrent_reads_and_writes_do_not_race_on_connection_config(tmp_path):
    storage = StorageBackend(db_path=tmp_path / "prov.db", artifact_dir=tmp_path / "art")
    rt = RuntimeTracker(storage=storage, session_id="t", history_name="concurrency-test")

    event_log = pd.DataFrame({"case:concept:name": ["A1", "A2"], "concept:name": ["a", "b"]})
    errors: list[Exception] = []
    stop = threading.Event()

    def writer():
        df = event_log
        while not stop.is_set():
            try:
                df = rt.trace_step(
                    func=lambda d: d.assign(x=len(d)),
                    func_name="assign", raw_line="df=assign(df)",
                    args=[df], kwargs={},
                )
            except Exception as e:  # pragma: no cover - failure path under test
                errors.append(e)
                return

    def reader():
        while not stop.is_set():
            try:
                storage.load_graph(history_id=rt._history.history_id)
            except Exception as e:  # pragma: no cover - failure path under test
                errors.append(e)
                return

    writer_thread = threading.Thread(target=writer)
    reader_threads = [threading.Thread(target=reader) for _ in range(3)]

    writer_thread.start()
    for t in reader_threads:
        t.start()

    # Let both sides race against the same DB file for a short, bounded window.
    threading.Event().wait(0.5)
    stop.set()

    writer_thread.join(timeout=5)
    for t in reader_threads:
        t.join(timeout=5)

    assert errors == [], f"Concurrent read/write connections raced: {errors[0]}"


def test_connect_lock_is_released_when_connect_itself_raises(tmp_path, monkeypatch):
    """
    A failed duckdb.connect() (e.g. corrupt/locked file, disk error) must not
    leave _connect_lock held — otherwise every later _connect() call in the
    process deadlocks forever, since nothing else would ever release it.
    """
    import tracker.storage as storage_module

    storage = StorageBackend(db_path=tmp_path / "prov.db", artifact_dir=tmp_path / "art")

    # Access the appropriate connection module based on what backend is being used
    if storage_module._BACKEND == "duckdb":
        import duckdb as _duckdb
    else:
        import sqlite3 as _duckdb

    real_connect = _duckdb.connect
    should_fail = {"value": True}

    def _flaky_connect(*args, **kwargs):
        if should_fail["value"]:
            should_fail["value"] = False
            raise RuntimeError("simulated connect failure")
        return real_connect(*args, **kwargs)

    monkeypatch.setattr(_duckdb, "connect", _flaky_connect)

    with pytest.raises(RuntimeError, match="simulated connect failure"):
        storage._connect()

    # If the lock leaked, this call hangs forever — bound it with a thread + join
    # timeout so the test fails loudly instead of hanging the whole suite.
    result: dict = {}

    def _try_reconnect():
        con = storage._connect()
        result["ok"] = True
        con.close()

    t = threading.Thread(target=_try_reconnect, daemon=True)
    t.start()
    t.join(timeout=5)

    assert not t.is_alive(), "storage._connect() hung — _connect_lock was leaked"
    assert result.get("ok") is True
