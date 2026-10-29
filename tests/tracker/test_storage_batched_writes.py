"""
The async writer batches its work.

Every artifact record, state binding and parameter used to be written on a DuckDB connection of its own
(connect, insert, close, under a process-wide lock). A notebook that tracks a few hundred steps queues tens of
thousands of such writes, and at ~100 ms each the queue took an hour to drain, which every synchronous write
(annotations, tags) and every `settle()` had to wait for. The writer now runs consecutive tasks on one connection
and writes artifact records and bindings with multi-row statements.
"""
import subprocess
import sys
import textwrap
import threading
import time
import uuid
from enum import Enum
from pathlib import Path
from types import SimpleNamespace

import pytest

import tracker.storage as storage_module
from tracker.storage import DuckDBSQLiteBackend

duckdb_only = pytest.mark.skipif(storage_module._BACKEND != "duckdb", reason="counts DuckDB connections")


class _Kind(Enum):
    DATAFRAME = "dataframe"


def _record(i: int, size=100):
    artifact = SimpleNamespace(artifact_id=f"art-{i % 7}", name=f"name-{i % 7}", artifact_type=_Kind.DATAFRAME)
    state = SimpleNamespace(
        artifact_state_id=f"as-{i}", artifact_id=artifact.artifact_id, analysis_state_id=f"s-{i % 11}",
        mime_type="application/parquet", checksum="c", content_ref="ref", size_bytes=size,
    )
    return artifact, state


def _backend(tmp_path) -> DuckDBSQLiteBackend:
    return DuckDBSQLiteBackend(db_path=tmp_path / "prov.db", artifact_dir=tmp_path / "art")


def _settle(storage) -> None:
    storage._executor.submit(lambda: None).result()


def _count(storage, table: str) -> int:
    con = storage._connect()
    try:
        return con.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
    finally:
        con.close()


@duckdb_only
def test_a_backlog_of_writes_uses_a_handful_of_connections(tmp_path, monkeypatch):
    storage = _backend(tmp_path)
    opened = []
    real_connect = storage_module._duckdb.connect
    monkeypatch.setattr(storage_module._duckdb, "connect", lambda *a, **k: opened.append(1) or real_connect(*a, **k))

    gate = threading.Event()
    storage._executor.submit(gate.wait)  # hold the writer so the whole backlog is queued before it starts
    for i in range(400):
        storage.save_artifact_records_async(*_record(i), "h")
    for i in range(50):
        storage.save_state_bindings_async(f"s-{i}", {"a": "as-1", "b": "as-2", "c": "as-3"})
    gate.set()
    _settle(storage)

    assert _count(storage, "artifact_states") == 400
    assert _count(storage, "artifacts") == 7
    assert _count(storage, "state_bindings") == 150
    assert len(opened) <= 20, f"{len(opened)} connections for 450 writes"


def test_records_queued_before_a_settle_are_visible_after_it(tmp_path):
    storage = _backend(tmp_path)
    for round_ in range(5):
        for i in range(round_ * 20, round_ * 20 + 20):
            storage.save_artifact_records_async(*_record(i), "h")
        _settle(storage)
        assert _count(storage, "artifact_states") == (round_ + 1) * 20


def test_a_failing_write_does_not_stop_the_writes_around_it(tmp_path):
    storage = _backend(tmp_path)

    def boom():
        raise RuntimeError("boom")

    storage.save_artifact_records_async(*_record(1), "h")
    failed = storage._executor.submit(boom)
    storage.save_artifact_records_async(*_record(2), "h")
    _settle(storage)

    assert isinstance(failed.exception(), RuntimeError)
    assert _count(storage, "artifact_states") == 2


def test_one_bad_record_does_not_lose_the_records_batched_with_it(tmp_path):
    storage = _backend(tmp_path)
    gate = threading.Event()
    storage._executor.submit(gate.wait)
    for i in range(10):
        storage.save_artifact_records_async(*_record(i, size=None if i == 4 else 100), "h")
    gate.set()
    _settle(storage)

    assert _count(storage, "artifact_states") == 9


def test_the_same_record_queued_twice_is_stored_once(tmp_path):
    storage = _backend(tmp_path)
    gate = threading.Event()
    storage._executor.submit(gate.wait)
    for _ in range(3):
        storage.save_artifact_records_async(*_record(1), "h")
    gate.set()
    _settle(storage)

    assert _count(storage, "artifact_states") == 1


def test_a_synchronous_write_is_not_stuck_behind_a_long_backlog(tmp_path):
    storage = _backend(tmp_path)
    gate = threading.Event()
    storage._executor.submit(gate.wait)
    for i in range(3000):
        storage.save_artifact_records_async(*_record(i), "h")
    gate.set()

    began = time.monotonic()
    storage.save_tag_sync("urgent")  # waits in the same queue as the 3,000 records
    assert time.monotonic() - began < 10


def test_reads_are_not_starved_while_the_writer_works_through_a_backlog(tmp_path):
    storage = _backend(tmp_path)
    gate = threading.Event()
    storage._executor.submit(gate.wait)
    for i in range(3000):
        storage.save_artifact_records_async(*_record(i), "h")
        storage.save_state_bindings_async(f"s-{i}", {"a": f"as-{i}"})
    gate.set()

    longest = 0.0
    for _ in range(30):
        began = time.monotonic()
        storage.load_graph(history_id="nope")
        longest = max(longest, time.monotonic() - began)
    assert longest < 2, f"a read waited {longest:.2f}s for the writer"
    _settle(storage)
    assert _count(storage, "artifact_states") == 3000


def test_writes_still_queued_when_the_process_exits_are_not_lost(tmp_path):
    root = Path(__file__).resolve().parents[2]
    script = textwrap.dedent(
        f"""
        import sys
        sys.path.insert(0, {str(root)!r})
        from types import SimpleNamespace
        from enum import Enum
        from tracker.storage import DuckDBSQLiteBackend

        class K(Enum):
            D = "d"

        storage = DuckDBSQLiteBackend(db_path={str(tmp_path / "exit.db")!r}, artifact_dir={str(tmp_path / "art2")!r})
        for i in range(200):
            a = SimpleNamespace(artifact_id="a", name="n", artifact_type=K.D)
            s = SimpleNamespace(artifact_state_id=f"as-{{i}}", artifact_id="a", analysis_state_id="s",
                                mime_type="m", checksum="c", content_ref="r", size_bytes=1)
            storage.save_artifact_records_async(a, s, "h")
        # no settle, no close: the interpreter exits with the writes still queued
        """
    )
    subprocess.run([sys.executable, "-c", script], check=True, timeout=120)

    storage = DuckDBSQLiteBackend(db_path=tmp_path / "exit.db", artifact_dir=tmp_path / "art2")
    assert _count(storage, "artifact_states") == 200
