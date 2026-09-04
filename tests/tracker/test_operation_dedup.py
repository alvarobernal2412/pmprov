import pandas as pd
import pytest
from tracker.storage import DuckDBSQLiteBackend as StorageBackend
from tracker.runtime import RuntimeTracker


@pytest.fixture
def storage(tmp_path):
    return StorageBackend(db_path=tmp_path / "prov.db", artifact_dir=tmp_path / "art")


def settle(rt):
    rt.storage._executor.submit(lambda: None).result()


def test_same_func_name_reuses_operation_id_across_sessions(storage):
    session1 = RuntimeTracker(storage=storage, session_id="s1", history_name="h1")
    session1.trace_step(func=lambda: pd.DataFrame({"a": [1]}), func_name="load",
                         raw_line="load()", args=[], kwargs={})
    settle(session1)
    op_id_1 = session1._operation_cache["load"][0].operation_id

    # A different history entirely -- Operations are global, not per-history.
    session2 = RuntimeTracker(storage=storage, session_id="s2", history_name="h2")
    session2.trace_step(func=lambda: pd.DataFrame({"a": [2]}), func_name="load",
                         raw_line="load()", args=[], kwargs={})
    settle(session2)
    op_id_2 = session2._operation_cache["load"][0].operation_id

    assert op_id_2 == op_id_1

    con = storage._connect(read_only=True)
    try:
        count = con.execute(
            "SELECT COUNT(*) FROM operations WHERE name = ?", ("load",)
        ).fetchone()[0]
    finally:
        con.close()
    assert count == 1


def test_different_func_names_get_different_operation_ids(storage):
    rt = RuntimeTracker(storage=storage, session_id="s1", history_name="h")
    rt.trace_step(func=lambda: pd.DataFrame({"a": [1]}), func_name="load",
                  raw_line="load()", args=[], kwargs={})
    rt.trace_step(func=lambda: pd.DataFrame({"b": [1]}), func_name="save",
                  raw_line="save()", args=[], kwargs={})
    settle(rt)

    assert (
        rt._operation_cache["load"][0].operation_id
        != rt._operation_cache["save"][0].operation_id
    )
