import pandas as pd
import pytest
from tracker.storage import DuckDBSQLiteBackend as StorageBackend
from tracker.runtime import RuntimeTracker


@pytest.fixture
def storage(tmp_path):
    return StorageBackend(db_path=tmp_path / "prov.db", artifact_dir=tmp_path / "art")


def settle(rt):
    rt.storage._executor.submit(lambda: None).result()


def test_find_latest_history_by_name_returns_none_when_no_match(storage):
    assert storage.find_latest_history_by_name("nope") is None


def test_find_latest_history_by_name_returns_the_only_match(storage):
    rt = RuntimeTracker(storage=storage, session_id="s1", history_name="my analysis")
    rt.trace_step(func=lambda: pd.DataFrame({"a": [1]}), func_name="load",
                  raw_line="load()", args=[], kwargs={})
    settle(rt)

    found = storage.find_latest_history_by_name("my analysis")
    assert found == rt._history.history_id


def test_find_latest_history_by_name_picks_most_recently_touched(storage):
    rt_old = RuntimeTracker(storage=storage, session_id="s1", history_name="dup")
    rt_old.trace_step(func=lambda: pd.DataFrame({"a": [1]}), func_name="load",
                       raw_line="load()", args=[], kwargs={})
    settle(rt_old)

    rt_new = RuntimeTracker(storage=storage, session_id="s2", history_name="dup")
    rt_new.trace_step(func=lambda: pd.DataFrame({"a": [2]}), func_name="load",
                       raw_line="load()", args=[], kwargs={})
    settle(rt_new)

    # Touch the OLD history again, after the new one -- it should now win.
    rt_old.trace_step(func=lambda: pd.DataFrame({"a": [3]}), func_name="load2",
                       raw_line="load2()", args=[], kwargs={})
    settle(rt_old)

    found = storage.find_latest_history_by_name("dup")
    assert found == rt_old._history.history_id


def test_load_history_returns_none_for_unknown_id(storage):
    assert storage.load_history("does-not-exist") is None


def test_load_history_returns_name_and_active_state(storage):
    rt = RuntimeTracker(storage=storage, session_id="s1", history_name="my analysis")
    rt.trace_step(func=lambda: pd.DataFrame({"a": [1]}), func_name="load",
                   raw_line="load()", args=[], kwargs={})
    settle(rt)

    row = storage.load_history(rt._history.history_id)
    assert row == {
        "history_id": rt._history.history_id,
        "name": "my analysis",
        "active_state_id": rt._current_state_id,
    }


def test_load_history_active_state_id_is_none_before_any_step(storage):
    rt = RuntimeTracker(storage=storage, session_id="s1", history_name="empty")
    row = storage.load_history(rt._history.history_id)
    assert row["active_state_id"] is None
