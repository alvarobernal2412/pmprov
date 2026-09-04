import pandas as pd
import pytest
from tracker.storage import DuckDBSQLiteBackend as StorageBackend
from tracker.runtime import RuntimeTracker


@pytest.fixture
def storage(tmp_path):
    return StorageBackend(db_path=tmp_path / "prov.db", artifact_dir=tmp_path / "art")


def settle(rt):
    rt.storage._executor.submit(lambda: None).result()


def _init_marimo_against(storage, **kwargs):
    # init_marimo() always builds its own DuckDBSQLiteBackend from db_path/
    # artifact_dir; tests need to target the SAME on-disk db as the fixture,
    # so reuse its paths rather than calling init_marimo with a bare name.
    from tracker.kernel_hooks import init_marimo
    return init_marimo(
        db_path=storage.db_path, artifact_dir=storage.artifact_dir, **kwargs
    )


def test_no_history_name_always_creates_fresh(storage):
    rt1 = _init_marimo_against(storage, history_name=None)
    rt2 = _init_marimo_against(storage, history_name=None)
    assert rt1._history.history_id != rt2._history.history_id


def test_first_call_with_a_name_creates(storage):
    rt = _init_marimo_against(storage, history_name="h")
    assert rt._history.name == "h"


def test_second_call_with_same_name_resumes_by_default(storage):
    rt1 = _init_marimo_against(storage, history_name="h")
    rt1.trace_step(func=lambda: pd.DataFrame({"a": [1]}), func_name="load",
                    raw_line="load()", args=[], kwargs={})
    settle(rt1)

    rt2 = _init_marimo_against(storage, history_name="h")

    assert rt2._history.history_id == rt1._history.history_id
    assert rt2._current_state_id == rt1._current_state_id


def test_fresh_true_ignores_existing_history(storage):
    rt1 = _init_marimo_against(storage, history_name="h")
    rt1.trace_step(func=lambda: pd.DataFrame({"a": [1]}), func_name="load",
                    raw_line="load()", args=[], kwargs={})
    settle(rt1)

    rt2 = _init_marimo_against(storage, history_name="h", fresh=True)

    assert rt2._history.history_id != rt1._history.history_id
    assert rt2._history.name == "h"
