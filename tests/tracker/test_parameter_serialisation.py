"""A parameter that cannot be JSON-encoded must not drop the step's other parameters."""
import pandas as pd
import pytest

from tracker.runtime import RuntimeTracker
from tracker.storage import DuckDBSQLiteBackend


@pytest.fixture
def rt(tmp_path):
    storage = DuckDBSQLiteBackend(db_path=tmp_path / "p.db", artifact_dir=tmp_path / "a")
    return RuntimeTracker(storage=storage, session_id="s", history_name="h")


def _read_csv(path, parse_dates=None, dtype=None):
    return pd.DataFrame({"a": ["x"]})


def test_a_type_valued_dict_parameter_is_recorded_with_its_siblings(rt, caplog):
    rt.trace_step(func=_read_csv, func_name="read_csv", raw_line="df = read_csv('f', parse_dates=['t'], dtype={'a': str})",
                  args=["f"], kwargs={"parse_dates": ["t"], "dtype": {"a": str}})
    rt.storage._executor.submit(lambda: None).result()

    state = rt._current_state_id
    params = {p["param_id"]: p for p in rt.storage.load_state_detail(state)["params"]}

    assert set(params) == {"read_csv:path", "read_csv:parse_dates", "read_csv:dtype"}
    assert "serialis" not in caplog.text.lower()
