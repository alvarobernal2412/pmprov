import pandas as pd
import pytest
from tracker.storage import DuckDBSQLiteBackend as StorageBackend
from tracker.runtime import RuntimeTracker

@pytest.fixture
def rt(tmp_path):
    s = StorageBackend(db_path=tmp_path / "prov.db", artifact_dir=tmp_path / "art")
    return RuntimeTracker(storage=s, session_id="integration", history_name="test")

def _settle(rt):
    rt.storage._executor.submit(lambda: None).result()

def test_trace_step_writes_state_and_step(rt):
    df = pd.DataFrame({"a": [1, 2, 3]})
    rt.trace_step(
        func=lambda x: x.assign(b=x["a"] * 2),
        func_name="df.assign",
        raw_line="df = df.assign(b=df['a']*2)",
        args=[df],
        kwargs={},
    )
    _settle(rt)
    graph = rt.storage.load_graph()
    assert len(graph["states"]) >= 1
    assert len(graph["steps"]) >= 1

def test_load_graph_keys(rt):
    graph = rt.storage.load_graph()
    assert "states" in graph and "steps" in graph
    assert "nodes" not in graph and "edges" not in graph

