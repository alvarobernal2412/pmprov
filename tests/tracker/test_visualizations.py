import pytest
import pandas as pd


@pytest.fixture
def rt(tmp_path):
    from tracker.storage import StorageBackend
    from tracker.runtime import RuntimeTracker
    s = StorageBackend(db_path=tmp_path / "prov.db", artifact_dir=tmp_path / "art")
    tracker = RuntimeTracker(storage=s, session_id="viz")
    df = pd.DataFrame({"a": [1, 2, 3]})
    tracker.trace_step(
        func=lambda d: d,
        func_name="identity",
        raw_line="df = identity(df)",
        args=[df],
        kwargs={},
    )
    tracker.storage._executor.submit(lambda: None).result()
    return tracker


def test_list_states_returns_dataframe(rt):
    result = rt.list_states()
    assert hasattr(result, "columns"), "list_states() must return a pandas DataFrame"
    required = {"state_id", "timestamp", "branch_name", "produced_by_step_id",
                "derived_from_state_id", "artifact_state_ids"}
    assert required.issubset(set(result.columns)), f"Missing columns: {required - set(result.columns)}"


def test_list_states_has_rows(rt):
    result = rt.list_states()
    assert len(result) >= 1, "list_states() must have at least 1 row after one trace_step"


def test_show_graph_does_not_raise(rt, tmp_path):
    rt.show_graph(save_path=str(tmp_path / "graph.svg"))


def test_to_networkx_returns_digraph(rt):
    import networkx as nx
    G = rt.storage.to_networkx()
    assert isinstance(G, nx.DiGraph)
    assert G.number_of_nodes() >= 1
