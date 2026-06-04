import pytest
from tracker.storage import StorageBackend

@pytest.fixture
def backend(tmp_path):
    return StorageBackend(db_path=tmp_path / "test.db", artifact_dir=tmp_path / "artifacts")

def _table_names(backend):
    try:
        import duckdb
        con = duckdb.connect(str(backend.db_path), read_only=True)
        rows = con.execute("SHOW TABLES").fetchall()
        con.close()
        return {r[0] for r in rows}
    except Exception:
        import sqlite3
        con = sqlite3.connect(str(backend.db_path))
        rows = con.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
        con.close()
        return {r[0] for r in rows}

def test_schema_creates_required_tables(backend):
    tables = _table_names(backend)
    required = {
        "analysis_histories", "analysis_branches", "analysis_states",
        "analysis_steps", "agents", "runtime_environments",
        "operations", "operation_types", "parameter_values",
        "artifacts", "artifact_states", "deltas",
        "pipelines", "pipeline_fragments",
    }
    assert required.issubset(tables), f"Missing tables: {required - tables}"

def test_no_nodes_or_edges_tables(backend):
    tables = _table_names(backend)
    assert "nodes" not in tables
    assert "edges" not in tables

def test_load_graph_returns_states_and_steps_keys(backend):
    graph = backend.load_graph()
    assert "states" in graph and "steps" in graph
    assert "nodes" not in graph and "edges" not in graph
