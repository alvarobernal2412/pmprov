import os
import json
import pandas as pd
import pytest
from tracker.storage import DuckDBSQLiteBackend


def _make_backend(tmp_path):
    db = str(tmp_path / "test.db")
    art = str(tmp_path / "artifacts")
    os.makedirs(art, exist_ok=True)
    return DuckDBSQLiteBackend(db_path=db, artifact_dir=art)


def test_filter_index_writes_json_not_parquet(tmp_path):
    backend = _make_backend(tmp_path)
    df = pd.DataFrame({"a": [1, 2, 3]}, index=["x", "y", "z"])
    path = backend.save_artifact("s1", df, kind="filter_index", parent_state_id="parent-001")
    art = str(tmp_path / "artifacts")
    assert os.path.isfile(os.path.join(art, "s1.filter.json"))
    assert not os.path.isfile(os.path.join(art, "s1.parquet"))
    assert path.endswith(".filter.json")


def test_filter_index_json_content(tmp_path):
    backend = _make_backend(tmp_path)
    df = pd.DataFrame({"v": [9, 8]}, index=["alpha", "beta"])
    backend.save_artifact("s2", df, kind="filter_index", parent_state_id="parent-abc")
    with open(str(tmp_path / "artifacts" / "s2.filter.json")) as f:
        data = json.load(f)
    assert data["parent_state_id"] == "parent-abc"
    assert data["index"] == ["alpha", "beta"]


def test_dataframe_no_index_companion(tmp_path):
    backend = _make_backend(tmp_path)
    df = pd.DataFrame({"a": [1, 2]})
    backend.save_artifact("s3", df, kind="dataframe")
    art = str(tmp_path / "artifacts")
    assert os.path.isfile(os.path.join(art, "s3.parquet"))
    assert not os.path.isfile(os.path.join(art, "s3.filter.json"))


def test_load_artifact_index_returns_list(tmp_path):
    backend = _make_backend(tmp_path)
    df = pd.DataFrame({"v": [9, 8, 7]}, index=[100, 200, 300])
    backend.save_artifact("s4", df, kind="filter_index", parent_state_id="p")
    result = backend.load_artifact_index("s4")
    assert result == [100, 200, 300]


def test_load_artifact_index_missing_returns_none(tmp_path):
    backend = _make_backend(tmp_path)
    assert backend.load_artifact_index("no-such-state") is None


def test_load_artifact_reconstructs_filter_from_parent(tmp_path):
    """
    Scenario: step S1 produces a 3-row DataFrame (parent_state).
              step S2 filters it to 2 rows (child_state, stored as filter_index).
    load_artifact for child_state must reconstruct the 2-row DataFrame by
    loading the parent Parquet and applying the stored index.
    """
    import uuid, hashlib
    backend = _make_backend(tmp_path)

    parent_df = pd.DataFrame({"val": [10, 20, 30]}, index=["a", "b", "c"])
    child_df = parent_df.loc[["a", "c"]]

    parent_state_id = str(uuid.uuid4())
    child_state_id = str(uuid.uuid4())
    history_id = str(uuid.uuid4())

    # Save parent as normal Parquet artifact
    parent_path = backend.save_artifact(parent_state_id, parent_df, kind="dataframe")
    assert parent_path is not None

    # Insert DB records for parent artifact_state
    checksum = hashlib.sha256(open(parent_path, "rb").read()).hexdigest()
    size = os.path.getsize(parent_path)
    parent_artifact_state_id = str(uuid.uuid4())
    parent_artifact_id = str(uuid.uuid4())
    con = backend._connect()
    con.execute("INSERT INTO artifacts VALUES (?,?,?,?)",
                (parent_artifact_id, history_id, "parent_df", "other"))
    con.execute("INSERT INTO artifact_states VALUES (?,?,?,?,?,?,?)",
                (parent_artifact_state_id, parent_artifact_id, parent_state_id,
                 "application/vnd.apache.parquet", checksum, parent_path, size))
    con.commit()
    con.close()

    # Save child as filter_index artifact
    child_path = backend.save_artifact(child_state_id, child_df,
                                       kind="filter_index", parent_state_id=parent_state_id)
    assert child_path is not None

    # Insert DB records for child artifact_state
    child_artifact_state_id = str(uuid.uuid4())
    child_artifact_id = str(uuid.uuid4())
    con = backend._connect()
    con.execute("INSERT INTO artifacts VALUES (?,?,?,?)",
                (child_artifact_id, history_id, "child_df", "other"))
    con.execute("INSERT INTO artifact_states VALUES (?,?,?,?,?,?,?)",
                (child_artifact_state_id, child_artifact_id, child_state_id,
                 "application/x-pmprov-filter-index+json", "na", child_path, 0))
    con.commit()
    con.close()

    # Reconstruct
    result = backend.load_artifact(child_artifact_state_id)
    assert result is not None
    assert list(result.index) == ["a", "c"]
    assert list(result["val"]) == [10, 30]


def test_trace_step_detects_filter_writes_index_only(tmp_path):
    """
    A step whose output has the same columns + fewer rows than its input
    must produce a .filter.json artifact and NO .parquet artifact.
    """
    import os
    from tracker.runtime import RuntimeTracker
    from tracker.storage import DuckDBSQLiteBackend

    db = str(tmp_path / "prov.db")
    art = str(tmp_path / "artifacts")
    os.makedirs(art)
    backend = DuckDBSQLiteBackend(db_path=db, artifact_dir=art)
    rt = RuntimeTracker(storage=backend, session_id="test-session")

    parent_df = pd.DataFrame({"v": [1, 2, 3]}, index=["a", "b", "c"])

    # Step 1: non-filter — should write Parquet
    parent_result = rt.trace_step(
        func=lambda: parent_df,
        args=[], kwargs={},
        func_name="load_data",
        raw_line="df = load_data()",
    )

    # Step 2: row filter — same columns, fewer rows — should write .filter.json, NOT .parquet
    child_result = rt.trace_step(
        func=lambda df: df.loc[["a", "c"]],
        args=[parent_result], kwargs={},
        func_name="filter_rows",
        raw_line="filtered = filter_rows(df)",
    )

    art_files = os.listdir(art)
    filter_files = [f for f in art_files if f.endswith(".filter.json")]
    parquet_files = [f for f in art_files if f.endswith(".parquet")]

    assert len(parquet_files) == 1, f"Expected 1 parquet (parent only), got: {parquet_files}"
    assert len(filter_files) == 1, f"Expected 1 filter.json (child only), got: {filter_files}"
    assert list(child_result.index) == ["a", "c"]
