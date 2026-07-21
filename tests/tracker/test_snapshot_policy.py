import pytest
from tracker.snapshot_policy import snapshot_policy, snapshot_policy_for_type, should_snapshot


def test_should_snapshot_defaults_to_true_when_unconfigured():
    assert should_snapshot("totally_unconfigured_func_abc", "unknown") is True


def test_snapshot_policy_never_overrides_default():
    snapshot_policy("cheap_op_xyz", "never")
    assert should_snapshot("cheap_op_xyz", "unknown") is False


def test_snapshot_policy_always_is_explicit_default():
    snapshot_policy("explicit_always_op", "always")
    assert should_snapshot("explicit_always_op", "unknown") is True


def test_snapshot_policy_matches_trailing_dotted_segment():
    snapshot_policy("assign_no_snapshot", "never")
    assert should_snapshot("df.assign_no_snapshot", "unknown") is False


def test_snapshot_policy_for_type_sets_type_level_default():
    snapshot_policy_for_type("cheap_type_xyz", "never")
    assert should_snapshot("some_func_using_cheap_type", "cheap_type_xyz") is False


def test_func_name_override_wins_over_type_default():
    snapshot_policy_for_type("mixed_type_xyz", "never")
    snapshot_policy("important_func_xyz", "always")
    assert should_snapshot("important_func_xyz", "mixed_type_xyz") is True


def test_snapshot_policy_rejects_invalid_mode():
    with pytest.raises(ValueError):
        snapshot_policy("some_func", "sometimes")


def test_snapshot_policy_for_type_rejects_invalid_mode():
    with pytest.raises(ValueError):
        snapshot_policy_for_type("some_type", "sometimes")


import pandas as pd
from tracker.storage import DuckDBSQLiteBackend as StorageBackend
from tracker.runtime import RuntimeTracker


@pytest.fixture
def rt(tmp_path):
    s = StorageBackend(db_path=tmp_path / "prov.db", artifact_dir=tmp_path / "art")
    return RuntimeTracker(storage=s, session_id="t", history_name="test")


@pytest.fixture
def event_log():
    return pd.DataFrame({"case:concept:name": ["A1", "A1"], "concept:name": ["a", "b"]})


def settle(rt):
    rt.storage._executor.submit(lambda: None).result()


def test_never_policy_skips_artifact_but_still_records_step(rt, event_log):
    snapshot_policy("no_snapshot_step", "never")

    output = rt.trace_step(
        func=lambda df: df.assign(x=1), func_name="no_snapshot_step",
        raw_line="df=no_snapshot_step(df)", args=[event_log], kwargs={},
    )
    state_id = rt._current_state_id
    settle(rt)

    assert output is not None  # user code result is unaffected

    detail = rt.describe_state(state_id)
    assert detail != {}  # AnalysisStep/AnalysisState were still recorded
    assert detail["func_name"] == "no_snapshot_step"

    assert rt.storage.load_output_artifact_state_id(state_id) is None


def test_default_policy_still_snapshots(rt, event_log):
    output = rt.trace_step(
        func=lambda df: df.assign(x=1), func_name="default_snapshot_step",
        raw_line="df=default_snapshot_step(df)", args=[event_log], kwargs={},
    )
    state_id = rt._current_state_id
    settle(rt)

    assert output is not None
    assert rt.storage.load_output_artifact_state_id(state_id) is not None
