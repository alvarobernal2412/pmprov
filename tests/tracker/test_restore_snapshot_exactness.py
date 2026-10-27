"""Acceptance tests: restore_state() returns an exact, selective snapshot.

Copy to pmprov/tests/tracker/test_restore_snapshot_exactness.py.

Scenario: a notebook that re-runs part of its analysis, the way a marimo
notebook does when a filter control changes. Every step assigns a variable
from a call, and later steps derive from earlier variables:

    1  event_log  = load()
    2  by_activity = keep_activities(event_log, A,B,C)     # row filter
    3  since_year  = keep_since(by_activity, 2005)         # row filter
    4  case_log    = summarise(since_year)                 # aggregation
    5  by_activity = keep_activities(event_log, A,B,C,D)   # re-run: name re-assigned
    6  since_year  = keep_since(by_activity, 2005)          # re-run

"The state after step N" is an immutable snapshot of the whole analysis: each
variable holds exactly the value it was bound to at that point, no matter what
later steps did.
"""
import pandas as pd
import pytest

from tracker.runtime import RuntimeTracker
from tracker.storage import DuckDBSQLiteBackend

import tracker  # noqa: F401 - patches restore_state() etc. onto RuntimeTracker


def load():
    return pd.DataFrame(
        {
            "case": range(100),
            "activity": ["A", "B", "C", "D"] * 25,
            "year": [2000 + (i % 10) for i in range(100)],
        }
    )


def keep_activities(df, activities):
    return df[df["activity"].isin(activities)].copy()


def keep_since(df, year):
    return df[df["year"] >= year].copy()


def summarise(df):
    return df.groupby("activity").agg(events=("case", "count"))


ABC = ["A", "B", "C"]
ABCD = ["A", "B", "C", "D"]


@pytest.fixture
def notebook(tmp_path):
    """Runs the six steps; returns (rt, [state_id after each step], truth).

    `truth[i]` is the exact namespace the analyst had after step i + 1.
    """
    storage = DuckDBSQLiteBackend(db_path=tmp_path / "prov.db", artifact_dir=tmp_path / "art")
    rt = RuntimeTracker(storage=storage, session_id="s", history_name="notebook")

    def step(func, raw_line, *args):
        out = rt.trace_step(func=func, func_name=func.__name__, raw_line=raw_line, args=list(args), kwargs={})
        storage._executor.submit(lambda: None).result()  # let the async writer settle
        return out

    ns, states, truth = {}, [], []

    def record():
        states.append(rt._current_state_id)
        truth.append({k: v.copy() for k, v in ns.items()})

    ns["event_log"] = step(load, "event_log = load()")
    record()
    ns["by_activity"] = step(keep_activities, "by_activity = keep_activities(event_log, ABC)", ns["event_log"], ABC)
    record()
    ns["since_year"] = step(keep_since, "since_year = keep_since(by_activity, 2005)", ns["by_activity"], 2005)
    record()
    ns["case_log"] = step(summarise, "case_log = summarise(since_year)", ns["since_year"])
    record()
    ns["by_activity"] = step(keep_activities, "by_activity = keep_activities(event_log, ABCD)", ns["event_log"], ABCD)
    record()
    ns["since_year"] = step(keep_since, "since_year = keep_since(by_activity, 2005)", ns["by_activity"], 2005)
    record()
    return rt, states, truth


# ---- 1. exactness -----------------------------------------------------------

@pytest.mark.parametrize("step_index", range(6))
def test_restore_state_returns_every_variable_exactly_as_bound_at_that_state(notebook, step_index):
    rt, states, truth = notebook

    restored = rt.restore_state(states[step_index])

    expected = truth[step_index]
    assert set(restored.variables) == set(expected), "exactly the variables that existed then"
    for name, value in expected.items():
        pd.testing.assert_frame_equal(
            restored.variables[name], value, obj=f"{name} at step {step_index + 1}"
        )


def test_restoring_an_old_state_after_later_steps_is_unchanged(notebook):
    """Immutability: step 2's snapshot must not pick up the re-run in step 5."""
    rt, states, truth = notebook

    restored = rt.restore_state(states[1])

    assert len(restored.variables["by_activity"]) == 75  # A,B,C only, not the later A,B,C,D
    assert "since_year" not in restored.variables and "case_log" not in restored.variables


def test_a_variable_not_reassigned_keeps_its_original_value_in_every_later_state(notebook):
    rt, states, truth = notebook

    for state in states:
        pd.testing.assert_frame_equal(rt.restore_state(state).variables["event_log"], truth[0]["event_log"])


# ---- 2. selective restore ---------------------------------------------------

def test_restore_state_with_names_returns_only_those_variables(notebook):
    rt, states, truth = notebook

    restored = rt.restore_state(states[3], names=["case_log"])

    assert set(restored.variables) == {"case_log"}
    pd.testing.assert_frame_equal(restored.variables["case_log"], truth[3]["case_log"])


def test_restore_state_with_names_loads_only_the_requested_artifacts(notebook, monkeypatch):
    rt, states, truth = notebook
    loaded = []
    original = rt.storage.load_artifact
    monkeypatch.setattr(rt.storage, "load_artifact", lambda artifact_state_id: (loaded.append(artifact_state_id), original(artifact_state_id))[1])

    rt.restore_state(states[5], names=["since_year"])

    # one variable asked for: its own content, plus at most what rebuilding it
    # structurally requires (a stored row filter needs its parent) - never the
    # whole namespace of 3 unrelated artifacts.
    assert len(loaded) <= 2


def test_restore_state_with_an_unknown_name_is_reported_not_raised(notebook):
    rt, states, _ = notebook

    restored = rt.restore_state(states[0], names=["event_log", "case_log"])  # case_log does not exist yet

    assert set(restored.variables) == {"event_log"}
    assert [s["name"] for s in restored.skipped] == ["case_log"]


# ---- 3. bindings ------------------------------------------------------------

def test_state_records_which_variable_name_was_bound_to_which_artifact(notebook):
    """Names must not be re-derived by parsing raw_line at read time."""
    rt, states, _ = notebook

    bindings = rt.storage.load_state_bindings(states[5])

    assert set(bindings) == {"event_log", "by_activity", "since_year", "case_log"}
    assert len(set(bindings.values())) == 4, "four distinct artifact versions"
    assert bindings != rt.storage.load_state_bindings(states[1]), "step 2 and step 6 differ"
