"""Tests for StepCategory registration, storage, and retrieval via load_state_detail."""
import pytest
import pandas as pd
from tracker.storage import DuckDBSQLiteBackend as StorageBackend
from tracker.runtime import RuntimeTracker
from tracker.operation_registry import step_category, lookup_category


@pytest.fixture
def rt(tmp_path):
    s = StorageBackend(db_path=tmp_path / "prov.db", artifact_dir=tmp_path / "art")
    return RuntimeTracker(storage=s, session_id="t", history_name="test")


@pytest.fixture
def event_log():
    return pd.DataFrame({
        "case:concept:name": ["A1", "A2"],
        "concept:name": ["Create Fine", "Create Fine"],
        "time:timestamp": pd.to_datetime(["2020-01-01", "2020-01-10"]),
        "org:resource": ["r1", "r1"],
    })


def settle(rt):
    rt.storage._executor.submit(lambda: None).result()


def test_lookup_category_returns_registered_name():
    step_category("log_enrichment", "attribute_derivation")
    assert lookup_category("attribute_derivation") == "log_enrichment"


def test_lookup_category_returns_none_when_unregistered():
    assert lookup_category("nonexistent_type_xyz") is None


def test_step_category_stored_on_operation(rt, event_log):
    from tracker.operation_registry import step_category, _registry
    step_category("log_enrichment", "attribute_derivation")
    _registry["enrich_step"] = "attribute_derivation"

    rt.trace_step(func=lambda df: df.assign(x=1), func_name="enrich_step",
                  raw_line="df=enrich_step(df)", args=[event_log], kwargs={})
    settle(rt)

    con = rt.storage._connect(read_only=True)
    cats = con.execute("SELECT name FROM step_categories").fetchall()
    con.close()
    assert any(r[0] == "log_enrichment" for r in cats)


def test_step_category_exposed_in_load_state_detail(rt, event_log):
    from tracker.operation_registry import step_category, _registry
    step_category("conformance_checking", "conformance_check")
    _registry["check_step"] = "conformance_check"

    rt.trace_step(func=lambda df: df.assign(ok=True), func_name="check_step",
                  raw_line="df=check_step(df)", args=[event_log], kwargs={})
    settle(rt)

    detail = rt.storage.load_state_detail(rt._current_state_id)
    assert detail["operation"]["category"] == "conformance_checking"

