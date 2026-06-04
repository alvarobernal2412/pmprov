"""Shared fixtures for acceptance-criteria tests."""
import pytest
import pandas as pd
from tracker.storage import StorageBackend
from tracker.runtime import RuntimeTracker


@pytest.fixture
def rt(tmp_path):
    s = StorageBackend(db_path=tmp_path / "prov.db", artifact_dir=tmp_path / "art")
    return RuntimeTracker(storage=s, session_id="ac-test", history_name="acceptance tests")


@pytest.fixture
def event_log():
    return pd.DataFrame({
        "case:concept:name": ["A1", "A1", "A2", "A2"],
        "concept:name": ["Create Fine", "Send Fine", "Create Fine", "Send Fine"],
        "time:timestamp": pd.to_datetime([
            "2020-01-01", "2020-04-15", "2020-01-10", "2020-02-01"
        ]),
        "org:resource": ["r1", "r2", "r1", "r3"],
    })


def settle(rt):
    """Wait for all async DB writes to complete."""
    rt.storage._executor.submit(lambda: None).result()
