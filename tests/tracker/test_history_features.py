"""Tests for update_annotation, history export, restore_state and review mode."""
import json

import pandas as pd
import pytest

from models.annotations import AnnotatableType
from tracker.runtime import RuntimeTracker
from tracker.storage import DuckDBSQLiteBackend as StorageBackend
import tracker  # noqa: F401 — patches every feature module onto RuntimeTracker


def settle(rt):
    rt.storage._executor.submit(lambda: None).result()


@pytest.fixture
def rt(tmp_path):
    s = StorageBackend(db_path=tmp_path / "prov.db", artifact_dir=tmp_path / "art")
    return RuntimeTracker(storage=s, session_id="t", history_name="test")


@pytest.fixture
def pipeline(rt):
    """event_log -> case_log -> case_log2 (three tracked steps); returns state ids."""
    df = pd.DataFrame({"a": [1, 2, 3, 4]})
    states = []
    out = rt.trace_step(func=lambda: df.copy(), func_name="load",
                        raw_line="event_log = load()", args=[], kwargs={})
    states.append(rt._current_state_id)
    out2 = rt.trace_step(func=lambda d: d.assign(b=d["a"] * 2), func_name="enrich",
                         raw_line="case_log = enrich(event_log)", args=[out], kwargs={})
    states.append(rt._current_state_id)
    rt.trace_step(func=lambda d: d.assign(c=1), func_name="more",
                  raw_line="case_log2 = more(case_log)", args=[out2], kwargs={})
    states.append(rt._current_state_id)
    settle(rt)
    return states


# ---- update_annotation ---------------------------------------------------

def test_update_annotation_changes_text_only(rt):
    target = (AnnotatableType.PIPELINE, "p1")
    annotation_id = rt.annotate([target], "old")
    before = rt.list_annotations(*target).iloc[0]

    rt.update_annotation(annotation_id, "new")

    after = rt.list_annotations(*target).iloc[0]
    assert after["text"] == "new"
    assert after["created_at"] == before["created_at"]
    assert after["agent_id"] == before["agent_id"]
    assert after["targets"] == before["targets"]


def test_update_annotation_unknown_id_raises(rt):
    with pytest.raises(ValueError):
        rt.update_annotation("nope", "x")


