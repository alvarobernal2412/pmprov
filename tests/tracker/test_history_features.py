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


# ---- get_history / export_history -----------------------------------------

def test_get_history_lists_events_in_order(rt, pipeline):
    h = rt.get_history()

    assert h["history"]["name"] == "test"
    assert h["history"]["root_state_id"] == rt._root_state_id
    assert [e["func_name"] for e in h["events"]] == ["load", "enrich", "more"]
    assert [e["sequence"] for e in h["events"]] == [0, 1, 2]
    assert [e["output_state_id"] for e in h["events"]] == pipeline
    assert h["events"][1]["input_state_id"] == pipeline[0]
    assert h["branches"][0]["name"] == "main"
    json.dumps(h, default=str)  # serialisable


def test_get_history_includes_annotations_and_tags(rt, pipeline):
    rt.annotate([(AnnotatableType.ANALYSIS_STATE, pipeline[1])], "check this")
    rt.tag(AnnotatableType.ANALYSIS_STATE, pipeline[1], "suspicious")

    event = rt.get_history()["events"][1]

    assert [a["text"] for a in event["state_notes"]["annotations"]] == ["check this"]
    assert event["state_notes"]["tags"] == ["suspicious"]


def test_export_history_json_and_jsonl(rt, pipeline, tmp_path):
    p_json = rt.export_history(tmp_path / "out" / "h.json", format="json")
    p_jsonl = rt.export_history(tmp_path / "h.jsonl", format="jsonl")

    assert len(json.loads(p_json.read_text(encoding="utf-8"))["events"]) == 3
    records = [json.loads(line) for line in p_jsonl.read_text(encoding="utf-8").splitlines()]
    assert [r["type"] for r in records] == ["history", "branch", "event", "event", "event"]


def test_export_history_rejects_unknown_format(rt, tmp_path):
    with pytest.raises(ValueError):
        rt.export_history(tmp_path / "h.xml", format="xml")


