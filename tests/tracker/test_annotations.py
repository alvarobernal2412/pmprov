"""Tests for RuntimeTracker annotation and tag methods."""
import pandas as pd
import pytest

from models.annotations import AnnotatableType
from tracker.storage import DuckDBSQLiteBackend as StorageBackend
from tracker.runtime import RuntimeTracker
import tracker.annotations  # noqa: F401 — patches methods onto RuntimeTracker


@pytest.fixture
def rt(tmp_path):
    s = StorageBackend(db_path=tmp_path / "prov.db", artifact_dir=tmp_path / "art")
    return RuntimeTracker(storage=s, session_id="t", history_name="test")


def test_annotate_single_target_round_trips(rt):
    annotation_id = rt.annotate([(AnnotatableType.ANALYSIS_HISTORY, rt._history.history_id)], "audit note")

    df = rt.list_annotations(AnnotatableType.ANALYSIS_HISTORY, rt._history.history_id)
    assert isinstance(df, pd.DataFrame)
    assert annotation_id in df["annotation_id"].values
    row = df[df["annotation_id"] == annotation_id].iloc[0]
    assert row["text"] == "audit note"
    assert row["agent_id"] == rt._agent.agent_id


def test_annotate_multiple_targets_shares_one_note(rt):
    annotation_id = rt.annotate(
        [(AnnotatableType.ANALYSIS_STATE, "s1"), (AnnotatableType.STEP, "step1")],
        "this set is suspicious",
    )

    from_state = rt.list_annotations(AnnotatableType.ANALYSIS_STATE, "s1")
    from_step = rt.list_annotations(AnnotatableType.STEP, "step1")
    assert annotation_id in from_state["annotation_id"].values
    assert annotation_id in from_step["annotation_id"].values


def test_list_annotations_empty_dataframe_for_unknown_target(rt):
    df = rt.list_annotations(AnnotatableType.BRANCH, "does-not-exist")
    assert isinstance(df, pd.DataFrame)
    assert df.empty


def test_remove_annotation(rt):
    annotation_id = rt.annotate([(AnnotatableType.PIPELINE, "p1")], "temp")
    rt.remove_annotation(annotation_id)

    df = rt.list_annotations(AnnotatableType.PIPELINE, "p1")
    assert df.empty


def test_tag_and_list_tags_round_trip(rt):
    assignment_id = rt.tag(AnnotatableType.ARTIFACT_STATE, "as1", "reviewed")

    df = rt.list_tags(AnnotatableType.ARTIFACT_STATE, "as1")
    assert assignment_id in df["assignment_id"].values
    assert "reviewed" in df["name"].values


def test_tag_reuses_existing_tag_by_name(rt):
    rt.tag(AnnotatableType.ARTIFACT_STATE, "as1", "reviewed")
    rt.tag(AnnotatableType.ARTIFACT_STATE, "as2", "reviewed")

    df1 = rt.list_tags(AnnotatableType.ARTIFACT_STATE, "as1")
    df2 = rt.list_tags(AnnotatableType.ARTIFACT_STATE, "as2")
    assert df1.iloc[0]["tag_id"] == df2.iloc[0]["tag_id"]


def test_untag_removes_assignment_by_name(rt):
    rt.tag(AnnotatableType.BRANCH, rt._branch.branch_id, "reviewed")

    rt.untag(AnnotatableType.BRANCH, rt._branch.branch_id, "reviewed")

    df = rt.list_tags(AnnotatableType.BRANCH, rt._branch.branch_id)
    assert df.empty


def test_untag_unknown_name_is_a_noop(rt):
    rt.tag(AnnotatableType.BRANCH, rt._branch.branch_id, "reviewed")

    rt.untag(AnnotatableType.BRANCH, rt._branch.branch_id, "not-a-real-tag")

    df = rt.list_tags(AnnotatableType.BRANCH, rt._branch.branch_id)
    assert "reviewed" in df["name"].values
