import uuid

from models.annotations import AnnotatableType, Annotation, AnnotationTarget, TagAssignment
from tracker.storage import DuckDBSQLiteBackend as StorageBackend


def _backend(tmp_path):
    return StorageBackend(db_path=tmp_path / "prov.db", artifact_dir=tmp_path / "art")


def test_save_and_load_annotation_single_target(tmp_path):
    s = _backend(tmp_path)
    annotation = Annotation(annotation_id=str(uuid.uuid4()), text="check this", agent_id="agent-1")
    targets = [AnnotationTarget(annotation_id=annotation.annotation_id,
                                 target_type=AnnotatableType.ANALYSIS_STATE, target_id="state-1")]

    s.save_annotation_sync(annotation, targets)
    loaded = s.load_annotations("analysis_state", "state-1")

    assert len(loaded) == 1
    assert loaded[0]["annotation_id"] == annotation.annotation_id
    assert loaded[0]["text"] == "check this"
    assert loaded[0]["agent_id"] == "agent-1"
    assert loaded[0]["targets"] == [{"target_type": "analysis_state", "target_id": "state-1"}]


def test_annotation_with_multiple_targets_is_visible_from_each(tmp_path):
    s = _backend(tmp_path)
    annotation = Annotation(annotation_id=str(uuid.uuid4()), text="this whole run is suspect", agent_id="agent-1")
    targets = [
        AnnotationTarget(annotation_id=annotation.annotation_id, target_type=AnnotatableType.ANALYSIS_STATE, target_id="s1"),
        AnnotationTarget(annotation_id=annotation.annotation_id, target_type=AnnotatableType.STEP, target_id="step1"),
    ]

    s.save_annotation_sync(annotation, targets)

    from_state = s.load_annotations("analysis_state", "s1")
    from_step = s.load_annotations("step", "step1")

    assert len(from_state) == 1
    assert len(from_step) == 1
    assert from_state[0]["annotation_id"] == from_step[0]["annotation_id"]
    assert {t["target_id"] for t in from_state[0]["targets"]} == {"s1", "step1"}


def test_load_annotations_returns_empty_for_unknown_target(tmp_path):
    s = _backend(tmp_path)
    assert s.load_annotations("analysis_state", "does-not-exist") == []


def test_remove_annotation_deletes_it_and_its_targets(tmp_path):
    s = _backend(tmp_path)
    annotation = Annotation(annotation_id=str(uuid.uuid4()), text="temp note", agent_id="agent-1")
    targets = [AnnotationTarget(annotation_id=annotation.annotation_id,
                                 target_type=AnnotatableType.BRANCH, target_id="branch-1")]
    s.save_annotation_sync(annotation, targets)

    s.remove_annotation_sync(annotation.annotation_id)

    assert s.load_annotations("branch", "branch-1") == []


def test_save_tag_dedups_by_name(tmp_path):
    s = _backend(tmp_path)
    tag_id_1 = s.save_tag_sync("reviewed")
    tag_id_2 = s.save_tag_sync("reviewed")

    assert tag_id_1 == tag_id_2


def test_save_and_load_tag_assignment(tmp_path):
    s = _backend(tmp_path)
    tag_id = s.save_tag_sync("outlier")
    assignment = TagAssignment(
        assignment_id=str(uuid.uuid4()), tag_id=tag_id,
        target_type=AnnotatableType.ARTIFACT_STATE, target_id="artifact-state-1",
    )

    s.save_tag_assignment_sync(assignment)
    loaded = s.load_tags("artifact_state", "artifact-state-1")

    assert len(loaded) == 1
    assert loaded[0]["tag_id"] == tag_id
    assert loaded[0]["name"] == "outlier"


def test_load_tags_returns_empty_for_unknown_target(tmp_path):
    s = _backend(tmp_path)
    assert s.load_tags("artifact_state", "does-not-exist") == []


def test_remove_tag_assignment(tmp_path):
    s = _backend(tmp_path)
    tag_id = s.save_tag_sync("outlier")
    assignment = TagAssignment(
        assignment_id=str(uuid.uuid4()), tag_id=tag_id,
        target_type=AnnotatableType.PIPELINE, target_id="pipeline-1",
    )
    s.save_tag_assignment_sync(assignment)

    s.remove_tag_assignment_sync(assignment.assignment_id)

    assert s.load_tags("pipeline", "pipeline-1") == []
