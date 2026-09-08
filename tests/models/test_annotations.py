from models.annotations import (
    AnnotatableType,
    Annotation,
    AnnotationTarget,
    Tag,
    TagAssignment,
)


def test_annotatable_type_values():
    assert AnnotatableType.ARTIFACT_STATE.value == "artifact_state"
    assert AnnotatableType.ANALYSIS_STATE.value == "analysis_state"
    assert AnnotatableType.BRANCH.value == "branch"
    assert AnnotatableType.ANALYSIS_HISTORY.value == "analysis_history"
    assert AnnotatableType.STEP.value == "step"
    assert AnnotatableType.PIPELINE.value == "pipeline"


def test_annotation_requires_no_target():
    annotation = Annotation(annotation_id="a1", text="worth revisiting", agent_id="agent-1")
    assert annotation.text == "worth revisiting"
    assert annotation.agent_id == "agent-1"
    assert annotation.created_at is not None


def test_annotation_target_links_annotation_to_object():
    target = AnnotationTarget(
        annotation_id="a1",
        target_type=AnnotatableType.ANALYSIS_STATE,
        target_id="state-1",
    )
    assert target.annotation_id == "a1"
    assert target.target_type == AnnotatableType.ANALYSIS_STATE
    assert target.target_id == "state-1"


def test_annotation_can_have_multiple_targets():
    targets = [
        AnnotationTarget(annotation_id="a1", target_type=AnnotatableType.ANALYSIS_STATE, target_id="s1"),
        AnnotationTarget(annotation_id="a1", target_type=AnnotatableType.STEP, target_id="step1"),
    ]
    assert {t.target_id for t in targets} == {"s1", "step1"}
    assert all(t.annotation_id == "a1" for t in targets)


def test_tag_holds_reusable_name():
    tag = Tag(tag_id="t1", name="reviewed")
    assert tag.name == "reviewed"


def test_tag_assignment_links_tag_to_object():
    assignment = TagAssignment(
        assignment_id="ta1",
        tag_id="t1",
        target_type=AnnotatableType.BRANCH,
        target_id="branch-1",
    )
    assert assignment.tag_id == "t1"
    assert assignment.target_type == AnnotatableType.BRANCH
    assert assignment.target_id == "branch-1"
    assert assignment.created_at is not None
