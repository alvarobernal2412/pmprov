"""
Annotation and tag methods for RuntimeTracker.

Imported as a side-effect from tracker/__init__.py.
Patches annotate(), list_annotations(), remove_annotation(), tag(), list_tags(),
untag() onto RuntimeTracker.
"""
from __future__ import annotations

import uuid
from typing import TYPE_CHECKING, Any

from models.annotations import AnnotatableType, Annotation, AnnotationTarget, TagAssignment

if TYPE_CHECKING:
    from tracker.runtime import RuntimeTracker


def _annotate(
    self: "RuntimeTracker",
    targets: list[tuple[AnnotatableType, str]],
    text: str,
) -> str:
    """
    Attach one free-text note to one or more Annotatable objects.

    Parameters
    ----------
    targets:
        List of (AnnotatableType, target_id) pairs. A single call may target
        several objects at once (e.g. a set of AnalysisStates and Steps) so
        the note is shared across all of them.
    text:
        The note's content.

    Returns
    -------
    The new annotation's annotation_id.
    """
    annotation = Annotation(annotation_id=str(uuid.uuid4()), text=text, agent_id=self._agent.agent_id)
    annotation_targets = [
        AnnotationTarget(annotation_id=annotation.annotation_id, target_type=target_type, target_id=target_id)
        for target_type, target_id in targets
    ]
    self.storage.save_annotation_sync(annotation, annotation_targets)
    return annotation.annotation_id


def _list_annotations(self: "RuntimeTracker", target_type: AnnotatableType, target_id: str) -> Any:
    """Return a DataFrame of annotations attached to (target_type, target_id)."""
    import pandas as pd

    rows = self.storage.load_annotations(target_type.value, target_id)
    return pd.DataFrame(rows) if rows else pd.DataFrame(
        columns=["annotation_id", "text", "agent_id", "created_at", "targets"]
    )


def _remove_annotation(self: "RuntimeTracker", annotation_id: str) -> None:
    """Delete an annotation and all of its targets."""
    self.storage.remove_annotation_sync(annotation_id)


def _tag(self: "RuntimeTracker", target_type: AnnotatableType, target_id: str, name: str) -> str:
    """
    Assign a short reusable label to an Annotatable object. Reuses the existing
    Tag row if *name* has already been used elsewhere.

    Returns the new tag assignment's assignment_id.
    """
    tag_id = self.storage.save_tag_sync(name)
    assignment = TagAssignment(
        assignment_id=str(uuid.uuid4()), tag_id=tag_id, target_type=target_type, target_id=target_id
    )
    self.storage.save_tag_assignment_sync(assignment)
    return assignment.assignment_id


def _list_tags(self: "RuntimeTracker", target_type: AnnotatableType, target_id: str) -> Any:
    """Return a DataFrame of tags assigned to (target_type, target_id)."""
    import pandas as pd

    rows = self.storage.load_tags(target_type.value, target_id)
    return pd.DataFrame(rows) if rows else pd.DataFrame(
        columns=["assignment_id", "tag_id", "name", "created_at"]
    )


def _untag(self: "RuntimeTracker", target_type: AnnotatableType, target_id: str, name: str) -> None:
    """Remove a tag assignment by (target_type, target_id, name). No-op if not found."""
    for row in self.storage.load_tags(target_type.value, target_id):
        if row["name"] == name:
            self.storage.remove_tag_assignment_sync(row["assignment_id"])


from tracker.runtime import RuntimeTracker  # noqa: E402

RuntimeTracker.annotate = _annotate
RuntimeTracker.list_annotations = _list_annotations
RuntimeTracker.remove_annotation = _remove_annotation
RuntimeTracker.tag = _tag
RuntimeTracker.list_tags = _list_tags
RuntimeTracker.untag = _untag
