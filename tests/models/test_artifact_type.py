from models.artifacts import ArtifactType

def test_dataframe_not_in_artifact_type():
    assert not hasattr(ArtifactType, "DATAFRAME"), (
        "DATAFRAME mixes technical storage with conceptual domain types"
    )

def test_conceptual_types_present():
    assert ArtifactType.EVENT_LOG.value == "event_log"
    assert ArtifactType.PROCESS_MODEL.value == "process_model"
    assert ArtifactType.KPI_REPORT.value == "kpi_report"
    assert ArtifactType.OTHER.value == "other"
