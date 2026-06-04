from models.artifacts import DataFrameDelta, ModificationType


def test_dataframe_delta_has_rows_delta():
    d = DataFrameDelta(
        delta_id="d1",
        modification_type=ModificationType.REMOVAL,
        root_artifact_state_id="s1",
        updated_artifact_state_id="s2",
        rows_delta=-5,
    )
    assert d.rows_delta == -5


def test_dataframe_delta_no_rows_added_removed():
    fields = DataFrameDelta.model_fields
    assert "rows_added" not in fields
    assert "rows_removed" not in fields
