import uuid
from tracker.storage import DuckDBSQLiteBackend as StorageBackend


def test_save_and_load_pruned_view_round_trips(tmp_path):
    s = StorageBackend(db_path=tmp_path / "prov.db", artifact_dir=tmp_path / "art")
    view_id = str(uuid.uuid4())
    config = {"group_by_category": True, "hidden_state_ids": ["a", "b"]}

    s.save_pruned_view_sync(view_id, "hist-1", "my-view", config)
    loaded = s.load_pruned_view_config(view_id)

    assert loaded is not None
    assert loaded["view_id"] == view_id
    assert loaded["history_id"] == "hist-1"
    assert loaded["name"] == "my-view"
    assert loaded["config"] == config


def test_load_pruned_view_config_returns_none_for_unknown_id(tmp_path):
    s = StorageBackend(db_path=tmp_path / "prov.db", artifact_dir=tmp_path / "art")
    assert s.load_pruned_view_config("does-not-exist") is None
