# tests/tracker/test_marimo_app_e2e.py
"""
E2E suite for PR #45 (session resume, branching, Operation dedup) using
local stand-ins shaped exactly like linear-continuous-process-mapper-marimo's
marimo_app.py: same operation_type registrations, same function names, same
transformation logic for apply_folds. No cross-repo dependency -- pmprov is
a published library and must not depend on a downstream app's code.
"""
import pandas as pd
import pytest
from tracker.kernel_hooks import init_marimo
from tracker.storage import DuckDBSQLiteBackend as StorageBackend
from tracker.visualizations import build_plotly_graph, format_params


@pytest.fixture
def db_paths(tmp_path):
    return tmp_path / "prov.db", tmp_path / "art"


def settle(rt):
    rt.storage._executor.submit(lambda: None).result()


LOG = pd.DataFrame({
    "case:concept:name": ["A", "A", "B", "B"],
    "concept:name": ["submit", "review", "submit", "approve"],
})


def read_csv_stub(_path):
    """Stands in for marimo_app.py's tracked pd.read_csv call."""
    return LOG.copy()


class _ServiceStub:
    def __init__(self, log):
        self._log = log

    def get_all_activities(self):
        return sorted(self._log["concept:name"].unique())

    def __str__(self):
        # Deterministic string representation based on activities,
        # not memory address, so fingerprinting works across sessions.
        return f"_ServiceStub(activities={self.get_all_activities()})"


def create_service_stub(log):
    """Stands in for marimo_app.py's create_process_analytics_service."""
    return _ServiceStub(log)


def apply_folds(log, fold_specs):
    """Copied verbatim from marimo_app.py's apply_folds (pure pandas,
    no app dependency needed to reuse the real logic)."""
    if not fold_specs:
        return log
    mapping = {
        act: fold["name"] for fold in fold_specs for act in fold["activities"]
    }
    folded = log.copy()
    folded["concept:name"] = folded["concept:name"].replace(mapping)
    return folded


def generate_sankey_figure(service, activities, builder, allow_loops, show_empty):
    """Stands in for ProcessAnalyticsService.generate_sankey_figure -- a
    small deterministic dict stands in for a Plotly figure, since the test
    only needs a distinct, reproducible output per param combination."""
    return {
        "activities": sorted(activities),
        "builder": builder,
        "allow_loops": allow_loops,
        "show_empty": show_empty,
    }


def _run_pipeline(rt, fold_specs, activities, builder="set_based",
                   allow_loops=True, show_empty=True):
    """One full marimo_app.py-shaped pipeline run: load -> service -> fold -> sankey."""
    log = rt.trace_step(func=read_csv_stub, func_name="read_csv_stub",
                         raw_line="pd.read_csv(...)", args=["log.csv"], kwargs={})
    service = rt.trace_step(func=create_service_stub, func_name="create_service_stub",
                             raw_line="create_service_stub(log)", args=[log], kwargs={})
    folded = rt.trace_step(func=apply_folds, func_name="apply_folds",
                            raw_line="apply_folds(log, fold_specs)",
                            args=[log, fold_specs], kwargs={})
    rt.trace_step(func=generate_sankey_figure, func_name="generate_sankey_figure",
                   raw_line="generate_sankey_figure(...)",
                   args=[service, activities, builder, allow_loops, show_empty],
                   kwargs={})
    return folded


def test_three_diagrams_one_session_three_branches(db_paths):
    db_path, artifact_dir = db_paths
    rt = init_marimo(db_path=db_path, artifact_dir=artifact_dir, history_name="app")
    history_id = rt._history.history_id

    _run_pipeline(rt, fold_specs=[], activities=["submit", "review", "approve"])
    branch1 = rt._branch.branch_id
    _run_pipeline(rt, fold_specs=[{"name": "F1", "activities": ["review"]}],
                  activities=["submit", "F1", "approve"])
    branch2 = rt._branch.branch_id
    _run_pipeline(rt, fold_specs=[{"name": "F2", "activities": ["approve"]}],
                  activities=["submit", "review", "F2"])
    branch3 = rt._branch.branch_id
    settle(rt)

    assert len({branch1, branch2, branch3}) == 3

    storage = StorageBackend(db_path=db_path, artifact_dir=artifact_dir)
    branches = storage.load_branches(history_id)
    assert len(branches) == 3

    con = storage._connect(read_only=True)
    try:
        count = con.execute(
            "SELECT COUNT(*) FROM operations WHERE name = 'read_csv_stub'"
        ).fetchone()[0]
    finally:
        con.close()
    assert count == 1

    fig = build_plotly_graph(storage, history_id)
    assert fig is not None


def test_stop_resume_default_forks_new_branch_same_history(db_paths):
    db_path, artifact_dir = db_paths
    rt1 = init_marimo(db_path=db_path, artifact_dir=artifact_dir, history_name="app")
    history_id = rt1._history.history_id
    _run_pipeline(rt1, fold_specs=[], activities=["submit", "review", "approve"])
    settle(rt1)
    branch1 = rt1._branch.branch_id

    rt2 = init_marimo(db_path=db_path, artifact_dir=artifact_dir, history_name="app")
    assert rt2._history.history_id == history_id
    assert rt2._branch.branch_id == branch1
    _run_pipeline(rt2, fold_specs=[{"name": "F1", "activities": ["review"]}],
                  activities=["submit", "F1", "approve"])
    settle(rt2)

    assert rt2._branch.branch_id != branch1

    storage = StorageBackend(db_path=db_path, artifact_dir=artifact_dir)
    assert len(storage.load_branches(history_id)) == 2


def test_stop_resume_identical_params_no_spurious_branch(db_paths):
    db_path, artifact_dir = db_paths
    rt1 = init_marimo(db_path=db_path, artifact_dir=artifact_dir, history_name="app")
    history_id = rt1._history.history_id
    _run_pipeline(rt1, fold_specs=[], activities=["submit", "review", "approve"])
    settle(rt1)
    branch1 = rt1._branch.branch_id

    rt2 = init_marimo(db_path=db_path, artifact_dir=artifact_dir, history_name="app")
    _run_pipeline(rt2, fold_specs=[], activities=["submit", "review", "approve"])
    settle(rt2)

    assert rt2._branch.branch_id == branch1

    storage = StorageBackend(db_path=db_path, artifact_dir=artifact_dir)
    assert len(storage.load_branches(history_id)) == 1


def test_fresh_true_is_only_override(db_paths):
    db_path, artifact_dir = db_paths
    rt1 = init_marimo(db_path=db_path, artifact_dir=artifact_dir, history_name="app")
    _run_pipeline(rt1, fold_specs=[], activities=["submit", "review", "approve"])
    settle(rt1)

    rt2 = init_marimo(db_path=db_path, artifact_dir=artifact_dir,
                       history_name="app", fresh=True)
    assert rt2._history.history_id != rt1._history.history_id

    storage = StorageBackend(db_path=db_path, artifact_dir=artifact_dir)
    con = storage._connect(read_only=True)
    try:
        count = con.execute(
            "SELECT COUNT(*) FROM analysis_histories WHERE name = 'app'"
        ).fetchone()[0]
    finally:
        con.close()
    assert count == 2


def test_format_params_readable_on_realistic_step(db_paths):
    db_path, artifact_dir = db_paths
    rt = init_marimo(db_path=db_path, artifact_dir=artifact_dir, history_name="app")
    _run_pipeline(rt, fold_specs=[{"name": "F1", "activities": ["review"]}],
                  activities=["submit", "F1", "approve"])
    settle(rt)

    storage = StorageBackend(db_path=db_path, artifact_dir=artifact_dir)
    con = storage._connect(read_only=True)
    try:
        output_state_id = con.execute(
            "SELECT output_state_id FROM analysis_steps WHERE func_name = 'apply_folds' LIMIT 1"
        ).fetchone()[0]
    finally:
        con.close()

    detail = storage.load_state_detail(output_state_id)
    summary = format_params(detail["params"])

    assert "__pmprov_state__" not in summary
    assert "fold_specs=" in summary
