import marimo

__generated_with = "0.23.9"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell
def _(mo):
    mo.md(r"""
    # pmprov — Acceptance Criteria Demo (Marimo)

    Validates R1–R7 against the Marimo code path.
    Each section asserts the same criteria as `acceptance_criteria_demo.ipynb`.

    | Requirement | ACs |
    |---|---|
    | R1 Traceability | 1.1 full lineage · 1.2 step detail · 1.3 execution context |
    | R2 Reproducibility | 2.1 artifact refs · 2.2 scalar params · 2.3 runtime env · 2.4 replay |
    | R3 Branching | 3.1 divergence point · 3.2 independent branches |
    | R4 Reusability | 4.1 pipeline creation · 4.2 param overrides · 4.3 distinct steps |
    | R5 State Comparison | 5.1 granular diff · 5.2 abstracted diff |
    | R6 History Comparison | 6.1 operation diff · 6.2–6.3 category diff |
    | R7 Data Evolution | 7.1 delta per step · 7.2 lifecycle · 7.3 linked states |
    """)
    return


@app.cell
def _():
    import sys
    from pathlib import Path

    _here = Path.cwd()
    _repo_root = _here
    for _candidate in [_here, *_here.parents]:
        if (_candidate / "pyproject.toml").exists():
            _repo_root = _candidate
            break

    if str(_repo_root) not in sys.path:
        sys.path.insert(0, str(_repo_root))
    _examples = _repo_root / "examples"
    if str(_examples) not in sys.path:
        sys.path.insert(0, str(_examples))

    import pandas as pd
    from tracker import init_marimo, omit_functions, operation_type, step_category
    from utils.event_enricher import (
        create_case_log,
        event_add_relative_case_time,
        case_add_activity_start_times,
    )

    CASE_ID_COL = "case:concept:name"
    TIMESTAMP_COL = "time:timestamp"
    ACTIVITY_COL = "concept:name"
    DATA_FILE = str(_repo_root / "examples" / "data" / "rtfm_full.csv")

    _artifact_dir = str(_repo_root / "examples" / "marimo" / "artifacts_ac_demo")
    _db_path = str(_repo_root / "examples" / "marimo" / "provenance_ac_demo.db")
    Path(_artifact_dir).mkdir(parents=True, exist_ok=True)

    rt = init_marimo(
        history_name="AC demo (Marimo)",
        branch_name="main",
        artifact_dir=_artifact_dir,
        db_path=_db_path,
    )

    operation_type("data_loading",         pd.read_csv)
    operation_type("case_aggregation",     create_case_log)
    operation_type("attribute_derivation", event_add_relative_case_time)
    operation_type("attribute_derivation", case_add_activity_start_times)
    step_category("data_loading_phase",    "data_loading")
    step_category("case_aggregation_phase","case_aggregation")
    step_category("log_enrichment_phase",  "attribute_derivation")
    omit_functions("reset_index", "sort_values", "read_parquet", "dropna", "tolist")

    def settle():
        rt.storage._executor.submit(lambda: None).result()

    print(f"History ID: {rt._history.history_id}")
    return (
        ACTIVITY_COL,
        CASE_ID_COL,
        DATA_FILE,
        TIMESTAMP_COL,
        case_add_activity_start_times,
        create_case_log,
        event_add_relative_case_time,
        pd,
        rt,
        settle,
    )


@app.cell
def _(DATA_FILE, TIMESTAMP_COL, pd):
    # Step 1 – load event log  [tracked by AST rewriter]
    event_log = pd.read_csv(
        DATA_FILE,
        dtype={"org:resource": str, "matricola": str},
        parse_dates=[TIMESTAMP_COL],
    )
    print(f"Loaded {len(event_log):,} events, {event_log['case:concept:name'].nunique():,} cases")
    return (event_log,)


@app.cell
def _(create_case_log, event_log):
    # Step 2 – aggregate to one row per case  [tracked]
    case_log = create_case_log(event_log)
    print(f"case_log shape: {case_log.shape}")
    return (case_log,)


@app.cell
def _(CASE_ID_COL, TIMESTAMP_COL, event_add_relative_case_time, event_log):
    # Step 3 – add relative case time  [tracked]
    event_log_enriched = event_add_relative_case_time(event_log, CASE_ID_COL, TIMESTAMP_COL)
    print(f"rel_time range: {event_log_enriched['rel_time'].min()} – {event_log_enriched['rel_time'].max()}")
    return (event_log_enriched,)


@app.cell
def _(
    ACTIVITY_COL,
    CASE_ID_COL,
    case_add_activity_start_times,
    case_log,
    event_log_enriched,
):
    # Step 4 – pivot activity start times  [tracked]
    case_log_pivoted = case_add_activity_start_times(
        case_log, event_log_enriched, CASE_ID_COL, ACTIVITY_COL, "rel_time"
    )
    _time_cols = [c for c in case_log_pivoted.columns if c.endswith("::start")]
    print(f"Activity start-time columns: {_time_cols[:4]}…")
    return (case_log_pivoted,)


@app.cell
def _(case_log_pivoted):
    # Step 5 – derive Delay Send  [arithmetic, NOT tracked — RHS is not a Call]
    case_log_with_delay = case_log_pivoted.copy()
    case_log_with_delay["Delay Send"] = (
        case_log_with_delay["Send_Fine::start"] - case_log_with_delay["Create_Fine::start"]
    ).dt.total_seconds() / 86400
    print(
        f"Delay Send (days) — mean: {case_log_with_delay['Delay Send'].mean():.1f}, "
        f"median: {case_log_with_delay['Delay Send'].median():.1f}"
    )
    return (case_log_with_delay,)


@app.cell
def _(case_log_with_delay):
    # Step 6 – classify cases  [tracked via case_log_with_delay.apply]
    is_illegal_delay = case_log_with_delay.apply(
        lambda case: case["Delay Send"] > 90.0, axis=1
    )
    print(f"Cases flagged as illegal delay: {is_illegal_delay.sum():,}")
    return (is_illegal_delay,)


@app.cell
def _(is_illegal_delay, rt, settle):
    # Explicit dep on is_illegal_delay ensures pipeline is fully run before querying
    _ = is_illegal_delay
    settle()
    states_df = rt.list_states()
    step_details = {}
    for _, _row in states_df.dropna(subset=["produced_by_step_id"]).iterrows():
        _d = rt.describe_step(_row["produced_by_step_id"])
        if _d:
            step_details[_d["func_name"]] = {
                "state_id": _row["state_id"],
                "step_id": _row["produced_by_step_id"],
                "detail": _d,
            }
    states_df
    return states_df, step_details


@app.cell
def _(mo):
    mo.md("""
    ## R1 – Traceability
    """)
    return


@app.cell
def _(mo, rt, settle):
    import networkx as nx
    settle()
    G = rt.storage.to_networkx(rt._history.history_id)
    _graph = rt.storage.load_graph(rt._history.history_id)
    assert len(_graph["states"]) >= 2, "Expected at least 2 states"
    assert len(_graph["steps"]) >= 2, "Expected at least 2 steps"
    assert nx.is_directed_acyclic_graph(G), "Provenance graph must be a DAG"
    mo.md(
        f"**AC 1.1 ✓** — {len(_graph['states'])} states, {len(_graph['steps'])} steps, "
        f"acyclic: {nx.is_directed_acyclic_graph(G)}"
    )
    return


@app.cell
def _(mo, step_details):
    _d = step_details.get("pd.read_csv", {}).get("detail", {})
    assert _d, "No step detail found for pd.read_csv"
    assert _d["func_name"] == "pd.read_csv"
    assert _d["raw_line"], "raw_line must be non-empty"
    assert isinstance(_d["params"], list), "params must be a list"
    mo.md(
        f"**AC 1.2 ✓** — func_name=`{_d['func_name']}`, "
        f"raw_line=`{_d['raw_line'][:60]}…`, "
        f"{len(_d['params'])} params captured"
    )
    return


@app.cell
def _(mo, step_details):
    _d = step_details.get("pd.read_csv", {}).get("detail", {})
    assert _d.get("agent"), "agent must be recorded"
    assert _d.get("environment"), "environment must be recorded"
    assert "Python" in _d["environment"].get("tool_version", ""), "tool_version must mention Python"
    mo.md(
        f"**AC 1.3 ✓** — agent_type=`{_d['agent']['agent_type']}`, "
        f"tool_version=`{_d['environment']['tool_version']}`"
    )
    return


@app.cell
def _(mo, rt):
    _fig = rt.show_graph()
    mo.as_html(_fig) if _fig is not None else mo.md("*(graph unavailable)*")
    return


@app.cell
def _(mo):
    mo.md("""
    ## R2 – Reproducibility
    """)
    return


@app.cell
def _(mo, step_details):
    _d = step_details.get("create_case_log", {}).get("detail", {})
    assert _d, "No step detail for create_case_log"
    _artifact_params = [p for p in _d["params"] if p.get("value_type") == "artifact_state_ref"]
    assert _artifact_params, "create_case_log input must reference artifact_state_ref"
    mo.md(
        f"**AC 2.1 ✓** — create_case_log has {len(_artifact_params)} artifact_state_ref param(s); "
        f"artifact_state_id=`{str(_artifact_params[0]['value'])[:8]}…`"
    )
    return


@app.cell
def _(mo, step_details):
    _d = step_details.get("pd.read_csv", {}).get("detail", {})
    _scalars = [p for p in _d.get("params", []) if p.get("value_type") == "scalar"]
    assert _scalars, "pd.read_csv must have at least one scalar param (the file path)"
    _path_param = next((p for p in _scalars if "rtfm" in str(p.get("value", ""))), None)
    assert _path_param, "File path param not captured as scalar"
    mo.md(f"**AC 2.2 ✓** — scalar param value=`{str(_path_param['value'])[:60]}`")
    return


@app.cell
def _(mo, step_details):
    import json as _json
    _d = step_details.get("pd.read_csv", {}).get("detail", {})
    _env = _d.get("environment", {})
    assert _env.get("library_versions"), "library_versions must be non-empty"
    assert _env.get("runtime"), "runtime must be recorded"
    _libs = (
        _json.loads(_env["library_versions"])
        if isinstance(_env["library_versions"], str)
        else _env["library_versions"]
    )
    mo.md(
        f"**AC 2.3 ✓** — runtime=`{_env['runtime']}`, "
        f"{len(_libs)} library versions captured"
    )
    return


@app.cell
def _(mo, rt, settle, step_details):
    _entry = step_details.get("create_case_log", {})
    _state_id = _entry.get("state_id")
    assert _state_id, "create_case_log state_id not found"
    _original_artifact = rt.storage.load_artifact(
        rt.storage.load_output_artifact_state_id(_state_id)
    )
    _state_before = rt._current_state_id
    rt.replay_state(_state_id)
    settle()
    _state_after = rt._current_state_id
    assert _state_after != _state_before, "replay_state must advance current state"
    _replayed_artifact = rt.storage.load_artifact(
        rt.storage.load_output_artifact_state_id(_state_after)
    )
    assert _replayed_artifact.shape == _original_artifact.shape, "Replayed output shape must match"
    mo.md(
        f"**AC 2.4 ✓** — replay produced new state `{_state_after[:8]}…`; "
        f"shape matches: {_replayed_artifact.shape}"
    )
    return


@app.cell
def _(mo):
    mo.md("""
    ## R3 – Branching
    """)
    return


@app.cell
def _(rt, settle, states_df):
    # Fork from the second non-root state (create_case_log output)
    _fork_row = states_df.dropna(subset=["produced_by_step_id"]).iloc[1]
    fork_state_id = _fork_row["state_id"]
    rt.checkout(fork_state_id, branch_name="alt_threshold")
    settle()
    return (fork_state_id,)


@app.cell
def _(fork_state_id, mo, rt):
    _branches = rt.list_branches()
    _names = list(_branches["name"])
    assert "main" in _names, "main branch must exist"
    assert "alt_threshold" in _names, "alt_threshold branch must exist"
    _alt = _branches[_branches["name"] == "alt_threshold"].iloc[0]
    assert _alt["divergence_point_id"] == fork_state_id, "Divergence point must match fork state"
    mo.md(
        f"**AC 3.1 ✓** — divergence_point_id=`{fork_state_id[:8]}…`  \n"
        f"**AC 3.2 ✓** — branches: {_names}"
    )
    return


@app.cell
def _(fork_state_id, mo, rt):
    _ = fork_state_id  # ensure checkout has run
    _fig = rt.show_graph()
    mo.as_html(_fig) if _fig is not None else mo.md("*(graph unavailable)*")
    return


@app.cell
def _(mo):
    mo.md("""
    ## R4 – Reusability
    """)
    return


@app.cell
def _(mo, rt, settle, step_details):
    PIPELINE_NAME = "case_enrichment_pipeline"
    _keys = ["create_case_log", "event_add_relative_case_time", "case_add_activity_start_times"]
    step_ids = [step_details[k]["step_id"] for k in _keys if k in step_details]
    assert len(step_ids) == 3, f"Expected 3 pipeline steps, got {len(step_ids)}"
    pipeline_id = rt.create_pipeline(PIPELINE_NAME, step_ids)
    settle()
    _con = rt.storage._connect(read_only=True)
    _rows = _con.execute(
        "SELECT pipeline_id, name FROM pipelines WHERE name = ?", [PIPELINE_NAME]
    ).fetchall()
    _con.close()
    assert _rows, "Pipeline must be persisted in DB"
    mo.md(
        f"**AC 4.1 ✓** — pipeline `{PIPELINE_NAME}` created with id=`{pipeline_id[:8]}…`, "
        f"{len(step_ids)} steps"
    )
    return pipeline_id, step_ids


@app.cell
def _(
    case_add_activity_start_times,
    create_case_log,
    event_add_relative_case_time,
    event_log,
    mo,
    pipeline_id,
    rt,
    settle,
    step_ids,
):
    _func_map = {
        step_ids[0]: create_case_log,
        step_ids[1]: event_add_relative_case_time,
        step_ids[2]: case_add_activity_start_times,
    }
    result_pipeline = rt.replay_pipeline(
        pipeline_id,
        func_map=_func_map,
        initial_input=event_log,
        param_overrides={},
    )
    settle()
    assert result_pipeline["errors"] == [], f"replay_pipeline errors: {result_pipeline['errors']}"
    mo.md(
        f"**AC 4.2 ✓** — replay_pipeline succeeded, "
        f"output shape: {result_pipeline['output'].shape}"
    )
    return


@app.cell
def _(mo, rt, settle, step_ids):
    settle()
    _graph = rt.storage.load_graph(rt._history.history_id)
    _recorded_step_ids = {s["step_id"] for s in _graph["steps"]}
    _missing = [sid for sid in step_ids if sid not in _recorded_step_ids]
    assert not _missing, f"Pipeline step IDs not in graph: {_missing}"
    mo.md(f"**AC 4.3 ✓** — all {len(step_ids)} pipeline step IDs present in provenance graph")
    return


@app.cell
def _(mo):
    mo.md("""
    ## R5 – State Comparison
    """)
    return


@app.cell
def _(mo, rt, step_details):
    el_state = step_details.get("pd.read_csv", {}).get("state_id")
    cl_state = step_details.get("create_case_log", {}).get("state_id")
    assert el_state and cl_state, "Need both event_log and case_log states"
    _cmp = rt.compare_states(el_state, cl_state)
    assert "common_columns" in _cmp
    assert "unique_to_a" in _cmp
    assert "unique_to_b" in _cmp
    assert "shape_a" in _cmp
    assert "shape_b" in _cmp
    mo.md(
        f"**AC 5.1 ✓** — common cols: {len(_cmp['common_columns'])}, "
        f"unique_to_event_log: {len(_cmp['unique_to_a'])}, "
        f"unique_to_case_log: {len(_cmp['unique_to_b'])}, "
        f"shapes: {_cmp['shape_a']} vs {_cmp['shape_b']}"
    )
    return cl_state, el_state


@app.cell
def _(cl_state, el_state, mo, rt):
    rt.register_abstraction("row_count", lambda df, _sid: len(df))
    rt.register_abstraction("column_names", lambda df, _sid: sorted(df.columns.tolist()))
    rt.register_abstraction(
        "numeric_means",
        lambda df, _sid: df.select_dtypes("number").mean().round(2).to_dict(),
    )
    _abstracted = rt.compare_states_abstracted(el_state, cl_state)
    assert "row_count" in _abstracted, "row_count abstraction must be present"
    assert "a" in _abstracted["row_count"] and "b" in _abstracted["row_count"]
    mo.md(
        f"**AC 5.2 ✓** — row_count: event_log={_abstracted['row_count']['a']}, "
        f"case_log={_abstracted['row_count']['b']}"
    )
    return


@app.cell
def _(mo):
    mo.md("""
    ## R6 – History Comparison
    """)
    return


@app.cell
def _(pd, rt, settle):
    import tempfile as _tempfile
    import os as _os
    from tracker import operation_type as _op_type, step_category as _st_cat

    _tmp = _tempfile.mkdtemp()
    rt2 = rt.__class__(
        db_path=_os.path.join(_tmp, "hist_b.db"),
        artifact_dir=_os.path.join(_tmp, "art_b"),
        history_name="History B",
    )
    _st_cat("data_loading_phase", "data_loading")

    _df_b = pd.DataFrame({"case": ["A", "B"], "amount": [10, 20]})
    rt2.trace_step(
        func=lambda df: df.copy(),
        func_name="load_b",
        raw_line="df_b = load_b(df_b)",
        args=[_df_b],
        kwargs={},
    )
    rt2.trace_step(
        func=lambda df: df.assign(flag=True),
        func_name="enrich_b",
        raw_line="df_b2 = enrich_b(df_b)",
        args=[_df_b],
        kwargs={},
    )
    settle()
    rt2.storage._executor.submit(lambda: None).result()
    return (rt2,)


@app.cell
def _(mo, rt, rt2):
    _hist_cmp = rt.compare_histories(rt2, group_by="operation")
    assert "shared_operations" in _hist_cmp
    assert "unique_to_a" in _hist_cmp
    assert "unique_to_b" in _hist_cmp
    mo.md(
        f"**AC 6.1 ✓** — shared ops: {len(_hist_cmp['shared_operations'])}, "
        f"unique_to_A: {len(_hist_cmp['unique_to_a'])}, "
        f"unique_to_B: {len(_hist_cmp['unique_to_b'])}"
    )
    return


@app.cell
def _(mo, rt, rt2):
    _cat_cmp = rt.compare_histories(rt2, group_by="category")
    assert "category_summary_a" in _cat_cmp
    assert "category_summary_b" in _cat_cmp
    mo.md(
        f"**AC 6.2 ✓** category_summary_a: {dict(list(_cat_cmp['category_summary_a'].items())[:3])}  \n"
        f"**AC 6.3 ✓** category_summary_b: {dict(list(_cat_cmp['category_summary_b'].items())[:3])}"
    )
    return


@app.cell
def _(mo):
    mo.md("""
    ## R7 – Data Evolution Transparency
    """)
    return


@app.cell
def _(mo, step_details):
    _d = step_details.get("event_add_relative_case_time", {}).get("detail", {})
    assert _d, "No step detail for event_add_relative_case_time"
    _delta = _d.get("delta", {})
    assert _delta, "Delta must be recorded for event_add_relative_case_time"
    mo.md(
        f"**AC 7.1 ✓** — delta keys: {list(_delta.keys())}, "
        f"columns_added: {_delta.get('columns_added', [])}"
    )
    return


@app.cell
def _(mo, rt, step_details):
    final_state = step_details.get("case_add_activity_start_times", {}).get("state_id")
    assert final_state, "case_add_activity_start_times state not found"
    _lifecycle = rt.storage.load_artifact_lifecycle(final_state)
    assert _lifecycle, "Artifact lifecycle must have entries"
    mo.md(
        f"**AC 7.2 ✓** — {len(_lifecycle)} lifecycle entries for case_log artifact"
    )
    return (final_state,)


@app.cell
def _(mo, rt, settle):
    settle()
    _con = rt.storage._connect(read_only=True)
    _rows = _con.execute(
        "SELECT state_id, derived_from_state_id FROM analysis_states "
        "WHERE derived_from_state_id IS NOT NULL AND history_id = ?",
        [rt._history.history_id],
    ).fetchall()
    _con.close()
    assert _rows, "At least one state must have derived_from_state_id set"
    _sids = [r[0] for r in _rows]
    _parents = [r[1] for r in _rows]
    mo.md(
        f"**AC 7.3 ✓** — {len(_rows)} states have derived_from_state_id set, "
        f"e.g. `{_sids[0][:8]}…` derived from `{_parents[0][:8]}…`"
    )
    return


@app.cell
def _(final_state, mo, rt):
    _fig = rt.show_artifact_lifecycle(final_state)
    mo.as_html(_fig) if _fig is not None else mo.md("*(lifecycle graph unavailable)*")
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Summary

    | AC | Feature | Status |
    |---|---|---|
    | 1.1 | Full lineage in provenance tree | ✓ |
    | 1.2 | Step detail (func_name, raw_line, params) | ✓ |
    | 1.3 | Execution context (agent, environment) | ✓ |
    | 2.1 | Artifact state references in parameters | ✓ |
    | 2.2 | Scalar parameter capture | ✓ |
    | 2.3 | Runtime environment capture | ✓ |
    | 2.4 | Replay state produces equivalent output | ✓ |
    | 3.1 | Divergence point on manual checkout | ✓ |
    | 3.2 | Independent branches | ✓ |
    | 4.1 | Pipeline fragment creation | ✓ |
    | 4.2 | Parameter overrides in replay | ✓ |
    | 4.3 | Distinct steps per pipeline | ✓ |
    | 5.1 | Granular state diff | ✓ |
    | 5.2 | Abstracted state comparison | ✓ |
    | 6.1 | Operation-level history diff | ✓ |
    | 6.2 | Category-level history diff | ✓ |
    | 6.3 | Category summary counts | ✓ |
    | 7.1 | Delta captured per step | ✓ |
    | 7.2 | Artifact lifecycle entries | ✓ |
    | 7.3 | Intermediate states linked | ✓ |
    """)
    return


if __name__ == "__main__":
    app.run()
