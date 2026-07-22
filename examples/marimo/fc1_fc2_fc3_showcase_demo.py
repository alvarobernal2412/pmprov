import marimo

__generated_with = "0.23.10"
app = marimo.App(width="medium")


@app.cell
def _():
    import sys
    from pathlib import Path

    _here = Path.cwd()
    PROJECT_ROOT = _here
    for _candidate in [_here, *_here.parents]:
        if (_candidate / "pyproject.toml").exists():
            PROJECT_ROOT = _candidate
            break

    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))

    EXAMPLES_DIR = PROJECT_ROOT / "examples"
    if str(EXAMPLES_DIR) not in sys.path:
        sys.path.insert(0, str(EXAMPLES_DIR))

    INPUT_FILE_NAME = "rtfm_full.csv"
    CASE_ID_COL = "case:concept:name"
    TIMESTAMP_COL = "time:timestamp"
    ACTIVITY_COL = "concept:name"

    data_path = PROJECT_ROOT / "examples" / "data" / INPUT_FILE_NAME

    print("Project root:", PROJECT_ROOT)
    print("Data file   :", data_path, "found" if data_path.exists() else "NOT FOUND")
    return ACTIVITY_COL, CASE_ID_COL, PROJECT_ROOT, TIMESTAMP_COL, data_path


@app.cell
def _(PROJECT_ROOT):
    import marimo as mo
    import pandas as pd

    from tracker import init_marimo, omit_functions, operation_type
    from tracker.operation_registry import step_category
    from tracker.snapshot_policy import snapshot_policy
    from utils.event_enricher import (
        create_case_log,
        event_add_relative_case_time,
        case_add_activity_start_times,
    )

    (PROJECT_ROOT / "examples" / "marimo" / "artifacts_fc_showcase").mkdir(
        parents=True, exist_ok=True
    )

    # init_marimo() is in OMIT_FUNCTIONS — not transformed by ProvTrackTransformer.
    # Own db_path/artifact_dir so this demo never collides with the other demos'
    # state files.
    rt = init_marimo(
        history_name="RTFM event log exploration — FC1/FC2/FC3 showcase (Marimo)",
        branch_name="main",
        db_path=str(PROJECT_ROOT / "examples" / "marimo" / "provenance_fc_showcase.db"),
        artifact_dir=str(PROJECT_ROOT / "examples" / "marimo" / "artifacts_fc_showcase"),
    )

    operation_type("data_loading", pd.read_csv)
    operation_type("case_aggregation", create_case_log)
    operation_type("attribute_derivation", event_add_relative_case_time)
    operation_type("attribute_derivation", case_add_activity_start_times)
    operation_type("case_filter", pd.DataFrame.apply)

    # StepCategory is the broader grouping FC-2's build_pruned_view(group_by_
    # category=True) actually collapses on — a separate, coarser registry from
    # OperationType above. Both attribute_derivation steps (3 and 4) share the
    # "log_enrichment" category, so they collapse into one edge; every other
    # step gets its own distinct category and stays separate.
    step_category("data_loading_phase", "data_loading")
    step_category("case_setup", "case_aggregation")
    step_category("log_enrichment", "attribute_derivation")
    step_category("delay_classification", "case_filter")

    # "number" omits the mo.ui.number(...) threshold widget constructor (UI
    # setup, not a pipeline operation — same reasoning as plotly_capture_demo.py
    # omitting "slider"). "DataFrame" omits this notebook's own debug-table
    # construction in the FC-2/FC-3 sections below (e.g. pd.DataFrame(_rows,
    # ...) building a summary table) — those calls happen on the very same `rt`
    # this demo showcases, so without this they'd pollute the provenance tree
    # being displayed with noise from the demo's own reporting code.
    omit_functions(
        "nunique", "mean", "sum", "min", "max",
        "vstack", "md", "as_html", "Html", "number", "DataFrame",
    )

    # ── FC-3: configurable snapshotting ──────────────────────────────────────
    # Step 2 (create_case_log) is a cheap, easily-recomputed aggregation — mark
    # it "never" so its output is never persisted as a Parquet artifact. Every
    # other step keeps the default "always" policy. The AnalysisStep itself is
    # still recorded either way — only artifact persistence is skipped.
    snapshot_policy("create_case_log", "never")

    def settle():
        rt.storage._executor.submit(lambda: None).result()

    print("Session ID   :", rt.session_id)
    print("History name :", rt._history.name)
    return (
        case_add_activity_start_times,
        create_case_log,
        event_add_relative_case_time,
        mo,
        pd,
        rt,
        settle,
    )


@app.cell
def _(mo):
    mo.md(r"""
    # FC-1 / FC-2 / FC-3 showcase

    Same RTFM pipeline as `provenance_demo.py`, with three added sections
    demonstrating each recent feature, **in the order FC-3 → FC-2 → FC-1** —
    each one builds on state the previous section produced:

    1. **FC-3 (configurable snapshotting)** — set up *before* the pipeline runs:
       `create_case_log`'s output is marked `snapshot_policy(..., "never")`, so
       we can later show its state has no artifact while every other step's does.
    2. **FC-2 (prune / curated view)** — after the pipeline (and its auto-branch)
       has run, build a pruned view that collapses same-category steps and hides
       the abandoned branch, then save and reload it.
    3. **FC-1 (shortest-path replay)** — pick the final branched state as a
       "finding" and package the minimal replay path into a brand-new,
       independent `AnalysisHistory`.
    """)
    return


@app.cell
def _(CASE_ID_COL, TIMESTAMP_COL, data_path, pd):
    # Step 1 – load the event log
    event_log = pd.read_csv(
        str(data_path),
        dtype={"org:resource": str, "matricola": str},
        parse_dates=[TIMESTAMP_COL],
    )
    print(f"Loaded {len(event_log):,} events across {event_log[CASE_ID_COL].nunique():,} cases")
    return (event_log,)


@app.cell
def _(create_case_log, event_log):
    # Step 2 – aggregate to one row per case.
    # snapshot_policy("create_case_log", "never") was set above — this step's
    # output will NOT get a Parquet artifact. The step itself is still recorded.
    case_log = create_case_log(event_log)
    print(f"Case log: {len(case_log):,} cases, {len(case_log.columns)} columns")
    return (case_log,)


@app.cell
def _(CASE_ID_COL, TIMESTAMP_COL, event_add_relative_case_time, event_log):
    # Step 3 – add relative case time
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
    # Step 4 – pivot activity start times
    case_log_pivoted = case_add_activity_start_times(
        case_log, event_log_enriched, CASE_ID_COL, ACTIVITY_COL, "rel_time"
    )
    time_cols = [c for c in case_log_pivoted.columns if c.endswith("::start")]
    print(f"Activity start-time columns: {time_cols}")
    return (case_log_pivoted,)


@app.cell
def _(case_log_pivoted):
    # Step 5 – derive Delay Send  [raw arithmetic — intentionally NOT tracked]
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
def _(mo):
    # Reactive threshold control — changing this re-runs Step 6 → auto-branch.
    threshold = mo.ui.number(
        value=90.0,
        start=1.0,
        stop=500.0,
        step=1.0,
        label="Illegal delay threshold (days)",
    )
    mo.vstack([mo.md("### Step 6 – Illegal delay threshold"), threshold])
    return (threshold,)


@app.cell
def _(case_log_with_delay, threshold):
    # Step 6 – classify cases with an illegal delay.
    # Moving the slider above changes this call's argument fingerprint — the
    # middleware detects the divergence from the prior run and auto-branches.
    is_illegal_delay = case_log_with_delay.apply(
        lambda case: case["Delay Send"] > threshold.value, axis=1
    )
    print(f"Threshold: {threshold.value} days")
    print(f"Cases flagged as illegal delay: {is_illegal_delay.sum():,}")
    return (is_illegal_delay,)


@app.cell
def _(is_illegal_delay, rt, settle):
    # Provenance graph — re-renders every time is_illegal_delay changes.
    _ = is_illegal_delay  # explicit dependency so Marimo re-runs this cell
    settle()
    rt.show_graph()
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 1. FC-3 in action — which states actually have an artifact?

    `create_case_log`'s output state has no Parquet artifact (we set
    `snapshot_policy("create_case_log", "never")` before the pipeline ran) —
    every other step's does, since the global default is `"always"`.
    """)
    return


@app.cell
def _(case_log, is_illegal_delay, mo, rt, settle):
    _ = is_illegal_delay
    settle()

    _con = rt.storage._connect(read_only=True)
    try:
        _rows = _con.execute("""
            SELECT s.func_name, ast.content_ref IS NOT NULL AS has_artifact
            FROM analysis_steps s
            LEFT JOIN artifact_states ast ON ast.analysis_state_id = s.output_state_id
            WHERE s.history_id = ?
            ORDER BY s.timestamp
        """, [rt._history.history_id]).fetchall()
    finally:
        _con.close()

    import pandas as _pd
    _ = case_log  # keep case_log in scope for readers following the pipeline above
    fc3_artifact_summary = _pd.DataFrame(_rows, columns=["func_name", "has_artifact"])
    mo.vstack([
        fc3_artifact_summary,
        mo.md(
            "`create_case_log` is the only step with `has_artifact = False` — "
            "its `AnalysisStep` was still recorded, but replaying it (rather than "
            "loading a snapshot) is what FC-1's `load_ancestor_chain` would fall "
            "back to if a curated history's chain ever passed through it."
        ),
    ])
    return (fc3_artifact_summary,)


@app.cell
def _(mo):
    mo.md(r"""
    ## 2. FC-2 in action — prune the tree into a curated view

    `build_pruned_view(group_by_category=True, hidden_branch_ids=[...])`
    collapses consecutive same-`OperationType` steps into single edges and hides
    the branch that the threshold slider's *previous* value produced (a
    stale/abandoned exploration path) — all computed in memory, never mutating
    the source history. `save_pruned_view` then persists just that
    configuration; `load_pruned_view` reloads it by re-rendering live against
    the (unchanged) source history.
    """)
    return


@app.cell
def _(fc3_artifact_summary, mo, rt, settle):
    _ = fc3_artifact_summary
    settle()

    _branches = rt.storage.load_branches(rt._history.history_id)
    _stale_branch_ids = [
        b["branch_id"] for b in _branches
        if b["name"] != "main" and b["divergence_point_id"] is not None
    ]

    pruned_view = rt.build_pruned_view(
        group_by_category=True,
        hidden_branch_ids=_stale_branch_ids,
    )
    view_id = rt.save_pruned_view(pruned_view, name="curated-illegal-delay-view")
    settle()

    reloaded_view = rt.load_pruned_view(view_id)

    import pandas as _pd
    edges_df = _pd.DataFrame([
        {
            "func_name": e["func_name"],
            "category": e["category"],
            "collapsed_step_ids": len(e["collapsed_step_ids"]),
        }
        for e in reloaded_view["edges"]
    ])

    mo.vstack([
        mo.md(
            f"Hid **{len(_stale_branch_ids)} stale branch(es)**. Pruned view "
            f"`{view_id[:8]}…` has **{len(reloaded_view['nodes'])} nodes** and "
            f"**{len(reloaded_view['edges'])} edges** (vs. the full tree's "
            f"unpruned step count above) — collapsed edges carry more than one "
            f"original `step_id` in `collapsed_step_ids`, so drill-down back to "
            f"the individual operations is never lost."
        ),
        edges_df,
    ])
    return (view_id,)


@app.cell
def _(mo):
    mo.md(r"""
    ## 3. FC-1 in action — package a finding as an independent history

    Take the **current tip of `main`** (the "finding") and ask for the minimal
    step sequence needed to reproduce it, then materialize that into a
    brand-new `AnalysisHistory` — its own history/branch/states/steps, never a
    reference into the source history's own rows. Loading it later never needs
    to query this notebook's history again.
    """)
    return


@app.cell
def _(mo, rt, settle, view_id):
    _ = view_id
    settle()

    target_state_id = rt._current_state_id
    replay_path = rt.find_shortest_replay_path(target_state_id)
    curated_history_id = rt.create_independent_history_from_state(
        target_state_id, name="curated-illegal-delay-finding"
    )
    settle()

    curated_graph = rt.storage.load_graph(history_id=curated_history_id)

    mo.md(
        f"- **Target state:** `{target_state_id[:8]}…`\n"
        f"- **Minimal replay path:** {len(replay_path)} step(s) "
        f"(only the nearest artifact-bearing ancestor onward — not the whole "
        f"6-step pipeline)\n"
        f"- **New independent history:** `{curated_history_id[:8]}…`, "
        f"`{len(curated_graph['states'])} states`, `{len(curated_graph['steps'])} steps`\n"
        f"- Querying `curated_history_id` never touches `{rt._history.history_id[:8]}…` "
        f"(this notebook's own history) again — confirmed by loading its graph "
        f"through the same `rt.storage`, keyed only by the new `history_id`."
    )
    return


if __name__ == "__main__":
    app.run()
