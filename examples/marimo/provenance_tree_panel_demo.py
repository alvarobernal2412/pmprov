import marimo

__generated_with = "0.23.14"
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
    from utils.event_enricher import (
        create_case_log,
        event_add_relative_case_time,
        case_add_activity_start_times,
    )

    (PROJECT_ROOT / "examples" / "marimo" / "artifacts_tree_panel_demo").mkdir(
        parents=True, exist_ok=True
    )

    # init_marimo() is in OMIT_FUNCTIONS — not transformed by ProvTrackTransformer.
    # Own db_path/artifact_dir so this demo never collides with provenance_demo.py's
    # or plotly_capture_demo.py's state files.
    rt = init_marimo(
        history_name="RTFM event log exploration — live tree panel (Marimo)",
        branch_name="main",
        db_path=str(PROJECT_ROOT / "examples" / "marimo" / "provenance_tree_panel.db"),
        artifact_dir=str(PROJECT_ROOT / "examples" / "marimo" / "artifacts_tree_panel_demo"),
    )

    operation_type("data_loading", pd.read_csv)
    operation_type("case_aggregation", create_case_log)
    operation_type("attribute_derivation", event_add_relative_case_time)
    operation_type("attribute_derivation", case_add_activity_start_times)
    operation_type("case_filter", pd.DataFrame.apply)

    omit_functions("nunique", "mean", "sum", "min", "max", "vstack", "md", "as_html", "Html")

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
    # Live provenance tree — fixed right panel

    Same RTFM pipeline as `provenance_demo.py`, but instead of a plain inline
    `rt.show_graph()` cell, the provenance tree is rendered in a **fixed panel on
    the right edge of the page** (`position: fixed`, not `mo.sidebar()` — this
    marimo version's sidebar renders in a different part of the DOM and can't be
    pinned to the right; see `plotly_capture_demo.py` for the same technique).

    The panel is the last cell in this notebook, but because it's `position: fixed`
    it stays visible while you scroll through the pipeline above. It depends on
    `states_df`, so every traced step — and every auto-branch from moving the
    threshold slider below — re-renders it immediately.
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
    # Step 2 – aggregate to one row per case
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
    # Reactive threshold control — changing this re-runs Step 6 and the tree panel.
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
    # Moving the slider above changes this call's argument fingerprint, which the
    # middleware detects as a divergence from the prior run — auto-branch fires,
    # and the live tree panel below picks up the new branch on its next render.
    is_illegal_delay = case_log_with_delay.apply(
        lambda case: case["Delay Send"] > threshold.value, axis=1
    )
    print(f"Threshold: {threshold.value} days")
    print(f"Cases flagged as illegal delay: {is_illegal_delay.sum():,}")
    return (is_illegal_delay,)


@app.cell
def _(is_illegal_delay, mo, rt, settle):
    # ── Live provenance tree — fixed right panel ──────────────────────────────
    # Depends on is_illegal_delay so every pipeline re-run (including auto-branch
    # from the threshold slider) refreshes the panel. settle() drains the async
    # write queue first so the graph reflects the step that was just traced.
    _ = is_illegal_delay
    settle()

    _graph_fig = rt.show_graph()
    _graph_html = (
        mo.as_html(_graph_fig)
        if _graph_fig is not None
        else mo.md("*(no provenance steps recorded yet)*")
    )

    _branches_df = rt.list_branches()
    _total_steps = int(_branches_df["step_count"].sum()) if not _branches_df.empty else 0
    _branch_count = len(_branches_df)

    _panel_content = mo.vstack([
        mo.md("### Live provenance tree"),
        mo.md(
            f"**{_total_steps} step{'s' if _total_steps != 1 else ''} recorded** "
            f"across **{_branch_count} branch{'es' if _branch_count != 1 else ''}**"
        ),
        _graph_html,
    ])

    # Fixed right-hand panel: plain CSS `position: fixed`, not mo.sidebar — sits
    # outside marimo's collapsible-sidebar shadow DOM entirely, so it can be
    # pinned to the right edge. Same technique as plotly_capture_demo.py.
    provenance_panel = mo.Html(f"""
    <div class="custom-right-sidebar">
        {_panel_content.text}
    </div>

    <style>
    .custom-right-sidebar {{
        position: fixed;
        top: 0;
        right: 0;
        width: 420px;
        height: 100vh;
        background-color: var(--bg-app, #f9f9f9);
        border-left: 1px solid var(--border-neutral, #e5e5e5);
        padding: 20px;
        box-sizing: border-box;
        z-index: 100;
        overflow-y: auto;
    }}

    .marimo-notebook, main {{
        margin-right: 440px !important;
    }}
    </style>
    """)

    provenance_panel
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
