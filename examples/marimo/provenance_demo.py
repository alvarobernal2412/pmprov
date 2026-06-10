import marimo

__generated_with = "0.23.9"
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

    from tracker import init_marimo, omit_functions, operation_type, enable_logging
    from utils.event_enricher import (
        create_case_log,
        event_add_relative_case_time,
        case_add_activity_start_times,
    )

    enable_logging(level="DEBUG")

    (PROJECT_ROOT / "examples" / "marimo" / "artifacts").mkdir(parents=True, exist_ok=True)

    # init_marimo() is in OMIT_FUNCTIONS — not transformed by ProvTrackTransformer.
    # It creates the RuntimeTracker and sets tracker.kernel_hooks._runtime = rt,
    # replacing the _NoOpRuntime that was active since sitecustomize.py ran.
    rt = init_marimo(
        history_name="RTFM event log exploration (Marimo)",
        branch_name="main",
        db_path=str(PROJECT_ROOT / "examples" / "marimo" / "provenance.db"),
        artifact_dir=str(PROJECT_ROOT / "examples" / "marimo" / "artifacts"),
    )

    operation_type("data_loading", pd.read_csv)
    operation_type("case_aggregation", create_case_log)
    operation_type("attribute_derivation", event_add_relative_case_time)
    operation_type("attribute_derivation", case_add_activity_start_times)
    operation_type("case_filter", pd.DataFrame.apply)

    omit_functions("nunique", "mean", "sum", "min", "max")

    print("Session ID   :", rt.session_id)
    print("History name :", rt._history.name)
    return (
        case_add_activity_start_times,
        create_case_log,
        event_add_relative_case_time,
        mo,
        pd,
        rt,
    )


@app.cell
def _(CASE_ID_COL, TIMESTAMP_COL, data_path, pd):
    # Step 1 – load the event log
    event_log = pd.read_csv(
        str(data_path),
        dtype={"org:resource": str, "matricola": str},
        parse_dates=[TIMESTAMP_COL],
    )
    print(f"Loaded {len(event_log):,} events across {event_log[CASE_ID_COL].nunique():,} cases")
    event_log.head(3)
    return (event_log,)


@app.cell
def _(create_case_log, event_log):
    # Step 2 – aggregate to one row per case
    case_log = create_case_log(event_log)
    print(f"Case log: {len(case_log):,} cases, {len(case_log.columns)} columns")
    case_log.head(3)
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
    # Reactive threshold control — changing this re-runs Step 6 and the graph.
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
    # Step 6 – classify cases with an illegal delay
    # ProvTrackTransformer rewrites this to:
    #   is_illegal_delay = __import__('tracker.kernel_hooks', fromlist=['_runtime'])
    #       ._runtime.trace_step(func=case_log_with_delay.apply,
    #                            func_name="case_log_with_delay.apply", ...)
    is_illegal_delay = case_log_with_delay.apply(
        lambda case: case["Delay Send"] > threshold.value, axis=1
    )
    print(f"Threshold: {threshold.value} days")
    print(f"Cases flagged as illegal delay: {is_illegal_delay.sum():,}")
    is_illegal_delay.head()
    return (is_illegal_delay,)


@app.cell
def _(is_illegal_delay, rt):
    # Provenance graph — re-renders every time is_illegal_delay changes.
    _ = is_illegal_delay  # explicit dependency so Marimo re-runs this cell
    rt.show_graph()
    return


if __name__ == "__main__":
    app.run()
