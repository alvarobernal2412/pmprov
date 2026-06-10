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
    # Provenance capture — Plotly + RTFM

    Demonstrates which calls the middleware captures automatically.

    | Call | Captured? | Why |
    |---|---|---|
    | `pd.read_csv(...)` | ✓ | top-level assignment, RHS is a Call |
    | `go.Figure()` | ✓ | top-level assignment, RHS is a Call |
    | `fig.add_trace(...)` | ✓ | top-level bare expression Call |
    | `fig.update_layout(...)` | ✓ | top-level bare expression Call |
    | `go.Histogram(...)` inside `add_trace` | ✗ | nested arg, not a top-level statement |
    | `fig.show()` | ✗ | `"show"` is in the default omit list |
    | arithmetic / subscript | ✗ | RHS is not a Call node |

    Moving the slider changes `nbinsx` → the middleware detects a parameter
    change on `fig.add_trace` and automatically creates a new branch.
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

    import pandas as pd
    import plotly.graph_objects as go
    from tracker import init_marimo, omit_functions, operation_type

    DATA_FILE = str(_repo_root / "examples" / "data" / "rtfm_full.csv")
    TIMESTAMP_COL = "time:timestamp"
    AMOUNT_COL = "amount"

    _artifact_dir = str(_repo_root / "examples" / "marimo" / "artifacts_plotly_demo")
    _db_path = str(_repo_root / "examples" / "marimo" / "provenance_plotly_demo.db")
    Path(_artifact_dir).mkdir(parents=True, exist_ok=True)

    rt = init_marimo(
        history_name="RTFM plotly capture demo",
        branch_name="main",
        artifact_dir=_artifact_dir,
        db_path=_db_path,
    )

    operation_type("data_loading", pd.read_csv)

    # Omit UI setup calls — widget constructors, not data operations.
    omit_functions("slider", "number", "vstack", "md", "as_html")

    def settle():
        rt.storage._executor.submit(lambda: None).result()

    print(f"History: {rt._history.history_id[:8]}…")
    return AMOUNT_COL, DATA_FILE, TIMESTAMP_COL, go, pd, rt, settle


@app.cell
def _(DATA_FILE, TIMESTAMP_COL, pd):
    # ── CAPTURED: pd.read_csv is a top-level Call assigned to event_log ──
    event_log = pd.read_csv(
        DATA_FILE,
        dtype={"org:resource": str, "matricola": str},
        parse_dates=[TIMESTAMP_COL],
    )
    print(f"Loaded {len(event_log):,} events, {event_log['case:concept:name'].nunique():,} cases")
    return (event_log,)


@app.cell
def _(AMOUNT_COL, event_log):
    # Subscript — NOT captured (RHS is not a Call node)
    amount_series = event_log[AMOUNT_COL].dropna()
    print(f"amount range: {amount_series.min():.0f} – {amount_series.max():.0f}")
    return (amount_series,)


@app.cell
def _(mo):
    # mo.ui.slider → omitted (UI setup, not a data operation)
    nbins = mo.ui.slider(start=10, stop=200, step=10, value=50, label="Number of bins (nbinsx)")
    mo.vstack([mo.md("### Histogram bin control"), nbins])
    return (nbins,)


@app.cell
def _(amount_series, go, nbins):
    # ── CAPTURED: go.Figure() — top-level assignment, RHS is a Call ──
    fig = go.Figure()

    # ── CAPTURED: fig.add_trace() — top-level bare expression Call ──
    # go.Histogram() is an *argument* here → NOT captured directly
    fig.add_trace(go.Histogram(x=amount_series, nbinsx=nbins.value))

    # ── CAPTURED: fig.update_layout() — top-level bare expression Call ──
    fig.update_layout(
        title=f"RTFM fine amounts — {nbins.value} bins",
        xaxis_title="Amount (€)",
        yaxis_title="Count",
        bargap=0.05,
    )

    # fig.show() would NOT be captured ("show" is in the default omit list)
    return (fig,)


@app.cell
def _(fig):
    # Marimo renders plotly figures natively when returned from a cell
    fig
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## What was captured

    The table below shows every step recorded so far.
    Notice:
    - `pd.read_csv` appears once (data loading phase)
    - `go.Figure`, `fig.add_trace`, `fig.update_layout` each appear once per
      slider position — moving the slider creates a new **auto-branch**
      because the `nbinsx` argument fingerprint changes
    """)
    return


@app.cell
def _(fig, rt, settle):
    _ = fig  # depend on fig so this cell reruns when slider changes
    settle()
    states_df = rt.list_states()
    states_df
    return (states_df,)


@app.cell
def _(mo, rt, states_df):
    _ = states_df  # rerun when states update
    _fig = rt.show_graph()
    mo.as_html(_fig) if _fig is not None else mo.md("*(graph unavailable)*")
    return


@app.cell
def _(mo, rt, settle, states_df):
    _ = states_df
    settle()
    # Show the detail of the most recent fig.add_trace step
    _add_trace_row = (
        states_df.dropna(subset=["produced_by_step_id"])
        .pipe(lambda df: df[df["produced_by_step_id"].apply(
            lambda sid: rt.describe_step(sid).get("func_name", "") == "fig.add_trace"
        )])
    )
    if not _add_trace_row.empty:
        _step_id = _add_trace_row.iloc[-1]["produced_by_step_id"]
        _detail = rt.describe_step(_step_id)
        mo.md(
            f"### Detail: most recent `fig.add_trace` step\n\n"
            f"- **func_name:** `{_detail.get('func_name')}`\n"
            f"- **raw_line:** `{_detail.get('raw_line', '')[:80]}`\n"
            f"- **branch:** `{_detail.get('branch_name')}`\n"
            f"- **params:** {len(_detail.get('params', []))} captured\n"
            f"- **delta:** `{_detail.get('delta')}`\n"
        )
    else:
        mo.md("*(no fig.add_trace step recorded yet)*")
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
