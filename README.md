# pmprov

**Analytic provenance tracking middleware for exploratory process mining notebooks.**

`pmprov` automatically records every meaningful decision an analyst makes during a Jupyter or Marimo session — which functions were called, with what arguments, on which data, and how the data changed — without requiring any modification to analyst code.

---

## How it works

```
Your cell code  →  AST rewriter  →  trace_step()  →  DuckDB/SQLite + Parquet
                   (transparent)     (non-blocking)
```

Every top-level function call is intercepted at the AST level, wrapped with a lightweight tracer, and recorded as a node in a provenance tree. The analyst's variables contain exactly the same values they would without the middleware.

---

## Getting started

**Prerequisites:** Python ≥ 3.13, [`uv`](https://docs.astral.sh/uv/)

```bash
git clone https://github.com/alvarobernal2412/pmprov.git
cd pmprov
uv pip install -e ".[full]"
```

**In a Jupyter notebook:**

```python
from tracker import init_jupyter, operation_type
import pandas as pd
import pm4py

# Register semantic labels for third-party functions (optional but recommended)
operation_type("data_loading",      pd.read_csv)
operation_type("process_discovery", pm4py.discover_petri_net_inductive)

# Start tracking — one call, no other changes to your code
rt = init_jupyter(history_name="My analysis")

# Your normal analysis code below — provenance is captured automatically
event_log = pd.read_csv("data/log.csv", parse_dates=["time:timestamp"])
net, im, fm = pm4py.discover_petri_net_inductive(event_log)
```

**In a Marimo notebook:**

Marimo support requires a one-time setup step. The middleware patches Marimo's AST compiler at Python startup via `sitecustomize.py`, which must be placed in your virtual environment (already in place if installed via `uv`).

**Structural requirement:** Marimo's reactive DAG is built from the original cell code *before* the AST rewriter runs. This means the rewriter cannot inject dependency edges automatically. To guarantee that `init_marimo()` runs before any tracked cell, **combine your path setup and `init_marimo()` call in one cell** and export all imports (`pd`, constants, etc.) from it. Downstream cells then depend on those names naturally, which implicitly orders them after init — no need to add `rt` to every cell's parameters.

```python
# ── Setup + init (one cell) ───────────────────────────────────────────────────
@app.cell
def _():
    import sys, pandas as pd
    from pathlib import Path
    from tracker import init_marimo, operation_type

    # ... path setup ...

    rt = init_marimo(history_name="My analysis")
    operation_type("data_loading", pd.read_csv)

    def settle():
        rt.storage._executor.submit(lambda: None).result()

    return pd, rt, settle, DATA_FILE  # export everything downstream cells need


# ── Analysis cells — no `rt` needed in params ─────────────────────────────────
@app.cell
def _(DATA_FILE, pd):          # depends on pd + DATA_FILE → init already ran
    event_log = pd.read_csv(DATA_FILE, parse_dates=["time:timestamp"])
    return (event_log,)
```

Run with:
```bash
uv run marimo run examples/marimo/provenance_demo.py
# or for interactive editing:
uv run marimo edit examples/marimo/provenance_demo.py
```

> **Note:** Provenance capture works in Marimo. Some introspection features (interactive widget, manual checkout) are not yet validated in the Marimo environment.

**Inspect the provenance graph:**

```python
rt.list_states()           # table of all recorded steps
rt.show_graph()            # static matplotlib tree
rt.show_graph_widget()     # interactive Jupyter widget (Jupyter only)
```

**Branch from a past state:**

```python
branch = rt.checkout("paste-state-id-here", branch_name="experiment-v2")
```

See [`examples/jupyter/provenance_demo.ipynb`](examples/jupyter/provenance_demo.ipynb) for a full Jupyter walkthrough and [`examples/marimo/provenance_demo.py`](examples/marimo/provenance_demo.py) for the Marimo equivalent, both using the RTFM dataset.

---

## Optional extras

| Extra | What it adds |
|---|---|
| `duckdb` | DuckDB storage backend (preferred) |
| `parquet` | DataFrame artifact persistence via pyarrow |
| `graph` | `show_graph()` via networkx + matplotlib |
| `widgets` | `show_graph_widget()` interactive panel |
| `marimo` | Marimo notebook support |
| `full` | Everything above |

```bash
uv pip install -e ".[full]"
```

---

## Project structure

```
models/    Pydantic v2 domain models (provenance entities)
tracker/   Runtime middleware (AST rewriter, tracer, storage)
examples/  Demo notebook + RTFM dataset helpers
```

---

## Running tests

```bash
uv sync --all-extras --dev
uv run pytest
```

## Git hooks (optional)

Enforces the test suite and conventional commit format (≤ 45 chars) on every commit:

```bash
uv run pre-commit install && uv run pre-commit install --hook-type commit-msg
```

Skip this step if you prefer to run checks manually.

---

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md).

## License

Apache 2.0 — see [LICENSE](LICENSE).
