"""
End-to-end test for the Marimo provenance integration.

Simulates what `marimo run` does without starting a real server:
  1. sitecustomize.py already patched mc.ast_compile (verified in subprocess)
  2. compile_cell() compiles each cell body
  3. exec() runs the bytecode in a shared globals dict
  4. Provenance DB is queried for recorded steps

Run with:
    uv run pytest tests/tracker/test_marimo_e2e.py -v
"""
import ast
import dis
import io
import os
import os.path
import subprocess
import sys
import pytest

pytest.importorskip("marimo")


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _has_trace_step(bytecode) -> bool:
    buf = io.StringIO()
    dis.dis(bytecode, file=buf)
    return "trace_step" in buf.getvalue()


def _compile(code: str, cell_id: str = "cell"):
    from marimo._ast.compiler import compile_cell
    return compile_cell(code, cell_id=cell_id)


# ---------------------------------------------------------------------------
# 1. Patch fires in a fresh subprocess (simulates kernel startup)
# ---------------------------------------------------------------------------

def test_sitecustomize_patches_ast_compile_in_subprocess(tmp_path):
    """sitecustomize.py in PYTHONPATH patches ast_compile in a fresh subprocess."""
    sc = tmp_path / "sitecustomize.py"
    sc.write_text(
        "from tracker.kernel_hooks import patch_marimo_ast_compile\n"
        "patch_marimo_ast_compile()\n"
    )
    env = os.environ.copy()
    env["PYTHONPATH"] = str(tmp_path) + os.pathsep + env.get("PYTHONPATH", "")
    result = subprocess.run(
        [sys.executable, "-c",
         "from marimo._ast import compiler as mc; print(mc.ast_compile.__name__)"],
        capture_output=True, text=True, env=env,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "_patched_ast_compile"


# ---------------------------------------------------------------------------
# 2. compile_cell transforms top-level assignments with non-omitted calls
# ---------------------------------------------------------------------------

def test_compile_cell_wraps_dataframe_call():
    result = _compile("df = pd.DataFrame({'a': [1, 2, 3]})", "c1")
    assert _has_trace_step(result.body), (
        "compile_cell did not produce trace_step bytecode — patch not active?"
    )


def test_compile_cell_does_not_wrap_omitted_calls():
    result = _compile("print('hello')", "c_omit")
    assert not _has_trace_step(result.body)


def test_compile_cell_does_not_wrap_init_marimo():
    result = _compile("rt = init_marimo(db_path='x.db')", "c_init")
    assert not _has_trace_step(result.body)


# ---------------------------------------------------------------------------
# 3. Full pipeline: init_marimo → exec cells → provenance recorded
# ---------------------------------------------------------------------------

@pytest.fixture()
def marimo_session(tmp_path):
    """Return (rt, shared_globals) after calling init_marimo."""
    import tracker.kernel_hooks as kh

    rt = kh.init_marimo(
        db_path=str(tmp_path / "prov.db"),
        artifact_dir=str(tmp_path / "art"),
        history_name="e2e_test",
    )
    # Shared globals dict — Marimo uses a single dict across all cells in a session
    glbls = {"__builtins__": __builtins__}
    return rt, glbls


def test_provenance_recorded_after_exec(marimo_session, tmp_path):
    """Compiling + exec'ing a cell should record exactly one provenance step."""
    import pandas as pd

    rt, glbls = marimo_session
    glbls["pd"] = pd

    cell = _compile("df = pd.DataFrame({'x': [1, 2, 3]})", "c_df")
    assert _has_trace_step(cell.body), "cell not transformed — fix patch first"

    exec(cell.body, glbls)  # noqa: S102

    assert "df" in glbls, "cell did not assign df"
    assert len(glbls["df"]) == 3

    # Allow async DB writes to flush
    rt.storage._executor.submit(lambda: None).result()

    graph = rt.storage.load_graph(rt._history.history_id)
    steps = graph["steps"]
    assert len(steps) == 1, f"expected 1 step, got {len(steps)}: {steps}"
    assert steps[0]["func_name"] == "pd.DataFrame"


def test_func_name_uses_call_site_expression(marimo_session):
    """func_name should be the exact source expression, not a generic class name."""
    import pandas as pd

    rt, glbls = marimo_session
    glbls["pd"] = pd
    glbls["my_df"] = pd.DataFrame({"a": [1, 2], "b": [3, 4]})

    # Simulate: is_illegal = my_df.apply(lambda r: r['a'] > 1, axis=1)
    cell = _compile(
        "result = my_df.apply(lambda r: r['a'] > 1, axis=1)", "c_apply"
    )
    assert _has_trace_step(cell.body)

    exec(cell.body, glbls)  # noqa: S102

    rt.storage._executor.submit(lambda: None).result()

    graph = rt.storage.load_graph(rt._history.history_id)
    steps = graph["steps"]
    assert len(steps) == 1
    # Must be "my_df.apply", NOT "DataFrame.apply"
    assert steps[0]["func_name"] == "my_df.apply", (
        f"got {steps[0]['func_name']!r} — monkey-patching fallback active?"
    )


def test_noop_runtime_before_init():
    """Calls that happen before init_marimo() must not raise — NoOpRuntime passthrough."""
    import tracker.kernel_hooks as kh

    original = kh._runtime
    from tracker.kernel_hooks import _NoOpRuntime
    kh._runtime = _NoOpRuntime()

    cell = _compile("x = dict(a=1)", "c_pre_init")
    glbls = {"__builtins__": __builtins__}
    exec(cell.body, glbls)  # noqa: S102
    assert glbls.get("x") == {"a": 1}

    kh._runtime = original  # restore


def test_multiple_cells_record_multiple_steps(marimo_session):
    """Each tracked cell execution adds one step to the DB."""
    import pandas as pd

    rt, glbls = marimo_session
    glbls["pd"] = pd

    for i, code in enumerate([
        "df1 = pd.DataFrame({'v': [1]})",
        "df2 = pd.DataFrame({'v': [2]})",
        "df3 = pd.DataFrame({'v': [3]})",
    ]):
        exec(_compile(code, f"c{i}").body, glbls)  # noqa: S102

    rt.storage._executor.submit(lambda: None).result()

    graph = rt.storage.load_graph(rt._history.history_id)
    assert len(graph["steps"]) == 3
