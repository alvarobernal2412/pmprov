#!/usr/bin/env python3
"""
Run the project's known-good demo notebooks end-to-end and fail loudly if any of
them errors at runtime. This exercises the real AST-rewriter + kernel-hook
integration path (Jupyter's ast_transformers / Marimo's ast_compile patch), which
unit tests that call RuntimeTracker.trace_step directly do not cover.

Deliberately excludes examples/jupyter/acceptance_criteria_demo.ipynb and
examples/marimo/acceptance_criteria_demo.py — both currently broken by pre-existing
bugs unrelated to this script (see docs/claude/checklist.md). Do not add them to
DEMOS below until those bugs are fixed; adding them here would turn CI red for a
reason unrelated to whatever change triggered the run.

Marimo demos get a second check beyond "did it render without error": `marimo
export html` renders each cell but — confirmed by direct experiment — never
executes a notebook's module-level code (only `@app.cell` bodies), so it cannot
exercise `patch_marimo_ast_compile()` when that call sits at module level per
kernel_hooks.py's own documented pattern. A notebook whose provenance tracking
was silently dead (e.g. via a never-reproducible sitecustomize.py hook — see
docs/claude/testing-plan-fc1-3.md item 8) would still render a clean HTML export
and report PASS here, which is exactly what happened until this check was added.
`python <notebook>.py` (the `if __name__ == "__main__": app.run()` path every
marimo notebook already supports) *does* run module-level code first, so it's
used here purely to verify at least one step actually got recorded — not as a
replacement for the export/render check above.

Usage:
    uv run python scripts/run_e2e_demos.py
"""
from __future__ import annotations

import subprocess
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

# (display_name, kind, relative_path, db_path_relative_to_repo_root)
# db_path is only meaningful for kind == "marimo" (used by the tracking-activation check).
DEMOS: list[tuple[str, str, str, str]] = [
    ("Jupyter: provenance_demo", "jupyter", "examples/jupyter/provenance_demo.ipynb", ""),
    (
        "Marimo: provenance_demo", "marimo", "examples/marimo/provenance_demo.py",
        "examples/marimo/provenance.db",
    ),
    (
        "Marimo: plotly_capture_demo", "marimo", "examples/marimo/plotly_capture_demo.py",
        "examples/marimo/provenance_plotly_demo.db",
    ),
]


def _run_jupyter(path: Path, out_dir: Path) -> subprocess.CompletedProcess:
    out_path = out_dir / f"{path.stem}.executed.ipynb"
    return subprocess.run(
        [
            sys.executable, "-m", "jupyter", "nbconvert",
            "--to", "notebook", "--execute",
            "--output", str(out_path),
            str(path),
        ],
        cwd=REPO_ROOT, capture_output=True, text=True, timeout=600,
    )


def _run_marimo(path: Path, out_dir: Path) -> subprocess.CompletedProcess:
    out_path = out_dir / f"{path.stem}.html"
    return subprocess.run(
        [
            sys.executable, "-m", "marimo", "export", "html",
            str(path), "-o", str(out_path), "--force",
        ],
        cwd=REPO_ROOT, capture_output=True, text=True, timeout=600,
    )


def _count_recorded_steps(db_path: Path) -> int:
    sys.path.insert(0, str(REPO_ROOT))
    from tracker.storage import DuckDBSQLiteBackend

    storage = DuckDBSQLiteBackend(db_path=db_path, artifact_dir=db_path.parent / "artifacts")
    # No history_id filter — the demo just ran and only wrote one history, and
    # this check only cares whether *anything* got recorded at all.
    return len(storage.load_graph(history_id=None)["steps"])


def _verify_tracking_activated(path: Path, db_path: Path) -> tuple[bool, str]:
    """Run a marimo demo in script mode (`python demo.py`) and confirm at
    least one step landed in its database — see module docstring for why
    the export/render check alone can't catch a dead tracking setup."""
    db_path.unlink(missing_ok=True)
    try:
        proc = subprocess.run(
            [sys.executable, str(path)],
            cwd=REPO_ROOT, capture_output=True, text=True, timeout=600,
        )
    except subprocess.TimeoutExpired as e:
        return False, f"Timed out after {e.timeout}s running {path.name} in script mode"

    if proc.returncode != 0:
        return False, f"`python {path.name}` exited {proc.returncode}:\n{proc.stderr}"

    if not db_path.exists():
        return False, f"`python {path.name}` produced no database at {db_path}"

    step_count = _count_recorded_steps(db_path)
    if step_count == 0:
        return False, (
            f"`python {path.name}` ran but recorded 0 steps — provenance "
            "tracking never activated (patch_marimo_ast_compile() likely "
            "isn't running, or isn't running before any traced call)."
        )
    return True, f"{step_count} step(s) recorded"


def main() -> int:
    results: list[tuple[str, bool, subprocess.CompletedProcess]] = []

    with tempfile.TemporaryDirectory() as tmp:
        out_dir = Path(tmp)
        for display_name, kind, rel_path, db_rel_path in DEMOS:
            path = REPO_ROOT / rel_path
            runner = _run_jupyter if kind == "jupyter" else _run_marimo
            print(f"::group::{display_name}", flush=True)
            try:
                proc = runner(path, out_dir)
                ok = proc.returncode == 0
            except subprocess.TimeoutExpired as e:
                proc = subprocess.CompletedProcess(
                    args=e.cmd, returncode=1, stdout=e.stdout or "",
                    stderr=f"Timed out after {e.timeout}s",
                )
                ok = False
            print(proc.stdout)
            print(proc.stderr, file=sys.stderr)

            if ok and kind == "marimo":
                tracking_ok, tracking_msg = _verify_tracking_activated(
                    path, REPO_ROOT / db_rel_path
                )
                print(f"[tracking check] {tracking_msg}")
                if not tracking_ok:
                    ok = False
                    proc = subprocess.CompletedProcess(
                        args=proc.args, returncode=1,
                        stdout=proc.stdout, stderr=f"{proc.stderr}\n{tracking_msg}",
                    )

            print("::endgroup::", flush=True)
            results.append((display_name, ok, proc))

    print("\n--- E2E demo summary ---")
    all_ok = True
    for display_name, ok, _ in results:
        status = "PASS" if ok else "FAIL"
        print(f"  [{status}] {display_name}")
        all_ok = all_ok and ok

    if not all_ok:
        print("\nFailing demo(s) full stderr:")
        for display_name, ok, proc in results:
            if not ok:
                print(f"\n=== {display_name} ===")
                print(proc.stderr)

    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
