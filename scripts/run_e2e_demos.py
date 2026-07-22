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

Usage:
    uv run python scripts/run_e2e_demos.py
"""
from __future__ import annotations

import subprocess
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

# (display_name, kind, relative_path)
DEMOS: list[tuple[str, str, str]] = [
    ("Jupyter: provenance_demo", "jupyter", "examples/jupyter/provenance_demo.ipynb"),
    ("Marimo: provenance_demo", "marimo", "examples/marimo/provenance_demo.py"),
    ("Marimo: plotly_capture_demo", "marimo", "examples/marimo/plotly_capture_demo.py"),
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


def main() -> int:
    results: list[tuple[str, bool, subprocess.CompletedProcess]] = []

    with tempfile.TemporaryDirectory() as tmp:
        out_dir = Path(tmp)
        for display_name, kind, rel_path in DEMOS:
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
