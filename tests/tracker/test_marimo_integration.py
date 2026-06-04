"""Smoke tests for the Marimo integration path (kernel_hooks.py).

The Marimo integration has never been tested end-to-end. These tests
verify the pure-Python logic (_is_marimo_cell, _PATCHED guard) without
running a live Marimo server.
"""
import builtins
import pytest
from tracker.kernel_hooks import _is_marimo_cell


def test_cell_filename_detected():
    assert _is_marimo_cell("<cell-1>") is True
    assert _is_marimo_cell("<cell_abc>") is True


def test_normal_filename_rejected():
    assert _is_marimo_cell("my_script.py") is False
    assert _is_marimo_cell("<stdin>") is False


def test_marimo_keyword_in_filename_detected():
    assert _is_marimo_cell("__marimo__something") is True


def test_compile_patch_applied_once(tmp_path):
    """init_marimo must not wrap builtins.compile more than once (_PATCHED guard).

    init_marimo signature: (db_path, artifact_dir, agent_id, history_name, branch_name)
    It creates its own StorageBackend internally.
    """
    from tracker.kernel_hooks import init_marimo
    import tracker.kernel_hooks as kh

    original_compile = builtins.compile
    original_patched = kh._PATCHED

    try:
        kh._PATCHED = False
        rt1 = init_marimo(
            db_path=str(tmp_path / "prov.db"),
            artifact_dir=str(tmp_path / "art"),
            history_name="m1",
        )
        after_first = builtins.compile

        # Second init_marimo must not add another wrapping layer
        rt2 = init_marimo(
            db_path=str(tmp_path / "prov.db"),
            artifact_dir=str(tmp_path / "art"),
            history_name="m2",
        )
        after_second = builtins.compile

        assert after_first is after_second, (
            "builtins.compile was wrapped twice — _PATCHED guard is broken"
        )
    finally:
        builtins.compile = original_compile
        kh._PATCHED = original_patched
