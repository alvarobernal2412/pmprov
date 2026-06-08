import os
os.environ.setdefault("MPLBACKEND", "Agg")

try:
    from tracker.kernel_hooks import patch_marimo_ast_compile
    patch_marimo_ast_compile()
except Exception:
    pass
