"""AST NodeTransformer that wraps top-level Call expressions with trace_step."""

from __future__ import annotations

import ast

# ---------------------------------------------------------------------------
# Omit list
# ---------------------------------------------------------------------------
# Function names (or trailing method names) that should never be tracked.
# Matching is done against both the full func_name (e.g. "plt.show") and the
# last dotted segment (e.g. "show" from "fig.show").  Analysts can extend this
# set at runtime via tracker.omit_functions().

OMIT_FUNCTIONS: set[str] = {
    # Python builtins – output / inspection
    "print",
    "repr",
    "len",
    "type",
    "isinstance",
    "hasattr",
    "getattr",
    "setattr",
    "vars",
    "dir",
    # Dict / object retrieval — never a data transformation
    "get",
    "items",
    "keys",
    "values",
    "copy",
    "update",
    # DataFrame / Series display methods
    "head",
    "tail",
    "sample",
    "info",
    "describe",
    "value_counts",
    "nunique",
    "dtypes",
    # Plotting display
    "show",         # plt.show(), fig.show()
    "savefig",
    # IPython / Jupyter display helpers
    "display",
    "clear_output",
    # ProvTrack's own inspection / configuration / replay methods
    "show_graph",
    "show_graph_widget",
    "show_artifact_lifecycle",
    "list_states",
    "list_branches",
    "load_graph",
    "to_networkx",
    "checkout",
    "describe_state",
    "describe_step",
    "replay_state",
    "replay_pipeline",
    "create_pipeline",
    "load_pipeline_steps",
    "compare_states",
    "compare_states_abstracted",
    "compare_histories",
    "register_abstraction",
    "operation_type",
    "step_category",
    "omit_functions",
    "enable_logging",
    "get_logger",
    "init_jupyter",
    "init_marimo",
    "settle",
}


def _is_omitted(func_name: str) -> bool:
    """Return True if *func_name* matches an entry in OMIT_FUNCTIONS."""
    if func_name in OMIT_FUNCTIONS:
        return True
    # Match on the trailing segment so "event_log.head" matches "head".
    tail = func_name.rsplit(".", 1)[-1]
    if tail in OMIT_FUNCTIONS:
        return True
    # Skip any call whose root object starts with __ — these are Python/Jupyter
    # internals (e.g. __DW_SCOPE__['dispose'], __builtins__['exec']) that are
    # never analyst analysis operations.
    root = func_name.split(".")[0].split("[")[0]
    return root.startswith("__")


class ProvTrackTransformer(ast.NodeTransformer):
    """
    Intercepts the cell's AST before execution and rewrites every top-level
    statement of the form

        target = func(*args, **kwargs)      # ast.Assign
        func(*args, **kwargs)               # ast.Expr

    into

        target = _provtrack_runtime.trace_step(
            func=func,
            func_name="<source representation>",
            raw_line="<original source line>",
            args=[...positional args...],
            kwargs={...keyword args...},
        )

    Calls whose func_name (or trailing method name) appears in OMIT_FUNCTIONS
    are left untouched — they produce no provenance node.

    Statements nested inside ``for`` loops, ``if`` blocks, or locally-defined
    functions within the cell are deliberately left untouched.
    """

    _RUNTIME = "_provtrack_runtime"

    def visit_Module(self, node: ast.Module) -> ast.Module:
        node.body = [self._maybe_wrap(stmt) for stmt in node.body]
        return node

    # ------------------------------------------------------------------
    # Top-level statement dispatch
    # ------------------------------------------------------------------

    def _maybe_wrap(self, stmt: ast.stmt) -> ast.stmt:
        if isinstance(stmt, ast.Assign) and isinstance(stmt.value, ast.Call):
            if _is_omitted(ast.unparse(stmt.value.func)):
                return stmt
            raw = ast.unparse(stmt)
            stmt.value = self._wrap_call(stmt.value, raw)
            ast.fix_missing_locations(stmt)
            return stmt

        if isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Call):
            if _is_omitted(ast.unparse(stmt.value.func)):
                return stmt
            raw = ast.unparse(stmt)
            stmt.value = self._wrap_call(stmt.value, raw)
            ast.fix_missing_locations(stmt)
            return stmt

        return stmt

    # ------------------------------------------------------------------
    # Call wrapper
    # ------------------------------------------------------------------

    def _wrap_call(self, call: ast.Call, raw_line: str) -> ast.Call:
        func_name = ast.unparse(call.func)

        # args list – preserves ast.Starred nodes so *unpacking still works:
        # [a, b, *c] is valid in a list literal.
        args_node = ast.List(elts=list(call.args), ctx=ast.Load())

        # kwargs dict – ast.Dict with None key represents **unpacking:
        # {"key": val, **d} → keys=[Constant("key"), None], values=[val, d]
        kw_keys: list = []
        kw_vals: list = []
        for kw in call.keywords:
            kw_keys.append(None if kw.arg is None else ast.Constant(value=kw.arg))
            kw_vals.append(kw.value)
        kwargs_node = ast.Dict(keys=kw_keys, values=kw_vals)

        wrapped = ast.Call(
            func=ast.Attribute(
                value=ast.Name(id=self._RUNTIME, ctx=ast.Load()),
                attr="trace_step",
                ctx=ast.Load(),
            ),
            args=[],
            keywords=[
                ast.keyword(arg="func",      value=call.func),
                ast.keyword(arg="func_name", value=ast.Constant(value=func_name)),
                ast.keyword(arg="raw_line",  value=ast.Constant(value=raw_line)),
                ast.keyword(arg="args",      value=args_node),
                ast.keyword(arg="kwargs",    value=kwargs_node),
            ],
        )
        ast.copy_location(wrapped, call)
        ast.fix_missing_locations(wrapped)
        return wrapped
