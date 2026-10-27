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
    "update",
    # DataFrame display methods that return nothing worth keeping
    "info",
    "dtypes",
    # Global configuration — changes how things print, not the data
    "set_option",
    "reset_option",
    # Plotting display
    "show",         # plt.show(), fig.show()
    "savefig",
    # Path / sys / OS system calls — never analysis operations
    "cwd",
    "mkdir",
    "insert",   # sys.path.insert
    "append",   # sys.path.append
    "exists",   # path.exists()
    "resolve",  # path.resolve()
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
    "find_shortest_replay_path",
    "create_independent_history_from_state",
    "create_pipeline",
    "load_pipeline_steps",
    "compare_states",
    "compare_states_abstracted",
    "compare_histories",
    "register_abstraction",
    "apply_abstractions",
    "build_pruned_view",
    "save_pruned_view",
    "load_pruned_view",
    "operation_type",
    "step_category",
    "snapshot_policy",
    "snapshot_policy_for_type",
    "set_default_snapshot_threshold",
    "omit_functions",
    "enable_logging",
    "get_logger",
    "init_jupyter",
    "init_marimo",
    "settle",
    # Storage read-back methods — never analyst operations
    "load_artifact",
    "load_output_artifact_state_id",
    "load_artifact_lifecycle",
    "load_artifact_path",
    "load_state_detail",
    "load_step_detail",
    "load_states_rich",
    "load_branches",
    "load_pipeline_by_name",
    # DB / connection primitives — never analyst operations
    "_connect",
    "execute",
    "fetchall",
    "fetchone",
    "close",
    # stdlib helpers that appear in notebooks but are not analysis steps
    "mkdtemp",
    "submit",
    "result",
}


# Inspection methods that are display-only as a bare statement (``df.head()``)
# but whose result is real data once it is bound (``x = df.copy()``,
# ``counts = df[c].value_counts()``): omitted only when the result is unbound.
OMIT_WHEN_UNBOUND: set[str] = {
    "copy",
    "head",
    "tail",
    "sample",
    "describe",
    "value_counts",
    "nunique",
}


def _is_omitted(func_name: str, bound: bool = False) -> bool:
    """Return True if *func_name* matches the omit list.

    *bound* says the call's result is assigned to a name; the
    OMIT_WHEN_UNBOUND entries then still count as analysis steps.
    """
    if func_name in OMIT_FUNCTIONS:
        return True
    # Match on the trailing segment so "event_log.info" matches "info".
    tail = func_name.rsplit(".", 1)[-1]
    if tail in OMIT_FUNCTIONS:
        return True
    if not bound and (func_name in OMIT_WHEN_UNBOUND or tail in OMIT_WHEN_UNBOUND):
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

    # Runtime lookup expression used in generated code.
    # We use __import__('tracker.kernel_hooks', fromlist=['_runtime'])._runtime
    # rather than a bare name like _provtrack_runtime so the reference always
    # resolves via Python's import system, regardless of the builtins dict that
    # the executing env (Marimo AppScriptRunner, Jupyter, plain Python) provides.
    _RUNTIME = "_provtrack_runtime"   # kept for Jupyter backward-compat only
    _USE_IMPORT_LOOKUP = True         # emit import-based lookup instead

    @staticmethod
    def _runtime_node() -> ast.expr:
        """AST node for ``__import__('tracker.kernel_hooks', fromlist=['_runtime'])._runtime``."""
        return ast.Attribute(
            value=ast.Call(
                func=ast.Name(id="__import__", ctx=ast.Load()),
                args=[ast.Constant(value="tracker.kernel_hooks")],
                keywords=[
                    ast.keyword(
                        arg="fromlist",
                        value=ast.List(
                            elts=[ast.Constant(value="_runtime")],
                            ctx=ast.Load(),
                        ),
                    )
                ],
            ),
            attr="_runtime",
            ctx=ast.Load(),
        )

    def visit_Module(self, node: ast.Module) -> ast.Module:
        body: list[ast.stmt] = []
        for stmt in node.body:
            wrapped = self._maybe_wrap(stmt)
            body.extend(wrapped if isinstance(wrapped, list) else [wrapped])
        node.body = body
        return node

    # ------------------------------------------------------------------
    # Top-level statement dispatch
    # ------------------------------------------------------------------

    def _maybe_wrap(self, stmt: ast.stmt):
        if isinstance(stmt, ast.Assign):
            return self._wrap_assign(stmt)

        if isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Call):
            if _is_omitted(ast.unparse(stmt.value.func)):
                return stmt
            raw = ast.unparse(stmt)
            targets = None
            if _has_inplace(stmt.value):
                root = _call_root(stmt.value.func)
                targets = [root.id] if root is not None else None
            stmt.value = self._wrap_call(stmt.value, raw, targets=targets)
            ast.fix_missing_locations(stmt)
            return stmt

        if isinstance(stmt, (ast.For, ast.While)):
            return self._wrap_loop(stmt)

        return stmt

    def _wrap_assign(self, stmt: ast.Assign):
        raw = ast.unparse(stmt)
        target = stmt.targets[0] if len(stmt.targets) == 1 else None

        # df['c'] = expr / df.loc[m, 'c'] = expr / df.columns = expr
        if isinstance(target, (ast.Subscript, ast.Attribute)):
            setter = _setter(target)
            if setter is not None:
                op, obj_expr, key_nodes = setter
                call = self._step_call(
                    op, raw, args=[obj_expr, *key_nodes, stmt.value], targets=[_root_name(target)],
                )
                new = ast.Expr(value=call)
                ast.copy_location(new, stmt)
                ast.fix_missing_locations(new)
                return new

        if isinstance(stmt.value, ast.Call):
            if _is_omitted(ast.unparse(stmt.value.func), bound=True):
                return stmt
            stmt.value = self._wrap_call(stmt.value, raw, targets=_assigned_names(stmt))
            ast.fix_missing_locations(stmt)
            return stmt

        # x = df[mask] / x = df[cols] / x = df[:n] / x = df.loc[...]
        if isinstance(stmt.value, ast.Subscript) and isinstance(target, ast.Name):
            op, obj_expr, key_node = _getter(stmt.value)
            stmt.value = self._step_call(op, raw, args=[obj_expr, key_node], targets=[target.id])
            ast.fix_missing_locations(stmt)
            return stmt

        # df2 = df: a second name for an object that is already tracked
        if isinstance(stmt.value, ast.Name) and isinstance(target, ast.Name):
            stmt.value = self._step_call("alias", raw, args=[stmt.value], targets=[target.id])
            ast.fix_missing_locations(stmt)
            return stmt

        # x = df['a'] > 3 / x = a & b / x = df.T / x = df[c].dt.date
        if isinstance(stmt.value, (ast.Compare, ast.BinOp, ast.UnaryOp, ast.BoolOp, ast.Attribute))                 and isinstance(target, ast.Name):
            stmt.value = self._step_call("value", raw, args=[stmt.value], targets=[target.id])
            ast.fix_missing_locations(stmt)
            return stmt

        return stmt

    def _wrap_loop(self, loop):
        names = _loop_mutations(loop)
        if not names:
            return loop
        raw = ast.unparse(loop)
        kind = "for" if isinstance(loop, ast.For) else "while"
        names_node = ast.List(elts=[ast.Constant(value=n) for n in names], ctx=ast.Load())
        globals_call = ast.Call(func=ast.Name(id="globals", ctx=ast.Load()), args=[], keywords=[])
        begin = ast.Expr(value=ast.Call(
            func=ast.Attribute(value=self._runtime_node(), attr="begin_block", ctx=ast.Load()),
            args=[ast.Constant(value=raw), names_node, globals_call], keywords=[],
        ))
        collected = ast.Call(
            func=ast.Attribute(
                value=ast.Call(
                    func=ast.Name(id="__import__", ctx=ast.Load()),
                    args=[ast.Constant(value="tracker.step_ops")],
                    keywords=[ast.keyword(arg="fromlist", value=ast.List(elts=[ast.Constant(value="collect")], ctx=ast.Load()))],
                ),
                attr="collect", ctx=ast.Load()),
            args=[globals_call, names_node], keywords=[],
        )
        end = ast.Expr(value=self._step_call(
            "block", raw, args=[collected], targets=names, named_outputs=True, func_name=kind,
        ))
        for node in (begin, end):
            ast.copy_location(node, loop)
            ast.fix_missing_locations(node)
        return [begin, loop, end]

    # ------------------------------------------------------------------
    # Call wrappers
    # ------------------------------------------------------------------

    _STEP_OPS = {
        "getitem": "getitem", "loc_getitem": "loc_getitem", "iloc_getitem": "iloc_getitem",
        "setitem": "setitem", "loc_setitem": "loc_setitem", "iloc_setitem": "iloc_setitem",
        "setattr": "setattr_", "block": "block_result", "value": "value", "alias": "alias",
    }
    # Name recorded for the step (what the history shows), by helper.
    _STEP_NAMES = {
        "loc_getitem": "loc", "iloc_getitem": "iloc", "value": "expr",
        "loc_setitem": "setitem", "iloc_setitem": "setitem", "setattr": "set_attr",
    }

    def _step_call(self, op: str, raw_line: str, *, args: list, targets: list,
                   named_outputs: bool = False, func_name: str | None = None) -> ast.Call:
        """``trace_step(func=tracker.step_ops.<op>, ...)`` for a statement that is not a call."""
        attr = self._STEP_OPS[op]
        func = ast.Attribute(
            value=ast.Call(
                func=ast.Name(id="__import__", ctx=ast.Load()),
                args=[ast.Constant(value="tracker.step_ops")],
                keywords=[ast.keyword(arg="fromlist", value=ast.List(elts=[ast.Constant(value=attr)], ctx=ast.Load()))],
            ),
            attr=attr, ctx=ast.Load(),
        )
        keywords = [
            ast.keyword(arg="func", value=func),
            ast.keyword(arg="func_name", value=ast.Constant(value=func_name or self._STEP_NAMES.get(op, op))),
            ast.keyword(arg="raw_line", value=ast.Constant(value=raw_line)),
            ast.keyword(arg="args", value=ast.List(elts=args, ctx=ast.Load())),
            ast.keyword(arg="kwargs", value=ast.Dict(keys=[], values=[])),
            ast.keyword(arg="targets", value=ast.List(elts=[ast.Constant(value=t) for t in targets], ctx=ast.Load())),
        ]
        if named_outputs:
            keywords.append(ast.keyword(arg="named_outputs", value=ast.Constant(value=True)))
        return ast.Call(
            func=ast.Attribute(value=self._runtime_node(), attr="trace_step", ctx=ast.Load()),
            args=[], keywords=keywords,
        )

    def _wrap_call(self, call: ast.Call, raw_line: str, targets: list | None = None) -> ast.Call:
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

        extra = []
        if targets:
            extra.append(ast.keyword(arg="targets", value=ast.List(
                elts=[ast.Constant(value=t) for t in targets], ctx=ast.Load())))
        root = _call_root(call.func)
        if root is not None:
            extra.append(ast.keyword(arg="root", value=ast.Name(id=root.id, ctx=ast.Load())))
            extra.append(ast.keyword(arg="root_name", value=ast.Constant(value=root.id)))

        wrapped = ast.Call(
            func=ast.Attribute(
                value=self._runtime_node(),
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
                *extra,
            ],
        )
        ast.copy_location(wrapped, call)
        ast.fix_missing_locations(wrapped)
        return wrapped


# ---------------------------------------------------------------------------
# Statement analysis helpers
# ---------------------------------------------------------------------------

def _has_inplace(call: ast.Call) -> bool:
    return any(
        kw.arg == "inplace" and isinstance(kw.value, ast.Constant) and kw.value.value is True
        for kw in call.keywords
    )


def _root_name(node: ast.AST) -> str | None:
    """The Name at the bottom of ``a.b[c].d`` style target chains."""
    while isinstance(node, (ast.Attribute, ast.Subscript, ast.Call)):
        node = node.func if isinstance(node, ast.Call) else node.value
    return node.id if isinstance(node, ast.Name) else None


def _call_root(func: ast.expr) -> ast.Name | None:
    """The Name a method call chain starts from (``df`` in ``df.groupby(k).agg``); None for plain functions."""
    if not isinstance(func, ast.Attribute):
        return None
    node = func.value
    while isinstance(node, (ast.Attribute, ast.Subscript, ast.Call)):
        node = node.func if isinstance(node, ast.Call) else node.value
    return node if isinstance(node, ast.Name) else None


def _assigned_names(stmt: ast.Assign) -> list[str]:
    names: list[str] = []
    for target in stmt.targets:
        elts = target.elts if isinstance(target, (ast.Tuple, ast.List)) else [target]
        for elt in elts:
            root = _root_name(elt)
            if root and root not in names:
                names.append(root)
    return names


def _key_node(node: ast.expr) -> ast.expr:
    """A subscript key as an ordinary expression (``a:b`` becomes ``slice(a, b)``)."""
    if isinstance(node, ast.Slice):
        parts = [node.lower, node.upper, node.step]
        return ast.Call(
            func=ast.Attribute(
                value=ast.Call(func=ast.Name(id="__import__", ctx=ast.Load()),
                               args=[ast.Constant(value="builtins")], keywords=[]),
                attr="slice", ctx=ast.Load()),
            args=[p if p is not None else ast.Constant(value=None) for p in parts], keywords=[],
        )
    if isinstance(node, ast.Tuple):
        return ast.Tuple(elts=[_key_node(e) for e in node.elts], ctx=ast.Load())
    return node


def _getter(sub: ast.Subscript):
    base = sub.value
    if isinstance(base, ast.Attribute) and base.attr in ("loc", "iloc"):
        return f"{base.attr}_getitem", base.value, _key_node(sub.slice)
    return "getitem", base, _key_node(sub.slice)


def _setter(target: ast.expr):
    """(op, object, [key/attribute nodes]) for an assignable ``name[...]``, ``name.loc[...]`` or ``name.attr``."""
    if isinstance(target, ast.Subscript):
        base = target.value
        if isinstance(base, ast.Attribute) and base.attr in ("loc", "iloc") and isinstance(base.value, ast.Name):
            return f"{base.attr}_setitem", base.value, [_key_node(target.slice)]
        if isinstance(base, ast.Name):
            return "setitem", base, [_key_node(target.slice)]
        return None
    if isinstance(target, ast.Attribute) and isinstance(target.value, ast.Name):
        return "setattr", target.value, [ast.Constant(value=target.attr)]
    return None


def _loop_mutations(loop: ast.stmt) -> list[str]:
    """Names a top-level loop may change: store targets and ``inplace=True`` receivers."""
    loop_vars: set[str] = set()
    if isinstance(loop, ast.For):
        loop_vars = {n.id for n in ast.walk(loop.target) if isinstance(n, ast.Name)}
    names: list[str] = []

    def add(name: str | None) -> None:
        if name and name not in loop_vars and name not in names:
            names.append(name)

    for node in ast.walk(loop):
        if isinstance(node, ast.Assign):
            targets = node.targets
        elif isinstance(node, (ast.AugAssign, ast.AnnAssign)):
            targets = [node.target]
        elif isinstance(node, ast.Call) and _has_inplace(node):
            root = _call_root(node.func)
            add(root.id if root else None)
            continue
        else:
            continue
        for target in targets:
            for elt in (target.elts if isinstance(target, (ast.Tuple, ast.List)) else [target]):
                add(_root_name(elt))
    return names
