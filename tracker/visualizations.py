"""
Provenance visualization methods for RuntimeTracker.

Imported as a side-effect from tracker/__init__.py, which patches
show_graph() and show_graph_widget() onto RuntimeTracker instances.
"""

from __future__ import annotations

from io import BytesIO
from typing import TYPE_CHECKING, Any, Optional

if TYPE_CHECKING:
    from tracker.runtime import RuntimeTracker


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _is_marimo() -> bool:
    """Heuristic: check whether we are inside a Marimo session."""
    try:
        import marimo  # noqa: F401
        return True
    except ImportError:
        pass
    import sys
    return any("marimo" in k for k in sys.modules)


def _step_nodes(graph_data: dict) -> list[dict]:
    """Return only analyst-facing step nodes (drop session_root and branch records)."""
    return [
        n for n in graph_data["nodes"]
        if n["metadata"].get("node_type") not in ("session_root", "branch")
        and not n["node_id"].startswith("branch-")
    ]


def _branch_name_map(graph_data: dict) -> dict[str, str]:
    """Return branch_id → branch_name from branch metadata nodes."""
    result: dict[str, str] = {}
    for n in graph_data["nodes"]:
        meta = n["metadata"]
        if meta.get("node_type") == "branch":
            b = meta.get("branch", {})
            result[b.get("branch_id", "")] = b.get("name", "unknown")
    # also pick up the main branch from session_root nodes
    for n in graph_data["nodes"]:
        meta = n["metadata"]
        if meta.get("node_type") == "session_root":
            b = meta.get("branch", {})
            result[b.get("branch_id", "")] = b.get("name", "main")
    return result


def _node_label(node: dict) -> str:
    ts = node.get("timestamp", "")
    try:
        from datetime import datetime
        dt = datetime.fromisoformat(ts)
        time_str = dt.strftime("%H:%M:%S")
    except Exception:
        time_str = ts[:8] if ts else "?"
    state_id = node["metadata"].get("output_state", {}).get("state_id", node["node_id"])
    return f"{time_str}\n{state_id[:8]}"


def _edge_label(func_name: str) -> str:
    return func_name[:28] if func_name else ""


def _build_display_graph(graph_data: dict):
    """Build a networkx DiGraph from step nodes only, with a virtual ROOT."""
    import networkx as nx

    steps = {n["node_id"]: n for n in _step_nodes(graph_data)}
    step_ids = set(steps)

    G = nx.DiGraph()
    for node_id, node in steps.items():
        G.add_node(node_id, **node)

    for e in graph_data["edges"]:
        if e["parent"] in step_ids and e["child"] in step_ids:
            G.add_edge(e["parent"], e["child"], func_name=e["type"])
        elif e["child"] in step_ids and e["parent"] not in step_ids:
            # edge from session_root → first step
            G.add_edge("__ROOT__", e["child"], func_name=e["type"])

    ROOT = "__ROOT__"
    # Add virtual root for any step with no in-edges
    root_children = [n for n in G.nodes() if n != ROOT and G.in_degree(n) == 0]
    if root_children:
        G.add_node(ROOT)
        for child in root_children:
            G.add_edge(ROOT, child, func_name="")

    return G, steps


def _compute_layout(G) -> dict:
    import networkx as nx

    try:
        from networkx.drawing.nx_pydot import graphviz_layout
        pos = graphviz_layout(G, prog="dot")
        # graphviz y increases downward — normalize so root is at top
        return pos
    except Exception:
        pass

    if nx.is_directed_acyclic_graph(G) and G.number_of_nodes() > 0:
        layers = list(nx.topological_generations(G))
        pos: dict = {}
        for layer_idx, layer in enumerate(layers):
            nodes = sorted(layer)
            count = max(len(nodes), 1)
            for offset, node_id in enumerate(nodes):
                x = (offset + 1) / (count + 1)
                y = -float(layer_idx) * 2.5
                pos[node_id] = (x, y)
        return pos

    return nx.spring_layout(G, seed=1)


def _text_fallback(graph_data: dict) -> None:
    steps = _step_nodes(graph_data)
    print(f"Provenance graph — {len(steps)} step(s), {len(graph_data['edges'])} edge(s)")
    print(f"{'Timestamp':<26} {'func_name'}")
    print("-" * 60)
    for n in steps:
        ts = n.get("timestamp", "")[:19]
        fn = n["metadata"].get("func_name", "?")
        print(f"{ts:<26} {fn}")


# ---------------------------------------------------------------------------
# show_graph
# ---------------------------------------------------------------------------

def _show_graph(
    self: "RuntimeTracker",
    save_path: Optional[str] = None,
) -> None:
    try:
        import networkx as nx  # noqa: F401
    except ImportError:
        print("show_graph() requires networkx. Install with: uv pip install -e '.[graph]'")
        return None

    graph_data = self.storage.load_graph()
    steps = _step_nodes(graph_data)

    if not steps:
        print("No provenance steps recorded yet.")
        return None

    try:
        import matplotlib.pyplot as plt
    except ImportError:
        _text_fallback(graph_data)
        return None

    try:
        G, step_map = _build_display_graph(graph_data)
        pos = _compute_layout(G)

        # Scale y axis for readability
        pos = {n: (x, y * 1.5) for n, (x, y) in pos.items()}

        node_labels = {}
        for node_id in G.nodes():
            if node_id == "__ROOT__":
                node_labels[node_id] = "ROOT"
            elif node_id in step_map:
                node_labels[node_id] = _node_label(step_map[node_id])
            else:
                node_labels[node_id] = node_id[:8]

        edge_labels = {
            (u, v): _edge_label(d.get("func_name", ""))
            for u, v, d in G.edges(data=True)
        }

        import networkx as nx
        layer_count = 1
        if nx.is_directed_acyclic_graph(G) and G.number_of_nodes() > 0:
            layer_count = len(list(nx.topological_generations(G)))

        fig, ax = plt.subplots(figsize=(10, max(5, 1.4 * layer_count)))

        nx.draw_networkx_edges(G, pos, ax=ax, edge_color="#5f6368", arrows=True)
        nx.draw_networkx_nodes(
            G, pos, ax=ax,
            node_color="#e8f0fe",
            node_size=900,
            edgecolors="#5f6368",
        )
        nx.draw_networkx_labels(G, pos, node_labels, ax=ax, font_size=7)
        nx.draw_networkx_edge_labels(
            G, pos, edge_labels=edge_labels, ax=ax, font_size=6, rotate=False
        )

        ax.axis("off")

        if save_path:
            from pathlib import Path
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(save_path, format="svg", bbox_inches="tight")

        plt.tight_layout()
        plt.show()

    except Exception as exc:
        print(f"show_graph() rendering error: {exc}")

    return None


# ---------------------------------------------------------------------------
# _render_graph_image (shared helper for widget)
# ---------------------------------------------------------------------------

def _render_graph_image(self: "RuntimeTracker") -> Any:
    try:
        import matplotlib.pyplot as plt
        import ipywidgets as widgets
    except ImportError:
        return None

    graph_data = self.storage.load_graph()
    steps = _step_nodes(graph_data)
    if not steps:
        return None

    try:
        G, step_map = _build_display_graph(graph_data)
        pos = _compute_layout(G)
        pos = {n: (x, y * 1.5) for n, (x, y) in pos.items()}

        node_labels = {
            node_id: ("ROOT" if node_id == "__ROOT__" else _node_label(step_map[node_id]))
            for node_id in G.nodes()
            if node_id == "__ROOT__" or node_id in step_map
        }
        edge_labels = {
            (u, v): _edge_label(d.get("func_name", ""))
            for u, v, d in G.edges(data=True)
        }

        import networkx as nx
        layer_count = max(1, len(list(nx.topological_generations(G))) if nx.is_directed_acyclic_graph(G) else 1)

        fig, ax = plt.subplots(figsize=(6, max(4, 1.2 * layer_count)))
        nx.draw_networkx_edges(G, pos, ax=ax, edge_color="#5f6368", arrows=True)
        nx.draw_networkx_nodes(G, pos, ax=ax, node_color="#e8f0fe", node_size=700, edgecolors="#5f6368")
        nx.draw_networkx_labels(G, pos, node_labels, ax=ax, font_size=7)
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, ax=ax, font_size=6, rotate=False)
        ax.axis("off")

        buf = BytesIO()
        fig.savefig(buf, format="png", bbox_inches="tight")
        plt.close(fig)
        buf.seek(0)

        return widgets.Image(value=buf.getvalue(), format="png")

    except Exception as exc:
        print(f"_render_graph_image() error: {exc}")
        return None


# ---------------------------------------------------------------------------
# show_graph_widget
# ---------------------------------------------------------------------------

def _show_graph_widget(
    self: "RuntimeTracker",
    preview_rows: int = 5,
) -> Any:
    if _is_marimo():
        print("show_graph_widget() is Jupyter-only. Falling back to show_graph().")
        return self.show_graph()

    try:
        import ipywidgets as widgets
        from IPython.display import display
    except ImportError:
        print("show_graph_widget() requires ipywidgets. Install with: uv pip install -e '.[widgets]'")
        print("Falling back to show_graph().")
        return self.show_graph()

    try:
        import networkx as nx  # noqa: F401
    except ImportError:
        print("show_graph_widget() requires networkx. Install with: uv pip install -e '.[graph]'")
        return None

    graph_data = self.storage.load_graph()
    steps = _step_nodes(graph_data)

    if not steps:
        print("No provenance steps recorded yet.")
        return None

    branch_names = _branch_name_map(graph_data)

    # Build selector options sorted by timestamp
    options = []
    for n in sorted(steps, key=lambda x: x.get("timestamp", "")):
        ts = n.get("timestamp", "")
        try:
            from datetime import datetime
            dt = datetime.fromisoformat(ts)
            time_str = dt.strftime("%H:%M:%S")
        except Exception:
            time_str = ts[:8]
        fn = n["metadata"].get("func_name", "?")
        branch_id = n["metadata"].get("output_state", {}).get("branch_id", "")
        branch = branch_names.get(branch_id, "main")
        label = f"{time_str} | {fn} | {branch}"
        options.append((label, n["node_id"]))

    # --- Widgets ---
    graph_image = self._render_graph_image()
    graph_panel = graph_image if graph_image is not None else widgets.HTML("<i>Graph unavailable</i>")

    selector = widgets.Select(
        options=options,
        rows=min(max(len(options), 1), 12),
        description="Step",
        layout=widgets.Layout(width="380px"),
    )
    details = widgets.HTML("<i>Select a step to view details.</i>")
    toggle = widgets.ToggleButtons(
        options=["Preview", "Summary"],
        value="Preview",
        description="Artifact",
        style={"button_width": "90px"},
    )
    artifact_output = widgets.Output()

    node_index = {n["node_id"]: n for n in steps}

    def _delta_html(delta: dict) -> str:
        if not delta or delta.get("kind") == "error":
            return "<i>none</i>"
        lines = []
        kind = delta.get("kind", "")
        if kind:
            lines.append(f"<b>kind</b>: {kind}")
        pre = delta.get("pre", {})
        post = delta.get("post", {})
        if pre.get("shape") and post.get("shape"):
            lines.append(f"<b>shape</b>: {pre['shape']} → {post['shape']}")
        cols_added = delta.get("columns_added", [])
        if cols_added:
            lines.append(f"<b>+cols</b>: {', '.join(str(c) for c in cols_added[:8])}"
                         + (f" +{len(cols_added)-8} more" if len(cols_added) > 8 else ""))
        cols_removed = delta.get("columns_removed", [])
        if cols_removed:
            lines.append(f"<b>-cols</b>: {', '.join(str(c) for c in cols_removed[:8])}"
                         + (f" +{len(cols_removed)-8} more" if len(cols_removed) > 8 else ""))
        dtype_changes = delta.get("dtype_changes", [])
        if dtype_changes:
            lines.append(f"<b>~dtypes</b>: {', '.join(str(c) for c in dtype_changes[:5])}")
        rows_delta = delta.get("rows_delta")
        if rows_delta is not None:
            lines.append(f"<b>rows Δ</b>: {rows_delta:+d}")
        return "<br/>".join(lines) if lines else "<i>none</i>"

    def update(change: Any = None) -> None:
        node_id = selector.value
        if node_id is None:
            return
        node = node_index.get(node_id)
        if node is None:
            return

        meta = node["metadata"]
        fn = meta.get("func_name", "?")
        raw = meta.get("raw_line", "")
        op_type = meta.get("operation_type", {}).get("name", "unknown")
        branch_id = meta.get("output_state", {}).get("branch_id", "")
        branch = branch_names.get(branch_id, "main")
        state_id = meta.get("output_state", {}).get("state_id", node_id)
        delta_html = _delta_html(meta.get("delta", {}))

        details.value = (
            f"<b>state_id</b>: <code>{state_id}</code><br/>"
            f"<b>branch</b>: {branch}<br/>"
            f"<b>func</b>: <code>{fn}</code><br/>"
            f"<b>op type</b>: {op_type}<br/>"
            f"<b>raw line</b>: <code>{raw}</code><br/>"
            f"<b>delta</b>:<br/>{delta_html}"
        )

        artifact_path = node.get("artifact_path")
        with artifact_output:
            artifact_output.clear_output(wait=True)
            if not artifact_path:
                print("No artifact for this step.")
                return
            try:
                import pandas as pd
                df = pd.read_parquet(artifact_path)
                from IPython.display import HTML
                if toggle.value == "Preview":
                    artifact_output.append_display_data(
                        HTML(f"<div style='overflow-x:auto'>{df.head(preview_rows).to_html()}</div>")
                    )
                else:
                    summary = (
                        f"<b>shape</b>: {df.shape}<br/>"
                        f"<b>columns</b>: {list(df.columns)}<br/>"
                    )
                    artifact_output.append_display_data(
                        HTML(summary + df.dtypes.to_frame("dtype").to_html())
                    )
            except Exception as exc:
                print(f"Could not load artifact: {exc}")

    selector.observe(update, names="value")
    toggle.observe(update, names="value")

    if options:
        selector.value = options[0][1]

    return widgets.HBox([
        widgets.VBox([graph_panel], layout=widgets.Layout(margin="0 20px 0 0")),
        widgets.VBox([selector, details, toggle, artifact_output]),
    ])


# ---------------------------------------------------------------------------
# list_states
# ---------------------------------------------------------------------------

def _list_states(self: "RuntimeTracker") -> Any:
    """
    Return a DataFrame listing every recorded analysis state in the current
    session, sorted by timestamp.

    Columns
    -------
    state_id        Full UUID — paste into rt.checkout() to branch from that state.
    timestamp       UTC time the state was created.
    branch          Name of the branch this state belongs to.
    func_name       Function call that produced this state.
    operation_type  Semantic category (e.g. 'case_aggregation', 'unknown').
    has_artifact    True when a Parquet artifact was saved for this state.
    """
    import pandas as pd

    graph_data = self.storage.load_graph()
    steps = _step_nodes(graph_data)

    if not steps:
        print("No provenance steps recorded yet.")
        return pd.DataFrame()

    branch_names = _branch_name_map(graph_data)

    rows = []
    for n in sorted(steps, key=lambda x: x.get("timestamp", "")):
        meta = n["metadata"]
        branch_id = meta.get("output_state", {}).get("branch_id", "")
        rows.append({
            "state_id":       meta.get("output_state", {}).get("state_id", n["node_id"]),
            "timestamp":      n.get("timestamp", "")[:19].replace("T", " "),
            "branch":         branch_names.get(branch_id, "main"),
            "func_name":      meta.get("func_name", "?"),
            "operation_type": meta.get("operation_type", {}).get("name", "unknown"),
            "has_artifact":   n.get("artifact_path") is not None,
        })

    df = pd.DataFrame(rows)

    # In a Jupyter cell `rt.list_states()` is a bare expression whose return
    # value Jupyter displays automatically — calling display() here too would
    # produce a second table.  Only print explicitly in non-interactive contexts
    # (plain Python scripts) where the return value is silently discarded.
    try:
        from IPython import get_ipython
        if get_ipython() is None:
            print(df.to_string(index=False))
    except ImportError:
        print(df.to_string(index=False))

    return df


# ---------------------------------------------------------------------------
# Patch onto RuntimeTracker
# ---------------------------------------------------------------------------

from tracker.runtime import RuntimeTracker  # noqa: E402

RuntimeTracker.show_graph = _show_graph
RuntimeTracker.show_graph_widget = _show_graph_widget
RuntimeTracker._render_graph_image = _render_graph_image
RuntimeTracker.list_states = _list_states
