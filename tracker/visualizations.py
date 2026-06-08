"""
Provenance visualization methods for RuntimeTracker.

Imported as a side-effect from tracker/__init__.py, which patches
show_graph(), show_graph_widget(), and list_states() onto RuntimeTracker.
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
    """Return analyst-facing steps (non-root states that were produced by a step)."""
    step_output_ids = {st["output_state_id"] for st in graph_data["steps"]}
    return [s for s in graph_data["states"] if s["state_id"] in step_output_ids]


def _state_label(state: dict) -> str:
    ts = state.get("timestamp", "")
    try:
        from datetime import datetime
        dt = datetime.fromisoformat(ts)
        time_str = dt.strftime("%H:%M:%S")
    except Exception:
        time_str = ts[:8] if ts else "?"
    return f"{time_str}\n{state['state_id'][:8]}"


def _edge_label(func_name: str) -> str:
    return func_name[:28] if func_name else ""


def _build_display_graph(graph_data: dict):
    """Build a networkx DiGraph from step-produced states only."""
    import networkx as nx

    step_output_ids = {st["output_state_id"] for st in graph_data["steps"]}
    state_map = {s["state_id"]: s for s in graph_data["states"]}

    G = nx.DiGraph()
    for sid in step_output_ids:
        if sid in state_map:
            G.add_node(sid, **state_map[sid])

    for st in graph_data["steps"]:
        src = st["input_state_id"]
        dst = st["output_state_id"]
        if dst in step_output_ids:
            if src in step_output_ids:
                G.add_edge(src, dst, func_name=st["func_name"])
            else:
                # edge from root/non-step state -> first step output
                G.add_edge("__ROOT__", dst, func_name=st["func_name"])

    ROOT = "__ROOT__"
    root_children = [n for n in G.nodes() if n != ROOT and G.in_degree(n) == 0]
    if root_children:
        G.add_node(ROOT)
        for child in root_children:
            G.add_edge(ROOT, child, func_name="")

    return G, state_map


def _compute_layout(G) -> dict:
    import networkx as nx

    try:
        from networkx.drawing.nx_pydot import graphviz_layout
        pos = graphviz_layout(G, prog="dot")
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
    steps = graph_data["steps"]
    states = {s["state_id"]: s for s in graph_data["states"]}
    print(f"Provenance graph — {len(steps)} step(s)")
    print(f"{'Timestamp':<26} {'func_name'}")
    print("-" * 60)
    for st in sorted(steps, key=lambda x: x.get("timestamp", "")):
        ts = st.get("timestamp", "")[:19]
        fn = st.get("func_name", "?")
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
    step_outputs = _step_nodes(graph_data)

    if not step_outputs:
        print("No provenance steps recorded yet.")
        return None

    try:
        import matplotlib.pyplot as plt
    except ImportError:
        _text_fallback(graph_data)
        return None

    try:
        G, state_map = _build_display_graph(graph_data)
        pos = _compute_layout(G)
        pos = {n: (x, y * 1.5) for n, (x, y) in pos.items()}

        node_labels = {}
        for node_id in G.nodes():
            if node_id == "__ROOT__":
                node_labels[node_id] = "ROOT"
            elif node_id in state_map:
                node_labels[node_id] = _state_label(state_map[node_id])
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
        plt.close(fig)
        return fig

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
    step_outputs = _step_nodes(graph_data)
    if not step_outputs:
        return None

    try:
        G, state_map = _build_display_graph(graph_data)
        pos = _compute_layout(G)
        pos = {n: (x, y * 1.5) for n, (x, y) in pos.items()}

        node_labels = {
            node_id: ("ROOT" if node_id == "__ROOT__" else _state_label(state_map[node_id]))
            for node_id in G.nodes()
            if node_id == "__ROOT__" or node_id in state_map
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
    step_outputs = _step_nodes(graph_data)

    if not step_outputs:
        print("No provenance steps recorded yet.")
        return None

    # Build step index for detail panel: keyed by output_state_id
    step_index = {st["output_state_id"]: st for st in graph_data["steps"]}
    state_index = {s["state_id"]: s for s in graph_data["states"]}

    # Build selector options sorted by timestamp
    options = []
    for st in sorted(graph_data["steps"], key=lambda x: x.get("timestamp", "")):
        ts = st.get("timestamp", "")
        try:
            from datetime import datetime
            dt = datetime.fromisoformat(ts)
            time_str = dt.strftime("%H:%M:%S")
        except Exception:
            time_str = ts[:8]
        fn = st.get("func_name", "?")
        state_id = st["output_state_id"]
        label = f"{time_str} | {fn}"
        options.append((label, state_id))

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
        state_id = selector.value
        if state_id is None:
            return
        step = step_index.get(state_id)
        if step is None:
            return

        fn = step.get("func_name", "?")
        raw = step.get("raw_line", "")
        ts = step.get("timestamp", "")[:19].replace("T", " ")

        details.value = (
            f"<b>state_id</b>: <code>{state_id}</code><br/>"
            f"<b>branch</b>: main<br/>"  # TODO: join analysis_branches once available
            f"<b>func</b>: <code>{fn}</code><br/>"
            f"<b>op type</b>: unknown<br/>"  # TODO: join operation_types
            f"<b>raw line</b>: <code>{raw}</code><br/>"
            f"<b>timestamp</b>: {ts}"
        )

        # Try to load artifact by path convention: artifact_dir/<state_id>.parquet
        artifact_path = str(self.storage.artifact_dir / f"{state_id}.parquet")
        with artifact_output:
            artifact_output.clear_output(wait=True)
            try:
                import pandas as pd
                from pathlib import Path
                if not Path(artifact_path).exists():
                    print("No artifact for this step.")
                    return
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
    Return a DataFrame listing every recorded analysis state in the current session.

    Columns
    -------
    state_id              Full UUID — paste into rt.checkout() to branch from that state.
    timestamp             UTC time the state was created.
    branch_name           Name of the branch this state belongs to.
    produced_by_step_id   Step that created this state (None for the root state).
    derived_from_state_id Parent state (None for the root state).
    artifact_state_ids    Comma-separated list of associated artifact state UUIDs.
    """
    import pandas as pd

    rows = self.storage.load_states_rich(self._history.history_id)
    if not rows:
        return pd.DataFrame(columns=[
            "state_id", "timestamp", "branch_name",
            "produced_by_step_id", "derived_from_state_id", "artifact_state_ids",
        ])

    df_rows = [
        {
            "state_id": r["state_id"],
            "timestamp": r["timestamp"][:19].replace("T", " "),
            "branch_name": r["branch_name"],
            "produced_by_step_id": r["produced_by_step_id"],
            "derived_from_state_id": r["derived_from_state_id"],
            "artifact_state_ids": ", ".join(r["artifact_state_ids"]),
        }
        for r in rows
    ]
    df = pd.DataFrame(df_rows).sort_values("timestamp").reset_index(drop=True)

    try:
        from IPython import get_ipython
        if get_ipython() is None:
            print(df.to_string(index=False))
    except ImportError:
        print(df.to_string(index=False))

    return df


# ---------------------------------------------------------------------------
# show_artifact_lifecycle
# ---------------------------------------------------------------------------

def _show_artifact_lifecycle(
    self: "RuntimeTracker",
    state_id: str,
    save_path: Optional[str] = None,
) -> None:
    """
    Render the lifecycle of the artifact associated with state_id as a
    matplotlib figure. Each node is an ArtifactState; edges show the
    transformation and delta summary.

    Falls back to a plain-text table when matplotlib or networkx are absent.

    Parameters
    ----------
    state_id:
        Any state_id with an associated artifact. Use rt.list_states() to find IDs.
    save_path:
        Optional path to save the figure as SVG.
    """
    lifecycle = self.storage.load_artifact_lifecycle(state_id)

    if not lifecycle:
        print(f"No artifact lifecycle found for state {state_id[:8]}.")
        return

    try:
        import matplotlib.pyplot as plt
        import networkx as nx
    except ImportError:
        print(f"Artifact lifecycle for state {state_id[:8]}:")
        print(f"{'Timestamp':<22} {'func_name':<20} {'modification':<18} rows_delta")
        print("-" * 75)
        for entry in lifecycle:
            ts = (entry.get("timestamp") or "")[:19]
            fn = entry.get("func_name") or "import"
            mt = entry.get("modification_type") or "-"
            rd = entry.get("rows_delta")
            rd_str = f"{rd:+d}" if rd is not None else "-"
            print(f"{ts:<22} {fn:<20} {mt:<18} {rd_str}")
        return

    G = nx.DiGraph()
    for i, entry in enumerate(lifecycle):
        sid = entry["state_id"]
        ts = (entry.get("timestamp") or "")
        try:
            from datetime import datetime
            dt = datetime.fromisoformat(ts)
            label = f"{dt.strftime('%H:%M:%S')}\n{sid[:8]}"
        except Exception:
            label = sid[:8]
        G.add_node(sid, label=label, entry=entry)
        if i > 0:
            prev = lifecycle[i - 1]
            mt = entry.get("modification_type") or "?"
            rd = entry.get("rows_delta")
            rd_str = f" ({rd:+d} rows)" if rd is not None else ""
            cols_added = entry.get("columns_added") or []
            fn = entry.get("func_name") or "?"
            edge_label = f"{fn}\n{mt}{rd_str}"
            if cols_added:
                names = ", ".join(str(c) for c in cols_added[:3])
                suffix = f"+{len(cols_added)-3} more" if len(cols_added) > 3 else ""
                edge_label += f"\n+{names}{suffix}"
            G.add_edge(prev["state_id"], sid, label=edge_label)

    try:
        from networkx.drawing.nx_pydot import graphviz_layout
        pos = graphviz_layout(G, prog="dot")
    except Exception:
        pos = {n: (0, -i * 2) for i, n in enumerate(G.nodes())}

    node_labels = {n: G.nodes[n]["label"] for n in G.nodes()}
    edge_labels = {(u, v): G[u][v]["label"] for u, v in G.edges()}

    n_nodes = max(G.number_of_nodes(), 1)
    fig, ax = plt.subplots(figsize=(8, max(4, n_nodes * 1.5)))
    nx.draw_networkx_nodes(G, pos, ax=ax, node_color="#fce8b2", node_size=1200, edgecolors="#f9a825")
    nx.draw_networkx_edges(G, pos, ax=ax, edge_color="#5f6368", arrows=True, arrowsize=15)
    nx.draw_networkx_labels(G, pos, node_labels, ax=ax, font_size=7)
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, ax=ax, font_size=6, rotate=False)
    ax.axis("off")
    ax.set_title(f"Artifact Lifecycle — anchor state {state_id[:8]}", fontsize=9)

    if save_path:
        from pathlib import Path
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, format="svg", bbox_inches="tight")

    plt.tight_layout()
    plt.show()
    plt.close(fig)


# ---------------------------------------------------------------------------
# Patch onto RuntimeTracker
# ---------------------------------------------------------------------------

from tracker.runtime import RuntimeTracker  # noqa: E402

RuntimeTracker.show_graph = _show_graph
RuntimeTracker.show_graph_widget = _show_graph_widget
RuntimeTracker._render_graph_image = _render_graph_image
RuntimeTracker.list_states = _list_states
RuntimeTracker.show_artifact_lifecycle = _show_artifact_lifecycle
