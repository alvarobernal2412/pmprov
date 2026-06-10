import pytest
import plotly.graph_objs as go
from tracker.snapshot_engine import capture_snapshot


def test_plotly_figure_snapshot_kind():
    fig = go.Figure(go.Bar(x=[1, 2], y=[3, 4]))
    snap = capture_snapshot(fig)
    assert snap["kind"] == "figure"


def test_plotly_figure_snapshot_fields():
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[1], y=[2]))
    fig.update_layout(title_text="test")
    snap = capture_snapshot(fig)
    assert snap["figure_type"] == "plotly"
    assert snap["trace_count"] == 1
    assert "object_id" in snap
    assert "layout_keys" in snap
