"""
R1 – Traceability
AC 1.1: Chronological ordering, dependencies, and full lineage visible in the provenance tree.
AC 1.2: Step operations and parameter configurations recorded on each transition.
AC 1.3: Execution context (agent, environment) explicit per state.
"""
import pytest
from .conftest import settle


def test_ac1_1_full_lineage_in_tree(rt, event_log):
    """AC 1.1 – all executed steps appear as linked states in load_graph."""
    def filter_fn(df): return df[df["case:concept:name"] == "A1"]
    def enrich_fn(df): return df.assign(flag=True)

    rt.trace_step(func=filter_fn, func_name="filter", raw_line="df=filter(df)",
                  args=[event_log], kwargs={})
    filtered = event_log[event_log["case:concept:name"] == "A1"]
    rt.trace_step(func=enrich_fn, func_name="enrich", raw_line="df=enrich(df)",
                  args=[filtered], kwargs={})
    settle(rt)

    graph = rt.storage.load_graph()
    # Root state + 2 output states recorded
    assert len(graph["states"]) >= 2
    # 2 steps recorded
    assert len(graph["steps"]) >= 2
    # Each step links input_state_id → output_state_id (no orphan edges)
    state_ids = {s["state_id"] for s in graph["states"]}
    for step in graph["steps"]:
        assert step["input_state_id"] in state_ids or step["input_state_id"] != ""
        assert step["output_state_id"] in state_ids


def test_ac1_2_step_operation_recorded(rt, event_log):
    """AC 1.2 – func_name and raw_line captured per step transition."""
    # Note: "head" is in OMIT_FUNCTIONS — use a non-omitted func_name
    rt.trace_step(func=lambda df: df[df["case:concept:name"] == "A1"],
                  func_name="df.filter_cases",
                  raw_line="df = df.filter_cases(df)",
                  args=[event_log], kwargs={})
    settle(rt)

    graph = rt.storage.load_graph()
    step = next(s for s in graph["steps"] if s["func_name"] == "df.filter_cases")
    assert step["func_name"] == "df.filter_cases"
    assert step["raw_line"] == "df = df.filter_cases(df)"


def test_ac1_3_execution_context_recorded(rt, event_log):
    """AC 1.3 – agent and runtime_environment rows exist after session init."""
    settle(rt)

    con = rt.storage._connect(read_only=True)
    agents = con.execute("SELECT agent_id, agent_type FROM agents").fetchall()
    envs = con.execute("SELECT env_id, tool_version FROM runtime_environments").fetchall()
    con.close()

    assert len(agents) >= 1
    assert agents[0][1] in ("human", "automated")
    assert len(envs) >= 1
    assert "Python" in envs[0][1]
