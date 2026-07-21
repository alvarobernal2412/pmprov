from tracker.pruning import _cascade_hidden_state_ids, _compute_pruned_graph


def _step(step_id, input_state_id, output_state_id, func_name="f"):
    return {"step_id": step_id, "input_state_id": input_state_id,
            "output_state_id": output_state_id, "func_name": func_name}


def test_cascade_hidden_state_ids_includes_all_descendants():
    steps = [
        _step("s1", "root", "a"),
        _step("s2", "a", "b"),
        _step("s3", "b", "c"),
        _step("s4", "root", "d"),  # sibling branch, unaffected
    ]
    result = _cascade_hidden_state_ids(steps, {"a"})
    assert result == {"a", "b", "c"}


def test_cascade_hidden_state_ids_no_descendants():
    steps = [_step("s1", "root", "a")]
    result = _cascade_hidden_state_ids(steps, {"a"})
    assert result == {"a"}


def test_compute_pruned_graph_excludes_hidden_states_and_their_steps():
    states = [{"state_id": sid} for sid in ("root", "a", "b", "d")]
    steps = [
        _step("s1", "root", "a"),
        _step("s2", "a", "b"),
        _step("s3", "root", "d"),
    ]
    result = _compute_pruned_graph(
        states, steps, categories_by_step_id={}, group_by_category=False,
        hidden_state_ids={"a", "b"},
    )
    node_ids = {n["state_id"] for n in result["nodes"]}
    edge_step_ids = {e["step_id"] for e in result["edges"]}
    assert node_ids == {"root", "d"}
    assert edge_step_ids == {"s3"}


def test_compute_pruned_graph_collapses_same_category_chain():
    states = [{"state_id": sid} for sid in ("root", "a", "b", "c")]
    steps = [
        _step("s1", "root", "a", "load"),
        _step("s2", "a", "b", "clean"),
        _step("s3", "b", "c", "clean_more"),
    ]
    categories = {"s1": "loading", "s2": "cleaning", "s3": "cleaning"}
    result = _compute_pruned_graph(
        states, steps, categories_by_step_id=categories, group_by_category=True,
        hidden_state_ids=set(),
    )
    edges = sorted(result["edges"], key=lambda e: e["input_state_id"])
    assert len(edges) == 2
    load_edge = next(e for e in edges if e["input_state_id"] == "root")
    clean_edge = next(e for e in edges if e["input_state_id"] == "a")
    assert load_edge["collapsed_step_ids"] == ["s1"]
    assert clean_edge["input_state_id"] == "a"
    assert clean_edge["output_state_id"] == "c"
    assert clean_edge["collapsed_step_ids"] == ["s2", "s3"]
    assert clean_edge["step_id"] is None  # synthetic — represents 2 original steps


def test_compute_pruned_graph_stops_collapse_at_branch_point():
    # 'a' has two children (branch point) — the chain through 'a' must not merge
    # across the branch, even if categories match on both sides.
    states = [{"state_id": sid} for sid in ("root", "a", "b", "c")]
    steps = [
        _step("s1", "root", "a", "load"),
        _step("s2", "a", "b", "clean"),
        _step("s3", "a", "c", "clean"),
    ]
    categories = {"s1": "loading", "s2": "cleaning", "s3": "cleaning"}
    result = _compute_pruned_graph(
        states, steps, categories_by_step_id=categories, group_by_category=True,
        hidden_state_ids=set(),
    )
    edge_step_ids = {tuple(e["collapsed_step_ids"]) for e in result["edges"]}
    assert edge_step_ids == {("s1",), ("s2",), ("s3",)}
