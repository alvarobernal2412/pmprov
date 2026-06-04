"""
R5 – Comparison of Histories (AC 5.1, 5.2)
R6 – Comparison of States (AC 6.1, 6.2, 6.3)

None of these are implemented in the current prototype.
StateAbstraction is defined as a Pydantic model but tracker/ does not yet
populate it. compare_histories and compare_states APIs do not exist yet.
Tests are commented out until the implementation is ready.
"""

# def test_ac5_1_history_divergences_exposed():
#     """AC 5.1 – divergences across steps between two histories are surfaced."""
#     # compare_histories() API not yet implemented
#     raise NotImplementedError

# def test_ac5_2_steps_grouped_by_category():
#     """AC 5.2 – steps are grouped into higher-level phases (StepCategory)."""
#     # classify_step() / StepCategory not yet implemented in tracker/
#     raise NotImplementedError

# def test_ac6_1_granular_state_diff():
#     """AC 6.1 – granular column/dtype/row differences between states exposed."""
#     # compare_states() API not yet implemented
#     raise NotImplementedError

# def test_ac6_2_abstractions_registered_and_stored():
#     """AC 6.2 – analyst-defined abstraction functions computed and stored per state."""
#     # StateAbstraction not yet populated in tracker/
#     raise NotImplementedError

# def test_ac6_3_abstracted_comparison():
#     """AC 6.3 – abstracted metrics structurally aligned for cross-state comparison."""
#     # compare_states_abstracted() not yet implemented
#     raise NotImplementedError
