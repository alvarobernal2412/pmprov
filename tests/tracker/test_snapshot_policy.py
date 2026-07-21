import pytest
from tracker.snapshot_policy import snapshot_policy, snapshot_policy_for_type, should_snapshot


def test_should_snapshot_defaults_to_true_when_unconfigured():
    assert should_snapshot("totally_unconfigured_func_abc", "unknown") is True


def test_snapshot_policy_never_overrides_default():
    snapshot_policy("cheap_op_xyz", "never")
    assert should_snapshot("cheap_op_xyz", "unknown") is False


def test_snapshot_policy_always_is_explicit_default():
    snapshot_policy("explicit_always_op", "always")
    assert should_snapshot("explicit_always_op", "unknown") is True


def test_snapshot_policy_matches_trailing_dotted_segment():
    snapshot_policy("assign_no_snapshot", "never")
    assert should_snapshot("df.assign_no_snapshot", "unknown") is False


def test_snapshot_policy_for_type_sets_type_level_default():
    snapshot_policy_for_type("cheap_type_xyz", "never")
    assert should_snapshot("some_func_using_cheap_type", "cheap_type_xyz") is False


def test_func_name_override_wins_over_type_default():
    snapshot_policy_for_type("mixed_type_xyz", "never")
    snapshot_policy("important_func_xyz", "always")
    assert should_snapshot("important_func_xyz", "mixed_type_xyz") is True


def test_snapshot_policy_rejects_invalid_mode():
    with pytest.raises(ValueError):
        snapshot_policy("some_func", "sometimes")


def test_snapshot_policy_for_type_rejects_invalid_mode():
    with pytest.raises(ValueError):
        snapshot_policy_for_type("some_type", "sometimes")
