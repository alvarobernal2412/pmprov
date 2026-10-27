"""Aliases and objects changed through another variable.

``df2 = df`` makes a second name for one object; ``ax.set_title(...)`` changes
the figure that ``fig`` names. Every state must still hold each variable
exactly as it was at that point.
"""
import pytest

from tests.tracker.test_course_patterns import Notebook, assert_same


@pytest.fixture
def nb(tmp_path, monkeypatch):
    notebook = Notebook(tmp_path, monkeypatch)
    notebook.run("df = pd.DataFrame({'a': [1, 2, 3], 'g': ['x', 'y', 'x']})")
    notebook.art_dir = tmp_path / "a"
    return notebook


def stored_files(nb):
    return sorted(p.name for p in nb.art_dir.iterdir() if not p.name.endswith(".ui_alias.json"))


# ---- aliases ------------------------------------------------------------------

def test_an_alias_is_a_step_bound_to_the_existing_artifact(nb):
    before = stored_files(nb)

    steps = nb.run("df2 = df")

    assert len(steps) == 1
    assert_same(nb.restored()["df2"], nb.ns["df"])
    assert stored_files(nb) == before, "an alias stores no new data"


def test_a_change_made_through_an_alias_reaches_both_names(nb):
    nb.run("df2 = df")
    aliased = nb.state()
    nb.run("df2['n'] = df2['a'] * 2")

    assert list(nb.restored(aliased)["df"].columns) == ["a", "g"]
    now = nb.restored()
    assert list(now["df"].columns) == ["a", "g", "n"]
    assert_same(now["df"], nb.ns["df"])
    assert_same(now["df2"], nb.ns["df2"])


def test_an_alias_of_a_plain_python_value_is_not_a_step(nb):
    nb.ns["d"] = {"a": 1}
    assert nb.run("d2 = d") == []


# ---- objects changed through another variable ---------------------------------

def test_a_figure_changed_through_its_axes_is_versioned(nb):
    pytest.importorskip("matplotlib")
    import matplotlib.pyplot as plt

    nb.ns["plt"] = plt
    nb.run("fig, ax = plt.subplots()")
    empty = nb.state()
    nb.run("bars = ax.bar([1, 2, 3], [3, 1, 2])")
    barred = nb.state()
    nb.run("ax.set_title('Release')")

    assert nb.restored(empty)["fig"].axes[0].get_title() == ""
    assert len(nb.restored(empty)["fig"].axes[0].patches) == 0
    assert len(nb.restored(barred)["fig"].axes[0].patches) == 3
    assert nb.restored(barred)["fig"].axes[0].get_title() == ""
    assert nb.restored()["fig"].axes[0].get_title() == "Release"
    plt.close("all")


def test_an_untouched_object_is_not_stored_again(nb):
    nb.ns["state"] = {"count": 0}
    nb.run("keep = dict(state)")
    before = stored_files(nb)

    nb.run("other = df.copy()")

    new = [f for f in stored_files(nb) if f not in before]
    assert len(new) == 1, "only the new frame is stored, not `keep` again"


def test_a_mutated_container_is_versioned(nb):
    nb.ns["make"] = lambda: {"count": 0}
    nb.run("box = make()")
    first = nb.state()
    nb.ns["box"]["count"] = 5  # changed by code pmprov does not intercept
    nb.run("other = df.copy()")

    assert nb.restored(first)["box"] == {"count": 0}
    assert nb.restored()["box"] == {"count": 5}
