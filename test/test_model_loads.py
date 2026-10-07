"""proteus.defaults loaders: fresh each run, shared within a run.

A numerics module that imports its physics module must get the module the
physics object was made from, so that what numerics does to physics (a
subgrid error adding to the coefficients' stencil, say) reaches the physics
that is solved. And loading the same files again, in the same process, must
execute them again from scratch.
"""
import sys
import textwrap

import pytest

from proteus import defaults


def write(directory, name, text):
    (directory / (name + ".py")).write_text(textwrap.dedent(text))


@pytest.fixture
def model(tmp_path):
    """A model directory: a counter module, a physics and a numerics module."""
    write(tmp_path, "shared", """
        executions = []          # one entry per execution of this file
        executions.append(1)
    """)
    write(tmp_path, "m_p", """
        import shared
        from proteus.default_p import *

        class Coefficients(object):
            def __init__(self):
                self.stencil = [set([0])]

        coefficients = Coefficients()
        T = 1.0
        name = None
    """)
    write(tmp_path, "m_n", """
        import shared
        import m_p
        from m_p import coefficients
        from proteus.default_n import *

        coefficients.stencil[0].add(1)   # numerics augments the physics
        m_p.T = 2.0                      # and rebinds a physics value
        shared_seen_by_numerics = shared
    """)
    write(tmp_path, "m_so", """
        from proteus.default_so import *
        pnList = [("m_p", "m_n")]
        name = "m"
    """)
    return tmp_path


def test_numerics_modifying_physics_reaches_the_physics_object(model):
    p = defaults.load_physics("m_p", str(model))
    n = defaults.load_numerics("m_n", str(model))
    assert p.coefficients.stencil == [set([0, 1])]       # the same object
    assert n.coefficients is p.coefficients
    assert p.T == 2.0                                    # rebinding carried over


def test_a_caller_edit_to_the_physics_object_survives_the_numerics_load(model):
    p = defaults.load_physics("m_p", str(model))
    p.name = "renamed"
    defaults.load_numerics("m_n", str(model))
    assert p.name == "renamed"


def test_each_load_executes_everything_again(model):
    p1 = defaults.load_physics("m_p", str(model))
    n1 = defaults.load_numerics("m_n", str(model))
    p2 = defaults.load_physics("m_p", str(model))        # a new run
    n2 = defaults.load_numerics("m_n", str(model))
    assert p2.coefficients is not p1.coefficients
    assert p1.coefficients.stencil == p2.coefficients.stencil == [set([0, 1])]
    # a helper both import is one module within a run, a fresh one across runs
    assert n1.shared_seen_by_numerics is not n2.shared_seen_by_numerics
    assert n1.shared_seen_by_numerics.executions == [1]
    assert n2.shared_seen_by_numerics.executions == [1]


def test_nothing_leaks_into_sys_modules_or_sys_path(model):
    before_path = list(sys.path)
    sentinel = object()
    sys.modules["shared"] = sentinel                     # someone else's module
    try:
        defaults.load_physics("m_p", str(model))
        defaults.load_numerics("m_n", str(model))
        assert sys.modules["shared"] is sentinel         # put back
        assert "m_p" not in sys.modules and "m_n" not in sys.modules
        assert sys.path == before_path
    finally:
        sys.modules.pop("shared", None)


def test_same_names_in_another_directory_do_not_cross_over(model, tmp_path_factory):
    other = tmp_path_factory.mktemp("other")
    write(other, "m_p", """
        from proteus.default_p import *
        class Coefficients(object):
            def __init__(self):
                self.stencil = [set([7])]
        coefficients = Coefficients()
        T = 7.0
        name = None
    """)
    write(other, "m_n", """
        import m_p
        from m_p import coefficients
        from proteus.default_n import *
        seen_T = m_p.T        # not a bare T: a star import may bind its own
    """)
    defaults.load_physics("m_p", str(model))
    p = defaults.load_physics("m_p", str(other))
    n = defaults.load_numerics("m_n", str(other))
    assert n.seen_T == 7.0 and n.coefficients is p.coefficients


def test_load_models_loads_the_set_named_by_the_so_module(model):
    so, pList, nList = defaults.load_models("m_so", str(model))
    assert so.name == "m"
    assert pList[0].name == "m_p"
    assert pList[0].coefficients is nList[0].coefficients
    assert pList[0].coefficients.stencil == [set([0, 1])]


# --- logging between runs ------------------------------------------------------------


def test_an_event_logged_while_the_log_is_closed_goes_to_the_next_log(tmp_path, monkeypatch):
    # Model modules are executed afresh on every load, so they can log between
    # one run's closeLog and the next run's openLog.
    from proteus import Profiling
    monkeypatch.setattr(Profiling, "procID", 0)
    monkeypatch.setattr(Profiling, "logFile", None)
    monkeypatch.setattr(Profiling, "preInitBuffer", [])
    Profiling.openLog(str(tmp_path / "first.log"), 2)
    Profiling.closeLog()
    Profiling.logEvent("between runs", level=1)          # used to raise on a closed file
    Profiling.openLog(str(tmp_path / "second.log"), 2)
    Profiling.closeLog()
    Profiling.openLog(str(tmp_path / "third.log"), 2)
    Profiling.closeLog()
    assert "between runs" in (tmp_path / "second.log").read_text()
    assert "between runs" not in (tmp_path / "third.log").read_text()   # written once
