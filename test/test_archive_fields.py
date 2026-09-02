"""Tests for proteus.ArchiveFields -- the declared-archive-field API.

These replace the sixteen `try/except: pass` blocks that used to live in
NumericalSolution.py. None of those blocks had a test; that is why a
NameError and a dead `phi_sp` declaration both survived in them for a long
time. See proteus/ArchiveFields.py for the full history.

Everything here runs against stub models rather than a real solve, so it
needs no compiled extensions and no mesh.
"""

import numpy as np
import pytest

from proteus.ArchiveFields import (
    CENTERINGS,
    RANKS,
    ArchiveField,
    ArchiveFieldError,
    archive_fields_for,
)


# --------------------------------------------------------------------------
# stubs
# --------------------------------------------------------------------------


class StubFemSpace(object):
    def __init__(self, tag="space"):
        self.tag = tag
        self.written = []

    def writeFunctionXdmf(self, ar, u, tCount=0, init=True):
        self.written.append((u.name, np.asarray(u.dof).copy(), tCount))


class StubU(object):
    def __init__(self, dof, femSpace):
        self.dof = dof
        self.femSpace = femSpace


class StubCoefficients(object):
    """Stands in for a TC_base subclass; `hook` supplies archiveFields."""

    def __init__(self, hook=None, **attrs):
        self._hook = hook
        for k, v in attrs.items():
            setattr(self, k, v)

    def archiveFields(self, lm):
        if self._hook is None:
            return iter(())
        return self._hook(self, lm)


class StubLevelModel(object):
    def __init__(self, coefficients, n_dof=4, name="stub_level", **attrs):
        self.coefficients = coefficients
        self.name = name
        space = StubFemSpace()
        self.u = {0: StubU(np.arange(n_dof, dtype="d"), space)}
        for k, v in attrs.items():
            setattr(self, k, v)


class StubModel(object):
    def __init__(self, lm, name="stub_model"):
        self.levelModelList = [lm]
        self.name = name


def make_model(hook=None, coeff_attrs=None, lm_attrs=None, name="stub_model"):
    coeffs = StubCoefficients(hook, **(coeff_attrs or {}))
    lm = StubLevelModel(coeffs, **(lm_attrs or {}))
    return StubModel(lm, name=name), lm


# --------------------------------------------------------------------------
# ArchiveField validation -- a malformed field is unrepresentable
# --------------------------------------------------------------------------


def test_a_valid_field_keeps_its_metadata():
    arr = np.zeros(3)
    f = ArchiveField("u", arr, center="Cell", rank="Vector", units="m/s", std_name="x")
    assert f.name == "u"
    assert f.value is arr
    assert f.center == "Cell"
    assert f.rank == "Vector"
    assert f.units == "m/s"
    assert f.std_name == "x"


def test_defaults_are_node_centered_scalars():
    f = ArchiveField("u", np.zeros(3))
    assert (f.center, f.rank, f.component) == ("Node", "Scalar", 0)


@pytest.mark.parametrize("center", sorted(CENTERINGS))
def test_every_documented_centering_is_accepted(center):
    assert ArchiveField("u", np.zeros(3), center=center).center == center


@pytest.mark.parametrize("rank", sorted(RANKS))
def test_every_documented_rank_is_accepted(rank):
    assert ArchiveField("u", np.zeros(3), rank=rank).rank == rank


def test_a_bogus_centering_is_rejected_at_declaration():
    # The old code could put any string into Center and only find out when
    # a viewer refused the archive.
    with pytest.raises(ArchiveFieldError, match="center='Vertex'"):
        ArchiveField("u", np.zeros(3), center="Vertex")


def test_a_bogus_rank_is_rejected_at_declaration():
    with pytest.raises(ArchiveFieldError, match="rank='Scalarr'"):
        ArchiveField("u", np.zeros(3), rank="Scalarr")


def test_an_empty_name_is_rejected():
    with pytest.raises(ArchiveFieldError, match="non-empty name"):
        ArchiveField("", np.zeros(3))


def test_a_none_value_is_rejected_and_says_what_to_do_instead():
    # This is the case the old `try/except: pass` was really expressing.
    # The message points at the replacement: guard with an `if` in the hook.
    with pytest.raises(ArchiveFieldError, match="guarded by an 'if'"):
        ArchiveField("u", None)


# --------------------------------------------------------------------------
# {model} expansion -- needed because several models can share one archive
# --------------------------------------------------------------------------


def test_model_placeholder_expands_to_the_model_name():
    def hook(self, lm):
        yield ArchiveField("quantDOFs_for_{model}", np.zeros(3))

    model, _ = make_model(hook, name="ls_model")
    (field,) = list(archive_fields_for(model.levelModelList[-1], model.name))
    assert field.archive_name == "quantDOFs_for_ls_model"


def test_a_name_without_the_placeholder_is_left_alone():
    f = ArchiveField("bathymetry", np.zeros(3))
    f.model_name = "sw_model"
    assert f.archive_name == "bathymetry"


def test_the_placeholder_without_a_model_name_is_an_error_not_a_literal():
    f = ArchiveField("quantDOFs_for_{model}", np.zeros(3))
    with pytest.raises(ArchiveFieldError, match="no model name was supplied"):
        f.archive_name


# --------------------------------------------------------------------------
# collection -- the base hook, and errors that used to be swallowed
# --------------------------------------------------------------------------


def test_a_model_declaring_nothing_yields_nothing():
    model, lm = make_model()
    assert list(archive_fields_for(lm, model.name)) == []


def test_coefficients_without_the_hook_at_all_yield_nothing():
    # A coefficients class predating TC_base.archiveFields must not break.
    class Ancient(object):
        pass

    lm = StubLevelModel.__new__(StubLevelModel)
    lm.coefficients = Ancient()
    assert list(archive_fields_for(lm, "m")) == []


def test_declared_fields_come_back_in_order():
    def hook(self, lm):
        yield ArchiveField("a", np.zeros(3))
        yield ArchiveField("b", np.ones(3))

    model, lm = make_model(hook)
    assert [f.name for f in archive_fields_for(lm, model.name)] == ["a", "b"]


def test_a_raising_hook_is_reported_not_swallowed():
    # THE central behaviour change. The old code wrapped each field in a
    # bare `except: pass`, so this was silence.
    def hook(self, lm):
        raise RuntimeError("boom")
        yield  # pragma: no cover

    model, lm = make_model(hook)
    with pytest.raises(ArchiveFieldError, match="boom"):
        list(archive_fields_for(lm, model.name))


def test_a_missing_attribute_in_a_hook_is_reported_not_swallowed():
    # This is precisely the phi_sp failure: the attribute stopped existing
    # and nobody found out for several releases.
    def hook(self, lm):
        yield ArchiveField("phi_sp", self.phi_sp)

    model, lm = make_model(hook)
    with pytest.raises(ArchiveFieldError, match="phi_sp"):
        list(archive_fields_for(lm, model.name))


def test_the_error_message_names_the_coefficients_class_and_model():
    def hook(self, lm):
        raise ValueError("nope")
        yield  # pragma: no cover

    model, lm = make_model(hook, name="my_model")
    with pytest.raises(ArchiveFieldError) as exc:
        list(archive_fields_for(lm, model.name))
    assert "StubCoefficients" in str(exc.value)
    assert "my_model" in str(exc.value)


def test_yielding_a_non_field_is_rejected():
    def hook(self, lm):
        yield ("bathymetry", np.zeros(3))

    model, lm = make_model(hook)
    with pytest.raises(ArchiveFieldError, match="not an ArchiveField"):
        list(archive_fields_for(lm, model.name))


def test_two_fields_with_the_same_name_are_rejected():
    # One would silently shadow the other in the archive.
    def hook(self, lm):
        yield ArchiveField("u", np.zeros(3))
        yield ArchiveField("u", np.ones(3))

    model, lm = make_model(hook)
    with pytest.raises(ArchiveFieldError, match="declared 'u' twice"):
        list(archive_fields_for(lm, model.name))


def test_names_colliding_only_after_expansion_are_still_rejected():
    def hook(self, lm):
        yield ArchiveField("q_for_m", np.zeros(3))
        yield ArchiveField("q_for_{model}", np.ones(3))

    model, lm = make_model(hook, name="m")
    with pytest.raises(ArchiveFieldError, match="twice"):
        list(archive_fields_for(lm, model.name))


# --------------------------------------------------------------------------
# femSpace resolution
# --------------------------------------------------------------------------


def test_fem_space_defaults_to_the_models_first_component():
    model, lm = make_model()
    f = ArchiveField("u", np.zeros(4))
    assert f.resolve_fem_space(lm) is lm.u[0].femSpace


def test_an_explicit_fem_space_wins():
    model, lm = make_model()
    other = StubFemSpace("other")
    f = ArchiveField("u", np.zeros(4), femSpace=other)
    assert f.resolve_fem_space(lm) is other


def test_a_missing_component_raises():
    model, lm = make_model()
    f = ArchiveField("u", np.zeros(4), component=7)
    with pytest.raises(ArchiveFieldError, match="no component 7"):
        f.resolve_fem_space(lm)


# --------------------------------------------------------------------------
# the values that used to be hacked into NumericalSolution.py
# --------------------------------------------------------------------------


def test_a_computed_field_is_evaluated_at_declaration_time():
    # `eta = b.dof + u[0].dof` is computed, not stored; the declaration must
    # reflect the state when archiveFields() runs.
    b = np.array([1.0, 1.0, 1.0, 1.0])

    def hook(self, lm):
        yield ArchiveField("eta", self.b_dof + lm.u[0].dof)

    model, lm = make_model(hook, coeff_attrs={"b_dof": b})
    (field,) = list(archive_fields_for(lm, model.name))
    np.testing.assert_array_equal(field.value, b + np.arange(4, dtype="d"))

    lm.u[0].dof[:] = 10.0
    (field2,) = list(archive_fields_for(lm, model.name))
    np.testing.assert_array_equal(field2.value, b + 10.0)


def test_a_field_owned_by_the_level_model_rather_than_the_coefficients():
    # CLSVOF's vofDOFs live on the level model; the declaration is on the
    # coefficients, which is what removes `if 'clsvof' in model.name`.
    def hook(self, lm):
        yield ArchiveField("vof", lm.vofDOFs)

    model, lm = make_model(hook, lm_attrs={"vofDOFs": np.full(4, 0.5)})
    (field,) = list(archive_fields_for(lm, model.name))
    np.testing.assert_array_equal(field.value, np.full(4, 0.5))


def test_a_flag_gated_field_is_an_ordinary_if():
    def hook(self, lm):
        if getattr(self, "outputQuantDOFs", False):
            yield ArchiveField("quantDOFs_for_{model}", lm.quantDOFs)

    off, lm_off = make_model(hook, coeff_attrs={"outputQuantDOFs": False},
                             lm_attrs={"quantDOFs": np.zeros(4)})
    assert list(archive_fields_for(lm_off, off.name)) == []

    on, lm_on = make_model(hook, coeff_attrs={"outputQuantDOFs": True},
                           lm_attrs={"quantDOFs": np.zeros(4)}, name="ncls")
    assert [f.archive_name for f in archive_fields_for(lm_on, on.name)] == [
        "quantDOFs_for_ncls"
    ]
