#!/usr/bin/env python
"""Flux boundary conditions returned as batched functions agree with per-point ones.

FluxBoundaryConditions collects, per exterior boundary quadrature point (ebNE,k),
the function g(x,t) a setter returned, and the residual evaluates all of them at
every call. A setter that returns the same ``batchedBC`` object for many points
gets them evaluated with one call per object; ordinary functions are called per
point exactly as before, and the two can be mixed.
"""
import numpy as np
import pytest

from proteus import FemTools
from proteus.FemTools import FluxBoundaryConditions, batchedBC


class Mesh(object):
    """Just what FluxBoundaryConditions reads."""
    def __init__(self, nExterior=7):
        self.nExteriorElementBoundaries_global = nExterior
        self.exteriorElementBoundariesArray = np.arange(nExterior, dtype='i')
        self.elementBoundaryMaterialTypes = np.arange(nExterior, dtype='i') % 3


NK = 4


def _x(seed=0, nExterior=7):
    return np.random.default_rng(seed).uniform(-1, 1, (nExterior, NK, 3))


@batchedBC
def linear(X, t):
    return X[:, 0] + 2.0 * X[:, 1] - t * X[:, 2]


@batchedBC
def zero(X, t):
    return np.zeros(len(X))


def linear_point(x, t):
    return x[0] + 2.0 * x[1] - t * x[2]


def _fbc(getDiffusive, getAdvective=None):
    return FluxBoundaryConditions(Mesh(), NK, _x(),
                                  getAdvectiveFluxBoundaryConditions=getAdvective,
                                  getDiffusiveFluxBoundaryConditions={0: getDiffusive})


def _evaluate(fbc, t=0.3):
    out = np.zeros((7, NK))
    fbc.evaluate(('diffusive', 0), fbc.diffusiveFluxBoundaryConditionsDictDict[0], _x(), out, t)
    return out


def test_one_point_call_matches_the_function():
    x = np.array([0.1, 0.2, 0.3])
    assert linear(x, 0.5) == pytest.approx(linear_point(x, 0.5))
    assert linear.batched is not None


def test_batched_agrees_with_per_point():
    reference = _evaluate(_fbc(lambda x, flag: linear_point))
    batched = _evaluate(_fbc(lambda x, flag: linear))
    np.testing.assert_array_equal(reference, batched)
    assert np.abs(batched).max() > 0


def test_mixed_batched_and_plain_points():
    def setter(x, flag):
        if flag == 0:
            return linear
        if flag == 1:
            return zero
        if x[0] > 0:
            return lambda x, t: 7.0 * t
        return None
    fbc = _fbc(setter)
    out = _evaluate(fbc, t=0.25)
    x = _x()
    expected = np.zeros((7, NK))
    for ebNE in range(7):
        flag = ebNE % 3
        for k in range(NK):
            if flag == 0:
                expected[ebNE, k] = linear_point(x[ebNE, k], 0.25)
            elif flag == 2 and x[ebNE, k, 0] > 0:
                expected[ebNE, k] = 7.0 * 0.25
    np.testing.assert_array_equal(out, expected)
    # one cached group per batched object, evaluated for every one of its points
    n, plain, batched, i0, i1 = fbc._groups(('diffusive', 0), fbc.diffusiveFluxBoundaryConditionsDictDict[0])
    assert len(batched) == 2 and all(len(b0) > 0 for g, b0, b1 in batched)


def test_a_batched_function_is_called_once_per_evaluation():
    calls = []

    @batchedBC
    def counting(X, t):
        calls.append(len(X))
        return np.zeros(len(X))
    fbc = _fbc(lambda x, flag: counting)
    _evaluate(fbc)
    _evaluate(fbc)
    assert calls == [7 * NK, 7 * NK]


def test_groups_follow_the_dictionary():
    fbc = _fbc(lambda x, flag: linear)
    _evaluate(fbc)
    conditions = fbc.diffusiveFluxBoundaryConditionsDictDict[0]
    del conditions[(0, 0)]
    out = _evaluate(fbc)
    assert out[0, 0] == 0.0 and out[0, 1] != 0.0


def test_wrong_number_of_values_is_refused():
    @batchedBC
    def short(X, t):
        return np.zeros(len(X) - 1)
    with pytest.raises(ValueError, match='batched boundary condition'):
        _evaluate(_fbc(lambda x, flag: short))


def test_advective_indices_cover_the_points():
    fbc = _fbc(lambda x, flag: None, getAdvective=lambda x, flag: zero)
    out = np.ones((7, NK))
    i0, i1 = fbc.evaluate('advective', fbc.advectiveFluxBoundaryConditionsDict, _x(), out, 0.0)
    assert len(i0) == len(i1) == 7 * NK
    assert not out.any()
