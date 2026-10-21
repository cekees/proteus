#!/usr/bin/env python
"""Batched coefficient functions agree with the per-point ones.

PoissonEquationCoefficients evaluates a(x) and f(x) at every quadrature point.
An ordinary function is called once per point; one decorated with
``TransportCoefficients.batched`` is called once with all the points. Both must
fill the quadrature arrays identically, on the element, element-boundary and
exterior-boundary quadrature.
"""
import numpy as np
import pytest

from proteus import TransportCoefficients as TC


def a_point(x):
    return np.array([[x[0] + 5.0, 0.0, 0.0], [0.0, x[1] + 5.0, 0.0], [0.0, 0.0, x[2] + 10.0]], 'd')


def f_point(x):
    return -2.0 * x[0] - 2 * (5. + x[0]) - 2. * x[1] - 2. * (5. + x[1]) - 2. * x[2] - 2. * (10 + x[2])


@TC.batched
def a_batched(X):
    a = np.zeros((X.shape[0], 3, 3))
    a[:, 0, 0] = X[:, 0] + 5.0
    a[:, 1, 1] = X[:, 1] + 5.0
    a[:, 2, 2] = X[:, 2] + 10.0
    return a


@TC.batched
def f_batched(X):
    return (-2.0 * X[:, 0] - 2 * (5. + X[:, 0]) - 2. * X[:, 1] - 2. * (5. + X[:, 1])
            - 2. * X[:, 2] - 2. * (10 + X[:, 2]))


def _quadrature(shape, seed=0, boundary=False):
    rng = np.random.default_rng(seed)
    q = {'x': rng.uniform(-3, 3, shape + (3,)),
         ('f', 0): np.ones(shape),
         ('r', 0): np.zeros(shape),
         ('u', 0): np.zeros(shape),
         ('a', 0, 0): np.zeros(shape + (9,))}
    return q


def _coefficients(a, f):
    return TC.PoissonEquationCoefficients({0: a}, {0: f}, nc=1, nd=3)


@pytest.mark.parametrize('shape', [(7, 8), (1, 1), (13,)])
def test_element_quadrature(shape):
    reference, batched = _quadrature(shape), _quadrature(shape)
    _coefficients(a_point, f_point).initializeElementQuadrature(0.0, reference)
    _coefficients(a_batched, f_batched).initializeElementQuadrature(0.0, batched)
    for key in (('r', 0), ('a', 0, 0), ('f', 0)):
        np.testing.assert_array_equal(reference[key], batched[key])
    assert np.abs(reference[('r', 0)]).max() > 0 and np.abs(reference[('a', 0, 0)]).max() > 0
    assert not reference[('f', 0)].any()


def test_boundary_quadratures_and_time_varying():
    for method, name in (('initializeElementBoundaryQuadrature', 'ebq'),
                         ('initializeGlobalExteriorElementBoundaryQuadrature', 'ebqe')):
        reference, batched = _quadrature((5, 4), seed=1), _quadrature((5, 4), seed=1)
        args = (0.0, reference, reference) if name == 'ebq' else (0.0, reference)
        getattr(_coefficients(a_point, f_point), method)(*args)
        args = (0.0, batched, batched) if name == 'ebq' else (0.0, batched)
        getattr(_coefficients(a_batched, f_batched), method)(*args)
        np.testing.assert_array_equal(reference[('r', 0)], batched[('r', 0)])
        np.testing.assert_array_equal(reference[('a', 0, 0)], batched[('a', 0, 0)])
    reference, batched = _quadrature((3, 6), seed=2), _quadrature((3, 6), seed=2)
    for q, (a, f) in ((reference, (a_point, f_point)), (batched, (a_batched, f_batched))):
        c = TC.PoissonEquationCoefficients({0: a}, {0: f}, nc=1, nd=3, timeVaryingCoefficients=True)
        c.evaluate(0.0, q)
    np.testing.assert_array_equal(reference[('r', 0)], batched[('r', 0)])
    np.testing.assert_array_equal(reference[('a', 0, 0)], batched[('a', 0, 0)])


def test_wrong_number_of_values_is_refused():
    @TC.batched
    def short(X):
        return np.zeros(X.shape[0] - 1)
    q = _quadrature((4, 4))
    with pytest.raises(ValueError, match='batched coefficient function'):
        _coefficients(a_point, short).initializeElementQuadrature(0.0, q)


def test_a_batched_function_is_called_once():
    calls = []

    @TC.batched
    def f(X):
        calls.append(X.shape)
        return np.zeros(X.shape[0])
    q = _quadrature((6, 5))
    _coefficients(a_batched, f).initializeElementQuadrature(0.0, q)
    assert calls == [(30, 3)]
