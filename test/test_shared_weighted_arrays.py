#!/usr/bin/env python
"""Weighted shape arrays that would hold the same values share one array.

``('grad(w)*dV_a',0,0)``, ``('grad(w)*dV_f',0)``, ``('w*dV_r',0)`` ... are the
shape function (or its gradient) times the quadrature weight and |det J|. When
two of them use the same reference quadrature weights their values are
identical, and each is a full (elements, points, dofs[, space]) array, so
OneLevelTransport stores one (``OneLevelTransport.shareWeightedArrays``). What
is pinned: the sharing happens on the Poisson problems; it never merges arrays
whose weights differ; and the answer does not change.
"""
import os

import numpy as np
import pytest

from proteus import Comm
Comm.init()
from proteus import Quadrature, Transport, default_s, default_so, NumericalSolution
from proteus.defaults import load_numerics, load_physics
from proteus.iproteus import opts as runopts

HERE = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'ci')


def _solve(numerics='poisson_3d_c0p1_n', quadrature=None, share=True):
    Transport.OneLevelTransport.shareWeightedArrays = share
    try:
        p = load_physics('poisson_3d_p', HERE)
        n = load_numerics(numerics, HERE)
        n.nnx = n.nny = n.nnz = 5
        n.nLevels = 1
        if quadrature is not None:
            n.elementQuadrature = quadrature
        so = default_so
        so.name = p.name = 'sharedweights'
        so.sList = [default_s]
        runopts.logLevel = 1
        runopts.verbose = False
        ns = NumericalSolution.NS_base(so, [p], [n], so.sList, runopts)
        ns.calculateSolution('sharedweights')
        lm = ns.modelList[0].levelModelList[-1]
        return lm, np.array(lm.u[0].dof)
    finally:
        Transport.OneLevelTransport.shareWeightedArrays = True


def _weighted(q, prefix):
    return [k for k in q if isinstance(k, tuple) and k[0].startswith(prefix)]


def _distinct(q, keys):
    return len({id(q[k]) for k in keys})


def test_equal_weights_share_storage():
    lm, _ = _solve()
    for prefix in ('grad(w)*dV_', 'w*dV_'):
        keys = _weighted(lm.q, prefix)
        assert len(keys) > 1
        assert _distinct(lm.q, keys) < len(keys), (prefix, keys)
    lm, _ = _solve(share=False)
    for prefix in ('grad(w)*dV_', 'w*dV_'):
        keys = _weighted(lm.q, prefix)
        assert _distinct(lm.q, keys) == len(keys)


@pytest.mark.parametrize('numerics', ['poisson_3d_c0p1_n', 'poisson_3d_c0q1_n', 'poisson_3d_c0p2_n'])
def test_solution_and_arrays_are_unchanged(numerics):
    lm, u = _solve(numerics)
    ref, uref = _solve(numerics, share=False)
    np.testing.assert_array_equal(u, uref)
    for key, value in ref.q.items():
        if isinstance(value, np.ndarray):
            np.testing.assert_array_equal(lm.q[key], value, err_msg=repr(key))


def test_different_weights_are_not_shared():
    # the 'f' integral on a different rule than the rest: its weights differ,
    # so its weighted arrays must not be merged with the others
    quadrature = {'default': Quadrature.SimplexGaussQuadrature(3, 4),
                  ('f', 0): Quadrature.SimplexGaussQuadrature(3, 2)}
    lm, u = _solve(quadrature=quadrature)
    w = lm.elementQuadratureWeights
    assert not np.array_equal(w[('f', 0)], w[('a', 0, 0)])
    assert lm.q[('grad(w)*dV_f', 0)] is not lm.q[('grad(w)*dV_a', 0, 0)]
    ref, uref = _solve(quadrature=quadrature, share=False)
    np.testing.assert_array_equal(u, uref)
