#!/usr/bin/env python
"""Serial baseline for the multilevel solvers, on the test problem of

    Kees, Farthing, Howington, Jenkins & Kelley (2006), *Nonlinear multilevel
    iterative methods for multiscale models of air/water flow in porous
    media*, CMWR XVI.

The paper's equation (2), ``div[b u^q - a grad(u^r)] = 0`` on (0,1) with
u(0)=1, u(1)=0, in both cases it solves analytically: ``r=1`` (linear, eq. 3)
and ``r=2`` (degenerate, eq. 4).  The degenerate case is the interesting one --
``dphi/du -> 0`` as ``u -> 0``, so the linearized operator loses ellipticity,
which is what makes air/water flow hard and what the paper is about.

What this pins:

* Newton-Multigrid reproduces the direct solve on both cases.  This guards a
  structural defect in which MGM could never drive a *nonlinear* solve at all:
  MGM builds its levels with ``convergenceTest='its'``, so ``its == maxIts`` is
  the expected exit, but the inherited ``LinearSolver.failed()`` reported that
  as a failure.  Newton saw ``linearSolverFailed`` on its first step and gave
  up, returning something close to the initial iterate.  Linear problems hid it
  by setting ``maxNonlinearIts = 1``.
* The multigrid convergence rate is independent of the mesh, which is the
  entire point of a multilevel method (paper section 1) and the property that
  has to hold in serial before any of it is worth parallelizing.

Measured rates (geometric-mean residual reduction per W-cycle, 2 pre- and 2
post-smooths), for the record:

    r=1, Jacobi : 0.128, 0.129, 0.129, 0.129   over 41 -> 321 dof
    r=1, StarILU: ~0.01 (converges to machine precision and stagnates)
    r=2, Jacobi : 0.096, 0.106, 0.134, 0.478
    r=2, StarILU: ~0.005 - 0.022

The ``r=2``/Jacobi degradation is the paper's own result, not a defect: damped
Jacobi is not a robust smoother for the degenerate operator, while the
Schwarz/ASM smoother (``StarILU``) is.  That is the motivation for FAS.  So the
mesh-independence assertion below is made on the linear case, and the
degenerate case is pinned by agreement with the direct solve instead.
"""
import os

import numpy as np
import pytest

from proteus import (Context, LinearSolvers, NumericalSolution, default_s,
                     default_so, defaults)
from proteus.iproteus import opts as runopts

modulepath = os.path.dirname(os.path.abspath(__file__))


def _run(r, nLevels, multilevel, level, smoother, tag, cycles=10):
    Context.contextOptionsString = "r=%d" % r
    p = defaults.load_physics('kees2006_adr_1d_p', modulepath)
    n = defaults.load_numerics('kees2006_adr_1d_n', modulepath)
    n.nLevels = nLevels
    n.multilevelLinearSolver = multilevel
    n.levelLinearSolver = level
    n.linearSmoother = smoother
    n.computeLinearSolverRates = True
    n.linearWCycles = cycles
    n.linearPreSmooths = 2
    n.linearPostSmooths = 2
    so = default_so
    so.name = p.name = tag
    so.sList = [default_s]
    runopts.logLevel = 1
    runopts.verbose = False
    ns = NumericalSolution.NS_base(so, [p], [n], so.sList, runopts)
    ns.calculateSolution(tag)

    fine = ns.modelList[0].levelModelList[-1]
    x = fine.mesh.nodeArray[:, 0]
    exact = np.array([p.analyticalSolution[0].uOfX([xi]) for xi in x])
    linear = ns.lsList[0]
    finest_cycle = getattr(linear, 'mgmSolver', None) or linear.solverList[-1]
    return dict(u=np.array(fine.u[0].dof, copy=True),
                exact=exact,
                rate=finest_cycle.rReductionFactor,
                rate_is_usable=not finest_cycle.convergenceHistoryIsCorrupt,
                newton=ns.nlsList[0].solverList[-1])


@pytest.mark.parametrize('r', [1, 2], ids=['linear_r1', 'degenerate_r2'])
def test_newton_multigrid_matches_direct_solve(r):
    """MGM must reach the same answer as a direct solve on the same mesh.

    Compared against LU at matched nLevels rather than against the analytical
    solution directly: on the degenerate problem the discretization error is
    not monotone in h, so only the solver-to-solver difference is a clean
    invariant.
    """
    direct = _run(r, 3, LinearSolvers.LU, LinearSolvers.LU, None, 'k06_lu_r%d' % r)
    mg = _run(r, 3, LinearSolvers.MGM, LinearSolvers.MGM, LinearSolvers.Jacobi,
              'k06_mgm_r%d' % r)

    discretization_error = np.abs(direct['u'] - direct['exact']).max()
    solver_difference = np.abs(mg['u'] - direct['u']).max()
    assert solver_difference < 0.1 * discretization_error, (
        "Newton-Multigrid differs from the direct solve by %.3e against a "
        "discretization error of %.3e" % (solver_difference, discretization_error))


def test_multigrid_actually_drives_the_newton_iteration():
    """The degenerate problem needs several Newton steps; MGM must allow them.

    This is the direct guard on the failed()/convergenceTest='its' defect. The
    symptom was subtle -- one Newton step, no exception, and a returned field
    still close to the initial iterate.
    """
    direct = _run(2, 3, LinearSolvers.LU, LinearSolvers.LU, None, 'k06_nl_lu')
    mg = _run(2, 3, LinearSolvers.MGM, LinearSolvers.MGM, LinearSolvers.Jacobi,
              'k06_nl_mgm')

    assert direct['newton'].its > 1, "test problem is not actually nonlinear"
    assert not mg['newton'].linearSolverFailed, (
        "MGM reported linear solver failure; running its prescribed cycle "
        "count is its convergence criterion, not a failure")
    assert not mg['newton'].failedFlag
    assert mg['newton'].its == direct['newton'].its, (
        "Newton took %d steps with MGM but %d with a direct solve"
        % (mg['newton'].its, direct['newton'].its))


def test_convergence_rate_is_mesh_independent():
    """A multigrid rate must not degrade as the mesh refines.

    Uses the linear case and damped Jacobi, where the rate is clean. Each extra
    level halves h, so this spans a 8x refinement.
    """
    rates = {}
    for nLevels in (2, 3, 4, 5):
        result = _run(1, nLevels, LinearSolvers.MGM, LinearSolvers.MGM,
                      LinearSolvers.Jacobi, 'k06_rate_%d' % nLevels)
        assert result['rate_is_usable'], (
            "residual stopped decreasing monotonically at nLevels=%d, so the "
            "measured rate is not meaningful" % nLevels)
        rates[nLevels] = result['rate']

    assert all(0.0 < rho < 0.25 for rho in rates.values()), (
        "multigrid rates outside the expected band: %r" % (rates,))

    # The real assertion: the rate must not grow with refinement. Measured
    # spread is under 2%, so 1.5x leaves ample room without being vacuous.
    assert max(rates.values()) < 1.5 * min(rates.values()), (
        "convergence rate degrades under refinement, so this is not behaving "
        "like multigrid: %r" % (rates,))
