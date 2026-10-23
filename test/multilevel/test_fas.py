#!/usr/bin/env python
"""FAS -- the Full Approximation Scheme of Kees et al. (2006), section 2.3.

The paper's second multilevel approach: rather than linearize first and apply
multigrid to the linear problem (Newton-Multigrid, see
test_kees2006_convergence.py), FAS puts the nonlinear problem itself in a
multilevel setting and relies on nonlinear smoothers.  Its coarse-grid equation
is the paper's equation (25),

    F_{l-1}( I~ u_l + v_{l-1} ) - F_{l-1}( I~ u_l ) - I F_l(u_l) = 0

where ``I~`` is the row-sum-normalized restriction used for *solutions* (as
opposed to ``I``, which restricts residuals).  ``FasTwoLevel.solve`` implements
exactly this, dividing the restricted solution by ``restrictSum``.

None of it had ever run.  Three defects, all found by running it:

* ``NonlinearSolvers`` never imported ``LinearSolver``, though NLJacobi,
  NLGaussSeidel, NLStarILU and FasTwoLevel each construct one as a placeholder.
  Every one raised NameError on construction.
* ``NLStarILU`` reached for ``self.csmoothers``.  In ``LinearSolvers`` each
  smoother does ``from . import csmoothers`` inside the class body, so
  ``self.csmoothers`` resolves there; in ``NonlinearSolvers`` the import is at
  module scope, and NLJacobi and NLGaussSeidel use the bare name correctly.
* The NLJacobi post-smoother and coarse solver were built with
  ``convergenceTest=smootherFullNewtonFlag`` -- a copy-paste of the
  ``fullNewton=`` argument below it.  ``convergenceTest`` then held ``True``
  rather than a string, and since ``converged()`` only recognises
  'its'/'rits'/'r'/'u' and ``failed()`` only flags ``['r','u']``, neither could
  ever fire: an unconditional infinite loop.  That is the worst failure mode
  here, so it gets a direct guard below.

Measured, |u - exact| at nLevels=3 (Newton reference in parentheses):

    r=1 (3.5e-4 at 2 levels, 2.36e-4 at 3): all three smoothers reach 2.36e-4
    r=2 (2.02e-3 at 3 levels):              all three smoothers reach 2.02e-3

with NLStarILU reaching it in the fewest cycles in every case -- the paper's
own conclusion, that the Schwarz/ASM smoother is the robust choice for the
degenerate operator while damped Jacobi is not.
"""
import inspect
import os

import numpy as np
import pytest

from proteus import (Context, LinearSolvers, NonlinearSolvers, NumericalSolution,
                     default_s, default_so, defaults)
from proteus.iproteus import opts as runopts

modulepath = os.path.dirname(os.path.abspath(__file__))

SMOOTHERS = {
    'NLJacobi': NonlinearSolvers.NLJacobi,
    'NLGaussSeidel': NonlinearSolvers.NLGaussSeidel,
    'NLStarILU': NonlinearSolvers.NLStarILU,
}

VALID_CONVERGENCE_TESTS = ('r', 'u', 'its', 'rits')


def _run(r, nLevels, level_solver, smoother, tag, cycles=10, smooths=5,
         return_solution=False, solve=True):
    Context.contextOptionsString = "r=%d" % r
    p = defaults.load_physics('kees2006_adr_1d_p', modulepath)
    n = defaults.load_numerics('kees2006_adr_1d_n', modulepath)
    n.nLevels = nLevels
    n.multilevelNonlinearSolver = NonlinearSolvers.NLNI
    n.levelNonlinearSolver = level_solver
    n.nonlinearSmoother = smoother
    n.multilevelLinearSolver = LinearSolvers.LU
    n.levelLinearSolver = LinearSolvers.LU
    n.nonlinearWCycles = cycles
    n.nonlinearPreSmooths = smooths
    n.nonlinearPostSmooths = smooths
    n.maxNonlinearIts = 50
    so = default_so
    so.name = p.name = tag
    so.sList = [default_s]
    runopts.logLevel = 1
    runopts.verbose = False
    # NS_base builds the solver stack in its constructor, so a test that only
    # needs to inspect the solvers must not solve -- the defect it is guarding
    # against is an infinite loop.
    ns = NumericalSolution.NS_base(so, [p], [n], so.sList, runopts)
    if not solve:
        return ns
    ns.calculateSolution(tag)
    if return_solution:
        return ns
    fine = ns.modelList[0].levelModelList[-1]
    x = fine.mesh.nodeArray[:, 0]
    exact = np.array([p.analyticalSolution[0].uOfX([xi]) for xi in x])
    return np.array(fine.u[0].dof, copy=True), exact


@pytest.fixture(scope='module')
def newton_reference():
    """Newton on each case, as the answer FAS has to reproduce."""
    return {r: _run(r, 3, NonlinearSolvers.Newton, None, 'fas_ref_r%d' % r)
            for r in (1, 2)}


@pytest.mark.parametrize('smoother_name', sorted(SMOOTHERS))
@pytest.mark.parametrize('r', [1, 2], ids=['linear_r1', 'degenerate_r2'])
def test_fas_reproduces_the_newton_solution(r, smoother_name, newton_reference):
    """FAS and Newton-Multigrid must agree: same discrete problem, same answer."""
    reference, exact = newton_reference[r]
    u, _ = _run(r, 3, NonlinearSolvers.FAS, SMOOTHERS[smoother_name],
                'fas_r%d_%s' % (r, smoother_name))

    discretization_error = np.abs(reference - exact).max()
    solver_difference = np.abs(u - reference).max()
    assert solver_difference < 0.1 * discretization_error, (
        "FAS/%s differs from Newton by %.3e against a discretization error of "
        "%.3e" % (smoother_name, solver_difference, discretization_error))


def test_schwarz_smoother_is_the_robust_one_on_the_degenerate_problem():
    """The paper's result: ASM converges where damped Jacobi still has not.

    On the degenerate problem with only two cycles, the Schwarz smoother is
    already at the Newton answer and Jacobi is nowhere near it.  This is the
    reason the paper recommends ASM, so it is worth pinning.
    """
    reference, exact = _run(2, 2, NonlinearSolvers.Newton, None, 'fas_rob_ref')
    discretization_error = np.abs(reference - exact).max()

    schwarz, _ = _run(2, 2, NonlinearSolvers.FAS, NonlinearSolvers.NLStarILU,
                      'fas_rob_asm', cycles=2, smooths=2)
    jacobi, _ = _run(2, 2, NonlinearSolvers.FAS, NonlinearSolvers.NLJacobi,
                     'fas_rob_jac', cycles=2, smooths=2)

    assert np.abs(schwarz - reference).max() < 0.1 * discretization_error
    assert np.abs(jacobi - reference).max() > np.abs(schwarz - reference).max()


def test_nonlinear_solvers_can_construct_a_linear_solver():
    """Each nonlinear smoother holds a placeholder LinearSolver.

    NonlinearSolvers never imported it, so they all raised NameError.
    """
    assert hasattr(NonlinearSolvers, 'LinearSolver')
    source = inspect.getsource(NonlinearSolvers.NLStarILU)
    assert 'self.csmoothers' not in source, (
        "NLStarILU must use the module-level csmoothers, as its sibling "
        "smoothers do; it has no csmoothers attribute of its own")


@pytest.mark.parametrize('smoother_name', sorted(SMOOTHERS))
def test_every_fas_solver_has_a_usable_convergence_test(smoother_name):
    """A non-string convergenceTest makes the solver loop forever.

    converged() recognises only 'its'/'rits'/'r'/'u' and failed() only flags
    ['r','u'], so anything else -- True, say -- means neither can ever fire.
    The NLJacobi post-smoother and coarse solver were built with
    convergenceTest=smootherFullNewtonFlag, and hung with no diagnostic.
    """
    ns = _run(2, 3, NonlinearSolvers.FAS, SMOOTHERS[smoother_name],
              'fas_ctest_%s' % smoother_name, cycles=1, smooths=1, solve=False)

    def walk(solver, seen):
        if id(solver) in seen:
            return
        seen.add(id(solver))
        test = getattr(solver, 'convergenceTest', None)
        if test is not None:
            assert test in VALID_CONVERGENCE_TESTS, (
                "%s was built with convergenceTest=%r, which neither "
                "converged() nor failed() can act on"
                % (type(solver).__name__, test))
        for attr in ('preSmoother', 'postSmoother', 'coarseSolver'):
            child = getattr(solver, attr, None)
            if child is not None:
                walk(child, seen)
        for attr in ('solverList', 'levelSolverList'):
            for child in getattr(solver, attr, []) or []:
                if child is not solver:
                    walk(child, seen)

    seen = set()
    for level_solver in ns.nlsList[0].solverList:
        walk(level_solver, seen)
    assert len(seen) > 1, "walked no nested solvers; the test proves nothing"
