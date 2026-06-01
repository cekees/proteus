"""
TADR numerics for the FluidFlower domain (start_again case).

STAB=2 + LUMPED_MASS_MATRIX -> RKEV + ExplicitLumpedMass.  femSpaces and
quadrature are inherited from domain.py.
"""
from __future__ import print_function
from __future__ import absolute_import

from proteus import *
from proteus.default_n import *
from domain import *
from transport_p import *


multilevelNonlinearSolver = Newton
if coefficients.STABILIZATION_TYPE in [0, 1]:
    levelNonlinearSolver = Newton
    maxLineSearches = 0
    fullNewtonFlag = True
    updateJacobian = True
    timeIntegration = BackwardEuler_cfl
elif coefficients.STABILIZATION_TYPE == 1:
    levelNonlinearSolver = TwoStageNewton
    fullNewtonFlag = False
    updateJacobian = False
    timeIntegration = BackwardEuler_cfl
elif coefficients.STABILIZATION_TYPE == 5:
    # ImplicitEV: backward-Euler implicit edge-based upwind scheme solved with
    # Newton (the kernel assembles the residual in calculateResidual and the
    # M-matrix Jacobian in calculateJacobian).  Must NOT use the explicit
    # lumped-mass solver here (see the STAB==5 asserts in TADR.py): that path
    # never solves the implicit system.
    #
    # Use BackwardEuler_cfl, NOT plain BackwardEuler.  Under
    # Sequential_MinModelStep the system dt is the MIN of each model's proposed
    # dt; plain BackwardEuler is non-adaptive, so it would freeze TADR's
    # proposal at the initial 1e-8 and cap the whole coupled system there.
    # BackwardEuler_cfl is adaptive (its dt proposal grows ~2x/step), so
    # mphase_co2 drives dt.  runCFL is set huge below so the CFL stays
    # NON-binding (implicit => unconditionally stable): TADR never throttles dt,
    # even at breakthrough; dt is bounded only by mphase_co2 and the tnList
    # output cadence.
    levelNonlinearSolver = Newton
    maxLineSearches = 0
    fullNewtonFlag = True
    updateJacobian = True
    maxNonlinearIts = 25     # give Newton room (like flow_n.py); the near-linear
                             # M-matrix system normally converges in 1-3 iters
    timeIntegration = BackwardEuler_cfl
else:
    fullNewtonFlag = False
    updateJacobian = False
    timeIntegration = TADR.RKEV
    if coefficients.LUMPED_MASS_MATRIX:
        levelNonlinearSolver = ExplicitLumpedMassMatrix
    else:
        levelNonlinearSolver = ExplicitConsistentMassMatrixForVOF

SSPOrder = 2
stepController = Min_dt_controller
if coefficients.STABILIZATION_TYPE == 5:
    # ImplicitEV is unconditionally stable -> keep the CFL non-binding so TADR
    # never caps the coupled system dt (mphase_co2 + tnList drive it).  The
    # 2x/step growth cap in BackwardEuler_cfl still ramps dt up smoothly from
    # the initial 1e-8.
    runCFL = 1.0e8
else:
    runCFL = 0.95
timeOrder = SSPOrder
nStagesTime = SSPOrder

numericalFluxType = TADR.NumericalFlux
shockCapturing = TADR.ShockCapturing(
    coefficients, nd,
    shockCapturingFactor=shockCapturingFactor_tadr,
    lag=lag_shockCapturing_tadr,
)

# --- Solver tolerances ----------------------------------------------------
# default_n leaves nl_atol_res = l_atol_res = 1.0, so the TADR Newton and its
# KSP "converge" at iteration 0 (the c-residual ~1e-5 is already << 1.0) and
# the linear system is barely solved.  Set real tolerances here (mirrors
# flow_n.py).  tolFac=0 -> use the absolute nl_atol_res directly (do not scale
# by the initial residual).  These are read for the Newton paths (STAB 0/1/5);
# the explicit RKEV path ignores them.
tolFac      = 0.0
nl_atol_res = 1.0e-8          # nonlinear (Newton) absolute residual tolerance
linTolFac   = 1.0e-3          # KSP relative tolerance -> 3-digit linear reduction
l_atol_res  = 1.0e-12         # tiny so it does NOT bind: c is a small field
                              # (residual ~1e-5), so let linTolFac (relative)
                              # drive the linear solve instead of a fixed atol.

matrix = SparseMatrix
if parallel:
    multilevelLinearSolver         = KSP_petsc4py
    levelLinearSolver              = KSP_petsc4py
    linear_solver_options_prefix   = 'tadr_'
    linearSolverConvergenceTest    = 'r-true'
else:
    multilevelLinearSolver = LU
    levelLinearSolver      = LU

if checkMass:
    auxiliaryVariables = [MassOverRegion()]

parallelPartitioningType = MeshParallelPartitioningTypes.node
