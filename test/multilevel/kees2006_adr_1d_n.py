"""Numerics for the Kees et al. (2006) degenerate 1D test problem.

Steady state, C0 P1, solved with Newton.  The linear solver is overridden per
test; the defaults here are the direct solve used as the reference.
"""
from proteus import *
from proteus.default_n import *
try:
    from .kees2006_adr_1d_p import *
except ImportError:
    from kees2006_adr_1d_p import *

timeIntegration = NoIntegration
nDTout = 1

femSpaces = {0: C0_AffineLinearOnSimplexWithNodalBasis}
elementQuadrature = SimplexGaussQuadrature(nd, 3)
elementBoundaryQuadrature = SimplexGaussQuadrature(nd - 1, 3)

nn = opts.nn
nLevels = 1

# The paper stabilizes the degenerate operator with a multiscale (ASGS)
# subgrid-error term; see its section 2.2, equations (17)-(20).
subgridError = AdvectionDiffusionReaction_ASGS(coefficients, nd, lag=False)
shockCapturing = None

numericalFluxType = Advection_DiagonalUpwind_Diffusion_SIPG_exterior

multilevelNonlinearSolver = Newton
levelNonlinearSolver = Newton
fullNewtonFlag = True
nl_atol_res = 1.0e-10
tolFac = 0.0
maxNonlinearIts = 50
maxLineSearches = 25

matrix = SparseMatrix
multilevelLinearSolver = LU
levelLinearSolver = LU
linearSmoother = None
linTolFac = 0.0
l_atol_res = 1.0e-12

conservativeFlux = None
