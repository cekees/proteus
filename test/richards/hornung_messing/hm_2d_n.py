from proteus import *
from proteus.default_n import *
from hm_2d_p import *

timeIntegrator = ForwardIntegrator
timeIntegration = Richards.ThetaScheme
timeIntegrationTheta = ct.theta_time
stepController = FixedStep
timeOrder = 1
DT = T / ct.nDT
tnList = [i * DT for i in range(ct.nDT + 1)]
nDTout = ct.nDT

nnx = nny = ct.nn
nLevels = 1
if ct.quad:
    # Q1 on squares, h = 1/(nn-1), 2x2 Gauss points (the kernel's Q1 build)
    quad = True
    femSpaces = {0: C0_AffineLinearOnCubeWithNodalBasis}
    elementQuadrature = CubeGaussQuadrature(nd, 2)
    elementBoundaryQuadrature = CubeGaussQuadrature(nd - 1, 2)
else:
    # structured right-triangle P1 mesh; all off-diagonal stiffness entries
    # are <= 0, so the edge-based low order needs no M-matrix clipping
    femSpaces = {0: C0_AffineLinearOnSimplexWithNodalBasis}
    elementQuadrature = SimplexGaussQuadrature(nd, 4)
    elementBoundaryQuadrature = SimplexGaussQuadrature(nd - 1, 4)
    triangleFlag = 0

subgridError = None
massLumping = False
shockCapturing = None
numericalFluxType = Richards_IIPG_exterior  # weak boundary flux, as in the WRR decks

multilevelNonlinearSolver = Newton
levelNonlinearSolver = Newton
nonlinearSmoother = None
fullNewtonFlag = True
tolFac = 0.0
nl_atol_res = 1.0e-10
maxNonlinearIts = 50
maxLineSearches = 20

matrix = SparseMatrix
multilevelLinearSolver = LU
levelLinearSolver = LU
linearSmoother = None
linTolFac = 0.001
conservativeFlux = None
