"""Numerics for ad_sine_p.py: P1/P2 Galerkin on regular simplices or Q1 on regular cubes, direct solve, no stabilization."""
import os
from proteus import *
from proteus.default_n import *
from ad_sine_p import *

timeIntegration = NoIntegration
nDTout = 1
femSpaces = {0: {'p1': C0_AffineLinearOnSimplexWithNodalBasis,
                 'p2': C0_AffineQuadraticOnSimplexWithNodalBasis,
                 'q1': C0_AffineLinearOnCubeWithNodalBasis}[SPACE]}
QORDER = int(os.environ.get('AD_QORDER', '4'))
Gauss = CubeGaussQuadrature if SPACE == 'q1' else SimplexGaussQuadrature
elementQuadrature = Gauss(nd, QORDER)
elementBoundaryQuadrature = Gauss(nd - 1, QORDER)
nnx = nny = nnz = int(os.environ.get('AD_NN', '9'))
hex = SPACE == 'q1'          # a regular hexahedral (3D) / quadrilateral (2D) mesh for Q1
quad = SPACE == 'q1'
nLevels = 1
subgridError = None
shockCapturing = None
# strong Dirichlet (the Dirichlet unknowns are eliminated) with advective outflow by the interior value and prescribed
# fluxes, or weak Dirichlet by the interior-penalty exterior fluxes (see ad_sine_p.py for AD_BC)
_WEAK = {'iipg': NumericalFlux.Advection_DiagonalUpwind_Diffusion_IIPG_exterior,
         'nipg': NumericalFlux.Advection_DiagonalUpwind_Diffusion_NIPG_exterior,
         'sipg': NumericalFlux.Advection_DiagonalUpwind_Diffusion_SIPG_exterior}
if BC == 'strong':
    numericalFluxType = None
elif BC in _WEAK:
    numericalFluxType = _WEAK[BC]
elif BC == 'robin':
    numericalFluxType = NumericalFlux.RobinFactory({0: getRobin}, base=NumericalFlux.StrongDirichletFactory({0: 'outFlow'}))
else:
    numericalFluxType = NumericalFlux.StrongDirichletFactory({0: 'outFlow'})
multilevelNonlinearSolver = Newton
levelNonlinearSolver = Newton
fullNewtonFlag = True
maxNonlinearIts = 5
maxLineSearches = 0
nl_atol_res = 1.0e-10
tolFac = 0.0
matrix = SparseMatrix
multilevelLinearSolver = LU
levelLinearSolver = LU
linearSmoother = None
linTolFac = 0.0
l_atol_res = 1.0e-12
conservativeFlux = None
