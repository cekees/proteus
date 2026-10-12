"""Converging numerics for heat_1d_p: backward Euler with fixed steps.

The mesh is the built-in uniform interval mesh (nd = 1), so no mesh
generator is involved and nothing needs to be pinned to a stored mesh.
"""
from proteus.default_n import *
from heat_1d_p import *

timeIntegration = BackwardEuler
stepController = FixedStep
DT = T/2
nDTout = 2
tnList = [i*T/nDTout for i in range(nDTout+1)]

femSpaces = {0: C0_AffineLinearOnSimplexWithNodalBasis}
elementQuadrature = SimplexGaussQuadrature(nd, 3)
elementBoundaryQuadrature = SimplexGaussQuadrature(nd-1, 3)
nn = 11
nLevels = 1

# weak Dirichlet conditions, as KSP_petsc4py requires
numericalFluxType = Advection_DiagonalUpwind_Diffusion_IIPG_exterior

multilevelNonlinearSolver = Newton
levelNonlinearSolver = Newton
fullNewtonFlag = True
tolFac = 0.0
nl_atol_res = 1.0e-8
maxNonlinearIts = 10

matrix = SparseMatrix
multilevelLinearSolver = KSP_petsc4py
levelLinearSolver = KSP_petsc4py
linearSmoother = None
linTolFac = 0.0
l_atol_res = 1.0e-12

parallelPartitioningType = MeshParallelPartitioningTypes.node
nLayersOfOverlapForParallel = 0
