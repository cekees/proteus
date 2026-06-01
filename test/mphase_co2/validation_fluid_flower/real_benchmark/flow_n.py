from proteus import *
from proteus.default_n import *
from flow_p import *
from proteus.default_so import *


timeIntegrator  = ForwardIntegrator
timeIntegration = mphase_co2.ThetaScheme
timeOrder       = 1


rtol_u[0] = 1.0e-3
atol_u[0] = 1.0e-3
rtol_u[1] = 1.0e-3
atol_u[1] = 1.0e-3

stepController = HeuristicNL_dt_controller
nonlinearIterationsFloor = 7 #6
nonlinearIterationsCeil  = 14
dtNLgrowFactor          = 1.2 #2.0
dtNLreduceFactor        = 0.5
dtNLfailureReduceFactor = 0.5
useInitialGuessPredictor = True
stepExact = True


femSpaces = {0: C0_AffineLinearOnSimplexWithNodalBasis,
             1: C0_AffineLinearOnSimplexWithNodalBasis}

elementQuadrature         = SimplexLobattoQuadrature(nd, 1)
elementBoundaryQuadrature = SimplexLobattoQuadrature(nd - 1, 1)


massLumping = False
numericalFluxType = Advection_DiagonalUpwind_Diffusion_IIPG_exterior
shockCapturing = None

multilevelNonlinearSolver = Newton
levelNonlinearSolver      = Newton
nonlinearSmoother         = NLJacobi

fullNewtonFlag  = True

tolFac          = 0.0
nl_atol_res     = 1.0e-8
maxNonlinearIts = 25        # was 20 -- give Newton room when r0 spikes at sharp fronts
maxLineSearches = 5

matrix = SparseMatrix
multilevelLinearSolver = KSP_petsc4py
levelLinearSolver      = KSP_petsc4py
linearSmoother         = None
linTolFac              = 1.0e-3  # was 1e-4 -- KSP must actually solve, not skate past atol=1.0
l_atol_res             = 1.0e-3  # absolute KSP tol; previously inherited PETSc default (1.0)

parallelPartitioningType = MeshParallelPartitioningTypes.node
