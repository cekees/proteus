from proteus import *
from proteus.default_n import *
from celia_p import *

timeIntegrator = ForwardIntegrator
tnList = [0.0]; nDTout = math.ceil(T/opts.dt)
DT = T/float(nDTout)
for i in range(nDTout):
    tnList.append(0.0+(i+1)*DT)    
timeIntegration = BackwardEuler

if opts.num in ['fct','low-order','implicit-fct']:
    timeIntegration = Richards.ThetaScheme
else:
    timeIntegration = BackwardEuler

timeOrder = 1

stepController = FixedStep
systemStepControllerType = SplitOperator.Sequential_tnList

femSpaces = {0:C0_AffineLinearOnSimplexWithNodalBasis}

if opts.num == 'low-order-galerkin':
    elementQuadrature = SimplexLobattoQuadrature(nd,1)
    elementBoundaryQuadrature = SimplexLobattoQuadrature(nd-1,1)
else:
    elementQuadrature = SimplexGaussQuadrature(nd,3)# can do 5 in 1D
    elementBoundaryQuadrature = SimplexGaussQuadrature(nd-1,3)

nn = opts.nnx
nnx=opts.nnx
nny=opts.nny
nnz=opts.nnz

nLevels = 1

subgridError = None
shockCapturing = None
numericalFluxType = Richards_IIPG_exterior #need weak for parallel and global conservation
multilevelNonlinearSolver = Newton
levelNonlinearSolver = Newton
nonlinearSmoother = NLStarILU
fullNewtonFlag = True

tolFac = 0.0

nl_atol_res = 1.0e-8

maxNonlinearIts = 1000
if opts.num == 'vms-sc-galerkin':
    maxLineSearches = 0
else:
    maxLineSearches = 100

matrix = SparseMatrix
multilevelLinearSolver = LU
computeEigenvalues = False
levelLinearSolver = LU
linearSmoother = StarILU
linTolFac = 0.001

#conservativeFlux = {0:'pwl'}
