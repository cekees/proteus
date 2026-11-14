"""Kirsch plate with a hole in plane strain, solved with proteus.elastoplastic on a one-element-thick tetrahedral slab.

Linear elasticity is obtained from the elastoplastic model with a cohesion far above any stress and P_a = 0.
Exact displacements are imposed on the outer boundary, symmetry on x = 0 and y = 0, w = 0 on the slab faces, and the hole
is traction free.
"""
import types
import numpy as np
from proteus import default_p, default_n, default_s, default_so
from proteus import Domain, FemTools, Quadrature, NumericalFlux, StepControl, TimeIntegration, NonlinearSolvers
from proteus import LinearSolvers, LinearAlgebraTools, MeshTools, NumericalSolution
from proteus.elastoplastic import ElastoPlastic
try:
    from . import kirsch_exact, slab_mesh
except ImportError:
    import kirsch_exact, slab_mesh

S, A, L, E, NU = 100.0, 1.0, 4.0, 2.0e5, 0.3
bf = slab_mesh.boundaryFlags


def _module(name, base):
    m = types.ModuleType(name)
    m.__dict__.update({k: v for k, v in base.__dict__.items() if not k.startswith('__')})
    return m


def make_pn(fileprefix, order=1):
    p = _module('kirsch_p', default_p)
    p.name = 'kirsch'
    p.nd = 3
    p.domain = Domain.MeshTetgenDomain(fileprefix)
    p.domain.boundaryFlags = bf
    p.genMesh = False
    smTypes = np.zeros((1, 14), 'd')
    smFlags = np.zeros((1,), 'i')
    D = E/((1.0 + NU)*(1.0 - 2.0*NU))
    smTypes[0, 0] = E
    smTypes[0, 1] = NU
    smTypes[0, 2] = D*(1.0 - NU)
    smTypes[0, 3] = D*NU
    smTypes[0, 4] = 0.5*D*(1.0 - 2.0*NU)
    smTypes[0, 5] = 0.0      # friction angle
    smTypes[0, 6] = 1.0e12   # cohesion: never yields
    smTypes[0, 7] = 0.0      # dilation angle
    smTypes[0, 12] = 0.0     # P_a = 0: constant modulus
    smTypes[0, 13] = 0.0     # density
    smFlags[0] = 1
    p.LevelModelType = ElastoPlastic.LevelModel
    p.coefficients = ElastoPlastic.Coefficients(modelType_block=smFlags, modelParams_block=smTypes,
                                                g=[0.0, 0.0, 0.0], rhow=0.0, pa=0.0, nd=3, SRF=1.0)
    p.initialConditions = None
    p.analyticalSolution = None

    def ux(x, t):
        return kirsch_exact.displacement(x[0], x[1], S, A, E, NU)[0]

    def uy(x, t):
        return kirsch_exact.displacement(x[0], x[1], S, A, E, NU)[1]

    def zero(x, t):
        return 0.0

    def getDBC_u(x, flag):
        if flag in (bf['right'], bf['top']):
            return ux
        if flag == bf['left']:
            return zero

    def getDBC_v(x, flag):
        if flag in (bf['right'], bf['top']):
            return uy
        if flag == bf['bottom']:
            return zero

    def getDBC_w(x, flag):
        if flag in (bf['back'], bf['front']):
            return zero

    def stress_u(x, flag):
        if flag in (bf['right'], bf['top'], bf['left']):
            return None
        return zero

    def stress_v(x, flag):
        if flag in (bf['right'], bf['top'], bf['bottom']):
            return None
        return zero

    def stress_w(x, flag):
        if flag in (bf['back'], bf['front']):
            return None
        return zero

    p.dirichletConditions = {0: getDBC_u, 1: getDBC_v, 2: getDBC_w}
    p.stressFluxBoundaryConditions = {0: stress_u, 1: stress_v, 2: stress_w}
    p.T = 1.0

    n = _module('kirsch_n', default_n)
    n.timeIntegration = TimeIntegration.NoIntegration
    n.stepControl = StepControl.Newton_controller
    n.tnList = [0.0, 1.0]
    if order == 1:
        space = FemTools.C0_AffineLinearOnSimplexWithNodalBasis
        n.elementQuadrature = Quadrature.SimplexGaussQuadrature(3, 3)
        n.elementBoundaryQuadrature = Quadrature.SimplexGaussQuadrature(2, 3)
    else:
        space = FemTools.C0_AffineQuadraticOnSimplexWithNodalBasis
        n.elementQuadrature = Quadrature.SimplexGaussQuadrature(3, 6)
        n.elementBoundaryQuadrature = Quadrature.SimplexGaussQuadrature(2, 6)
    n.femSpaces = {0: space, 1: space, 2: space}
    n.subgridError = None
    n.massLumping = False
    n.numericalFluxType = NumericalFlux.Stress_IIPG_exterior
    n.shockCapturing = None
    n.multilevelNonlinearSolver = NonlinearSolvers.Newton
    n.levelNonlinearSolver = NonlinearSolvers.Newton
    n.fullNewtonFlag = True
    n.maxNonlinearIts = 25
    n.maxLineSearches = 0
    n.tolFac = 0.0
    n.atol = 1.0e-5
    n.nl_rtol_res = 0.0
    n.nl_atol_res = 1.0e-5  # roundoff floor is ~1e-7 with E = 2e5
    n.matrix = LinearAlgebraTools.SparseMatrix
    n.multilevelLinearSolver = LinearSolvers.LU
    n.levelLinearSolver = LinearSolvers.LU
    n.linTolFac = 0.0
    n.conservativeFlux = None
    n.parallelPartitioningType = MeshTools.MeshParallelPartitioningTypes.node
    n.nLayersOfOverlapForParallel = 1
    return p, n


def solve(fileprefix, nRefine, order, opts):
    """Generate the mesh, solve, and return the level model."""
    slab_mesh.quarter_plate_with_hole(fileprefix, nRefine, a=A, L=L)
    p, n = make_pn(fileprefix, order)
    so = _module('kirsch_so', default_so)
    so.name = p.name = 'kirsch_n%d_p%d' % (nRefine, order)
    so.sList = [default_s]
    so.tnList = n.tnList
    ns = NumericalSolution.NS_base(so, [p], [n], so.sList, opts)
    ns.calculateSolution(so.name)
    return ns, ns.modelList[0].levelModelList[-1]


def errors(model):
    """Displacement and stress errors of the solution against the exact one, from the vertex values (P1 interpolant).

    Returns h, relative L2 displacement error, relative energy-type (stress L2) error, max nodal displacement error."""
    mesh = model.mesh
    X = mesh.nodeArray
    nN = X.shape[0]
    u = np.array([model.u[0].dof[:nN], model.u[1].dof[:nN], model.u[2].dof[:nN]]).T
    ue = np.zeros_like(u)
    ue[:, 0], ue[:, 1] = kirsch_exact.displacement(X[:, 0], X[:, 1], S, A, E, NU)
    tets = mesh.elementNodesArray
    P = X[tets]                                  # (nE, 4, 3)
    J = P[:, 1:, :] - P[:, :1, :]                # rows: edge vectors
    vol = np.abs(np.linalg.det(J))/6.0
    Jinv = np.linalg.inv(J)
    U = u[tets]
    dU = U[:, 1:, :] - U[:, :1, :]               # (nE, 3 edges, 3 comps)
    grad = np.einsum('eij,ejk->eki', Jinv, dU)   # grad[e, comp, dir]
    eps = 0.5*(grad + np.transpose(grad, (0, 2, 1)))
    lam = E*NU/((1 + NU)*(1 - 2*NU))
    G = E/(2*(1 + NU))
    tr = np.trace(eps, axis1=1, axis2=2)
    sig = 2*G*eps + lam*tr[:, None, None]*np.eye(3)[None]
    xc = P.mean(axis=1)
    # 4-point degree-2 quadrature for the exact stress and displacement
    aq, bq = 0.5854101966249685, 0.1381966011250105
    W = np.full((4, 4), bq) + (aq - bq)*np.eye(4)
    errS = np.zeros(len(tets))
    normS = np.zeros(len(tets))
    errU = np.zeros(len(tets))
    normU = np.zeros(len(tets))
    for q in range(4):
        xq = np.einsum('n,enk->ek', W[q], P)
        uq = np.einsum('n,enk->ek', W[q], U)
        sxx, syy, szz, sxy = kirsch_exact.stress(xq[:, 0], xq[:, 1], S, A, NU)
        errS += 0.25*vol*((sig[:, 0, 0] - sxx)**2 + (sig[:, 1, 1] - syy)**2 + (sig[:, 2, 2] - szz)**2
                          + 2*(sig[:, 0, 1] - sxy)**2 + 2*sig[:, 0, 2]**2 + 2*sig[:, 1, 2]**2)
        normS += 0.25*vol*(sxx**2 + syy**2 + szz**2 + 2*sxy**2)
        uex, uey = kirsch_exact.displacement(xq[:, 0], xq[:, 1], S, A, E, NU)
        errU += 0.25*vol*((uq[:, 0] - uex)**2 + (uq[:, 1] - uey)**2 + uq[:, 2]**2)
        normU += 0.25*vol*(uex**2 + uey**2)
    h = (vol.sum()/len(tets))**(1.0/3.0)
    return (h, np.sqrt(errU.sum()/normU.sum()), np.sqrt(errS.sum()/normS.sum()), np.abs(u - ue).max(),
            np.abs(sig[:, 0, 0]).max())
