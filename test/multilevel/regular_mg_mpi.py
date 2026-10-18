"""Helper run under mpiexec by test_generated_hierarchy.py; not collected.

proteus's multigrid (MGM, Jacobi) for P1/Q1 Poisson on a regular grid of
triangles (triangleFlag 0), quadrilaterals, tetrahedra or hexahedra, from the
coarsest node count NN per axis and LEVELS levels. Prints a checksum of the
globally assembled fine-level solution, and whether the hierarchy was generated
in pieces (MeshTools.MultilevelMesh.generateRegularGrids, set from
PROTEUS_GENERATE_REGULAR_GRIDS) and whether anything initialized PETSc, so runs
can be compared across rank counts and across the two ways of building the
hierarchy.

The 3D problem lives on [-3,3]^3, so a generated hierarchy's origin is
exercised; the 2D one on the unit square, which its Dirichlet conditions need.

Usage: regular_mg_mpi.py KIND NN LEVELS [scan]

scan: after the solve, look through every numpy array this process holds (the
C++ mesh arrays are wrapped as numpy views, so they are included) for any with
at least half as many entries as the finest level has nodes -- a global-sized
array -- and report how many there are and the largest, as
``global_arrays=<count over ranks> largest=<entries>``.
"""
import os
import resource
import sys

import numpy as np
from proteus import Comm
comm = Comm.init()
from mpi4py import MPI
from proteus.iproteus import opts as runopts
from proteus import (MeshTools, defaults, default_n, default_s, default_so,
                     NumericalSolution, LinearSolvers, cpartitioning)

rank, size = comm.rank(), comm.size()
mpi = comm.comm.tompi4py()
testroot = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
kind = sys.argv[1]
nn = int(sys.argv[2])
nLevels = int(sys.argv[3])
scan = 'scan' in sys.argv[4:]
generated = []
_generate = MeshTools.MultilevelMesh.generateCartesianHierarchy
def _record(self, *args, **kwargs):
    generated.append(args[0])
    return _generate(self, *args, **kwargs)
MeshTools.MultilevelMesh.generateCartesianHierarchy = _record
if kind in ('hex', 'tet'):
    TESTDIR = os.path.join(testroot, "ci")
    p = defaults.load_physics('poisson_3d_p', TESTDIR)
    n = defaults.load_numerics('poisson_3d_c0q1_n' if kind == 'hex' else 'poisson_3d_c0p1_n', TESTDIR)
    n.nnx = n.nny = n.nnz = nn
    n.linearSolverConvergenceTest = 'r'
else:
    TESTDIR = os.path.join(testroot, "poisson_2d")
    p = defaults.load_physics('poisson_het_2d_p', TESTDIR)
    n = defaults.load_numerics('poisson_het_2d_c0pk_n', TESTDIR)
    p.x0 = (0., 0., 0.)
    p.L = (1., 1., 1.)
    p.meshfile = None
    n.nn = nn
    n.triangleFlag = 0
    if kind == 'quad':
        n.quad = True
        n.femSpaces[0] = default_n.C0_AffineLinearOnCubeWithNodalBasis
        n.elementQuadrature = default_n.CubeGaussQuadrature(p.nd, 3)
        n.elementBoundaryQuadrature = default_n.CubeGaussQuadrature(p.nd - 1, 3)
    else:
        n.quad = False
        n.femSpaces[0] = default_n.C0_AffineLinearOnSimplexWithNodalBasis
n.nLevels = nLevels
n.parallelPartitioningType = MeshTools.MeshParallelPartitioningTypes.node
n.nLayersOfOverlapForParallel = 1
n.multilevelLinearSolver = LinearSolvers.MGM
n.levelLinearSolver = LinearSolvers.MGM
n.linearSmoother = LinearSolvers.Jacobi
n.linearWCycles = 1
n.linearPreSmooths = n.linearPostSmooths = 3
so = default_so
so.name = p.name = "regularmg_%s_%d_%d_%d" % (kind, nn, nLevels, size)
so.sList = [default_s]
runopts.logLevel = 1
runopts.verbose = False
ns = NumericalSolution.NS_base(so, [p], [n], so.sList, runopts)
ns.calculateSolution("regularmg")
lm = ns.modelList[0].levelModelList[-1]
dm = lm.u[0].femSpace.dofMap
N = dm.nDOF_all_processes
g = np.zeros(N)
s2g = np.asarray(dm.subdomain2global)
off = np.asarray(dm.dof_offsets_subdomain_owned)
for i in range(int(off[rank + 1] - off[rank])):
    g[int(s2g[i])] = lm.u[0].dof[i]
G = np.zeros(N)
mpi.Allreduce(g, G, op=MPI.SUM)
petsc = mpi.allreduce(int(cpartitioning.petscInitialized()), op=MPI.MAX)
nGlobal = 0
largest = 0
if scan:
    import gc
    #A global-sized array is one indexed by a global entity: its leading
    #dimension is a global count of some level (nodes, elements, element
    #boundaries, edges, dofs) larger than this rank's subdomain has. numpy
    #arrays are not tracked by the garbage collector, so walk everything
    #reachable from the solution object (models, meshes, hierarchies, solvers,
    #operators); this helper's own assembled vectors g and G are left out.
    mlMesh = ns.mlMesh_nList[0]
    globalCounts, localMax = set(), 0
    for l, mesh in enumerate(mlMesh.meshList):
        sd = mesh.subdomainMesh
        globalCounts |= {mesh.nNodes_global, mesh.nElements_global, mesh.nElementBoundaries_global,
                         mesh.nEdges_global}
        localMax = max(localMax, sd.nNodes_global, sd.nElements_global, sd.nElementBoundaries_global,
                       sd.nEdges_global)
    for lm_ in ns.modelList[0].levelModelList:
        globalCounts.add(lm_.u[0].femSpace.dofMap.nDOF_all_processes)
    globalCounts = set(c for c in globalCounts if c > localMax)
    seen = {id(g), id(G)}
    stack = [ns]
    found = []
    while stack:
        obj = stack.pop()
        if id(obj) in seen or isinstance(obj, (type, type(sys))):
            continue
        seen.add(id(obj))
        if isinstance(obj, np.ndarray):
            if obj.ndim > 0 and obj.shape[0] in globalCounts:
                nGlobal += 1
                largest = max(largest, obj.size)
                found.append(obj.shape)
            continue
        stack.extend(gc.get_referents(obj))
    if found and (rank == 0 or len(found) < 5):
        print("global-sized arrays on rank %d (shapes):" % rank, sorted(set(found))[:10], flush=True)
    nGlobal = mpi.allreduce(nGlobal, op=MPI.SUM)
    largest = mpi.allreduce(largest, op=MPI.MAX)
peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024.0 if sys.platform != 'darwin' else 1024.0 ** 2)
peak = mpi.allreduce(peak, op=MPI.MAX)
#the sum and the sum of squares: neither depends on how the dofs are numbered
if rank == 0:
    print("RESULT kind=%s nn=%d levels=%d ranks=%d N=%d generated=%d checksum=%.12e sumsq=%.12e petsc=%d rss_max_mb=%.1f"
          " global_arrays=%d largest=%d"
          % (kind, nn, nLevels, size, N, int(bool(generated)), G.sum(), (G * G).sum(), petsc, peak, nGlobal, largest))
    sys.stdout.flush()
