"""Helper run under mpiexec by test_parallel_nested_iteration.py; not collected.

Builds the real MultilevelTransport for a small Poisson problem and checks the
*_bcListDict operators -- the ones nonlinear nested iteration is handed -- and
then runs the nested-iteration solve itself.

The operators are checked against the subdomain meshes, in the numbering the
solvers' dof arrays use, and ghost entries are part of the check because the
solvers read them straight away to form the next residual. That is the lesson
of nodal_prolongation_mpi.py: a test that samples everything in one numbering
agrees with itself while being unusable with a partitioned vector.

An operator whose dimensions do not match the vectors it is given used to
write past the end of the array (the C loop is compiled without bounds
checks), which is how this was found: NLNI prolonged a 7-entry coarse solution
with a 25x9 global matrix. Hence the explicit shape assertions first.

Prints a checksum of the globally assembled fine solution, like
parallel_smoother_mpi.py, so rank counts can be compared.
"""
import os
import sys

import numpy as np
from proteus import Comm
comm = Comm.init()
from mpi4py import MPI
from proteus.iproteus import opts as runopts
from proteus import (MeshTools, defaults, default_n, default_s, default_so,
                     NumericalSolution, LinearSolvers)
from proteus.superluWrappers import SparseMatrix
from multilevel_problems import make_uncoupled_components

rank, size = comm.rank(), comm.size()
mpi = comm.comm.tompi4py()
TESTDIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "poisson_2d")
nLevels = int(sys.argv[1]) if len(sys.argv) > 1 else 3
space = sys.argv[2] if len(sys.argv) > 2 else 'p1'
assert space in ('p1', 'p2', 'p1x2'), space
NC = 2 if space == 'p1x2' else 1

p = defaults.load_physics('poisson_het_2d_p', TESTDIR)
n = defaults.load_numerics('poisson_het_2d_c0pk_n', TESTDIR)
p.meshfile = os.path.join(TESTDIR, 'square4x4')
p.x0 = (0., 0., 0.)
p.L = (1., 1., 1.)
n.nLevels = nLevels
n.femSpaces = dict((i, {'p1': default_n.C0_AffineLinearOnSimplexWithNodalBasis,
                        'p1x2': default_n.C0_AffineLinearOnSimplexWithNodalBasis,
                        'p2': default_n.C0_AffineQuadraticOnSimplexWithNodalBasis}[space])
                   for i in range(NC))
if NC == 2:
    # two uncoupled copies of the same Poisson problem: exercises the
    # per-component operators, and the components must agree with each other
    make_uncoupled_components(p, n, NC, default_n.C0_AffineLinearOnSimplexWithNodalBasis)
n.nLevels = nLevels
n.femSpaces = dict((i, {'p1': default_n.C0_AffineLinearOnSimplexWithNodalBasis,
                        'p1x2': default_n.C0_AffineLinearOnSimplexWithNodalBasis,
                        'p2': default_n.C0_AffineQuadraticOnSimplexWithNodalBasis}[space])
                   for i in range(NC))
if NC == 2:
    # two uncoupled copies of the same Poisson problem: exercises the
    # per-component operators, and the components must agree with each other
    from proteus import TransportCoefficients
    p.nc = NC
    rng_ = range(NC)
    p.analyticalSolution = dict((i, p.u5Ex()) for i in rng_)
    p.dirichletConditions = dict((i, p.getDBC5) for i in rng_)
    p.aOfX = dict((i, p.a5) for i in rng_)
    p.fOfX = dict((i, p.f5) for i in rng_)
    p.advectiveFluxBoundaryConditions = dict((i, p.getAdvFluxBC5) for i in rng_)
    p.diffusiveFluxBoundaryConditions = dict((i, {i: p.getDiffFluxBC5}) for i in rng_)
    p.analyticalSolutionVelocity = dict((i, p.velEx(p.analyticalSolution[i], p.aOfX[i])) for i in rng_)
    p.coefficients = TransportCoefficients.PoissonEquationCoefficients(p.aOfX, p.fOfX, NC, p.nd)
    p.coefficients.variableNames = ['u%d' % i for i in rng_]
    p.fluxBoundaryConditions = dict((i, 'setFlow') for i in rng_)
n.numericalFluxType = default_n.Advection_DiagonalUpwind_Diffusion_SIPG_exterior
n.multilevelLinearSolver = LinearSolvers.KSP_petsc4py
n.levelLinearSolver = LinearSolvers.KSP_petsc4py
n.linearSmoother = None
n.parallelPartitioningType = MeshTools.MeshParallelPartitioningTypes.node
n.nLayersOfOverlapForParallel = 1
so = default_so
so.name = p.name = "bct_%s_%d_%d" % (space, nLevels, size)
so.sList = [default_s]
runopts.logLevel = 1
runopts.verbose = False
try:
    ns = NumericalSolution.NS_base(so, [p], [n], so.sList, runopts)
except NotImplementedError as e:
    # the documented refusal for spaces the nodal operators cannot describe
    if rank == 0:
        print("REFUSED %s" % (str(e).replace("\n", " "),))
        sys.stdout.flush()
    sys.exit(0)
model = ns.modelList[0]
mt = model.meshTransfers


def field(x):
    """Linear, so prolongation (linear interpolation) must reproduce it exactly."""
    return 3.0 * x[:, 0] - 2.0 * x[:, 1] + 0.75


def wiggly(x):
    """Not linear, so only an operator that copies values can reproduce it."""
    return np.sin(5.0 * x[:, 0]) * np.cos(3.0 * x[:, 1]) + x[:, 0] * x[:, 1]


def owned_dot(par, a, b):
    k = par.dim_proc
    return mpi.allreduce(float(np.dot(a[:k], b[:k])), op=MPI.SUM)


if size > 1 and space != 'p2':
    for l in range(1, nLevels):
        for name in ('prolong_bcListDict', 'restrict_bcListDict',
                     'scaled_restrict_bcListDict', 'interp_bcListDict'):
            for cj in range(NC):
                op = getattr(mt, name)[cj][l]
                assert not isinstance(op, SparseMatrix), (
                    "%s[%i][%i] is still the global matrix; it cannot act on a "
                    "subdomain's vectors" % (name, cj, l))

for cj in range(NC if space != 'p2' else 0):
    for l in range(nLevels - 1):
        coarse_lm = model.levelModelList[l]
        fine_lm = model.levelModelList[l + 1]
        xc = np.asarray(coarse_lm.mesh.nodeArray)
        xf = np.asarray(fine_lm.mesh.nodeArray)
        nc_, nf_ = xc.shape[0], xf.shape[0]
        par_c = coarse_lm.u[cj].par_dof
        par_f = fine_lm.u[cj].par_dof
        P = mt.prolong_bcListDict[cj][l + 1]
        R = mt.restrict_bcListDict[cj][l + 1]
        Rs = mt.scaled_restrict_bcListDict[cj][l + 1]
        I = mt.interp_bcListDict[cj][l + 1]
        sums = mt.restrict_bcSumListDict[cj][l + 1]

        if size > 1:
            # shapes first: a mismatch here is the heap corruption this guards against
            assert list(P.shape) == [nf_, nc_], (l, P.shape, nf_, nc_)
            assert list(R.shape) == [nc_, nf_], (l, R.shape, nf_, nc_)
            assert list(Rs.shape) == [nc_, nf_], (l, Rs.shape, nf_, nc_)
            assert list(I.shape) == [nc_, nf_], (l, I.shape, nf_, nc_)
            assert sums.size == nc_, (l, sums.size, nc_)

        # prolong: a linear field must survive exactly, ghosts included
        out = np.zeros(nf_)
        P.matvec(field(xc), out)
        err = mpi.allreduce(float(np.abs(out - field(xf)).max()), op=MPI.MAX)
        assert err < 1.0e-12, "level %i prolong_bc: linear field off by %r" % (l, err)

        # interp: picks the fine values that sit on coarse nodes, so any function
        # sampled on the fine mesh comes back as itself on the coarse mesh
        back = np.zeros(nc_)
        I.matvec(wiggly(xf), back)
        err = mpi.allreduce(float(np.abs(back - wiggly(xc)).max()), op=MPI.MAX)
        assert err < 1.0e-12, "level %i interp_bc: injection off by %r" % (l, err)

        # restrict is the adjoint of prolong, over the owned entries
        rng = np.random.default_rng(23 + l)
        c = rng.random(nc_)
        f = rng.random(nf_)
        Pc = np.zeros(nf_)
        Rf = np.zeros(nc_)
        P.matvec(c, Pc)
        R.matvec(f, Rf)
        lhs, rhs = owned_dot(par_f, Pc, f), owned_dot(par_c, c, Rf)
        assert abs(lhs - rhs) <= 1.0e-12 * max(1.0, abs(lhs)), (
            "level %i restrict_bc is not the adjoint of prolong_bc: %r vs %r" % (l, lhs, rhs))

        # the row sums are R applied to ones -- including their ghost entries --
        # and the scaled restriction divides by them, so it preserves constants
        ones_f = np.ones(nf_)
        Rones = np.zeros(nc_)
        R.matvec(ones_f, Rones)
        err = mpi.allreduce(float(np.abs(Rones - sums).max()), op=MPI.MAX)
        assert err < 1.0e-12, "level %i restrict_bcSum: row sums off by %r" % (l, err)
        scaled = np.zeros(nc_)
        Rs.matvec(ones_f, scaled)
        err = mpi.allreduce(float(np.abs(scaled - 1.0).max()), op=MPI.MAX)
        assert err < 1.0e-12, "level %i scaled_restrict_bc: constants not preserved, off by %r" % (l, err)


ns.calculateSolution("bct")
lm = ns.modelList[0].levelModelList[-1]
sums_by_component = []
for cj in range(NC):
    dm = lm.u[cj].femSpace.dofMap
    N = dm.nDOF_all_processes
    g = np.zeros(N)
    s2g = np.asarray(dm.subdomain2global)
    off = np.asarray(dm.dof_offsets_subdomain_owned)
    nown = int(off[rank + 1] - off[rank])
    dof = np.asarray(lm.u[cj].dof)
    for i in range(nown):
        g[int(s2g[i])] = dof[i]
    G = np.zeros(N)
    mpi.Allreduce(g, G, op=MPI.SUM)
    sums_by_component.append(G.sum())
if rank == 0:
    print(f"RESULT space={space} levels={nLevels} ranks={size} N={N} checksum={sums_by_component[0]:.12e} "
          f"checksums={','.join('%.12e' % c for c in sums_by_component)}")
    sys.stdout.flush()
