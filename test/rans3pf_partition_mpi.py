#!/usr/bin/env python
"""The ProjScheme_with_EV manufactured-solution case (RANS3PF, pressure increment, pressure, initial pressure) for a few steps.

Run under mpiexec by test_rank_interface_flux.py. RANS3PF is a C++ kernel with its own exterior loop. On a
face with no condition its momentum flux has a one-sided inflow term, and the faces a rank shares with its
neighbour are such faces when the partition has no overlap. FluxBoundaryConditions specifies a zero advective
flux there; NOZERO=1 turns that off, as a control.

On more than one rank the pressure models are solved with PETSc (the case solves them with SuperLU), and the
pressure increment, which pins a dof in serial, keeps its constant in the null space instead; the velocity
does not depend on the pressure increment's constant. Prints RESULT with the L2 norms squared of u and v.
PART_TYPE / NLAYERS pick the partition (mesh and assembly alike); NSTEPS the number of time levels.
"""
import os, sys
from proteus import Comm
comm = Comm.init()
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import proteus
from proteus import default_s, MeshTools, LinearSolvers
if os.environ.get('NOZERO'):
    from proteus import FemTools
    FemTools.FluxBoundaryConditions._noAdvectiveFluxOnRankInterfaces = lambda self, *a, **k: None
from proteus.iproteus import opts
from ProjScheme_with_EV import parameters
parameters.ct.USE_SUPG_NS = 0; parameters.ct.ARTIFICIAL_VISCOSITY_NS = 0; parameters.ct.INT_BY_PARTS_PRESSURE = 0
from ProjScheme_with_EV import NS_convergence as C
C.useSuperlu = (comm.size() == 1 and not os.environ.get("FORCEKSP"))
ptype = getattr(MeshTools.MeshParallelPartitioningTypes, os.environ.get("PART_TYPE", "element"))
nl = int(os.environ.get("NLAYERS", "0"))
C.parallelPartitioningType = ptype; C.nLayersOfOverlapForParallel = nl
C.domain.MeshOptions.parallelPartitioningType = ptype
C.domain.MeshOptions.nLayersOfOverlapForParallel = nl
from ProjScheme_with_EV import (NS_convergence_so as so, twp_navier_stokes_p, twp_navier_stokes_n, pressureincrement_p, pressureincrement_n, pressure_p, pressure_n, pressureInitial_p, pressureInitial_n)
pn = [(twp_navier_stokes_p, twp_navier_stokes_n), (pressureincrement_p, pressureincrement_n), (pressure_p, pressure_n), (pressureInitial_p, pressureInitial_n)]
pl, nl_ = [], []
for p, n in pn:
    if p.name is None: p.name = p.__name__
    if not C.useSuperlu:
        n.multilevelLinearSolver = n.levelLinearSolver = LinearSolvers.KSP_petsc4py
        n.linearSmoother = None
    n.parallelPartitioningType = ptype; n.nLayersOfOverlapForParallel = nl
    pl.append(p); nl_.append(n)
if not C.useSuperlu:
    pc = pressureincrement_p.coefficients
    pc.fixNullSpace = False
    pc.nullSpace = 'ConstantComponentsNullSpace'
    pc.nullSpaceComponents = [0]
so.tnList = so.tnList[:int(os.environ.get("NSTEPS", "5"))]
so.name += "_r3"
ns = proteus.NumericalSolution.NS_base(so, pl, nl_, [default_s]*4, opts)
ns.calculateSolution("r3")
from mpi4py import MPI
c = comm.comm.tompi4py()
out = []
for mi, ci in ((0, 0), (0, 1)):
    m = ns.modelList[mi].levelModelList[-1]
    k = ('u', ci)
    own = m.mesh.nElements_owned
    out.append(c.allreduce(float((m.q[k][:own]**2 * m.q['dV'][:own]).sum()), op=MPI.SUM))
if comm.rank() == 0:
    print("RESULT", " ".join("%.12e" % v for v in out), "own=%d" % ns.modelList[0].levelModelList[-1].mesh.nElements_owned, flush=True)
m = ns.modelList[0].levelModelList[-1].mesh
print("RANK", comm.rank(), "ifaces", int((m.elementBoundaryMaterialTypes[m.exteriorElementBoundariesArray] == 0).sum()), flush=True)

