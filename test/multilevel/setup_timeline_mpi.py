"""Rank-resolved timeline of model setup, by statistical sampling; run under mpiexec, not collected.

Why setup time grows with the rank count at fixed work per rank (CXX_REFACTOR_setup_and_assembly_
SCOPING.md section 3.3, P0). The max over ranks that setup_assembly_profile_mpi.py reports cannot
say whether the growth is imbalance (one slow rank), a cost every rank pays more of (more ghosts,
more collectives), or the node itself (64 processes sharing memory bandwidth). This records, per
rank, wall time and a sampled profile of NumericalSolution's construction, and the sizes that
could explain a difference (owned and ghost nodes and elements, neighbouring ranks).

Sampling, not cProfile or tracemalloc: those inflate per-call costs a thousandfold on this code.
A thread looks at the main thread's stack every INTERVAL seconds and credits the time since the previous
sample to every function on the stack (inclusive) and to the innermost one (self).

Usage: setup_timeline_mpi.py KIND NN LEVELS OUTDIR [interval=0.005] [solve]
writes OUTDIR/rank_<r>.json (r is the MPI rank, or $TIMELINE_TAG for independent serial jobs).
"""
import mpi_failfast  # noqa: F401  (an uncaught exception aborts every rank)
import json
import os
import sys
import threading
import time

import numpy as np
from proteus import Comm
comm = Comm.init()
from mpi4py import MPI
from proteus import (MeshTools, defaults, default_s, default_so, NumericalSolution, LinearSolvers,
                     Transport)
from proteus.iproteus import opts as runopts

rank, size = comm.rank(), comm.size()
mpi = comm.comm.tompi4py()
kind, nn, levels, outdir = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), sys.argv[4]
interval = 0.005
solve = False
for flag in sys.argv[5:]:
    if flag.startswith('interval='):
        interval = float(flag.split('=')[1])
    if flag == 'solve':
        solve = True
tag = os.environ.get('TIMELINE_TAG', str(rank))
os.makedirs(outdir, exist_ok=True)

inclusive, selfcount = {}, {}
samples = [0]
stop = threading.Event()
mainThread = threading.main_thread().ident
root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def label(code):
    f = code.co_filename
    if f.startswith(root):
        f = f[len(root) + 1:]
    return "%s:%s" % (f, code.co_name)


def sampler():
    # Each sample is weighted by the time since the previous one. The sampling thread needs the GIL, so
    # a long native call that holds it delays the next sample; the stack seen afterwards still contains
    # the functions that were running through that call, so the inclusive time of every enclosing
    # function stays right, though the "self" time of the innermost one is not (it lands on the caller).
    last = time.perf_counter()
    while not stop.wait(interval):
        now = time.perf_counter()
        dt, last = now - last, now
        frame = sys._current_frames().get(mainThread)
        if frame is None:
            continue
        samples[0] += 1
        seen = set()
        first = True
        while frame is not None:
            name = label(frame.f_code)
            if first:
                selfcount[name] = selfcount.get(name, 0.0) + dt
                first = False
            if name not in seen:
                seen.add(name)
                inclusive[name] = inclusive.get(name, 0.0) + dt
            frame = frame.f_back


testroot = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
p = defaults.load_physics('poisson_3d_p', os.path.join(testroot, 'ci'))
n = defaults.load_numerics('poisson_3d_c0q1_n' if kind == 'hex' else 'poisson_3d_c0p1_n', os.path.join(testroot, 'ci'))
n.nnx = n.nny = n.nnz = nn
n.nLevels = levels
n.parallelPartitioningType = MeshTools.MeshParallelPartitioningTypes.node
n.nLayersOfOverlapForParallel = 1
n.multilevelLinearSolver = LinearSolvers.KSP_petsc4py
n.levelLinearSolver = LinearSolvers.KSP_petsc4py
n.linearSmoother = LinearSolvers.petsc_MG
n.petscOptions = {'ksp_type': 'cg', 'pc_type': 'mg', 'mg_levels_ksp_type': 'chebyshev', 'mg_levels_ksp_max_it': '2',
                  'mg_levels_pc_type': 'jacobi', 'ksp_rtol': '1e-8'}
n.nonlinearSolverConvergenceTest = 'its'
n.levelNonlinearSolverConvergenceTest = 'its'
n.useEisenstatWalker = False
Transport.MultilevelTransport.distributedTransfers = True
so = default_so
so.name = p.name = 'setuptimeline'
so.sList = [default_s]
runopts.logLevel = 1
runopts.verbose = False

if size > 1:
    mpi.Barrier()
thread = threading.Thread(target=sampler, daemon=True)
t0 = time.perf_counter()
thread.start()
ns = NumericalSolution.NS_base(so, [p], [n], so.sList, runopts)
tSetup = time.perf_counter() - t0
stop.set()
thread.join()
tSolve = None
if solve:
    t1 = time.perf_counter()
    ns.calculateSolution('setuptimeline')
    tSolve = time.perf_counter() - t1
if size > 1:
    mpi.Barrier()

mlMesh = ns.mlMesh_nList[0]
fine = mlMesh.meshList[-1]
sizes = {}
for l, mesh in enumerate(mlMesh.meshList):
    sd = getattr(mesh, 'subdomainMesh', mesh)
    sizes['level%d' % l] = {
        'nodes_owned': int(getattr(sd, 'nNodes_owned', sd.nNodes_global)), 'nodes_local': int(sd.nNodes_global),
        'elements_local': int(sd.nElements_global),
        'elements_owned': int(getattr(sd, 'nElements_owned', sd.nElements_global)),
        'neighbours': int(len(getattr(sd, 'nodeSendList', []) or [])) if hasattr(sd, 'nodeSendList') else None}
record = {'rank': rank, 'size': size, 'tag': tag, 'kind': kind, 'nn': nn, 'levels': levels,
          'setup_seconds': tSetup, 'solve_seconds': tSolve, 'samples': samples[0], 'interval': interval,
          'inclusive': inclusive, 'self': selfcount, 'sizes': sizes}
with open(os.path.join(outdir, 'rank_%s.json' % tag), 'w') as f:
    json.dump(record, f)
setups = mpi.allgather(tSetup) if size > 1 else [tSetup]
if rank == 0:
    print("TIMELINE ranks=%d nn=%d levels=%d setup seconds: min %.2f median %.2f max %.2f (argmax rank %d)"
          % (size, nn, levels, min(setups), float(np.median(setups)), max(setups), int(np.argmax(setups))))
    sys.stdout.flush()
