"""Whole-solve ranks x threads: the 3-D advection-diffusion problem of ad_solve_device.py, distributed.

Each MPI rank owns a brick of the Kuhn-tetrahedra cube mesh (the cells whose lowest node it owns, plus a ghost layer of
nodes on the high side of each axis), assembles its cells with the Kokkos colored kernel on KOKKOS_NUM_THREADS threads, and
sends its matrix and residual to PETSc through the COO interface (ghost rows are added into their owners); PETSc then
solves in parallel with GMRES + GAMG on all ranks. Run R ranks x T threads at a fixed number of cores to see where cores are
better spent: more solver ranks, or more assembly threads per rank.

  mpiexec -n R -bind-to core:T env KOKKOS_NUM_THREADS=T OMP_NUM_THREADS=1 python ad_solve_mpi.py [--repeats n] nn

Reported per phase: the maximum over ranks (what the run waits for). 'local' is the numpy mesh of the rank's brick, 'setup'
the coloring and CSR pattern in the module, 'assemble' the best-of-n colored assembly, 'coo' the matrix preallocation and value
exchange and the residual's, into PETSc, 'bc' the Dirichlet rows, then the KSP setup and solve.
"""
import argparse
import json
import resource
import time

import numpy as np
from mpi4py import MPI

import petsc4py
petsc4py.init([])
from petsc4py import PETSc

import adrh_kokkos

B = np.array([1.0, 0.5, 0.25]); KAPPA = 0.2; OMEGA = np.array([3.0, 2.0, 4.0]); OMEGA0 = 0.3
comm = MPI.COMM_WORLD
rank, size = comm.rank, comm.size


def exact(X):
    return np.sin(X @ OMEGA + OMEGA0)


def factor3(n):
    """px*py*pz = n with the smallest surface (the most cubical bricks)."""
    best = None
    for pz in range(1, n + 1):
        if n % pz:
            continue
        for py in range(1, n // pz + 1):
            if (n // pz) % py:
                continue
            px = n // pz // py
            surf = px * py + py * pz + pz * px
            if best is None or surf < best[0]:
                best = (surf, px, py, pz)
    return best[1:]


def kuhn_box(nn, lo, n):
    """Nodes and Kuhn tetrahedra of the node box [lo, lo+n) (per axis) of the nn^3 grid, local ids i + ni (j + nj k)."""
    ni, nj, nk = n
    g = np.linspace(0.0, 1.0, nn)
    K, J, I = np.meshgrid(np.arange(nk), np.arange(nj), np.arange(ni), indexing="ij")
    ii, jj, kk = I.ravel() + lo[0], J.ravel() + lo[1], K.ravel() + lo[2]
    X = np.stack([g[ii], g[jj], g[kk]], axis=1)
    ck, cj, ci = np.meshgrid(np.arange(nk - 1), np.arange(nj - 1), np.arange(ni - 1), indexing="ij")
    base = (ci + ni * (cj + nj * ck)).ravel()
    off = {0: 1, 1: ni, 2: ni * nj}
    perms = [(0, 1, 2), (0, 2, 1), (1, 0, 2), (1, 2, 0), (2, 0, 1), (2, 1, 0)]
    conn = np.empty((base.size, 6, 4), dtype=np.int32)
    for p, (a, b, c) in enumerate(perms):
        conn[:, p, 0] = base
        conn[:, p, 1] = base + off[a]
        conn[:, p, 2] = base + off[a] + off[b]
        conn[:, p, 3] = base + off[a] + off[b] + off[c]
    return X, conn.reshape(-1, 4), (ii, jj, kk)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("nn", type=int)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--json")
    a = ap.parse_args()
    nn = a.nn
    p = factor3(size)
    edges = [np.linspace(0, nn, q + 1).round().astype(int) for q in p]       # owned node ranges per axis
    ra = (rank % p[0], (rank // p[0]) % p[1], rank // (p[0] * p[1]))
    own_lo = [edges[d][ra[d]] for d in range(3)]
    own_hi = [edges[d][ra[d] + 1] for d in range(3)]
    own_n = [own_hi[d] - own_lo[d] for d in range(3)]
    # owned nodes per brick, and the global row offset of every brick
    cnt = np.empty(size, dtype=np.int64)
    for r in range(size):
        c = (r % p[0], (r // p[0]) % p[1], r // (p[0] * p[1]))
        cnt[r] = np.prod([edges[d][c[d] + 1] - edges[d][c[d]] for d in range(3)])
    offset = np.concatenate([[0], np.cumsum(cnt)])
    N = int(offset[-1])
    nOwned = int(cnt[rank])

    comm.Barrier()
    t0 = time.perf_counter()
    # local brick: the cells whose lowest node is owned, so nodes up to and including one ghost layer on the high side
    loc_hi = [min(own_hi[d], nn - 1) for d in range(3)]
    loc_n = [loc_hi[d] - own_lo[d] + 1 for d in range(3)]
    X, conn, (ii, jj, kk) = kuhn_box(nn, own_lo, loc_n)
    owner = []
    for d, idx in enumerate((ii, jj, kk)):
        owner.append(np.searchsorted(edges[d], idx, side="right") - 1)
    ownerRank = owner[0] + p[0] * (owner[1] + p[1] * owner[2])
    # the owner's local index of each node: its position in the owner's owned box
    sizes = [np.diff(e) for e in edges]
    lo_o = [edges[d][owner[d]] for d in range(3)]
    ni_o, nj_o = sizes[0][owner[0]], sizes[1][owner[1]]
    gid = (offset[ownerRank] + (ii - lo_o[0]) + ni_o * ((jj - lo_o[1]) + nj_o * (kk - lo_o[2]))).astype(PETSc.IntType)
    bmask_loc = ((X < 1e-12) | (X > 1 - 1e-12)).any(axis=1)
    U0 = np.where(bmask_loc, exact(X), 0.0)
    tLocal = time.perf_counter() - t0

    qp, qw = adrh_kokkos.quad_tet4()
    asm = adrh_kokkos.AdSineAssembler3D(X, conn, qp, qw)
    res = asm.assemble(B, KAPPA, OMEGA, OMEGA0, U0, a.repeats)
    indptr, indices = asm.indptr, asm.indices
    rowOf = np.repeat(np.arange(X.shape[0]), np.diff(indptr))
    tSetup, tAsm = asm.t_setup, res["t_assemble"]
    nE = conn.shape[0]; nColors = asm.n_colors
    asm.close()

    comm.Barrier()
    t0 = time.perf_counter()
    A = PETSc.Mat().create(PETSc.COMM_WORLD)
    A.setSizes(((nOwned, N), (nOwned, N)))
    A.setType("aij")
    A.setPreallocationCOO(gid[rowOf], gid[indices])
    A.setValuesCOO(res["values"], PETSc.InsertMode.ADD_VALUES)
    rhs = PETSc.Vec().create(PETSc.COMM_WORLD)
    rhs.setSizes((nOwned, N)); rhs.setType("mpi")
    rhs.setValues(gid, -res["R"], PETSc.InsertMode.ADD_VALUES)       # ghost entries are added into their owners at assemble()
    rhs.assemble()
    tCoo = time.perf_counter() - t0

    comm.Barrier()
    t0 = time.perf_counter()
    # owned boundary nodes: their global rows become identity rows, with a zero right-hand side
    own = (ownerRank == rank)
    brows = gid[own & bmask_loc]
    A.zeroRows(brows, diag=1.0)
    rhs.setValues(brows, np.zeros(brows.size))
    rhs.assemble()
    tBC = time.perf_counter() - t0

    ksp = PETSc.KSP().create(PETSc.COMM_WORLD)
    ksp.setOperators(A)
    ksp.setType("gmres"); ksp.setTolerances(rtol=1e-10, max_it=500)
    ksp.getPC().setType("gamg")
    comm.Barrier()
    t0 = time.perf_counter()
    ksp.setUp()
    tKspSetup = time.perf_counter() - t0
    x = A.createVecRight()
    comm.Barrier()
    t0 = time.perf_counter()
    ksp.solve(rhs, x)
    tSolve = time.perf_counter() - t0
    its = ksp.getIterationNumber()

    # error against the exact solution on the owned nodes
    xl = x.getArray(readonly=True)
    delta = np.zeros(X.shape[0]); delta[own] = 0.0
    ownedIdx = np.nonzero(own)[0]
    order = np.argsort(gid[ownedIdx])                                  # owned rows in PETSc's local order
    err = U0[ownedIdx][order] + xl - exact(X[ownedIdx][order])
    h = 1.0 / (nn - 1)
    l2 = float(np.sqrt(comm.allreduce(h ** 3 * np.sum(err ** 2), MPI.SUM)))
    mx = float(comm.allreduce(np.abs(err).max(), MPI.MAX))

    vals = np.array([tLocal, tSetup, tAsm, tCoo, tBC, tKspSetup, tSolve, resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0])
    vmax = np.zeros_like(vals); comm.Allreduce(vals, vmax, op=MPI.MAX)
    rss = comm.allreduce(vals[7], MPI.SUM)
    nEtot = comm.allreduce(nE, MPI.SUM); nElMax = comm.allreduce(nE, MPI.MAX)
    ghost = comm.allreduce(int(X.shape[0] - nOwned), MPI.SUM)
    if rank == 0:
        info = adrh_kokkos.kokkos_info()
        rec = dict(nn=nn, ranks=size, threads=info["concurrency"], bricks=list(p), tets=int(nEtot), nodes=N, colors=int(nColors),
                   local=vmax[0], setup=vmax[1], assemble=vmax[2], coo=vmax[3], bc=vmax[4], ksp_setup=vmax[5], ksp_solve=vmax[6],
                   iterations=int(its), l2_error=l2, max_error=mx, rss_total_mb=float(rss), rss_max_mb=float(vmax[7]),
                   ghost_nodes=int(ghost), elem_imbalance=float(nElMax * size / nEtot))
        rec["total"] = rec["assemble"] + rec["coo"] + rec["bc"] + rec["ksp_setup"] + rec["ksp_solve"]
        print("R=%d x T=%d  nn=%d  bricks %s  tets %d  nodes %d  ghost rows %d  imbalance %.2f" % (size, rec["threads"], nn, p, nEtot, N, ghost, rec["elem_imbalance"]))
        print("   assemble %.4f  coo %.4f  bc %.4f  ksp setup %.4f  solve %.4f (%d its)  | total %.3f s   [local mesh %.3f, kokkos setup %.3f]" % (
            rec["assemble"], rec["coo"], rec["bc"], rec["ksp_setup"], rec["ksp_solve"], its, rec["total"], rec["local"], rec["setup"]))
        print("   max error %.3e  L2 %.3e   rss total %.0f MB (max rank %.0f MB)" % (mx, l2, rss, vmax[7]))
        if a.json:
            open(a.json, "a").write(json.dumps(rec) + "\n")
    ksp.destroy(); A.destroy(); rhs.destroy(); x.destroy()


if __name__ == "__main__":
    main()
