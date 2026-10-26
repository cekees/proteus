"""Stage-1 device assembly, measured end to end: assemble on the Kokkos execution space, solve on the host.

The 3-D steady advection-diffusion problem of ad_regression.py (manufactured solution sin(omega.x + omega0), strong
Dirichlet data on the whole boundary) on the unit cube split into Kuhn tetrahedra. The Kokkos module assembles the
global residual and the CSR Jacobian with the colored kernel (kokkos_kernel.h); the driver imposes the Dirichlet rows
and solves with PETSc on the host. Each phase is timed on its own, so that what a device buys (assembly) can be
set against what it costs (the copies) and what it leaves alone (the solve):

  setup      once: coloring, CSR pattern, upload of coordinates and quadrature
  h2d        the nodal vector to the execution space
  assemble   residual and Jacobian, best of --repeats
  d2h        residual and Jacobian back to the host
  bc         Dirichlet rows on the host
  petsc      the matrix into PETSc, then KSP setup and solve

usage: [KOKKOS_NUM_THREADS=n] python ad_solve_device.py [--check] [--repeats n] [--pc gamg|ilu|lu] nn [nn ...]
  --check   at small nn, compare the device-assembled system with the host engine's (adrh_proto + numpy scatter)
"""
import argparse
import json
import sys
import time

import numpy as np
import scipy.sparse as sp

import adrh_kokkos

B = np.array([1.0, 0.5, 0.25]); KAPPA = 0.2; OMEGA = np.array([3.0, 2.0, 4.0]); OMEGA0 = 0.3


def exact(X):
    return np.sin(X @ OMEGA + OMEGA0)


def cube_tets(nn):
    """Nodes (nn^3, 3) and Kuhn tetrahedra (6 (nn-1)^3, 4) of the unit cube; node id = i + nn (j + nn k)."""
    g = np.linspace(0.0, 1.0, nn)
    K, J, I = np.meshgrid(np.arange(nn), np.arange(nn), np.arange(nn), indexing="ij")
    X = np.stack([g[I.ravel()], g[J.ravel()], g[K.ravel()]], axis=1)
    ne = nn - 1
    ck, cj, ci = np.meshgrid(np.arange(ne), np.arange(ne), np.arange(ne), indexing="ij")
    base = (ci + nn * (cj + nn * ck)).ravel()
    off = {0: 1, 1: nn, 2: nn * nn}
    perms = [(0, 1, 2), (0, 2, 1), (1, 0, 2), (1, 2, 0), (2, 0, 1), (2, 1, 0)]
    conn = np.empty((base.size, 6, 4), dtype=np.int32)
    for p, (a, b, c) in enumerate(perms):
        conn[:, p, 0] = base
        conn[:, p, 1] = base + off[a]
        conn[:, p, 2] = base + off[a] + off[b]
        conn[:, p, 3] = base + off[a] + off[b] + off[c]
    return X, conn.reshape(-1, 4)


def boundary_mask(X):
    return ((X < 1e-12) | (X > 1 - 1e-12)).any(axis=1)


def host_system(X, conn, qp, qw, U):
    """The same system from the host engine's element kernel and a numpy scatter (the reference for --check)."""
    import adrh_proto
    nE = conn.shape[0]
    xn = X[conn]
    u = U[conn].reshape(nE, 1, 4)
    R, J = adrh_proto.advection_diffusion_sine(3, "p1", B, KAPPA, OMEGA, OMEGA0, xn, u, qp, qw)
    n = X.shape[0]
    rows = np.repeat(conn, 4, axis=1).ravel(); cols = np.tile(conn, (1, 4)).ravel()
    Jm = sp.coo_matrix((J.reshape(nE, 4, 4).ravel(), (rows, cols)), shape=(n, n)).tocsr()
    Rg = np.zeros(n); np.add.at(Rg, conn.ravel(), R.reshape(nE, 4).ravel())
    Jm.sort_indices()
    return Rg, Jm


def solve_petsc(indptr, indices, values, rhs, pc, rtol):
    from petsc4py import PETSc
    t0 = time.perf_counter()
    n = rhs.size
    A = PETSc.Mat().createAIJ(size=(n, n), csr=(indptr.astype(PETSc.IntType), indices.astype(PETSc.IntType), values))
    A.assemble()
    bvec = A.createVecLeft(); bvec.array[:] = rhs
    x = A.createVecRight()
    tCreate = time.perf_counter() - t0
    ksp = PETSc.KSP().create(PETSc.COMM_SELF)
    ksp.setOperators(A)
    if pc == "lu":
        ksp.setType("preonly"); ksp.getPC().setType("lu")
    else:
        ksp.setType("gmres"); ksp.setTolerances(rtol=rtol, max_it=500)
        ksp.getPC().setType("gamg" if pc == "gamg" else "ilu")
    t0 = time.perf_counter()
    ksp.setUp()
    tSetup = time.perf_counter() - t0
    t0 = time.perf_counter()
    ksp.solve(bvec, x)
    tSolve = time.perf_counter() - t0
    out = x.array.copy(), ksp.getIterationNumber(), tCreate, tSetup, tSolve
    ksp.destroy(); A.destroy(); bvec.destroy(); x.destroy()
    return out


def run(nn, repeats, pc, check):
    t0 = time.perf_counter()
    X, conn = cube_tets(nn)
    tMesh = time.perf_counter() - t0
    qp, qw = adrh_kokkos.quad_tet4()
    asm = adrh_kokkos.AdSineAssembler3D(X, conn, qp, qw)
    try:
        bmask = boundary_mask(X)
        U0 = np.where(bmask, exact(X), 0.0)
        res = asm.assemble(B, KAPPA, OMEGA, OMEGA0, U0, repeats)
        R, vals = res["R"], res["values"]
        indptr, indices = asm.indptr, asm.indices
        if check:
            Rh, Jh = host_system(X, conn, qp, qw, U0)
            assert np.array_equal(indptr, Jh.indptr) and np.array_equal(indices, Jh.indices), "pattern differs"
            print("   check: device system vs host engine + scatter: residual %.1e  Jacobian %.1e (relative max)" % (
                np.abs(R - Rh).max() / np.abs(Rh).max(), np.abs(vals - Jh.data).max() / np.abs(Jh.data).max()))
        # Dirichlet rows: identity, zero update on the boundary
        t0 = time.perf_counter()
        rowOf = np.repeat(np.arange(X.shape[0]), np.diff(indptr))
        bm = bmask[rowOf]
        v = vals.copy(); v[bm] = 0.0; v[bm & (indices == rowOf)] = 1.0
        rhs = -R; rhs[bmask] = 0.0
        tBC = time.perf_counter() - t0
        delta, its, tCreate, tKspSetup, tSolve = solve_petsc(indptr, indices, v, rhs, pc, 1e-10)
        u = U0 + delta
        err = u - exact(X)
        h = 1.0 / (nn - 1)
        rec = dict(nn=nn, nodes=int(X.shape[0]), tets=int(conn.shape[0]), nnz=int(indices.size), colors=asm.n_colors,
                   space=adrh_kokkos.kokkos_info()["space"], threads=adrh_kokkos.kokkos_info()["concurrency"], pc=pc,
                   mesh=tMesh, setup=asm.t_setup, h2d=res["t_h2d"], assemble=res["t_assemble"], d2h=res["t_d2h"], bc=tBC,
                   petsc_create=tCreate, ksp_setup=tKspSetup, ksp_solve=tSolve, iterations=its,
                   l2_error=float(np.sqrt(h ** 3 * np.sum(err ** 2))), max_error=float(np.abs(err).max()))
    finally:
        asm.close()
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("nn", type=int, nargs="+")
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--pc", default="gamg", choices=["gamg", "ilu", "lu"])
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--json", help="append one JSON line per size to this file")
    a = ap.parse_args()
    for nn in a.nn:
        r = run(nn, a.repeats, a.pc, a.check)
        host = r["ksp_setup"] + r["ksp_solve"] + r["petsc_create"] + r["bc"]
        dev = r["h2d"] + r["assemble"] + r["d2h"]
        print("nn=%d  %s x%d  tets %d  nodes %d  nnz %d  colors %d" % (r["nn"], r["space"], r["threads"], r["tets"], r["nodes"], r["nnz"], r["colors"]))
        print("   assemble %.4f s (%.1f ns/tet)   h2d %.4f   d2h %.4f   | bc %.4f   petsc create %.4f   ksp setup %.4f   solve %.4f s (%d its, %s)"
              % (r["assemble"], 1e9 * r["assemble"] / r["tets"], r["h2d"], r["d2h"], r["bc"], r["petsc_create"], r["ksp_setup"], r["ksp_solve"], r["iterations"], r["pc"]))
        print("   assembly share of (assemble + host side) %.0f%%;  copies are %.0f%% of the device-side time;  L2 error %.3e  max %.3e"
              % (100 * r["assemble"] / (r["assemble"] + host), 100 * (r["h2d"] + r["d2h"]) / dev, r["l2_error"], r["max_error"]))
        if a.json:
            open(a.json, "a").write(json.dumps(r) + "\n")


if __name__ == "__main__":
    main()
