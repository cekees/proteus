r"""
L2-only spatial convergence table for the Tracy transient Richards problem.

Same quadrature and exact solution as analyze_convergence_fast.py, but the table
carries only  h, L2(psi) and the observed rate p.

Env vars:  REF_MAX=ref_5   OUT_TXT=...   CHUNK=400000
"""
import os
import xml.etree.ElementTree as ET
from math import pi, sqrt

import numpy as np
import quadpy
import h5py

# ----------------------------------------------------------------------
Lx, Ly = 10.0, 10.0

TRACY = dict(a=10.0, L=10.0, alpha=0.164, psi_r=-15.24,
             theta_s=0.301, theta_r=0.093, Ks_day=2.04, n_terms=200)

ALL_REFS = [("ref_2", 41), ("ref_3", 81), ("ref_4", 161),
            ("ref_5", 321), ("ref_6", 641)]
_names   = [r[0] for r in ALL_REFS]
_ref_min = os.environ.get("REF_MIN", "ref_2")
_ref_max = os.environ.get("REF_MAX", "ref_6")
ref_info = ALL_REFS[_names.index(_ref_min): _names.index(_ref_max) + 1]
# only the refinements that were actually run
ref_info = [(r, n) for (r, n) in ref_info
            if os.path.exists(f"{r}/re_vgm_sand_10x10m_2d.h5")]

# 300 -> 3.75e-4 d, 350 -> 4.375e-4 d, 400 -> 5.0e-4 d
T_INDICES = [int(s) for s in os.environ.get("T_INDICES", "300,350,400").split(",")]
OUT_TXT   = os.environ.get("OUT_TXT", "Tracy_L2_convergence.txt")
CHUNK     = int(os.environ.get("CHUNK", "400000"))
DEGREE    = 5


# ----------------------------------------------------------------------
def get_time_days_from_h5(f, t_idx):
    """Physical time (days) from the Mesh_Spatial_Domain_<t_idx> XML in the h5."""
    mesh_key = f"Mesh_Spatial_Domain_{t_idx}"
    if mesh_key not in f:
        raise KeyError(f"Missing dataset '{mesh_key}' in H5 file.")
    root = ET.fromstring(f[mesh_key][:])
    return float(root.find("Time").attrib["Value"])


def _tracy_precompute_params(t_days, a, L, alpha, psi_r,
                             theta_s, theta_r, Ks_day, n_terms):
    h0 = 1.0 - np.exp(alpha * psi_r)        # \bar h_0
    Ks = Ks_day / 86400.0                   # m/s
    c  = alpha * (theta_s - theta_r) / Ks   # 1/s
    t  = t_days * 86400.0                   # s
    k_idx  = np.arange(1, n_terms + 1, dtype=float)
    lam    = k_idx * pi / L
    gamma1 = (lam**2 + (alpha**2) / 4.0) / c
    gamma2 = ((2.0 * pi / a)**2 + lam**2 + (alpha**2) / 4.0) / c
    sign = (-1.0) ** k_idx
    A = sign * lam * (1.0 / gamma1) * np.exp(-gamma1 * t)
    B = sign * lam * (1.0 / gamma2) * np.exp(-gamma2 * t)
    root_ss = sqrt((alpha / 2.0)**2 + (2.0 * pi / a)**2)
    return dict(h0=h0, Ks=Ks, c=c, t=t, lam=lam, A=A, B=B, root_ss=root_ss,
                alpha=alpha, psi_r=psi_r, a=a, L=L)


# ----------------------------------------------------------------------
def tracy_exact_chunk(x, z, params, geom=None):
    """
    Vectorised Tracy exact psi at points (x,z).  If `geom` (a dict with the
    time-independent pieces for THIS chunk) is passed it is reused; otherwise it
    is built and returned so the caller can reuse it across output times.
    Returns (psi_exact, geom).
    """
    h0, c, lam = params["h0"], params["c"], params["lam"]
    A, B, root = params["A"], params["B"], params["root_ss"]
    alpha, psi_r, a, L = (params["alpha"], params["psi_r"],
                          params["a"], params["L"])

    if geom is None:
        top = np.abs(z - L) < 1e-12
        cos_x = np.cos(2.0 * np.pi * x / a)
        exp_factor = np.exp(0.5 * alpha * (L - z))
        h_bar_ss = (0.5 * h0 * exp_factor *
                    (np.sinh(0.5 * alpha * z) / np.sinh(0.5 * alpha * L)
                     - cos_x * np.sinh(root * z) / np.sinh(root * L)))
        sin_mat = np.sin(lam[:, None] * z[None, :])   # (K, n)
        geom = dict(top=top, cos_x=cos_x, exp_factor=exp_factor,
                    h_bar_ss=h_bar_ss, sin_mat=sin_mat)

    top      = geom["top"]
    cos_x    = geom["cos_x"]
    exp_f    = geom["exp_factor"]
    h_bar_ss = geom["h_bar_ss"]
    sin_mat  = geom["sin_mat"]

    SA = A @ sin_mat
    SB = B @ sin_mat
    s  = SA - cos_x * SB
    phi_bar = (h0 / (L * c)) * exp_f * s
    h_bar = h_bar_ss + phi_bar
    psi = (1.0 / alpha) * np.log(np.exp(alpha * psi_r) + h_bar)

    if np.any(top):
        psi[top] = (1.0 / alpha) * np.log(
            np.exp(alpha * psi_r)
            + 0.5 * h0 * (1.0 - np.cos(2.0 * np.pi * x[top] / a)))
    return psi, geom


# ----------------------------------------------------------------------
def mesh_geometry(nodes, elements, scheme):
    """Time-independent quadrature geometry for one mesh."""
    bary = np.asarray(scheme.points)          # (3, nq) barycentric
    w    = np.asarray(scheme.weights)         # (nq,)
    coords = nodes[elements][:, :, :2]        # (Nel,3,2)

    v1 = coords[:, 1] - coords[:, 0]
    v2 = coords[:, 2] - coords[:, 0]
    detJ = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    area = 0.5 * np.abs(detJ)                 # (Nel,)

    X = np.einsum("bq,ebd->eqd", bary, coords)       # (Nel,nq,2)
    Xflat = X.reshape(-1, 2)
    cflat = (area[:, None] * w[None, :]).reshape(-1)  # integration coeff / point
    return Xflat[:, 0].copy(), Xflat[:, 1].copy(), cflat, bary


# ----------------------------------------------------------------------
def analyse_mesh(ref_name, nnx, params_by_t, scheme):
    h5_path = f"{ref_name}/re_vgm_sand_10x10m_2d.h5"
    with h5py.File(h5_path, "r") as f:
        elements = f["elementsSpatial_Domain0"][:]
        nodes    = f["nodesSpatial_Domain0"][:]
        psi_by_t = {t: f[f"pressure_head_t{t}"][:] for t in params_by_t}
    if elements.min() == 1:
        elements = elements - 1

    qx, qz, cflat, bary = mesh_geometry(nodes, elements, scheme)
    uloc = {t: psi_by_t[t][elements] for t in params_by_t}          # (Nel,3)
    uq   = {t: np.einsum("bq,eb->eq", bary, uloc[t]).reshape(-1)
            for t in params_by_t}                                    # (Nel*nq,)

    Np = qx.shape[0]
    L2sq = {t: 0.0 for t in params_by_t}
    for s0 in range(0, Np, CHUNK):
        s1 = min(s0 + CHUNK, Np)
        xs, zs, cs = qx[s0:s1], qz[s0:s1], cflat[s0:s1]
        geom = None
        for t, params in params_by_t.items():
            ex, geom = tracy_exact_chunk(xs, zs, params, geom)
            diff = uq[t][s0:s1] - ex
            L2sq[t] += np.dot(cs, diff * diff)

    ndof = nodes.shape[0]
    h    = Lx / (nnx - 1)
    return {t: dict(ref=ref_name, nnx=nnx, ndof=ndof, h=h,
                    L2=np.sqrt(L2sq[t])) for t in params_by_t}


# ----------------------------------------------------------------------
def add_rates(rows):
    for k, r in enumerate(rows):
        if k == 0:
            r["pL2"] = None
            continue
        lr = np.log(rows[k-1]["h"] / rows[k]["h"])
        r["pL2"] = np.log(rows[k-1]["L2"] / rows[k]["L2"]) / lr


def format_table(t_idx, t_days, rows):
    lines = ["=" * 48,
             "Tracy transient Richards — FCT (STAB=2, FCT=True)",
             f"L2 error at output index t_idx={t_idx}, physical t={t_days:.6e} days"]
    hdr = f"{'ref':<6}{'nnx':>5}{'ndof':>9}{'h':>9}{'L2(psi)':>13}{'p':>6}"
    lines += [hdr, "-" * len(hdr)]
    fp = lambda p: "   -  " if p is None else f"{p:6.2f}"
    for r in rows:
        lines.append(f"{r['ref']:<6}{r['nnx']:>5}{r['ndof']:>9}{r['h']:>9.4f}"
                     f"{r['L2']:>13.4e}{fp(r['pL2'])}")
    return "\n".join(lines) + "\n"


def main():
    scheme = quadpy.t2.get_good_scheme(DEGREE)
    with h5py.File(f"{ref_info[0][0]}/re_vgm_sand_10x10m_2d.h5", "r") as f0:
        t_days = {t: get_time_days_from_h5(f0, t) for t in T_INDICES}
    params_by_t = {t: _tracy_precompute_params(t_days[t], **TRACY)
                   for t in T_INDICES}

    per_t_rows = {t: [] for t in T_INDICES}
    for ref_name, nnx in ref_info:
        res = analyse_mesh(ref_name, nnx, params_by_t, scheme)
        for t in T_INDICES:
            per_t_rows[t].append(res[t])
        print(f"done {ref_name}: "
              + "  ".join(f"t{t}: L2={res[t]['L2']:.4e}" for t in T_INDICES),
              flush=True)

    blocks = []
    for t in T_INDICES:
        add_rates(per_t_rows[t])
        blocks.append(format_table(t, t_days[t], per_t_rows[t]))
    text = "\n".join(blocks)
    print("\n" + text)
    with open(OUT_TXT, "w") as fout:
        fout.write(text)
    print(f"Wrote {OUT_TXT}")


if __name__ == "__main__":
    main()
