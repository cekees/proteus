import os
import numpy as np
import h5py
import xml.etree.ElementTree as ET


# -----------------------------
# Analytical solution settings (smooth Gaussian pulse)
# -----------------------------
C0 = 1.0
v = 0.3
D = 0.007
x0 = 0.5
sigma0 = 0.08

# H5 files from coarse to fine (edit as needed)
REF_FILES = [
    #"tadr_level_0.h5",
    #"tadr_level_1.h5",
    #"tadr_level_2.h5",
    "tadr_level_3.h5",
    "tadr_level_4.h5",
    "tadr_level_5.h5",
    "tadr_level_6.h5",
    "tadr_level_7.h5",
    "tadr_level_8.h5",
    

    
    
]

# Output times by solution index (u_t<idx>)
TIME_INDICES = [10, 20, 50, 100, 150, 199]  # edit as needed based on available u_t<idx> in H5

OUT_TXT = "Error_rate_TADR.txt"
QUAD_ORDER = 4


def c_gaussian_ic(x, t, c0=C0, vel=v, diff=D, center=x0, sigma=sigma0):
    """
    1D ADE analytical solution for Gaussian IC on R:
      C(x,0)=C0*exp(-(x-x0)^2/(2*sigma0^2)).
    """
    x = np.asarray(x)
    var_t = sigma**2 + 2.0*diff*t
    amp = c0*np.sqrt((sigma**2)/var_t)
    dx = x - (center + vel*t)
    return amp*np.exp(-0.5*(dx*dx)/var_t)


def get_time_from_h5(f, t_idx):
    """
    Read physical time from Mesh_Spatial_Domain_<t_idx> XML in HDF5.
    Falls back to None if not found.
    """
    mesh_key = f"Mesh_Spatial_Domain_{t_idx}"
    if mesh_key not in f:
        return None
    root = ET.fromstring(f[mesh_key][:])
    node = root.find("Time")
    if node is None:
        return None
    return float(node.attrib["Value"])


def linear_interp_segment(xq, x0, x1, u0, u1):
    if abs(x1 - x0) < 1.0e-30:
        return np.full_like(xq, u0)
    xi = (xq - x0) / (x1 - x0)
    return (1.0 - xi) * u0 + xi * u1


def compute_errors_1d(nodes, elements, u_num, t_val, quad_order=4):
    """
    Compute L2 and Linf errors for 1D P1 solution on line segments.
    """
    x_nodes = nodes[:, 0]

    # Ensure 0-based connectivity
    if elements.min() == 1:
        elements = elements - 1

    qx_ref, qw_ref = np.polynomial.legendre.leggauss(quad_order)  # [-1,1]

    l1 = 0.0
    l2_sq = 0.0
    linf = 0.0

    for e in elements:
        i0, i1 = int(e[0]), int(e[1])
        x0, x1 = x_nodes[i0], x_nodes[i1]
        u0, u1 = u_num[i0], u_num[i1]

        # vertex Linf
        u_ex_v = c_gaussian_ic(np.array([x0, x1]), t_val)
        linf = max(linf, float(np.max(np.abs(np.array([u0, u1]) - u_ex_v))))

        # map quadrature points to [x0,x1]
        xq = 0.5 * (x1 - x0) * qx_ref + 0.5 * (x1 + x0)
        jac = 0.5 * abs(x1 - x0)

        u_num_q = linear_interp_segment(xq, x0, x1, u0, u1)
        u_ex_q = c_gaussian_ic(xq, t_val)
        diff = u_num_q - u_ex_q

        l1 += np.sum(qw_ref * np.abs(diff)) * jac
        l2_sq += np.sum(qw_ref * diff * diff) * jac
        linf = max(linf, float(np.max(np.abs(diff))))

    return l1, np.sqrt(max(l2_sq, 0.0)), linf


def infer_h_from_nodes(nodes):
    x = np.unique(np.round(nodes[:, 0], 14))
    if x.size < 2:
        return np.nan, x.size
    return float(np.min(np.diff(x))), int(x.size)


def available_u_indices(f):
    idx = []
    for k in f.keys():
        if k.startswith("u_t"):
            try:
                idx.append(int(k[3:]))
            except ValueError:
                pass
    return sorted(idx)


def main():
    existing = [p for p in REF_FILES if os.path.exists(p)]
    if not existing:
        raise FileNotFoundError(
            f"No H5 files found from REF_FILES={REF_FILES}. Run from Step_analytical_solution or edit REF_FILES."
        )

    with open(OUT_TXT, "w") as fout:
        fout.write("1D TADR vs ADE analytical Gaussian solution\n")
        fout.write(f"C0={C0}, v={v}, D={D}, x0={x0}, sigma0={sigma0}\n")

        # determine valid time indices from first available file
        with h5py.File(existing[0], "r") as f0:
            u_idx_all = set(available_u_indices(f0))
        eval_indices = [i for i in TIME_INDICES if i in u_idx_all]
        if not eval_indices:
            raise ValueError(f"None of TIME_INDICES={TIME_INDICES} found in {existing[0]}")

        for t_idx in eval_indices:
            with h5py.File(existing[0], "r") as f0:
                t_val = get_time_from_h5(f0, t_idx)
            if t_val is None:
                raise KeyError(f"Missing Mesh_Spatial_Domain_{t_idx} time metadata in {existing[0]}")

            fout.write("\n" + "=" * 72 + "\n")
            fout.write(f"Errors at t = {t_val:.8f} (index={t_idx})\n")
            fout.write("=" * 72 + "\n")

            rows = []
            for h5_path in existing:
                with h5py.File(h5_path, "r") as f:
                    if f"u_t{t_idx}" not in f:
                        continue
                    nodes = f["nodesSpatial_Domain0"][:]
                    elements = f["elementsSpatial_Domain0"][:]
                    u_num = f[f"u_t{t_idx}"][:]

                h, nnx = infer_h_from_nodes(nodes)
                l1, l2, linf = compute_errors_1d(nodes, elements, u_num, t_val, QUAD_ORDER)
                rows.append((h5_path, nnx, h, l1, l2, linf))

            rows.sort(key=lambda r: r[2], reverse=True)  # coarse -> fine by h

            p_l1 = [None]
            p_l2 = [None]
            p_linf = [None]
            for k in range(1, len(rows)):
                h_c, h_f = rows[k - 1][2], rows[k][2]
                e_c_l1, e_f_l1 = rows[k - 1][3], rows[k][3]
                e_c_l2, e_f_l2 = rows[k - 1][4], rows[k][4]
                e_c_li, e_f_li = rows[k - 1][5], rows[k][5]

                rate_l1 = np.log(e_c_l1 / e_f_l1) / np.log(h_c / h_f) if e_f_l1 > 0 else None
                rate_l2 = np.log(e_c_l2 / e_f_l2) / np.log(h_c / h_f) if e_f_l2 > 0 else None
                rate_li = np.log(e_c_li / e_f_li) / np.log(h_c / h_f) if e_f_li > 0 else None
                p_l1.append(rate_l1)
                p_l2.append(rate_l2)
                p_linf.append(rate_li)

            fout.write(f"{'file':<22} {'nnx':>6} {'h':>12} {'L1':>14} {'L2':>14} {'Linf':>14} {'p_L1':>10} {'p_L2':>10} {'p_Linf':>10}\n")
            fout.write("-" * 122 + "\n")
            for (h5_path, nnx, h, l1, l2, linf), r1, r2, rinf in zip(rows, p_l1, p_l2, p_linf):
                r1s = "-" if r1 is None else f"{r1:.3f}"
                r2s = "-" if r2 is None else f"{r2:.3f}"
                ris = "-" if rinf is None else f"{rinf:.3f}"
                fout.write(
                    f"{os.path.basename(h5_path):<22} {nnx:6d} {h:12.6e} {l1:14.6e} {l2:14.6e} {linf:14.6e} {r1s:>10} {r2s:>10} {ris:>10}\n"
                )

    print(f"Wrote error table: {OUT_TXT}")


if __name__ == "__main__":
    main()
