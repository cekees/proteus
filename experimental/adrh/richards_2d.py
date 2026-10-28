"""The 2D Richards deck: proteus's test/richards/re_vgm_sand_10x10m_2d (a 10 m x 10 m block of sand, water ponded on the middle third of the top
for 0.35 d), solved with the K-upwinded low-order scheme of richards_nd.py, optionally with FCT and with a lens of a finer soil.

The deck: Mualem-van Genuchten sand (alpha 5.47, n 4.264, theta_r 0.093, theta_s 0.301, K_s 5.04 m/d), gravity down, hydrostatic initial head psi = -y (water table
at the bottom), Dirichlet psi = 0.1 m under the pond (the middle third of the top), 0 on the bottom, 0.1 - y on the two sides, no flow elsewhere on the top. Strong
Dirichlet rows, as the K-upwinded scheme requires. There is no saved reference for this deck; what is checked is that the head stays inside the data's bounds
[-10, 0.1] (homogeneous sand; a lens perches water, so the head rises above the ponding there). --lens puts a 4 m x 2 m lens of a finer soil (alpha 1.0, n 1.6, theta_r 0.08, theta_s 0.43,
K_s a tenth of the sand's) in the path of the wetting front; --interface picks the interface rule of richards_nd.py.

usage: [ADRH_MODULE=adrh_kokkos] python richards_2d.py [--n 21] [--steps 35] [--fct] [--lens] [--interface node|element]
"""
import argparse
import time

import numpy as np

import richards_nd as nd
from richards_regression import dimensionless_ks

L, PONDING, T_END = 10.0, 0.1, 0.35


def build(n, lens, interface):
    x, conn = nd.rectangle_triangles(n, n, L, L)
    X, Y = x[:, 0], x[:, 1]
    top = (Y == L) & (X >= L / 3) & (X <= 2 * L / 3); bottom = (Y == 0.0); side = (X == 0.0) | (X == L)
    gD = np.where(top, PONDING, np.where(bottom, 0.0, np.where(side, PONDING - Y, 0.0)))
    fixed = np.where(top | bottom | side)[0]
    cx, cy = x[conn, 0].mean(axis=1), x[conn, 1].mean(axis=1)
    material = ((np.abs(cx - L / 2) < 2.0) & (np.abs(cy - 6.0) < 1.0)).astype('i') if lens else np.zeros(conn.shape[0], dtype='i')
    ks = dimensionless_ks(5.04)
    model = nd.KUpwindRichards(x, conn, material, alpha=[5.47, 1.0], n_vg=[4.264, 1.6], theta_r=[0.093, 0.08], theta_s=[0.301, 0.43],
                               ks=[[ks, ks], [0.1 * ks, 0.1 * ks]], g=[0.0, -1.0], dirichlet=(fixed, gD[fixed]), interface=interface)
    u0 = np.where(top, PONDING, -Y)
    return model, u0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n', type=int, default=21, help='nodes per side')
    ap.add_argument('--steps', type=int, default=35)
    ap.add_argument('--fct', action='store_true')
    ap.add_argument('--lens', action='store_true')
    ap.add_argument('--interface', default='element', choices=['node', 'element'])
    a = ap.parse_args()
    model, u0 = build(a.n, a.lens, a.interface)
    t0 = time.perf_counter()
    u, its = model.run(u0, T_END, a.steps, fct=a.fct, tol=1e-8, norm=2)
    wall = time.perf_counter() - t0
    print("%s: %d x %d nodes (%d triangles), %d steps of %.4f d%s%s, interface=%r: %d Newton iterations (%.1f per step), %.1f s with %s" %
          ("sand 10 x 10 m, 2D deck", a.n, a.n, model.nE, a.steps, T_END / a.steps, ", FCT" if a.fct else "", ", lens" if a.lens else "", a.interface,
           its, its / a.steps, wall, nd.engine.__name__))
    rise = u - u0
    print("  psi in [%.4f, %.4f]%s; mass %.6f -> %.6f; nodes whose head rose by more than 0.5 m: %d; deepest at the pond centre: y = %.2f m" %
          (u.min(), u.max(), "" if a.lens else " (data: [-10, %.1f])" % PONDING, model.mass(u0), model.mass(u), int((rise > 0.5).sum()),
           model.x[(rise > 0.5) & (np.abs(model.x[:, 0] - L / 2) < 1e-9), 1].min() if ((rise > 0.5) & (np.abs(model.x[:, 0] - L / 2) < 1e-9)).any() else float('nan')))


if __name__ == '__main__':
    main()
