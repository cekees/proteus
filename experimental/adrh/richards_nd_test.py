"""The K-upwinded Richards scheme on a general P1 mesh (richards_nd.py) against what is known.

1. One dimension: KUpwindRichards reproduces richards_regression.py's Column (the scheme pinned to proteus's saved profiles) on the Celia deck, low-order and
   with FCT, to the Newton tolerance.
2. Two dimensions: the same column as a strip of triangles, uniform in x, gives the 1D solution in every column of nodes (the low-order scheme reduces to
   the 1D one exactly, because the right-angled diagonals carry no stiffness and the horizontal edges no head difference); with FCT the solution stays
   in the bounds of the data and the FCT correction keeps the mass sum_i M_L,i m_i exactly.
3. Heterogeneous media: the two-layer steady column of richards_layers.py (an ODE solution that uses no finite elements). With proteus's rule (an interface node
   has one retention curve and K_r is taken from it) the error falls only at rate 0.3-0.7 and is 0.14-0.58 m; with K_r taken from each element's own material
   (interface='element') it falls at rate 1 and is 100 times smaller. The strip equals the column for both. A transient layered case with FCT stays in bounds
   and conserves mass.
4. FCT bounds: on the head (continuous across soils) they are the moisture bounds for one material, and across a soil interface they stop the moisture
   bound's undershoot (a lens node drying below every head around it).
5. Nodal materials: the node's material is the first incident element's.

usage: [ADRH_MODULE=adrh_kokkos] python richards_nd_test.py
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import richards_layers as RL
import richards_nd as nd
import richards_regression as rr

engine = nd.engine
failures = []


def check(ok, text):
    print(("  ok   " if ok else "  FAIL ") + text)
    if not ok:
        failures.append(text)


def celia_parameters(case):
    return dict(alpha=[case.alpha], n_vg=[case.n_vg], theta_r=[case.theta_r], theta_s=[case.theta_s], ks=[[case.ks]], g=[-1.0], psk=case.psk)


def one_dimension():
    print("1D, Celia deck, 41 nodes, 40 steps of 0.025 d: KUpwindRichards against Column")
    case = rr.CASES['celia']; nn, steps = 41, 40
    x = np.linspace(0.0, case.L, nn)[:, None]; conn = np.stack([np.arange(nn - 1), np.arange(1, nn)], axis=1)
    for fct in (False, True):
        col = rr.Column(nn, 'strong', fct=fct, case=case, upwind=True)
        u = case.initial(col.x).astype(float); dt = case.t_end / steps
        for _ in range(steps):
            uold = u
            u, _ = rr.newton(col, u, 0.0, dt, 1e-12, 'inf')
            if fct:
                u, _, _ = col.fct_correct(u, uold, 0.0, dt)
        model = nd.KUpwindRichards(x, conn, np.zeros(nn - 1, dtype='i'), dirichlet=([0, nn - 1], [case.bottom, case.top]), **celia_parameters(case))
        v, _ = model.run(case.initial(col.x), case.t_end, steps, fct=fct)
        check(np.abs(u - v).max() < 1e-8, "%s: max |difference| %.2e" % ("FCT" if fct else "low-order", np.abs(u - v).max()))


def strip(nx, ny, case):
    x, conn = nd.rectangle_triangles(nx, ny, 0.25, case.L)
    j = np.arange(x.shape[0]) % ny
    fixed = np.concatenate([np.where(j == 0)[0], np.where(j == ny - 1)[0]])
    return x, conn, fixed


def two_dimensions():
    print("2D, the Celia column as a strip of triangles (5 x 41 nodes), 40 steps")
    case = rr.CASES['celia']; nx, ny, steps = 5, 41, 40
    x, conn, fixed = strip(nx, ny, case)
    gD = np.where(x[fixed, 1] == 0.0, case.bottom, case.top)
    p = celia_parameters(case); p['ks'] = [[case.ks, case.ks]]; p['g'] = [0.0, -1.0]
    model = nd.KUpwindRichards(x, conn, np.zeros(conn.shape[0], dtype='i'), dirichlet=(fixed, gD), **p)
    u0 = np.where(x[:, 1] == case.L, case.top, case.bottom)
    col = rr.Column(ny, 'strong', case=case, upwind=True); u1 = case.initial(col.x).astype(float); dt = case.t_end / steps
    for _ in range(steps):
        u1, _ = rr.newton(col, u1, 0.0, dt, 1e-12, 'inf')
    u, _ = model.run(u0, case.t_end, steps)
    U = u.reshape(nx, ny)
    check(np.abs(U - u1[None, :]).max() < 1e-7, "low-order: every column equals the 1D solution, max |difference| %.2e" % np.abs(U - u1[None, :]).max())
    lo, hi = min(case.bottom, case.top), max(case.bottom, case.top)
    # FCT: one limited step from the low-order one, bounds and exact mass conservation
    uold = u0; uL, _ = model.newton(u0, uold, dt)
    uF, mF = model.fct_correct(uL, uold, dt)
    mL, _ = model.m(uL)
    check(abs(float(model.ML @ mF) - float(model.ML @ mL)) < 1e-12 * abs(float(model.ML @ mL)), "FCT keeps sum M_L m exactly (%.2e relative)" %
          (abs(float(model.ML @ mF) - float(model.ML @ mL)) / abs(float(model.ML @ mL))))
    check(uF.min() >= lo - 1e-9 and uF.max() <= hi + 1e-9, "FCT stays in the data's bounds [%.2f, %.2f]: psi in [%.4f, %.4f]" % (lo, hi, uF.min(), uF.max()))
    uF, _ = model.run(u0, case.t_end, steps, fct=True)
    check(uF.min() >= lo - 1e-9 and uF.max() <= hi + 1e-9, "FCT, 40 steps: psi in [%.4f, %.4f]" % (uF.min(), uF.max()))


def layered(nn, ny=None, interface='node'):
    """The two-layer steady column of richards_layers.py with nodal materials: 1D on nn nodes, or a strip of nn nodes in y (ny set: strip of ny columns of nodes)."""
    tol = dict(alpha=RL.ALPHA, n_vg=RL.N_VG, theta_r=RL.THETA_R, theta_s=RL.THETA_S)
    if ny is None:
        x = np.linspace(0.0, 1.0, nn)[:, None]; conn = np.stack([np.arange(nn - 1), np.arange(1, nn)], axis=1)
        mid = x[conn, 0].mean(axis=1); fixed = np.array([0, nn - 1]); y = x[:, 0]
        ks = RL.KS.reshape(2, 1); g = [-1.0]
    else:
        x, conn = nd.rectangle_triangles(ny, nn, 0.25, 1.0)
        mid = x[conn, 1].mean(axis=1); y = x[:, 1]
        fixed = np.where((y == 0.0) | (y == 1.0))[0]
        ks = np.stack([RL.KS, RL.KS], axis=1); g = [0.0, -1.0]
    material = (mid > RL.X_INTERFACE).astype('i')
    gD = np.where(y[fixed] == 0.0, RL.PSI_BOTTOM, RL.PSI_TOP)
    model = nd.KUpwindRichards(x, conn, material, dirichlet=(fixed, gD), ks=ks, g=g, interface=interface, **tol)
    u0 = RL.PSI_BOTTOM + (RL.PSI_TOP - RL.PSI_BOTTOM) * y
    u, its = model.newton(u0, u0, None, tol=1e-11, maxit=100)
    return model, y, u, its


def layers():
    print("two layers (richards_layers.py), steady K-upwinded scheme: the nodal error against the ODE solution")
    results = {}
    for iface in ('node', 'element'):
        errs = []
        print("    interface=%r (%s)" % (iface, "proteus's rule: K_r from the node's material" if iface == 'node' else "K_r from each element's own material"))
        for nn in (21, 41, 81, 161):
            model, y, u, its = layered(nn, interface=iface)
            _, ex = RL.exact_profile(y)
            errs.append(np.abs(u - ex).max())
            print("      nn %4d  %2d Newton iterations  max |error| %.3e%s" % (nn, its, errs[-1], "" if len(errs) == 1 else "   rate %.2f" % np.log2(errs[-2] / errs[-1])))
        results[iface] = errs
        check(all(a > b for a, b in zip(errs, errs[1:])), "interface=%r: the error falls under refinement" % iface)
    rate = np.log2(results['element'][-2] / results['element'][-1])
    check(rate > 0.9, "interface='element': first order (rate %.2f on the finest pair)" % rate)
    check(all(e < 0.1 * n for e, n in zip(results['element'], results['node'])), "interface='element' is at least 10x more accurate than the node rule on every mesh")
    for iface in ('node', 'element'):
        model, y, u, _ = layered(41, ny=3, interface=iface)
        m1, y1, u1, _ = layered(41, interface=iface)
        U = u.reshape(3, 41)
        check(np.abs(U - u1[None, :]).max() < 1e-7, "interface=%r: the strip (3 x 41 nodes) equals the column, max |difference| %.2e" % (iface, np.abs(U - u1[None, :]).max()))
    check(model.nodeMat.reshape(3, 41)[:, :21].max() == 0, "the nodes up to the interface take the sand's material")


def layered_transient():
    print("two layers, transient, FCT with interface='element': bounds and mass")
    nn = 41
    x = np.linspace(0.0, 1.0, nn)[:, None]; conn = np.stack([np.arange(nn - 1), np.arange(1, nn)], axis=1)
    material = (x[conn, 0].mean(axis=1) > RL.X_INTERFACE).astype('i')
    model = nd.KUpwindRichards(x, conn, material, dirichlet=([0, nn - 1], [RL.PSI_BOTTOM, -0.2]), ks=RL.KS.reshape(2, 1), g=[-1.0],
                               alpha=RL.ALPHA, n_vg=RL.N_VG, theta_r=RL.THETA_R, theta_s=RL.THETA_S, interface='element')
    u0 = np.full(nn, -6.0); u0[0] = RL.PSI_BOTTOM; u0[-1] = -0.2
    dt = 0.01; u = u0.copy(); worst = 0.0
    for _ in range(20):
        uold = u
        uL, _ = model.newton(u, uold, dt)
        u, mnew = model.fct_correct(uL, uold, dt)
        mL, _ = model.m(uL)
        worst = max(worst, abs(float(model.ML @ mnew) - float(model.ML @ mL)) / abs(float(model.ML @ mL)))
    check(u.min() >= -6.0 - 1e-9 and u.max() <= -0.2 + 1e-9, "psi stays in [-6, -0.2]: [%.4f, %.4f]" % (u.min(), u.max()))
    check(worst < 1e-12, "the FCT step keeps sum M_L m exactly (worst relative change %.1e)" % worst)


def head_bounds():
    print("FCT bounds on the head (bounds='head') against on the moisture content (bounds='moisture')")
    case = rr.CASES['celia']; nx, ny = 5, 41
    x, conn, fixed = strip(nx, ny, case)
    gD = np.where(x[fixed, 1] == 0.0, case.bottom, case.top)
    p = celia_parameters(case); p['ks'] = [[case.ks, case.ks]]; p['g'] = [0.0, -1.0]
    model = nd.KUpwindRichards(x, conn, np.zeros(conn.shape[0], dtype='i'), dirichlet=(fixed, gD), **p)
    u0 = np.where(x[:, 1] == case.L, case.top, case.bottom); dt = 0.025
    uL, _ = model.newton(u0, u0, dt)
    _, mh = model.fct_correct(uL, u0, dt, bounds='head'); _, mm = model.fct_correct(uL, u0, dt, bounds='moisture')
    check(np.abs(mh - mm).max() < 1e-12, "one material: the two bounds agree, max |difference| in moisture content %.1e" % np.abs(mh - mm).max())
    import richards_2d as R2
    lo = None
    for bounds in ('moisture', 'head'):
        model, u0 = R2.build(41, True, 'element')
        lo = u0.min()
        u, _ = model.run(u0, R2.T_END, 70, fct=True, tol=1e-8, norm=2, bounds=bounds)
        below = int((u < lo - 1e-9).sum())
        print("      lens, 41 x 41 nodes, 70 steps, bounds=%r: psi in [%.4f, %.4f], %d nodes below the initial minimum %.0f" % (bounds, u.min(), u.max(), below, lo))
    check(below == 0 and u.max() <= R2.PONDING + 1e-9, "the lens, head bounds: no node leaves [%.0f, %.1f]" % (lo, R2.PONDING))


def materials():
    print("the node material is the first incident element's")
    conn = np.array([[0, 1], [1, 2], [2, 3]]); material = np.array([0, 1, 2])
    got = nd.first_incident_material(conn, material, 4)
    check(list(got) == [0, 0, 1, 2], "nodes 0..3 -> %s (expected [0, 0, 1, 2])" % list(got))


if __name__ == '__main__':
    print("module:", engine.__name__)
    one_dimension()
    two_dimensions()
    layers()
    layered_transient()
    head_bounds()
    materials()
    if failures:
        print("%d check(s) failed" % len(failures))
        sys.exit(1)
    print("all checks passed")
