"""Transient regression: ADRH engine vs the Proteus model (ad_dirac_p/n.py) vs the exact Dirac-IC Gaussian.

    u_t + div(b u - kappa grad u) + c u = 0 on the unit square or cube, exact solution the Gaussian of
    AnalyticalSolutions.LinearADR_Decay_DiracIC (c = 0: LinearAD_DiracIC; both verified in #90), started at
    tStart so it is resolved, run to T with backward Euler in NSTEPS fixed steps; boundary data from the exact
    solution in the mode AD_BC says (regression_common.py; the same seven as the steady regression).

The engine takes the same steps as Proteus (BackwardEuler + FixedStep): initial condition interpolated at the
Lagrange nodes with the Dirichlet dofs set at t = 0, then per step one linear solve of R(u; u_old, t) = 0 from the
element kernel (mass term (m - m_old)/dt) and the face kernel with the data at t.

Proteus's first step is special: its old mass m_last for step one comes from a residual evaluation made after the
strongly imposed Dirichlet dofs were already set to their values at t_1, so in the first step the boundary dofs have
no time derivative (later steps take m_last from the previous step's final evaluation, which is right). The engine
does not do that by default; AD_FIRST_STEP=proteus makes it mimic Proteus (the old state's Dirichlet dofs at t_1),
which is how round-off agreement with Proteus is shown. In the modes without eliminated dofs (iipg, nipg, sipg) the
two agree to round-off either way. Checks:

  1. engine and Proteus agree at the final time to solver round-off (with AD_FIRST_STEP=proteus in the modes with
     strong Dirichlet dofs; without it the difference is the first-step effect, O(dt^2) and local to the boundary);
  2. the error against the exact solution converges: in h at fixed small dt (space rate), in dt at fixed fine h
     (backward Euler: rate 1).

usage: [AD_ND=2|3] [AD_SPACE=p1|p2|q1] [AD_BC=mode] [AD_NSTEPS=n] [AD_FIRST_STEP=proteus] python ad_dirac_regression.py [nn ...]
       AD_SWEEP=dt [AD_NN=nn] python ad_dirac_regression.py [nsteps ...]        (the dt sweep at one mesh)
"""
import importlib
import os
import sys

import numpy as np

from regression_common import (ND, SPACE, BC, nD, nV, qp, qw, qpf, qwf, WEAK_PARAMS, ROBIN_ALPHA, ROBIN_BETA,
                               adrh_proto, run_proteus, boundary_faces, dirichlet_dofs, assemble,
                               solve_with_dirichlet, l2_error)

# the problem (exported for ad_dirac_p.py): a pulse that stays inside the unit domain and is resolved from the start
b = np.array([0.5, 0.25, 0.125])[:ND]; kappa = 0.02; c = 0.5
tStart = 0.1; u0 = 1.0; xc = np.array([0.3, 0.4, 0.5])[:ND]       # xc: the pulse's centre at time 0 (x0 is proteus's domain origin)
T_END = float(os.environ.get('AD_T', '0.3'))
for k, v in (('AD_B', ','.join(repr(float(x)) for x in b)), ('AD_KAPPA', repr(kappa)), ('AD_C', repr(c)),
             ('AD_TSTART', repr(tStart)), ('AD_U0', repr(u0)), ('AD_XC', ','.join(repr(float(x)) for x in xc)),
             ('AD_T', repr(T_END))):
    os.environ[k] = v


def exact(X, T):
    """The Gaussian at points X (..., 3) and time T (proteus's uOfXT, including its exponent cut-off at 100)."""
    t = T + tStart
    y = X[..., :ND] - xc - b * t
    arg = (y * y).sum(axis=-1) / (4.0 * kappa * t)
    return np.where(arg > 100.0, 0.0, u0 * np.exp(-np.minimum(arg, 100.0)) / (4.0 * kappa * np.pi * t) ** (0.5 * ND)
                    * np.exp(-c * t))


# the kernels: the host engine, or the Kokkos port (ADRH_MODULE=adrh_kokkos)
engine = importlib.import_module(os.environ.get('ADRH_MODULE', 'adrh_proto'))


def engine_solution(nodes, conn, nsteps):
    nE, nN = len(conn), len(nodes)
    dt = T_END / nsteps
    xn = np.ascontiguousarray(nodes[conn])
    faces = boundary_faces(nodes, conn)
    onb = dirichlet_dofs(nodes)
    u = exact(nodes, 0.0)                                              # interpolated initial condition
    zeros = np.zeros((nE, 1, nD))
    for step in range(1, nsteps + 1):
        t = step * dt
        if step == 1 and FIRST_STEP == 'proteus':
            u = u.copy(); u[onb] = exact(nodes, t)[onb]                # Proteus's first m_last: Dirichlet dofs at t_1
        uold = np.ascontiguousarray(u[conn][:, None, :])
        R, J = engine.transient_adr(ND, SPACE, b, kappa, c, xn, zeros, uold, qp, qw, t, dt)
        K, rhs = assemble(conn, R, J, nN)
        if faces is not None:
            fe, fk, kind = faces
            face_conn = conn[fe]
            pc, pp, sig = WEAK_PARAMS
            Rf, Jf = engine.transient_adr_faces(ND, SPACE, b, kappa, c, tStart, u0, xc, fk, kind,
                                                    np.ascontiguousarray(nodes[face_conn]), np.zeros((len(fe), 1, nD)),
                                                    qpf, qwf, t, pc, pp, sig, ROBIN_ALPHA, ROBIN_BETA)
            Kf, rhs_f = assemble(face_conn, Rf, Jf, nN)
            K = K + Kf; rhs = rhs + rhs_f
        u = solve_with_dirichlet(K, rhs, onb, exact(nodes, t))
    return u, xn


FIRST_STEP = os.environ.get('AD_FIRST_STEP', 'exact')
assert FIRST_STEP in ('exact', 'proteus'), FIRST_STEP
SWEEP = os.environ.get('AD_SWEEP', 'h')
if SWEEP == 'dt':
    nn = int(os.environ.get('AD_NN', '33' if ND == 2 else '17'))
    runs = [(nn, int(a)) for a in sys.argv[1:]] or [(nn, 4), (nn, 8), (nn, 16), (nn, 32)]
else:
    nsteps = int(os.environ.get('AD_NSTEPS', '16'))
    runs = [(int(a), nsteps) for a in sys.argv[1:]] or [(nn, nsteps) for nn in ((9, 17, 33) if ND == 2 else (5, 9, 17))]
print("AD_ND=%d AD_SPACE=%s AD_BC=%s T=%g sweep=%s first_step=%s" % (ND, SPACE, BC, T_END, SWEEP, FIRST_STEP))
print("%4s %6s %9s %12s %12s %14s %14s %9s %9s" % ("nn", "steps", "elements", "|eng-prot|", "eng-vs-ex L2", "prot-vs-ex L2", "eng max err", "L2 rate", "max rate"))
prev = None
for nn, nsteps in runs:
    os.environ['AD_NSTEPS'] = str(nsteps)
    up, nodes, conn = run_proteus('ad_dirac_p', 'ad_dirac_n', 'addirac', nn)
    ue, xn = engine_solution(nodes, conn, nsteps)
    at_T = lambda X: exact(X, T_END)
    l2e, l2p = l2_error(ue, conn, xn, at_T), l2_error(up, conn, xn, at_T)
    mxe = np.abs(ue - exact(nodes, T_END)).max()
    param = T_END / nsteps if SWEEP == 'dt' else 1.0 / (nn - 1)
    rate = ("%9.2f %9.2f" % (np.log(prev[0] / l2e) / np.log(prev[1] / param), np.log(prev[2] / mxe) / np.log(prev[1] / param))) if prev else "%9s %9s" % ("-", "-")
    print("%4d %6d %9d %12.2e %12.3e %14.3e %14.3e %s" % (nn, nsteps, len(conn), np.abs(ue - up).max(), l2e, l2p, mxe, rate))
    prev = (l2e, param, mxe)
