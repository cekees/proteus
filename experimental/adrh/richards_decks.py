"""The three 1D Richards benchmark decks of test/richards (Celia 1990 Fig 6b, HYDRUS-1D 20 m, Szymkiewicz 2009 test 1) against proteus and a fine-mesh reference.

For each deck and each saved output time, the moisture content theta(psi) (the conserved variable; the pressure head spans ten orders of magnitude of
conductivity) of every scheme is compared with

  * the profile proteus saved for STABILIZATION_TYPE=2 (which is the K-upwinded low-order scheme with the lumped mass) and for FCT, in
    test/richards/comparison_files, and
  * a reference: the engine's K-upwinded FCT scheme on a mesh `refine` times finer with `refine` times as many steps, with a second, still finer
    solution to show the reference has converged.

The columns: the largest and the L1 (integral) difference of theta from the reference, and the position of the front (the first place theta falls
through the middle between the wet boundary value and the driest initial value, scanning down from the wet boundary). The engine schemes are the Galerkin discretization with the consistent mass
(weak Dirichlet data, as the saved plain-Galerkin columns), the Jacobian-based discrete upwinding of lowOrder.h, and the K-upwinded low-order scheme
and its FCT correction (strong Dirichlet data, as proteus does for them), with the limiter's local bounds taken over the new and old low-order moisture
content (ours) or over the new one alone (proteus's FCTStep).

usage: [ADRH_MODULE=adrh_kokkos] python richards_decks.py [celia|hydrus|szymkiewicz ...] [--refine 8]
"""
import argparse
import sys

import numpy as np

import richards_regression as R

SCHEMES = (  # label, Column keywords, run options
    ('Galerkin, consistent mass', dict(bc='weak')),
    ('Jacobian upwinding, lumped', dict(bc='weak', stabilize=True)),
    ('K-upwind, lumped', dict(bc='strong', upwind=True)),
    ('K-upwind + FCT', dict(bc='strong', upwind=True, fct=True)),
    ('K-upwind + FCT, proteus bounds', dict(bc='strong', upwind=True, fct=True, proteus_limiter=True)),
)


def _theta(case, psi):
    pc = -psi
    sr = case.theta_s - case.theta_r
    th = np.full_like(psi, case.theta_s)
    if case.psk == 0:
        w = pc > 0
        m = 1 - 1 / case.n_vg
        th[w] = case.theta_r + sr * (1 + (case.alpha * pc[w]) ** case.n_vg) ** (-m)
    else:
        a = case.alpha * pc; w = a > 1
        th[w] = case.theta_r + sr * a[w] ** (-case.n_vg)
    return th


def front(x, th, level):
    """First position, scanning down from the wet end (the last node), where theta falls through `level` (linear interpolation)."""
    for i in range(len(x) - 1, 0, -1):
        if th[i] >= level > th[i - 1]:
            return x[i - 1] + (x[i] - x[i - 1]) * (level - th[i - 1]) / (th[i] - th[i - 1])
    return float('nan')


def l1(x, a, b):
    return np.trapezoid(np.abs(a - b), x)


def solve(case, nn, steps, frames, **kw):
    """The profiles at the saved fractions of the run: {frame: (x, psi)}; frame numbers are in units of case.steps."""
    col = R.Column(nn, case=case, **kw)
    at = {(steps * f) // case.steps: f for f in frames}
    out, its, wall = R.run(col, steps, 1e-8, 2, tuple(at))
    return {at[k]: (col.x, v) for k, v in out.items()}, its / steps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('cases', nargs='*', default=['celia', 'hydrus', 'szymkiewicz'])
    ap.add_argument('--refine', type=int, default=8)
    a = ap.parse_args()
    for name in a.cases:
        case = R.CASES[name]
        xs, ref = R.read_reference(case)
        psi0 = case.initial(xs)
        level = 0.5 * (_theta(case, np.array([max(case.bottom, case.top)]))[0] + _theta(case, np.array([psi0.min()]))[0])
        k = a.refine
        nn_f = (case.nn - 1) * k + 1
        print("\n=== %s: L = %g m, %d nodes, %d steps of %.4g d; theta front level %.3f; reference %d nodes x %d steps (and %d x %d)" % (
            name, case.L, case.nn, case.steps, case.t_end / case.steps, level, nn_f, case.steps * k, (nn_f - 1) * 2 + 1, case.steps * k * 2))
        fine, _ = solve(case, nn_f, case.steps * k, case.frames, bc='strong', upwind=True, fct=True)  # the reference keeps the wider bounds
        finer, _ = solve(case, (nn_f - 1) * 2 + 1, case.steps * k * 2, case.frames, bc='strong', upwind=True, fct=True)
        sols = {}
        for label, kw in SCHEMES:
            try:
                sols[label], its = solve(case, case.nn, case.steps, case.frames, **kw)
            except RuntimeError as e:
                sols[label] = None
                print("  %s: %s" % (label, str(e).split(':')[0]))
        for f in case.frames:
            tag = 't%d' % f
            xf, pf = fine[f]; xff, pff = finer[f]
            thf = _theta(case, pf)
            ref_on_coarse = np.interp(xs, xf, thf)
            thff = np.interp(xf, xff, _theta(case, pff))
            print("\n  frame %d (t = %.4g d)   reference front at %.4f m; reference vs the finer one: max %.1e, L1 %.1e" % (
                f, f * case.t_end / case.steps, front(xf, thf, level), np.abs(thf - thff).max(), l1(xf, thf, thff)))
            print("  %-34s %10s %10s %10s %10s" % ('', 'max |dtheta|', 'L1 dtheta', 'front (m)', 'psi min'))
            rows = []
            for label, _ in SCHEMES:
                if sols[label] is None:
                    rows.append((label, None)); continue
                x, psi = sols[label][f]
                rows.append((label, psi))
            for kcol, label in (('stab_2', "proteus, STABILIZATION_TYPE=2"), ('FCT', 'proteus, FCT'), ('stab_0', 'proteus, Galerkin')):
                if kcol + '_' + tag in ref: rows.append((label, ref[kcol + '_' + tag]))
            for label, psi in rows:
                if psi is None:
                    print("  %-34s %10s" % (label, 'Newton failed')); continue
                th = _theta(case, psi)
                print("  %-34s %10.2e %10.2e %10.4f %10.3f" % (label, np.abs(th - ref_on_coarse).max(), l1(xs, th, ref_on_coarse), front(xs, th, level), psi.min()))


if __name__ == '__main__':
    main()
