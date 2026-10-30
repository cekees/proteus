"""Implicit FCT (richards_nd.py, implicit_fct_step; Dielmann, Kuzmin and Kees, proteus's FCT_TYPE='implicit') on the three 1D benchmark decks, against the
post-step FCT and the low-order scheme, and a fine-mesh reference.

Each scheme is run on the deck's own mesh and time step; the reference is the post-step K-upwinded FCT on a mesh `refine` times finer with `refine` times as
many steps (the reference of richards_decks.py). Reported at the first saved output time of each deck: the L1 difference of the moisture content from the
reference (the conserved variable; the head spans ten orders of magnitude of conductivity), the front position, the lowest head, the Newton iterations per step
(both solves of an implicit step) and how often the Galerkin predictor failed and the low-order one stood in.

usage: [ADRH_MODULE=adrh_kokkos] python richards_implicit_fct.py [celia|hydrus|szymkiewicz ...] [--refine 8]
"""
import argparse
import sys

import numpy as np

import richards_nd as nd
import richards_regression as rr
from richards_decks import front, l1


def build(case, nn):
    x = np.linspace(0.0, case.L, nn)[:, None]; conn = np.stack([np.arange(nn - 1), np.arange(1, nn)], axis=1)
    return nd.KUpwindRichards(x, conn, np.zeros(nn - 1, dtype='i'), dirichlet=([0, nn - 1], [case.bottom, case.top]), alpha=[case.alpha], n_vg=[case.n_vg],
                              theta_r=[case.theta_r], theta_s=[case.theta_s], ks=[[case.ks]], g=[-1.0], psk=case.psk)


def march(model, case, steps, upto, kind, tol=1e-8, bounds='local'):
    """The head after `upto` of `steps` steps of one scheme ('low', 'post', 'implicit'); (head, iterations, predictor fallbacks)."""
    u = case.initial(model.x[:, 0]).astype(float); dt = case.t_end / steps; its = fb = 0
    for _ in range(upto):
        uold = u
        if kind == 'implicit':
            u, c = model.implicit_fct_step(uold, dt, tol=tol, norm=2, bounds=bounds); its += c['predictor'] + c['final']; fb += c['fallback']
        else:
            u, k = model.newton(u, uold, dt, tol, 2); its += k
            if kind == 'post':
                u, _ = model.fct_correct(u, uold, dt)
    return u, its, fb


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('cases', nargs='*', default=['celia', 'hydrus', 'szymkiewicz'])
    ap.add_argument('--refine', type=int, default=8)
    a = ap.parse_args()
    for name in a.cases:
        case = rr.CASES[name]; k = a.refine; frame = case.frames[0]
        coarse = build(case, case.nn); nn_f = (case.nn - 1) * k + 1; fine = build(case, nn_f)
        theta = lambda model, u: model.m(u)[0]
        psi0 = case.initial(coarse.x[:, 0])
        level = 0.5 * (theta(coarse, np.full(case.nn, max(case.bottom, case.top)))[0] + theta(coarse, np.full(case.nn, psi0.min()))[0])
        uf, _, _ = march(fine, case, case.steps * k, frame * k, 'post')
        thf = theta(fine, uf)
        ref_on_coarse = np.interp(coarse.x[:, 0], fine.x[:, 0], thf)
        print("\n=== %s: %d nodes, %d steps of %.4g d, frame %d (t = %.4g d); reference %d nodes x %d steps, front at %.4f m" % (
            name, case.nn, case.steps, case.t_end / case.steps, frame, frame * case.t_end / case.steps, nn_f, case.steps * k, front(fine.x[:, 0], thf, level)))
        print("  %-28s %10s %10s %10s %10s %10s" % ('', 'L1 dtheta', 'front (m)', 'psi min', 'its/step', 'fallbacks'))
        for label, kind, bounds in (('K-upwind, lumped', 'low', 'global'), ('K-upwind + post-step FCT', 'post', 'global'), ('implicit FCT', 'implicit', 'local'),
                                    ("implicit FCT, proteus's bounds", 'implicit', 'global')):
            try:
                u, its, fb = march(coarse, case, case.steps, frame, kind, bounds=bounds)
            except RuntimeError as e:
                print("  %-28s %s" % (label, str(e).split(':')[0])); continue
            th = theta(coarse, u)
            print("  %-28s %10.2e %10.4f %10.3f %10.2f %10d" % (label, l1(coarse.x[:, 0], th, ref_on_coarse), front(coarse.x[:, 0], th, level), u.min(), its / frame, fb))


if __name__ == '__main__':
    main()
