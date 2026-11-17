"""DIAGNOSTIC ONLY (branch richards-nan-diag, never merged): pytest plugin that traces the Richards implicit FCT
levee test and prints, to stderr, compact lines starting with RICHARDS_NAN_DIAG.

Run:  PYTHONPATH=test python -m pytest -s -p richards_instr <the levee test>

What it prints
  NEWTON   one line per Newton.solve: model, t, dt, fct phase, iterations, ||r0||, ||r||, failed flags, and the
           history (max|u|, ||r||) of every residual evaluation of the solve (first 10 and last 3)
  DT       every change of dt (or of t) between two solves of one model
  MASS     the first residual calls of Richards: old-time mass (m_last) against the mass at the iterate and at u_old
  FIRST    the first non-finite quantity of a kind (residual, Jacobian entry, iterate, dt, du, matrix), with the model,
           time step, solve and iteration, the first bad node and the values that produced it, every non-finite array of
           the model, and the last residual calls before it
"""
import os
import sys

import numpy as np

TAG = 'RICHARDS_NAN_DIAG'
S = {'lines': 0, 'ncall': {}, 'ring': [], 'first': set(), 'cur': None, 'solve': 0, 'last_dt': {},
     'last_model': None, 'last_phase': None}
MAXLINES = int(os.environ.get('RICHARDS_NAN_DIAG_MAXLINES', '2500'))


def P(*a):
    S['lines'] += 1
    if S['lines'] > MAXLINES:
        return
    print(TAG, *a, file=sys.stderr, flush=True)
    if S['lines'] == MAXLINES:
        print(TAG, 'LINE LIMIT REACHED', file=sys.stderr, flush=True)


def name(model):
    return type(model).__module__.split('.')[-1]


def fin(x):
    return bool(np.all(np.isfinite(x)))


def st(x):
    x = np.asarray(x, dtype=float)
    ok = np.isfinite(x)
    nb = int((~ok).sum())
    if ok.any():
        return '%d/%.3e/%.3e' % (nb, x[ok].min(), x[ok].max())   # nbad/min/max
    return '%d/allbad' % nb


def amax(x):
    x = np.asarray(x, dtype=float)
    ok = np.isfinite(x)
    return np.abs(x[ok]).max() if ok.any() else float('nan')


def tinfo(m):
    ti = m.timeIntegration
    return 't=%.9e dt=%.4e' % (ti.t, ti.dt)


def ring_add(s):
    S['ring'].append(s)
    del S['ring'][:-8]


def scan_model(m, i=None):
    """every non-finite ndarray attribute of the model, its time integration and coefficients"""
    out = []
    for owner, obj in (('model', m), ('ti', m.timeIntegration), ('coef', m.coefficients)):
        for k, v in list(vars(obj).items()):
            vs = [(k, v)]
            if isinstance(v, dict):
                vs = [('%s[%s]' % (k, kk), vv) for kk, vv in v.items()]
            for kk, vv in vs:
                if isinstance(vv, np.ndarray) and vv.dtype.kind == 'f' and vv.size and not fin(vv):
                    bad = np.flatnonzero(~np.isfinite(vv.ravel()))
                    out.append('%s.%s nbad=%d first=%d(%r)' % (owner, kk, bad.size, bad[0], vv.ravel()[bad[0]]))
                elif isinstance(vv, float) and not np.isfinite(vv):
                    out.append('%s.%s=%r' % (owner, kk, vv))
    return out


NODE_FIELDS = ('u_dof_old', 'mn', 'mLow', 'fluxCorrection', 'm_fct', 'ML', 'alpha_nv', 'Rpos', 'Rneg',
               'rho_dof_in', 'rho_nodal', 'bc_mask', 'nodalSource', 'psi_old')


def node_values(m, i):
    v = []
    u = m.u[0].dof
    v.append('u=%r' % (u[i],))
    for k in NODE_FIELDS:
        a = getattr(m, k, None)
        if isinstance(a, np.ndarray) and a.ndim == 1 and a.shape[0] > i and a.shape[0] == u.shape[0]:
            v.append('%s=%r' % (k, a[i]))
    ti = m.timeIntegration
    for k in ('m_last', 'm_tmp', 'beta_bdf'):
        a = getattr(ti, k, None)
        if a is not None:
            try:
                v.append('%s=%r' % (k, a[0][i]))
            except Exception:
                pass
    return ' '.join(v)


def first_event(kind, m, it_info, i=None):
    """print once per (kind, model): the full context of the first non-finite quantity"""
    key = (kind, name(m))
    if key in S['first']:
        return
    S['first'].add(key)
    ti = m.timeIntegration
    P('FIRST', kind, name(m), tinfo(m), 'phase=%s' % getattr(m, 'fct_phase', None), 'solve#%d' % S['solve'],
      'residual_calls=%d' % S['ncall'].get(name(m), 0), it_info)
    P('FIRST   alpha_bdf=%r dt_last=%r' % (getattr(ti, 'alpha_bdf', None), getattr(ti, 'dt_last', None)),
      'nonfinite:', '; '.join(scan_model(m)) or 'none')
    if i is not None:
        P('FIRST   node %d x=%s: %s' % (i, m.mesh.nodeArray[i][:3].tolist(), node_values(m, i)))
    for s in S['ring']:
        P('FIRST   ring:', s)


def newton_iteration_hint():
    c = S['cur']
    return 'newton_res_eval#%d' % (len(c['hist']) if c else -1)


def install():
    from proteus import LinearSolvers, NonlinearSolvers
    from proteus.richards import Richards
    from proteus.mprans import TADR

    for mod in (Richards, TADR):
        LM = mod.LevelModel
        if getattr(LM, '_rnd_patched', False):
            continue
        LM._rnd_patched = True

        def make(LM=LM):
            orig_res = LM.getResidual
            orig_jac = LM.getJacobian

            def getResidual(self, u, r):
                nm = name(self)
                S['ncall'][nm] = S['ncall'].get(nm, 0) + 1
                nc = S['ncall'][nm]
                u_in_fin = fin(u)
                out = orig_res(self, u, r)
                rn = float(np.linalg.norm(r)) if fin(r) else float('nan')
                ti = self.timeIntegration
                phase = getattr(self, 'fct_phase', None)
                S['last_model'], S['last_phase'] = nm, phase
                line = '%s#%d %s phase=%s |u|max=%.3e r(nbad/min/max)=%s ||r||=%.4e' % (
                    nm, nc, tinfo(self), phase, amax(u), st(r), rn)
                ring_add(line)
                if S['cur'] is not None and S['cur']['model'] == nm:
                    S['cur']['hist'].append((amax(u), rn))
                if nc <= 40:
                    P('RES', line, 'mLow=%s fluxCorr=%s' % (st(getattr(self, 'mLow', 0.0)),
                                                              st(getattr(self, 'fluxCorrection', 0.0))))
                if nm == 'Richards' and nc <= 3:
                    mlast = ti.m_last[0]
                    mtmp = ti.m_tmp[0]
                    q_rho = self.q['rho'] if 'rho' in self.q else None
                    P('MASS', 'call#%d' % nc, tinfo(self),
                      'm_last(nbad/min/max)=%s m(u)=%s max|m(u)-m_last|=%.4e (m(u)-m_last)/dt max=%.4e' % (
                          st(mlast), st(mtmp), amax(mtmp - mlast), amax(mtmp - mlast) / max(ti.dt, 1e-300)),
                      'q_rho=%s' % (st(q_rho) if q_rho is not None else None),
                      'u_dof_old=%s' % st(self.u_dof_old), '|u-u_old|max=%.3e' % amax(u - self.u_dof_old))
                if not u_in_fin:
                    bad = np.flatnonzero(~np.isfinite(u))
                    first_event('NONFINITE_ITERATE_INPUT_TO_RESIDUAL', self, newton_iteration_hint(), int(bad[0]))
                if not fin(r):
                    bad = np.flatnonzero(~np.isfinite(r))
                    first_event('NONFINITE_RESIDUAL', self, newton_iteration_hint(), int(bad[0]))
                return out

            def getJacobian(self, jacobian):
                out = orig_jac(self, jacobian)
                rowptr, colind, nz = jacobian.getCSRrepresentation()
                nm = name(self)
                if S['ncall'].get('jac_' + nm, 0) < 12:
                    P('JAC', nm, tinfo(self), 'phase=%s' % getattr(self, 'fct_phase', None), 'nz(nbad/min/max)=%s' % st(nz))
                S['ncall']['jac_' + nm] = S['ncall'].get('jac_' + nm, 0) + 1
                if not fin(nz):
                    bad = ~np.isfinite(nz)
                    rows = np.repeat(np.arange(len(rowptr) - 1), np.diff(rowptr))
                    badrows = np.unique(rows[bad])
                    i = int(badrows[0])
                    first_event('NONFINITE_JACOBIAN', self, '%d bad entries of %d, %d bad rows of %d, first rows %s; row %d nz=%s' % (
                        bad.sum(), nz.size, badrows.size, len(rowptr) - 1, badrows[:6].tolist(), i,
                        nz[rowptr[i]:rowptr[i + 1]].tolist()), i)
                return out

            LM.getResidual = getResidual
            LM.getJacobian = getJacobian
        make()

    # ---- linear solver: matrix and du --------------------------------------------------------------------------
    orig_prep = LinearSolvers.LU.prepare
    orig_solve = LinearSolvers.LU.solve

    def prepare(self, b=None, newton_its=None):
        if type(self.L).__name__ == 'SparseMatrix':
            rowptr, colind, nz = self.L.getCSRrepresentation()
            if not fin(nz):
                P('LU.prepare NONFINITE MATRIX n=%d model(last residual)=%s phase=%s nz(nbad/min/max)=%s b=%s' % (
                    self.n, S['last_model'], S['last_phase'], st(nz), st(b) if b is not None else None))
        return orig_prep(self, b, newton_its)

    def solve(self, u, *a, **k):
        out = orig_solve(self, u, *a, **k)
        if not fin(u):
            b = k.get('b')
            P('LU.solve du NONFINITE n=%d du=%s b=%s failed=%s' % (self.n, st(u), st(b) if b is not None else None,
                                                                   self.failed()))
        return out

    LinearSolvers.LU.prepare = prepare
    LinearSolvers.LU.solve = solve

    # ---- one line per Newton solve ------------------------------------------------------------------------------
    def wrap_newton(cls):
        if 'solve' not in vars(cls) or getattr(cls.solve, '_rnd', False):
            return
        orig = cls.solve

        def newton_solve(self, u, *a, **k):
            F = self.F
            nm = name(F)
            if nm not in ('Richards', 'TADR'):
                return orig(self, u, *a, **k)
            S['solve'] += 1
            ti = F.timeIntegration
            key = nm
            last = S['last_dt'].get(key)
            if last is None or last != (ti.t, ti.dt):
                P('DT', nm, 'solve#%d' % S['solve'], 'phase=%s' % getattr(F, 'fct_phase', None),
                  'before: t=%.9e dt=%.4e' % last if last else 'first', '-> t=%.9e dt=%.4e' % (ti.t, ti.dt))
                S['last_dt'][key] = (ti.t, ti.dt)
            if not (np.isfinite(ti.dt) and ti.dt > 0):
                first_event('NONFINITE_OR_NONPOSITIVE_DT', F, 'at Newton.solve entry')
            if not fin(u):
                first_event('NONFINITE_ITERATE_AT_SOLVE_ENTRY', F, 'at Newton.solve entry', int(np.flatnonzero(~np.isfinite(u))[0]))
            outer = S['cur']
            S['cur'] = {'model': nm, 'hist': []}
            try:
                return orig(self, u, *a, **k)
            finally:
                cur = S['cur']
                h = cur['hist']
                hs = h if len(h) <= 13 else h[:10] + [None] + h[-3:]
                if S['solve'] <= 400 or S['solve'] % 25 == 0 or not fin(u):
                    P('NEWTON', nm, 'solve#%d' % S['solve'], tinfo(F), 'phase=%s' % getattr(F, 'fct_phase', None),
                      'its=%s' % getattr(self, 'its', None), 'norm_r0=%.4e norm_r=%.4e' % (
                          getattr(self, 'norm_r0', float('nan')), getattr(self, 'norm_r', float('nan'))),
                      'failedFlag=%s linFailed=%s' % (getattr(self, 'failedFlag', None), getattr(self, 'linearSolverFailed', None)),
                      '|u|max_out=%.3e' % amax(u), 'hist(|u|max,||r||)=' + ' '.join(
                          '..' if e is None else '(%.2e,%.2e)' % e for e in hs))
                S['cur'] = outer
        newton_solve._rnd = True
        cls.solve = newton_solve

    def all_subclasses(c):
        for s in c.__subclasses__():
            yield s
            yield from all_subclasses(s)

    wrap_newton(NonlinearSolvers.Newton)
    P('plugin installed')


def pytest_configure(config):
    install()
