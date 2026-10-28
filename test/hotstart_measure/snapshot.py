"""Flatten the numerical state of a NumericalSolution into {path: value},
and put a saved state back.

Used by run_case.py (HS_SNAP=file) to compare the state of an uninterrupted
run at step N with the state a hotstart rebuilds, variable by variable, and
(HS_ORACLE=groups) to overwrite groups of the hotstarted state with the true
values and see which group removes the error.
"""
import numbers
import pickle

import numpy as np


def _flat(obj, path, out, parent=None, key=None, depth=0):
    """out[path] = (value, setter). setter(v) writes v back."""
    if isinstance(obj, np.ndarray):
        if obj.dtype.kind in "fiub":
            def set_arr(v, a=obj):
                a[...] = v
            out[path] = (obj.copy(), set_arr)
    elif isinstance(obj, (bool, numbers.Number)):
        if parent is not None and not isinstance(parent, tuple):
            def set_sc(v, p=parent, k=key, t=type(obj)):
                p[k] = t(v)
            out[path] = (float(obj), set_sc)
    elif isinstance(obj, dict) and depth < 3:
        for k, v in list(obj.items()):
            _flat(v, "%s[%s]" % (path, k), out, obj, k, depth + 1)
    elif isinstance(obj, (list, tuple)) and depth < 3 and len(obj) < 64:
        for i, v in enumerate(obj):
            _flat(v, "%s[%d]" % (path, i), out, obj, i, depth + 1)


def _attrs(obj, path, out):
    d = vars(obj)
    for k, v in list(d.items()):
        if not k.startswith("__"):
            _flat(v, "%s.%s" % (path, k), out, d, k)


def collect(ns):
    out = {}
    _attrs(ns.systemStepController, "system", out)
    for m in ns.modelList:
        mp = m.name
        _attrs(m.stepController, mp + ".stepController", out)
        # the Newton initial guess and the residual live on the model
        for name in ("uList", "rList"):
            _flat(getattr(m, name), "%s.%s" % (mp, name), out)
        lm = m.levelModelList[-1]
        _attrs(lm.timeIntegration, mp + ".timeIntegration", out)
        for ci, u in lm.u.items():
            for a in ("dof", "dof_last", "dof_last_last"):
                if hasattr(u, a):
                    _flat(getattr(u, a), "%s.u[%d].%s" % (mp, ci, a), out)
        for k, v in lm.q.items():
            if isinstance(v, np.ndarray) and v.dtype.kind == "f":
                _flat(v, "%s.q[%s]" % (mp, k), out)
        _attrs(lm.coefficients, mp + ".coefficients", out)
        for name in ("shockCapturing", "subgridError", "stabilization",
                     "numericalFlux"):
            o = getattr(lm, name, None)
            if o is not None:
                _attrs(o, "%s.%s" % (mp, name), out)
        for name in ("ebqe", "ebq", "ebq_global", "ebq_global_u"):
            d = getattr(lm, name, None)
            if isinstance(d, dict):
                for k, v in d.items():
                    _flat(v, "%s.%s[%s]" % (mp, name, k), out)
        _attrs(lm, mp + ".levelModel", out)
    return out


def snapshot(ns, fname):
    out = collect(ns)
    with open(fname, "wb") as f:
        pickle.dump({k: v[0] for k, v in out.items()}, f)


GROUPS = {
    "udof": lambda p: ".u[" in p,
    "q": lambda p: ".q[" in p,
    "ti": lambda p: ".timeIntegration." in p,
    "ctrl": lambda p: ".stepController." in p or p.startswith("system."),
    "coef": lambda p: ".coefficients." in p,
    "sc": lambda p: any(t in p for t in (".shockCapturing.", ".subgridError.", ".stabilization.", ".numericalFlux.")),
    "ebq": lambda p: any(t in p for t in (".ebqe[", ".ebq[", ".ebq_global")),
    "lm": lambda p: ".levelModel." in p,
    "all": lambda p: True,
    "first": lambda p: False,
    "fixalias": lambda p: False,
    "ulist": lambda p: ".uList[" in p or ".rList[" in p,
}


def restore(ns, fname, groups):
    """Overwrite the live state with the saved one for the named groups."""
    saved = pickle.load(open(fname, "rb"))
    live = collect(ns)
    n = 0
    for p, (_, setter) in live.items():
        if p in saved and any(GROUPS[g](p) for g in groups):
            try:
                setter(saved[p])
                n += 1
            except Exception:
                pass
    # NCLS.ShockCapturing starts with numDiff_last aliased to numDiff and
    # swaps in separate copies when it switches to lagging; writing values
    # into aliased arrays cannot restore that, so rebuild the copies.
    if "fixalias" in groups:
        for m in ns.modelList:
            sc = getattr(m.levelModelList[-1], "shockCapturing", None)
            if sc is not None and getattr(sc, "nStepsToDelay", None) is not None and sc.lag:
                sc.numDiff_last = [a.copy() for a in sc.numDiff]
                for ci in range(len(sc.numDiff_last)):
                    sc.numDiff_last[ci][...] = saved["%s.shockCapturing.numDiff_last[%d]" % (m.name, ci)]
                    sc.numDiff[ci][...] = saved["%s.shockCapturing.numDiff[%d]" % (m.name, ci)]
                n += 1
    return n
