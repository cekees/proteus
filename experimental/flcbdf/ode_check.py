"""FLCBDF on y' = -lam*y: does it integrate, adapt the step and the order, and
how does the error scale with tolerance. No mesh, no proteus models.

    PYTHONPATH=experimental/flcbdf python experimental/flcbdf/ode_check.py
"""
import numpy as np
import flcbdfWrappers


def integrate(rtol, lam=1.0, tout=10.0, order=None, fixed_dt=None):
    y = np.array([1.0])
    yp = np.array([-lam])
    dv = np.ones(1)
    f = flcbdfWrappers.FLCBDF_integrator(y, "ode")
    f.setTolerances(1.0e-10, rtol, dv)
    if order:
        f.set_order(order)
    dt = f.initialize_dt(0.0, tout, y, yp)
    if fixed_dt:
        dt = f.set_dt(fixed_dt)
    f.initializeTimeHistory(y, yp)
    t, nsteps, nfail = 0.0, 0, 0
    ynew = np.zeros(1)
    D = np.zeros(1)
    ypn = np.zeros(1)
    while t < tout - 1e-12:
        t_new = min(t + dt, tout)
        if t_new - t < dt:
            dt = f.set_dt(t_new - t)
        f.setInitialGuess(ynew)                  # predictor
        for _ in range(2):                       # BDF residual is linear: Newton is exact in one
            f.calculate_yprime(ynew, D, ypn, D)
            # ypn = alpha*(y-yn)+yn' ; residual ypn + lam*y = 0 -> Newton step
            a = f.getCurrentAlpha()
            ynew[:] = ynew - (ypn + lam * ynew) / (a + lam)   # in place: the wrapper keeps the buffers
        f.calculate_yprime(ynew, D, ypn, D)
        if fixed_dt is None and not f.lastStepErrorOk(ynew):
            dt = f.retryStep_errorFailure()
            nfail += 1
            continue
        f.stepTaken(ynew)
        t = t_new
        nsteps += 1
        y[:] = ynew
        if fixed_dt is None:
            dt = f.choose_dt(t, tout)
        if nsteps > 100000:
            raise RuntimeError("runaway")
    return abs(y[0] - np.exp(-lam * tout)), nsteps, nfail


if __name__ == "__main__":
    for rtol in (1e-2, 1e-3, 1e-4, 1e-5, 1e-6):
        e, n, nf = integrate(rtol)
        print("rtol %.0e  error %.2e  steps %5d  retried %d" % (rtol, e, n, nf))
