"""FLCBDF on a diagonal ODE y_i' = -lam_i y_i split over the MPI ranks unevenly:
the step sequence must not depend on the number of ranks (the error norm is
global).  Prints the step sizes as JSON on rank 0.

    mpiexec -n 2 python experimental/flcbdf/ode_parallel.py
"""
import json
import numpy as np
from mpi4py import MPI
import flcbdfWrappers

N = 17
LAM = 1.0 + 0.3 * np.arange(N)


def run(rtol=1.0e-4, tout=5.0):
    comm = MPI.COMM_WORLD
    r, p = comm.Get_rank(), comm.Get_size()
    cuts = [0] + [int(round(N * (i + 1) ** 1.3 / p ** 1.3)) for i in range(p)]
    lo, hi = cuts[r], cuts[r + 1]
    lam = LAM[lo:hi].copy()
    y = np.ones(hi - lo)
    yp = -lam * y
    f = flcbdfWrappers.FLCBDF_integrator(y, "odepar")
    f.setTolerances(1.0e-10, rtol, np.ones(hi - lo))
    dt = f.initialize_dt(0.0, tout, y, yp)
    f.initializeTimeHistory(y, yp)
    t, dts = 0.0, []
    ynew, D, ypn = np.zeros_like(y), np.zeros_like(y), np.zeros_like(y)
    while t < tout - 1e-12:
        if t + dt > tout:
            dt = f.set_dt(tout - t)
        f.setInitialGuess(ynew)
        for _ in range(2):
            f.calculate_yprime(ynew, D, ypn, D)
            ynew[:] = ynew - (ypn + lam * ynew) / (f.getCurrentAlpha() + lam)
        f.calculate_yprime(ynew, D, ypn, D)
        if not f.lastStepErrorOk(ynew):
            dt = f.retryStep_errorFailure()
            continue
        f.stepTaken(ynew)
        t += dt
        dts.append(dt)
        y[:] = ynew
        dt = f.choose_dt(t, tout)
    err = comm.allreduce(float(np.max(np.abs(y - np.exp(-lam * tout)))) if len(y) else 0.0, op=MPI.MAX)
    return dts, err


if __name__ == "__main__":
    dts, err = run()
    if MPI.COMM_WORLD.Get_rank() == 0:
        print(json.dumps({"dts": dts, "err": err}))
