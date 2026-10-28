"""Tests of the experimental FLCBDF build (skipped when the extension is not built).

    PYTHONPATH=experimental/flcbdf python -m pytest experimental/flcbdf/test_flcbdf.py
"""
import os
import re
import subprocess
import sys

import numpy as np
import pytest

here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, here)
flcbdfWrappers = pytest.importorskip("flcbdfWrappers")
import ode_check  # noqa: E402


@pytest.fixture(autouse=True)
def in_tmp(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)  # the integrator writes a data_*.txt log


def test_ode_error_falls_and_steps_grow_with_tolerance():
    results = [ode_check.integrate(rtol) for rtol in (1e-2, 1e-4, 1e-6)]
    errors = [r[0] for r in results]
    steps = [r[1] for r in results]
    assert errors[0] > errors[1] > errors[2]
    assert steps[0] < steps[1] < steps[2]
    assert errors[2] < 1e-8


def test_order_one_with_fixed_steps_is_backward_euler():
    """Order 1 and a constant step is backward Euler. The step is chosen by the
    integrator and then set to another (what a system of models does when
    another model's step is smaller); that used to leave the coefficients of
    the chosen step in place when the set step equalled the previous one."""
    H, lam = 0.1, 1.0
    y = np.array([1.0])
    yp = np.array([-lam])
    D = np.zeros(1)
    ypn = np.zeros(1)
    ynew = np.zeros(1)
    f = flcbdfWrappers.FLCBDF_integrator(y, "be_equivalence")
    f.setTolerances(1.0e-10, 1.0e3, np.ones(1))
    f.set_order(1)
    f.initialize_dt(0.0, 10.0, y, yp)
    f.set_dt(H)
    f.initializeTimeHistory(y, yp)
    for n in range(8):
        f.setInitialGuess(ynew)
        f.calculate_yprime(ynew, D, ypn, D)
        ynew[:] = ynew - (ypn + lam * ynew) / (f.getCurrentAlpha() + lam)
        f.calculate_yprime(ynew, D, ypn, D)
        f.lastStepErrorOk(ynew)
        f.stepTaken(ynew)
        assert abs(ynew[0] - (1.0 / (1.0 + H)) ** (n + 1)) < 1e-13
        f.choose_dt(0.0, 10.0)
        f.set_dt(H)


def test_step_size_at_machine_precision_raises():
    y = np.array([1.0])
    f = flcbdfWrappers.FLCBDF_integrator(y, "tiny_step")
    f.setTolerances(1.0e-10, 1.0e-4, np.ones(1))
    f.initialize_dt(0.0, 10.0, y, np.array([-1.0]))
    f.initializeTimeHistory(y, np.array([-1.0]))
    with pytest.raises(RuntimeError, match="machine precision"):
        for _ in range(2000):
            f.retryStep_errorFailure()


def _vortex(tmp_path, ti_ls, ti_vof, tol):
    root = os.path.abspath(os.path.join(here, "..", ".."))
    env = dict(os.environ, HS_TI_LS=ti_ls, HS_TI_VOF=ti_vof, HS_FLCBDF_TOL=str(tol),
               PYTHONPATH=os.pathsep.join([root, here, os.environ.get("PYTHONPATH", "")]))
    run = os.path.join(root, "test", "hotstart_measure", "run_case.py")
    out = tmp_path / ("run_%s_%s" % (ti_ls, ti_vof))
    subprocess.run([sys.executable, run, str(out), "flcbdf", "1", "full"],
                   env=env, check=True, capture_output=True)
    log = (out / "proteus_default.log").read_text()
    times = [float(t) for t in re.findall(r"Step Taken, System time step t= *([0-9.e+-]+)", log)]
    return times, log


@pytest.mark.parametrize("ti_ls,ti_vof,tol", [("flcbdf", "be", 1e-2),
                                              ("be", "flcbdf", 1e-2),
                                              ("flcbdf", "flcbdf", 1e-2)])
def test_vortex2d_with_flcbdf_reaches_the_end(tmp_path, ti_ls, ti_vof, tol):
    """vortex2D to t = 8 with FLCBDF on the level set, on the volume of fluid,
    and on both, with no Newton failure. Known gaps: the level set at the
    default 1e-4 drives dt toward zero at the steep front, and at 1e-3 the
    redistancing model's Newton iteration stalls near its tolerance."""
    times, log = _vortex(tmp_path, ti_ls, ti_vof, tol)
    assert times and times[-1] >= 8.0 - 1e-8
    assert "FAILED" not in log
