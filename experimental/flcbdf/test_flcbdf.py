"""Tests of the experimental FLCBDF build (skipped when the extension is not built).

    PYTHONPATH=experimental/flcbdf python -m pytest experimental/flcbdf/test_flcbdf.py
"""
import os
import re
import subprocess
import sys

import pytest

here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, here)
flcbdfWrappers = pytest.importorskip("flcbdfWrappers")
import ode_check  # noqa: E402


def test_ode_error_falls_and_steps_grow_with_tolerance(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)  # the integrator writes a data_*.txt log
    results = [ode_check.integrate(rtol) for rtol in (1e-2, 1e-4, 1e-6)]
    errors = [r[0] for r in results]
    steps = [r[1] for r in results]
    assert errors[0] > errors[1] > errors[2]
    assert steps[0] < steps[1] < steps[2]
    assert errors[2] < 1e-8


def test_vortex2d_vof_with_flcbdf_reaches_the_end(tmp_path):
    """Level set by backward Euler, volume of fluid by FLCBDF (the level set
    with FLCBDF does not converge with these tolerances), to t = 8."""
    root = os.path.abspath(os.path.join(here, "..", ".."))
    env = dict(os.environ, HS_TI_LS="be",
               PYTHONPATH=os.pathsep.join([root, here, os.environ.get("PYTHONPATH", "")]))
    run = os.path.join(root, "test", "hotstart_measure", "run_case.py")
    out = tmp_path / "run"
    subprocess.run([sys.executable, run, str(out), "flcbdf", "1", "full"],
                   env=env, check=True, capture_output=True)
    log = (out / "proteus_default.log").read_text()
    times = [float(t) for t in re.findall(r"Step Taken, System time step t= *([0-9.e+-]+)", log)]
    assert times and times[-1] >= 8.0 - 1e-8
