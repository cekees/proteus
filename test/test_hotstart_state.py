"""What a hot start restores beyond the fields (serial), per time integrator, and where it is kept.

* Backward differentiation of order 2 (VBDF) needs the last two masses and step sizes: rebuilt from
  the restart state they make the first steps first order (vortex2D drifts 0.17 after one step).
* An SSP Runge-Kutta integrator with edge-based entropy viscosity (RKEV) needs nothing extra: its
  stage history is rebuilt exactly from the solution. The test keeps it that way.
* The solver state is one section, rewritten at every archived step and stamped with its step:
  the file does not grow with the number of steps, and a hot start from any other step says so
  and restarts instead of reading the wrong state.

The runs use the hot start measurement harness, test/hotstart_measure, in subprocesses.
"""
import os
import re
import subprocess
import sys

import numpy as np
import pytest

h5py = pytest.importorskip("h5py")

here = os.path.dirname(os.path.abspath(__file__))
harness = os.path.join(here, "hotstart_measure")


def _env(**extra):
    env = dict(os.environ, OMP_NUM_THREADS="1",
               PYTHONPATH=os.pathsep.join([os.path.dirname(here), harness, os.environ.get("PYTHONPATH", "")]))
    env.update({k: str(v) for k, v in extra.items()})
    return env


def _vortex(out, ti, order, mode, n, endn=None):
    env = _env(**({"HS_ENDN": endn} if endn is not None else {}))
    cmd = [sys.executable, os.path.join(harness, "run_case.py"), str(out), ti, str(order), mode] + \
          ([str(n)] if mode != "full" else [])
    subprocess.run(cmd, env=env, check=True, capture_output=True)


def _rkev(out, mode, n, endn=None):
    env = _env(**({"HS_ENDN": endn} if endn is not None else {}))
    subprocess.run([sys.executable, os.path.join(harness, "run_rkev.py"), str(out), mode, str(n)],
                   env=env, check=True, capture_output=True)


def _difference(a, b, step, fields):
    with h5py.File(a, "r") as fa, h5py.File(b, "r") as fb:
        worst = 0.0
        for f in fields:
            x = np.array(fa["%s_t%d" % (f, step)]).ravel()
            y = np.array(fb["%s_t%d" % (f, step)]).ravel()
            worst = max(worst, np.linalg.norm(x - y) / np.linalg.norm(x))
        return worst


def test_vbdf2_hot_start_matches_the_run_that_was_not_stopped(tmp_path):
    n, end = 6, 12
    _vortex(tmp_path / "full", "vbdf", 2, "full", n, endn=end)
    _vortex(tmp_path / "hot", "vbdf", 2, "stop", n)
    _vortex(tmp_path / "hot", "vbdf", 2, "hot", n, endn=end)
    h5 = "hs_vbdf2.h5"
    assert _difference(tmp_path / "full" / h5, tmp_path / "hot" / h5, end, ("u", "vof")) < 1e-10


def test_rkev_hot_start_matches_the_run_that_was_not_stopped(tmp_path):
    n, end = 8, 16
    _rkev(tmp_path / "full", "full", n, endn=end)
    _rkev(tmp_path / "hot", "stop", n)
    _rkev(tmp_path / "hot", "hot", n, endn=end)
    assert _difference(tmp_path / "full" / "hs_rkev.h5", tmp_path / "hot" / "hs_rkev.h5", end, ("u",)) < 1e-12


def test_solver_state_is_one_section_and_a_hot_start_from_another_step_restarts(tmp_path):
    n = 6
    _vortex(tmp_path / "hot", "be", 1, "stop", n)
    with h5py.File(tmp_path / "hot" / "hs_be1.h5", "r") as f:
        models = list(f["solver_state"].keys())
        assert models and not any(k.isdigit() for k in models), models
        steps = {int(f["solver_state/%s/step_p0" % m][...][0]) for m in models}
    assert steps == {n}
    #hot start from step 4 while the archive holds the state of step 6
    _vortex(tmp_path / "hot", "be", 1, "hot", 4, endn=8)
    log = (tmp_path / "hot" / "proteus_default.log").read_text()
    assert re.search(r"Hot start: no solver state to restore at step 4 for model \S+ "
                     r"\(the archive holds the state of step 6 only\)", log)
