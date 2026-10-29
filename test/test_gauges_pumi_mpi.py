"""Gauges on a PUMI mesh partitioned over two ranks, under real MPI.

``gauges_pumi_mpi.py`` loads the two-part ``squarepar`` and ``cubepar`` meshes
through PUMI and runs point, line and line integral gauges on them; every value
has a closed form (u = x + 10 y, or x + 10 y + 100 z in 3D), so the answer does
not depend on the partition.

The tolerances are looser than in ``test_gauges_mpi.py``: PUMI places boundary
nodes on its geometric model to about 1e-12 (the node meant to be at (1, 0.25)
is 6e-13 off), so the interpolated values carry that much noise.  Those same
nodes lie on the gauge line, which cuts the line into slivers shorter than a
gauge's segment tolerance; see ``test_slivers_at_the_end_of_a_line...``.
"""
import json
import os
import shutil
import subprocess
import sys

import numpy as np
import numpy.testing as npt
import pytest

HELPER = os.path.join(os.path.dirname(os.path.abspath(__file__)), "gauges_pumi_mpi.py")
ADAPT_HELPER = os.path.join(os.path.dirname(os.path.abspath(__file__)), "gauges_pumi_adapt_mpi.py")

try:
    from proteus.MeshAdaptPUMI import MeshAdapt  # noqa: F401
except ImportError:
    MeshAdapt = None


def _mpiexec():
    """The launcher beside this interpreter, which matches its mpi4py."""
    beside = os.path.join(os.path.dirname(sys.executable), "mpiexec")
    return beside if os.path.exists(beside) else shutil.which("mpiexec")


MPIEXEC = _mpiexec()


def run_ranks(tmp_path, nranks=2, helper=HELPER):
    env = dict(os.environ)
    env.setdefault("OMPI_MCA_rmaps_base_oversubscribe", "1")
    env.setdefault("PRTE_MCA_rmaps_default_mapping_policy", ":oversubscribe")
    cmd = [MPIEXEC, "-n", str(nranks), sys.executable, helper]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=300, env=env, cwd=str(tmp_path))
    assert result.returncode == 0, "%d ranks failed:\nSTDOUT:\n%s\nSTDERR:\n%s" % (
        nranks, result.stdout[-2000:], result.stderr[-4000:])
    lines = result.stdout.splitlines()
    observed = [int(l.split("=")[1]) for l in lines if l.startswith("OBSERVED_SIZE=")]
    # a launcher that does not match mpi4py starts N independent one-rank jobs
    assert observed == [nranks], "asked for %d ranks, ran with %s" % (nranks, observed)
    payload = [l for l in lines if l.startswith("GAUGE_RESULT=")]
    assert payload, "no result in:\n%s" % result.stdout[-2000:]
    return json.loads(payload[0].split("=", 1)[1])


def xs(result):
    return np.array([float(n.split("[")[1].split()[0]) for n in result["names"]])


@pytest.mark.skipif(MPIEXEC is None, reason="mpiexec not available")
@pytest.mark.skipif(MeshAdapt is None, reason="PUMI is not built")
def test_gauges_on_a_pumi_mesh_do_not_depend_on_the_partition(tmp_path):
    result = run_ranks(tmp_path)
    u = lambda x, y: x + 10 * y
    tol = dict(rtol=1e-9, atol=1e-9)
    # points, the last two a roundoff outside the square and snapped onto it
    npt.assert_allclose(result["points"]["values"],
                        [u(0.37, 0.22), u(0.5, 0.5), u(0.25, 0.75), u(0.5, 0), u(1, 0.5)], **tol)
    # a line that leaves the mesh at both ends: points only inside, none repeated
    # (a mesh node on the line used to give a point per triangle around it)
    x = xs(result["line"])
    npt.assert_allclose([x.min(), x.max()], [0.0, 1.0], atol=1e-9)
    assert len(np.unique(np.round(x, 9))) == len(x)
    # the x read back from the header is printed to five digits
    npt.assert_allclose(result["line"]["values"], u(x, 0.25), atol=1e-5)
    # the integral of x + 2.5 over [0, 1]; thousands of repeated slivers at the
    # node on the right edge used to add 1e-8
    npt.assert_allclose(result["integral"]["values"], [3.0], rtol=1e-10)
    # the vertical line crosses the cut: integral of x + 10 y at x = 0.31 over y in [0, 1]
    npt.assert_allclose(result["vertical_integral"]["values"], [0.31 + 5.0], rtol=1e-10)
    # the diagonal of the unit cube, through both parts
    x = xs(result["diagonal_3d"])
    npt.assert_allclose(x, [0.0, 1.0], atol=1e-9)
    npt.assert_allclose(result["diagonal_3d"]["values"], 111 * x, **tol)


@pytest.mark.skipif(MPIEXEC is None, reason="mpiexec not available")
@pytest.mark.skipif(MeshAdapt is None, reason="PUMI is not built")
@pytest.mark.parametrize("nranks", [2, 4])
def test_gauges_on_a_mesh_adapted_in_parallel_do_not_depend_on_the_partition(tmp_path, nranks):
    """The mesh is adapted to a circle and rebalanced by ParMA, so the parts are
    ones nobody wrote down, and the gauges are attached to it as they are after
    each adapt of a run."""
    result = run_ranks(tmp_path, nranks, ADAPT_HELPER)
    u = lambda x, y: x + 10 * y
    tol = dict(rtol=1e-9, atol=1e-9)
    points = ((0.5, 0.2), (0.8, 0.5), (0.5, 0.8), (0.2, 0.5), (0.37, 0.22),
              (0.05, 0.05), (0.95, 0.95), (0.5, 0.5), (0.25, 0.75), (1.0, 0.5))
    npt.assert_allclose(result["points"]["values"], [u(*p) for p in points], **tol)
    x = xs(result["line"])
    npt.assert_allclose([x.min(), x.max()], [0.0, 1.0], atol=1e-9)
    assert len(np.unique(np.round(x, 9))) == len(x)
    # the x read back from the header is printed to five digits
    npt.assert_allclose(result["line"]["values"], u(x, 0.25), atol=1e-5)
    npt.assert_allclose(result["integral"]["values"], [3.0], rtol=1e-10)
    npt.assert_allclose(result["vertical_integral"]["values"], [5.5], rtol=1e-10)
