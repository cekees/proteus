"""mprans kernels with their own exterior loop on the faces between ranks.

An element partition with no overlap leaves the faces a rank shares with its
neighbour as exterior faces of its subdomain. A kernel that finds neither a
Dirichlet value nor a flux condition there takes its open-boundary branch,
which is one-sided -- outflow gets ``u b.n``, inflow gets none -- so the two
ranks' terms do not cancel. FluxBoundaryConditions specifies a zero advective
flux on those faces, which every model that reads its advective flux
dictionary then applies as a prescribed flux.
"""
import os
import shutil
import subprocess
import sys

import numpy
import pytest

HELPER = os.path.join(os.path.dirname(os.path.abspath(__file__)), "vof_partition_mpi.py")


def _mpiexec():
    bindir = os.path.dirname(sys.executable)
    for name in ("mpiexec", "mpirun"):
        candidate = os.path.join(bindir, name)
        if os.path.exists(candidate):
            return candidate
    return shutil.which("mpiexec") or shutil.which("mpirun")


MPIEXEC = _mpiexec()


def _run(ranks, cwd, **overrides):
    """One run of vof_partition_mpi.py, with the partition pinned.

    -mat_partitioning_type current keeps the contiguous initial partition, the
    same every time (PETSc's default is not reproducible), and the linear
    solve is converged far enough that the answers can be compared to 1e-8.
    """
    env = dict(os.environ, OMP_NUM_THREADS="1", PART_TYPE="element", NLAYERS="0",
               PETSC_OPTIONS="-mat_partitioning_type current -vof_ksp_rtol 1e-14 "
                             "-vof_ksp_atol 1e-14 -vof_ksp_type gmres")
    env.setdefault("FI_PROVIDER", "tcp")
    env.setdefault("OMPI_MCA_rmaps_base_oversubscribe", "1")
    env.setdefault("PRTE_MCA_rmaps_default_mapping_policy", ":oversubscribe")
    env["PYTHONPATH"] = os.pathsep.join(
        [p for p in sys.path if p] + [env.get("PYTHONPATH", "")]).strip(os.pathsep)
    env.update({k: str(v) for k, v in overrides.items()})
    argv = [sys.executable, HELPER] if ranks == 1 else \
        [MPIEXEC, "-n", str(ranks), sys.executable, HELPER]
    out = subprocess.run(argv, capture_output=True, text=True, timeout=900, env=env,
                         cwd=str(cwd))
    assert out.returncode == 0, out.stdout[-2000:] + out.stderr[-2000:]
    line = [l for l in out.stdout.splitlines() if l.startswith("RESULT")][0]
    result = {k: float(v) for k, v in (item.split("=") for item in line.split()[1:])}
    result["interfaces"] = sum(
        int(dict(i.split("=") for i in l.split()[1:])["interfaces"])
        for l in out.stdout.splitlines() if l.startswith("RANK"))
    return result


@pytest.mark.skipif(MPIEXEC is None, reason="mpiexec not available")
def test_vof_on_an_element_partition_without_overlap_gives_the_serial_answer(tmp_path):
    """Mass used to leave through the cut: 2.9e-02 on two ranks against 4.5e-02 on one."""
    serial = _run(1, tmp_path)
    assert serial["interfaces"] == 0
    parallel = _run(2, tmp_path)
    assert parallel["interfaces"] > 0, parallel
    numpy.testing.assert_allclose(parallel["mass"], serial["mass"], rtol=1e-8)
    numpy.testing.assert_allclose(parallel["l2"], serial["l2"], rtol=1e-8)


@pytest.mark.skipif(MPIEXEC is None, reason="mpiexec not available")
def test_vof_with_one_layer_of_overlap_is_unchanged(tmp_path):
    """The control: with overlap the cut faces are ghost-layer faces and already harmless."""
    serial = _run(1, tmp_path)
    parallel = _run(2, tmp_path, NLAYERS=1)
    numpy.testing.assert_allclose(parallel["mass"], serial["mass"], rtol=1e-8)
