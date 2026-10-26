"""The periodic duct on two ranks.

It segfaulted in object construction: duct.py asked n for an element partition
without overlap (what the parallel periodic numbering needs) but left the
domain's MeshOptions at the default node partition, so the mesh was generated in
pieces with no global copy and the global trial space got None for its mesh
arrays. Then SimpleNavierStokes2D, the Schur complement preconditioner it uses
with KSP_petsc4py, was refused as a linear smoother in parallel.
"""
import os
import shutil
import subprocess
import sys

import numpy
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
HELPER = os.path.join(HERE, "duct_partition_mpi.py")
OPTIONS = os.path.join(HERE, "petsc", "petsc.options.schur.selfp_petsc.superlu")


def _mpiexec():
    bindir = os.path.dirname(sys.executable)
    for name in ("mpiexec", "mpirun"):
        candidate = os.path.join(bindir, name)
        if os.path.exists(candidate):
            return candidate
    return shutil.which("mpiexec") or shutil.which("mpirun")


MPIEXEC = _mpiexec()


def _run(ranks, cwd, **overrides):
    options = OPTIONS
    if ranks > 1:
        # the serial direct solver does not run on several ranks
        options = os.path.join(str(cwd), "petsc.options.superlu_dist")
        with open(OPTIONS) as f, open(options, "w") as g:
            g.write(f.read().replace("superlu\n", "superlu_dist\n"))
    env = dict(os.environ, OMP_NUM_THREADS="1", SOLVER_OPTIONS=options,
               PETSC_OPTIONS="-mat_partitioning_type current")
    env.setdefault("FI_PROVIDER", "tcp")
    env.setdefault("OMPI_MCA_rmaps_base_oversubscribe", "1")
    env.setdefault("PRTE_MCA_rmaps_default_mapping_policy", ":oversubscribe")
    env["PYTHONPATH"] = os.pathsep.join(
        [p for p in sys.path if p] + [env.get("PYTHONPATH", "")]).strip(os.pathsep)
    env.update({k: str(v) for k, v in overrides.items()})
    argv = [sys.executable, HELPER] if ranks == 1 else \
        [MPIEXEC, "-n", str(ranks), sys.executable, HELPER]
    out = subprocess.run(argv, capture_output=True, text=True, timeout=900, env=env, cwd=str(cwd))
    assert out.returncode == 0, out.stdout[-2000:] + out.stderr[-2000:]
    line = [l for l in out.stdout.splitlines() if l.startswith("RESULT")][0]
    return {k: float(v) for k, v in (item.split("=") for item in line.split()[1:])}


@pytest.mark.skipif(MPIEXEC is None, reason="mpiexec not available")
def test_duct_runs_on_two_ranks_and_matches_the_converged_serial_answer(tmp_path):
    """The serial answer at the file's tolerance is not converged (u2 is 10% off the
    tight one), so the reference is a serial run with a tight Newton tolerance; the
    parallel run at the file's tolerance lands within 1e-3 of it."""
    from petsc4py import PETSc
    if not PETSc.Sys.hasExternalPackage("superlu_dist"):
        pytest.skip("needs superlu_dist")
    reference = _run(1, tmp_path, TIGHT=1)
    parallel = _run(2, tmp_path)
    numpy.testing.assert_allclose(parallel["u2"], reference["u2"], rtol=1e-3)
    numpy.testing.assert_allclose(parallel["us"], reference["us"], rtol=1e-3)
