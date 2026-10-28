"""Proteus mesh -> PUMI mesh in parallel.

reconstructFromProteus2 is serial: it derives the geometric model from the whole
boundary, and on a subdomain a partition boundary would become a model face. The
parallel route gathers the mesh on rank 0, builds and derives there, and spreads
it over the ranks by Proteus's own element partition. The model must be the
domain's (a unit square: 1 face, 4 edges, 4 vertices; a cube: 1 region, 6 faces,
12 edges, 8 vertices) and each rank must hold the elements Proteus gave it.
"""
import os
import shutil
import subprocess
import sys

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
HELPER = os.path.join(HERE, "reconstruct_parallel_mpi.py")
pytest.importorskip("proteus.MeshAdaptPUMI.MeshAdapt")


def _mpiexec():
    bindir = os.path.dirname(sys.executable)
    for name in ("mpiexec", "mpirun"):
        candidate = os.path.join(bindir, name)
        if os.path.exists(candidate):
            return candidate
    return shutil.which("mpiexec") or shutil.which("mpirun")


def _run(ranks, cwd, **env_overrides):
    env = dict(os.environ, OMP_NUM_THREADS="1", PETSC_OPTIONS="-mat_partitioning_type current")
    env.setdefault("FI_PROVIDER", "tcp")
    env.setdefault("OMPI_MCA_rmaps_base_oversubscribe", "1")
    env.setdefault("PRTE_MCA_rmaps_default_mapping_policy", ":oversubscribe")
    env["PYTHONPATH"] = os.pathsep.join(
        [p for p in sys.path if p] + [env.get("PYTHONPATH", "")]).strip(os.pathsep)
    env.update({k: str(v) for k, v in env_overrides.items()})
    out = subprocess.run([_mpiexec(), "-n", str(ranks), sys.executable, HELPER],
                         capture_output=True, text=True, timeout=600, env=env, cwd=str(cwd))
    assert out.returncode == 0, out.stdout[-2000:] + out.stderr[-2000:]
    rows = []
    for line in out.stdout.splitlines():
        if line.startswith("RANK"):
            rows.append(dict(item.split("=") for item in line.split()[1:]))
    assert len(rows) == ranks
    return rows


@pytest.mark.parametrize("shape,model", [("tri", "0,1,4,4"), ("tet", "1,6,12,8")])
@pytest.mark.parametrize("ranks", [2, 4])
@pytest.mark.parametrize("layers", [0, 1])
def test_parallel_reconstruction_matches_proteus_partition(tmp_path, shape, model, ranks, layers):
    rows = _run(ranks, tmp_path, SHAPE=shape, NLAYERS=layers)
    for row in rows:
        assert row["pumi_owned"] == row["proteus_owned"]
        assert row["pumi_global"] == row["global"] == row["returned"]
        assert row["model"] == model
