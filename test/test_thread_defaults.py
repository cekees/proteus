"""One thread per MPI rank by default; threads left alone without MPI.

proteus/_threads.py gives the thread-count variables of numpy's BLAS and
OpenMP runtimes the value 1 when the job has more than one rank and the user
did not set them.  The tests check the rules on a fake environment, then the
effect on real processes: that ``import proteus`` sets the variables for a
launcher that announces its size, that a user's value survives, that a single
process is untouched, and -- under a real ``mpiexec`` -- that a rank's BLAS
runs on one core, whether numpy or proteus was imported first.
"""
import json
import os
import shutil
import subprocess
import sys
import textwrap

import pytest

from proteus import _threads

VARIABLES = tuple(_threads.THREAD_VARIABLES)
LAUNCHER = _threads.LAUNCHER_SIZE_VARIABLES


def clean_environment(**extra):
    """This process's environment without thread settings or launcher announcements."""
    env = {k: v for k, v in os.environ.items()
           if k not in VARIABLES and k not in LAUNCHER and not k.startswith(("OMPI_", "PMIX_", "PMI_"))}
    env.update(extra)
    return env


# --- the rules, on a fake environment --------------------------------------------------------

def test_launched_size_reads_the_launchers_variables():
    assert _threads.launched_size({}) == 1
    assert _threads.launched_size({"OMPI_COMM_WORLD_SIZE": "4"}) == 4
    assert _threads.launched_size({"PMI_SIZE": "8"}) == 8
    assert _threads.launched_size({"SLURM_STEP_NUM_TASKS": "16"}) == 16
    assert _threads.launched_size({"PMI_SIZE": "not a number"}) == 1


@pytest.fixture
def fresh_state(monkeypatch):
    monkeypatch.setattr(_threads, "defaulted", [])
    monkeypatch.setattr(_threads, "numpy_was_loaded", False)


def test_one_rank_changes_nothing(fresh_state):
    env = {}
    assert _threads.apply_defaults(1, env) == []
    assert env == {}


def test_several_ranks_set_every_unset_variable_to_one(fresh_state):
    env = {}
    assert sorted(_threads.apply_defaults(4, env)) == sorted(VARIABLES)
    assert env == {name: "1" for name in VARIABLES}


def test_a_value_the_user_set_is_kept(fresh_state):
    env = {"OMP_NUM_THREADS": "4", "OPENBLAS_NUM_THREADS": "2"}
    names = _threads.apply_defaults(4, env)
    assert env["OMP_NUM_THREADS"] == "4" and env["OPENBLAS_NUM_THREADS"] == "2"
    assert "OMP_NUM_THREADS" not in names and "OPENBLAS_NUM_THREADS" not in names
    assert env["MKL_NUM_THREADS"] == "1"


def test_nothing_to_warn_about_when_the_user_set_everything(fresh_state):
    env = {name: "2" for name in VARIABLES}
    assert _threads.after_mpi_init(4) is None
    assert _threads.apply_defaults(4, env) == []


# --- import proteus, no MPI ------------------------------------------------------------------

def report_environment(**extra):
    out = subprocess.run(
        [sys.executable, "-c",
         "import json, os, proteus; print(json.dumps({k: os.environ.get(k) for k in %r}))" % (VARIABLES,)],
        env=clean_environment(**extra), capture_output=True, text=True, timeout=120)
    assert out.returncode == 0, out.stderr
    return json.loads(out.stdout.splitlines()[-1])


def test_import_proteus_without_mpi_leaves_threads_alone():
    assert set(report_environment().values()) == {None}


@pytest.mark.parametrize("launcher", ["OMPI_COMM_WORLD_SIZE", "PMI_SIZE"])
def test_import_proteus_under_a_launcher_with_several_ranks_sets_one_thread(launcher):
    assert report_environment(**{launcher: "4"}) == {name: "1" for name in VARIABLES}


def test_import_proteus_under_a_launcher_with_one_rank_leaves_threads_alone():
    assert set(report_environment(OMPI_COMM_WORLD_SIZE="1").values()) == {None}


def test_import_proteus_keeps_the_users_value():
    got = report_environment(OMPI_COMM_WORLD_SIZE="4", OMP_NUM_THREADS="3")
    assert got["OMP_NUM_THREADS"] == "3" and got["OPENBLAS_NUM_THREADS"] == "1"


# --- real ranks ------------------------------------------------------------------------------

def _find_mpiexec():
    beside = os.path.join(os.path.dirname(sys.executable), "mpiexec")
    return beside if os.path.exists(beside) else shutil.which("mpiexec")


MPIEXEC = _find_mpiexec()

RANK_SCRIPT = textwrap.dedent('''
    import json, os, sys, time
    if sys.argv[1] == "numpy-first":
        import numpy as np
    from proteus import Comm
    import numpy as np
    size = Comm.get().size()
    a = np.random.rand(1200, 1200)
    a @ a
    wall0, cpu0 = time.perf_counter(), time.process_time()
    for _ in range(4):
        a @ a
    wall, cpu = time.perf_counter() - wall0, time.process_time() - cpu0
    print("RANK " + json.dumps({"size": size, "cpu_per_wall": cpu / wall,
          "env": {k: os.environ.get(k) for k in %r}}), flush=True)
''' % (VARIABLES,))


def run_ranks(tmp_path, ranks, order, **extra):
    pytest.importorskip("proteus.Comm")
    if MPIEXEC is None:
        pytest.skip("no mpiexec")
    script = tmp_path / "rank.py"
    script.write_text(RANK_SCRIPT)
    out = subprocess.run([MPIEXEC, "-n", str(ranks), sys.executable, str(script), order],
                         env=clean_environment(**extra), capture_output=True, text=True, timeout=300)
    assert out.returncode == 0, out.stdout + out.stderr
    reports = [json.loads(line[5:]) for line in out.stdout.splitlines() if line.startswith("RANK ")]
    assert len(reports) == ranks, out.stdout + out.stderr
    return reports


def test_ranks_run_their_blas_on_one_core(tmp_path):
    """proteus imported first: the variables are set before numpy loads its BLAS."""
    for report in run_ranks(tmp_path, 2, "proteus-first"):
        assert report["size"] == 2
        assert report["env"] == {name: "1" for name in VARIABLES}
        assert report["cpu_per_wall"] < 1.5, report


def test_ranks_run_their_blas_on_one_core_when_numpy_was_imported_first(tmp_path):
    """The variables are read at load, so too late; threadpoolctl limits the loaded libraries."""
    pytest.importorskip("threadpoolctl")
    for report in run_ranks(tmp_path, 2, "numpy-first"):
        assert report["cpu_per_wall"] < 1.5, report


def test_a_single_rank_keeps_its_threads(tmp_path):
    for report in run_ranks(tmp_path, 1, "proteus-first"):
        assert report["size"] == 1
        assert set(report["env"].values()) == {None}


def test_a_value_the_user_set_survives_under_mpi(tmp_path):
    for report in run_ranks(tmp_path, 2, "proteus-first", OMP_NUM_THREADS="2"):
        assert report["env"]["OMP_NUM_THREADS"] == "2"
        assert report["env"]["OPENBLAS_NUM_THREADS"] == "1"
