"""Fixtures for the multilevel tests."""
import contextlib
import fcntl
import os
import subprocess
import tempfile
import time

import pytest

# The MPI tests here launch 2-8 ranks, and CI runners have 2 cores. Open MPI 5
# refuses to start more ranks than slots ("not enough slots available"), which
# fails every such test there while passing on a workstation; MPICH always
# oversubscribes and ignores these. Set once for the whole directory, before any
# test builds the environment it hands to mpiexec.
os.environ.setdefault('OMPI_MCA_rmaps_base_oversubscribe', '1')
os.environ.setdefault('PRTE_MCA_rmaps_default_mapping_policy', ':oversubscribe')
# Open MPI 5 picks UCX for point-to-point when it can, and on some CI runners (the ones
# whose network card is a MANA verbs device) UCX fails to open it ("uct_iface_open(ud_verbs/
# mana_0:1) failed: Address not valid"), which fails or hangs the launched ranks even for
# single-node jobs. The plain ob1 layer (self, shared memory, tcp) needs none of that.
# MPICH ignores these.
os.environ.setdefault('OMPI_MCA_pml', 'ob1')


# How the MPI jobs these tests launch share the machine under pytest-xdist (CI runs
# ``-n auto --dist loadgroup``). PROTEUS_TEST_MPI picks: 'slots' (default), 'group' or 'none'.
#
# One worker per core, each launching its own 2-4 rank job, puts several times more busy-polling
# MPI processes on a runner than it has cores: on the 4-core macOS Intel runner 4-rank tests that
# take seconds on an idle machine took up to 17 minutes and four hit their 300 s limit.
#   slots: a cross-process counting semaphore over MPI *ranks*. A launch holds as many slots as it
#          has ranks (``mpiexec -n N``) while it runs, and there are PROTEUS_TEST_MPI_SLOTS of them
#          (default: twice the core count). Jobs that fit together still run side by side, and the
#          machine is never more than 2x oversubscribed. On 4 cores pinned, 91 of these tests ran
#          in 63 s with no cap, 72 s at 2x, 107 s at 1x and 139 s on one worker (MPICH, Linux).
#   group: every test in this directory on one xdist worker, so at most one job at a time. The
#          setting that fixed the Intel runner, kept as the fallback.
#   none:  no limit.
_MODE = os.environ.get('PROTEUS_TEST_MPI', 'slots')
_SLOTS = max(1, int(os.environ.get('PROTEUS_TEST_MPI_SLOTS', 2 * (os.cpu_count() or 2))))
# One directory per user, so every pytest run on the machine shares the slots; PROTEUS_TEST_MPI_SLOT_DIR gives a run
# (the test of the limiter itself) slots of its own, which the other runs on a shared runner cannot hold.
_SLOT_DIR = os.environ.get('PROTEUS_TEST_MPI_SLOT_DIR') or os.path.join(tempfile.gettempdir(), 'proteus_mpi_slots_%d' % os.getuid())


def _ranks(cmd):
    """The rank count of an ``mpiexec -n N ...`` command, or None for anything else."""
    try:
        if os.path.basename(str(cmd[0])) not in ('mpiexec', 'mpirun', 'mpiexec.hydra'):
            return None
        for i, arg in enumerate(cmd[1:-1], 1):
            if arg in ('-n', '-np', '--np'):
                return int(cmd[i + 1])
    except (TypeError, IndexError, ValueError):
        return None
    return 1


@contextlib.contextmanager
def _rank_slots(n):
    """Hold n of the machine's rank slots (all of them if n exceeds them) until the block ends.

    Slots are flock'd files, so a killed worker (``--forked`` children, a timeout) frees its
    slots with its process. Only one launch at a time collects slots, under the gate lock, so two
    launches that each hold some and wait for more cannot deadlock."""
    n = min(n, _SLOTS)
    os.makedirs(_SLOT_DIR, exist_ok=True)
    held = []
    with open(os.path.join(_SLOT_DIR, 'gate'), 'w') as gate:
        fcntl.flock(gate, fcntl.LOCK_EX)
        try:
            while len(held) < n:
                for k in range(_SLOTS):
                    if len(held) == n:
                        break
                    f = open(os.path.join(_SLOT_DIR, 'slot%d' % k), 'w')
                    try:
                        fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
                        held.append(f)
                    except OSError:
                        f.close()
                if len(held) < n:
                    time.sleep(0.05)
        except BaseException:
            for f in held:
                f.close()
            raise
    try:
        yield
    finally:
        for f in held:
            f.close()


@pytest.fixture(autouse=True)
def _launch_policy(monkeypatch, _retry_an_mpi_launch_that_never_started):
    """Every ``subprocess.run`` of an mpiexec from these tests holds its rank slots while it runs
    (mode 'slots'). It wraps the retry test/conftest.py puts on every launch, so a launch that
    never started (an Open MPI runner flake) is launched again under the same slots."""
    run = subprocess.run

    def launch(cmd, *args, **kwargs):
        ranks = _ranks(cmd) if _MODE == 'slots' else None
        if ranks is None:
            return run(cmd, *args, **kwargs)
        with _rank_slots(ranks):
            return run(cmd, *args, **kwargs)
    monkeypatch.setattr(subprocess, 'run', launch)
    yield


@pytest.hookimpl(tryfirst=True)  # before xdist turns the group marks into the '@group' part of the test id
def pytest_collection_modifyitems(config, items):
    if _MODE != 'group':
        return
    here = os.path.dirname(os.path.abspath(__file__))
    for item in items:
        if str(getattr(item, 'path', item.fspath)).startswith(here + os.sep):
            item.add_marker(pytest.mark.xdist_group('multilevel-mpi'))


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "xdist_group(name): run the tests with this name on one pytest-xdist worker (--dist loadgroup)")
    config.addinivalue_line(
        "markers",
        "global_hierarchy: tests the global hierarchy builder itself (refinement, "
        "partitionLevels, the global node parents and vertex interpolation), so it runs "
        "with MultilevelMesh.regularGrid='global', not hierarchies generated in pieces")


def _globally(node):
    return node.get_closest_marker('global_hierarchy') is not None


@pytest.fixture(autouse=True, scope='module')
def _global_hierarchy_module(request):
    """Build hierarchies globally for modules marked global_hierarchy; the rest
    of the suite runs on hierarchies generated in pieces, the default.
    Module-scoped, so that a module's own module-scoped fixtures, which build
    their hierarchies before any test's fixtures run, see it too. (The MPI
    helpers those modules launch select the global path themselves.)"""
    if not _globally(request.node):
        yield
        return
    from proteus import MeshTools, default_n
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(MeshTools.MultilevelMesh, 'regularGrid', 'global')
        mp.setattr(default_n, 'regularGrid', 'global')
        yield


@pytest.fixture(autouse=True)
def _global_hierarchy(request, monkeypatch):
    """The same for each test marked global_hierarchy. default_n too: the
    numerics a test loads take regularGrid from it, and NumericalSolution
    passes that to the mesh constructors explicitly."""
    if _globally(request.node):
        from proteus import MeshTools, default_n
        monkeypatch.setattr(MeshTools.MultilevelMesh, 'regularGrid', 'global')
        monkeypatch.setattr(default_n, 'regularGrid', 'global')
    yield
