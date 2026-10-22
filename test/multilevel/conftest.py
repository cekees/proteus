"""Fixtures for the multilevel tests."""
import os

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


@pytest.hookimpl(tryfirst=True)  # before xdist turns the group marks into the '@group' part of the test id
def pytest_collection_modifyitems(config, items):
    """Keep the multilevel tests on one pytest-xdist worker (CI runs ``-n auto --dist loadgroup``).

    Most of them launch 2-4 MPI ranks. One worker per core each launching its own
    4-rank job puts several times more busy-polling processes on the runner than it
    has cores: on the 4-core macOS Intel runner 4-rank tests that take seconds
    elsewhere took up to 17 minutes and four timed out at their 300 s limit, while
    the same directory run sequentially (as it was before the suite went to xdist)
    took minutes. One group means one worker, so at most one of these MPI jobs runs
    at a time, next to the single-process tests on the other workers."""
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
