"""Fixtures for the multilevel tests."""
import pytest


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "global_hierarchy: tests the global hierarchy builder itself (refinement, "
        "partitionLevels, the global node parents and vertex interpolation), so it runs "
        "with hierarchies generated in pieces switched off")


def _globally(node):
    return node.get_closest_marker('global_hierarchy') is not None


@pytest.fixture(autouse=True, scope='module')
def _global_hierarchy_module(request):
    """Build hierarchies globally for modules marked global_hierarchy, even when
    PROTEUS_GENERATE_REGULAR_GRIDS=1 runs the rest of the suite on hierarchies
    generated in pieces. Module-scoped, so that a module's own module-scoped
    fixtures, which build their hierarchies before any test's fixtures run,
    see it too."""
    if not _globally(request.node):
        yield
        return
    from proteus import MeshTools
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(MeshTools.MultilevelMesh, 'generateRegularGrids', False)
        mp.setenv('PROTEUS_GENERATE_REGULAR_GRIDS', '0')
        yield


@pytest.fixture(autouse=True)
def _global_hierarchy(request, monkeypatch):
    """The same for each test marked global_hierarchy -- here and in the MPI
    helpers the test launches."""
    if _globally(request.node):
        from proteus import MeshTools
        monkeypatch.setattr(MeshTools.MultilevelMesh, 'generateRegularGrids', False)
        monkeypatch.setenv('PROTEUS_GENERATE_REGULAR_GRIDS', '0')
    yield
