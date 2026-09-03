"""Regression tests for calls left behind by the PyTables-to-h5py migration.

Several places kept calling PyTables methods on h5py objects long after the
switch. ``h5py.File`` has no ``createArray``, ``createGroup`` or
``get_node``, and ``h5py.Dataset`` lost ``.value`` in h5py 3.0, so each of
these raised ``AttributeError`` on every call. None of them was covered by
a test, which is why they survived:

* ``proteus/AuxiliaryVariables.py`` -- ``hdfFile.createArray`` in
  ``PressureProfile``, twice.
* ``proteus/FemTools.py`` -- ``hdfFileGlb.get_node`` in the hot-start
  reader (removed entirely) and ``Dataset.value`` in the per-subdomain
  read for the C0P2 space.
* ``test/POD/read_hdf5.py`` and ``test/POD/deim_utils.py`` --
  ``hdfFile.get_node(label).read()``, called with an h5py file by every
  caller.
* ``scripts/extractSolution.py`` -- a whole script on PyTables, whose
  camelCase names PyTables itself removed in 3.0, and which imported an
  undeclared dependency besides.
* ``scripts/readFun.py`` -- ``outputGrid.get_node``, unrelated to PyTables
  but the same kind of stale name: the method is ``getNode``.

These tests are cheap and shallow on purpose. The point is that each call
path executes at all.
"""

import os
import sys

import numpy as np
import pytest

h5py = pytest.importorskip("h5py")

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPTS = os.path.join(REPO_ROOT, "scripts")
POD = os.path.join(REPO_ROOT, "test", "POD")


# --------------------------------------------------------------------------
# AuxiliaryVariables.PressureProfile
# --------------------------------------------------------------------------


class _StubArchive:
    def __init__(self, hdf):
        self.hdfFile = hdf


def test_a_profile_is_written_as_an_h5py_dataset(tmp_path):
    from proteus.AuxiliaryVariables import _write_profile

    path = tmp_path / "a.h5"
    with h5py.File(path, "w") as f:
        _write_profile(_StubArchive(f), "theta", [1.0, 2.0, 3.0])
    with h5py.File(path, "r") as f:
        np.testing.assert_array_equal(f["theta"][:], [1.0, 2.0, 3.0])


def test_a_profile_write_is_idempotent(tmp_path):
    # calculate() is called once per timestep and names datasets by tCount,
    # but re-running a step must not fail on an existing name
    from proteus.AuxiliaryVariables import _write_profile

    path = tmp_path / "a.h5"
    with h5py.File(path, "w") as f:
        _write_profile(_StubArchive(f), "pressure0", [1.0])
        _write_profile(_StubArchive(f), "pressure0", [2.0, 3.0])
        np.testing.assert_array_equal(f["pressure0"][:], [2.0, 3.0])


def test_an_empty_profile_is_skipped_not_crashed(tmp_path):
    # No node matching the flag is normal, and h5py cannot make a dataset
    # from an empty untyped list.
    from proteus.AuxiliaryVariables import _write_profile

    path = tmp_path / "a.h5"
    with h5py.File(path, "w") as f:
        _write_profile(_StubArchive(f), "theta", [])
        assert "theta" not in f


def test_no_hdf_file_is_a_no_op():
    from proteus.AuxiliaryVariables import _write_profile

    _write_profile(_StubArchive(None), "theta", [1.0])


def _attribute_names_used(module_path):
    """Every attribute name the module actually accesses.

    AST rather than text search: the code that replaced these calls
    documents what it replaced, so grepping the source finds the names in
    prose and reports a failure that is not one.
    """
    import ast

    tree = ast.parse(open(module_path).read())
    return {node.attr for node in ast.walk(tree)
            if isinstance(node, ast.Attribute)}


def test_pressure_profile_no_longer_calls_createarray():
    import proteus.AuxiliaryVariables as module

    used = _attribute_names_used(module.__file__)
    assert "createArray" not in used
    assert "create_dataset" in used


# --------------------------------------------------------------------------
# the POD helpers
# --------------------------------------------------------------------------


@pytest.fixture
def pod_file(tmp_path):
    path = tmp_path / "pod.h5"
    with h5py.File(path, "w") as f:
        f["u0"] = np.arange(6, dtype="d")
    return path


@pytest.mark.parametrize("module_name", ["read_hdf5", "deim_utils"])
def test_pod_readers_accept_an_h5py_file(pod_file, module_name):
    """Every caller passes archive.hdfFile, which is an h5py.File."""
    sys.path.insert(0, POD)
    try:
        module = __import__(module_name)
    finally:
        sys.path.remove(POD)
    with h5py.File(pod_file, "r") as f:
        np.testing.assert_array_equal(
            module.read_from_hdf5(f, "/u0"), np.arange(6))


@pytest.mark.parametrize("module_name", ["read_hdf5", "deim_utils"])
def test_pod_readers_apply_a_dof_map(pod_file, module_name):
    sys.path.insert(0, POD)
    try:
        module = __import__(module_name)
    finally:
        sys.path.remove(POD)
    with h5py.File(pod_file, "r") as f:
        got = module.read_from_hdf5(f, "/u0", dof_map=np.array([5, 4, 3]))
    np.testing.assert_array_equal(got, [5.0, 4.0, 3.0])


# --------------------------------------------------------------------------
# scripts/extractSolution.py, end to end
# --------------------------------------------------------------------------


def test_extract_solution_splits_and_composes(tmp_path, monkeypatch):
    """Split per-rank archives apart, then compose them back with XMF.

    Runs the whole script rather than checking it imports: the conversion
    touched file opening, group creation, dataset creation and reading, and
    only running it exercises all four.
    """
    monkeypatch.chdir(tmp_path)
    size, step = 2, 0
    for proc in range(size):
        with h5py.File("sol%d.h5" % proc, "w") as f:
            f["elementsSpatial_Domain%d" % step] = np.arange(8, dtype="i").reshape(2, 4)
            f["nodesSpatial_Domain%d" % step] = np.zeros((5, 3), dtype="d")
            for name in ("u", "v", "w", "p"):
                f["%s%d" % (name, step)] = np.zeros(5, dtype="d")
        with h5py.File("phi%d.h5" % proc, "w") as f:
            f["phid%d" % step] = np.zeros(5, dtype="d")

    sys.path.insert(0, SCRIPTS)
    try:
        import extractSolution
    finally:
        sys.path.remove(SCRIPTS)
    extractSolution.splitH5all("sol", "phi", size, step, step, 1)

    with h5py.File("solution.%d.h5" % step, "r") as f:
        assert sorted(f.keys()) == ["p0", "p1"]
        assert sorted(f["p0"].keys()) == [
            "elements", "nodes", "p", "phid", "u", "v", "w"]
        assert f["p0"]["elements"].shape == (2, 4)
        assert f["p1"]["nodes"].shape == (5, 3)

    from xml.etree.ElementTree import parse

    root = parse("solution.xmf").getroot()
    assert root.findall(".//Grid"), "composed .xmf has no grids"


def test_extract_solution_no_longer_imports_pytables():
    """PyTables is not declared in any environment file.

    So importing it here was an undeclared dependency on top of calling
    names PyTables 3.0 removed.
    """
    import ast

    path = os.path.join(SCRIPTS, "extractSolution.py")
    tree = ast.parse(open(path).read())
    imported = {alias.name.split(".")[0]
                for node in ast.walk(tree)
                if isinstance(node, (ast.Import, ast.ImportFrom))
                for alias in node.names}
    assert "tables" not in imported
    assert "h5py" in imported
    used = _attribute_names_used(path)
    assert not {"createArray", "createGroup", "get_node"} & used, used


# --------------------------------------------------------------------------
# scripts/readFun.py
# --------------------------------------------------------------------------


def test_rectangular_grid_node_accessor_is_spelled_correctly():
    """readFun.py called get_node; RectangularGrid provides getNode."""
    used = _attribute_names_used(os.path.join(SCRIPTS, "readFun.py"))
    assert "getNode" in used
    assert "get_node" not in used
