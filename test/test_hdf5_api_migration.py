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


# --------------------------------------------------------------------------
# scripts/clearh5.py -- truncating an archive
# --------------------------------------------------------------------------


def _write_archive(path_base, n_steps=5, collections=("Mesh Spatial_Domain",
                                                      "Mesh_c0p2_Lagrange")):
    """A .ymf + .h5 pair shaped like a real proteus archive."""
    from ymf.archive import (add_collection, add_uniform_step, attribute,
                             data_item, geometry, new_domain, topology,
                             write_ymf)

    domain = None
    datasets = {}
    for ci, name in enumerate(collections):
        suffix = "" if ci == 0 else "_c0p2"
        if domain is None:
            domain = new_domain(name)
        else:
            add_collection(domain, name)
        for step in range(n_steps):
            e = "elements%s%d" % (suffix, step)
            n = "nodes%s%d" % (suffix, step)
            u = "u%s_t%d" % (suffix, step)
            datasets[e] = np.zeros((2, 3), dtype="i")
            datasets[n] = np.zeros((4, 3), dtype="d")
            datasets[u] = np.zeros(4, dtype="d")
            add_uniform_step(
                domain, float(step),
                topology("Triangle", 2,
                         data_item([2, 3], "%s.h5:/%s" % (path_base, e),
                                   data_type="Int")),
                geometry(data_item([4, 3], "%s.h5:/%s" % (path_base, n),
                                   precision=8)),
                [attribute("u", data_item([4], "%s.h5:/%s" % (path_base, u),
                                          precision=8))],
                collection=ci)
    write_ymf(domain, path_base + ".ymf")
    with h5py.File(path_base + ".h5", "w") as f:
        for name, data in datasets.items():
            f[name] = data
        f.attrs["ymf_archive_metadata_version"] = 2
        f.attrs["ymf_archive_collections"] = "\n".join(collections)
    return len(datasets)


@pytest.fixture
def clearh5_module():
    sys.path.insert(0, SCRIPTS)
    try:
        import clearh5
    finally:
        sys.path.remove(SCRIPTS)
    return clearh5


def test_clearh5_truncates_every_collection(tmp_path, monkeypatch, clearh5_module):
    """Truncation must apply to all collections, not just the first.

    The previous version removed grids from Domain[0] whichever collection
    they came from, so truncating an archive with a quadratic space
    corrupted it.
    """
    monkeypatch.chdir(tmp_path)
    _write_archive("run", n_steps=5)
    clearh5_module.clearh5("run", tCount=3)

    from ymf.archive import read_ymf

    domain, _ = read_ymf("run_clean.ymf")
    assert [(c["Name"], len(c["Data"])) for c in domain["TimeCollections"]] == [
        ("Mesh Spatial_Domain", 3), ("Mesh_c0p2_Lagrange", 3)]


def test_clearh5_keeps_only_the_referenced_datasets(tmp_path, monkeypatch,
                                                    clearh5_module):
    monkeypatch.chdir(tmp_path)
    total = _write_archive("run", n_steps=5)
    clearh5_module.clearh5("run", tCount=2)
    with h5py.File("run_clean.h5", "r") as f:
        kept = set(f.keys())
    # 2 collections x 2 steps x 3 datasets
    assert len(kept) == 12
    assert len(kept) < total
    assert "u_t4" not in kept and "u_t0" in kept


def test_clearh5_leaves_no_dangling_reference(tmp_path, monkeypatch,
                                              clearh5_module):
    """Every reference the truncated archive keeps must still resolve."""
    monkeypatch.chdir(tmp_path)
    _write_archive("run", n_steps=4)
    clearh5_module.clearh5("run", tCount=2)

    from ymf.archive import read_ymf

    domain, _ = read_ymf("run_clean.ymf")
    with h5py.File("run_clean.h5", "r") as f:
        for collection in domain["TimeCollections"]:
            for step in collection["Data"]:
                for grid in step.get("SpatialCollection", [step]):
                    nodes = [grid["Topology"], grid["Geometry"]]
                    nodes += grid.get("Attributes", [])
                    for node in nodes:
                        ref = node["DataItem"]["Data"].split(":/")[-1]
                        assert ref in f, ref


def test_clearh5_carries_the_archive_attributes_across(tmp_path, monkeypatch,
                                                       clearh5_module):
    # otherwise the truncated copy is a bag of datasets rather than a
    # readable archive
    monkeypatch.chdir(tmp_path)
    _write_archive("run", n_steps=3)
    clearh5_module.clearh5("run", tCount=2)
    with h5py.File("run_clean.h5", "r") as f:
        assert int(f.attrs["ymf_archive_metadata_version"]) == 2
        assert "Mesh_c0p2_Lagrange" in f.attrs["ymf_archive_collections"]


def test_clearh5_requires_a_positive_tcount(tmp_path, monkeypatch, clearh5_module):
    monkeypatch.chdir(tmp_path)
    _write_archive("run", n_steps=2)
    with pytest.raises(SystemExit, match="positive"):
        clearh5_module.clearh5("run", tCount=-1)


def test_clearh5_no_longer_matches_digits_out_of_dataset_names(clearh5_module):
    """It decided what to keep with int(re.search(r'\\d+', name).group()).

    That reads the first digit group of a name like ``u_p0_t12`` -- the
    rank, not the step -- and raises on any name with no digits at all.
    Datasets to keep now come from the archive's own references.
    """
    import ast
    import inspect

    tree = ast.parse(inspect.getsource(clearh5_module.clearh5))
    calls = [n for n in ast.walk(tree)
             if isinstance(n, ast.Attribute) and n.attr == "search"]
    assert not calls, "clearh5 still pattern-matches dataset names"
