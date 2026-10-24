"""Every DataItem the XDMF writers produce declares the type of the array it names.

The mesh writers in Archiver.py and MeshTools.py used to declare ``Int``
(4 bytes) and ``Float``/``8`` whatever they wrote, and a dozen ``Float``
items declared no ``Precision`` at all, which XDMF reads as 4 bytes, over
8-byte data. Each declaration now comes from the array, or from what the
array is made from, through :func:`proteus.Archiver.xdmf_number_type`.

This is the first step of C33 (numeric types). Proteus cannot yet make a
float32 mesh, int64 connectivity or narrow material types, so these tests
fabricate them and push them through the real writers and a real archive.
That pins the format before the solver can produce such arrays, instead of
finding out afterwards with archives already written.
"""
import os

import numpy as np
import pytest

h5py = pytest.importorskip("h5py")
from ymf.archive import YmfArchiveError, inline_domain, load_array, read_ymf

from proteus import Comm
from proteus import Archiver, MeshTools
from proteus.Archiver import xdmf_number_type


@pytest.mark.parametrize("dtype, declared", [
    ("int32", ("Int", "4")),
    ("int64", ("Int", "8")),
    ("int16", ("Int", "2")),
    ("uint8", ("UInt", "1")),
    ("float32", ("Float", "4")),
    ("float64", ("Float", "8")),
    ("bool", ("UChar", "1")),
])
def test_the_declared_type_is_the_dtype(dtype, declared):
    expected = {"DataType": declared[0], "Precision": declared[1]}
    assert xdmf_number_type(np.dtype(dtype)) == expected
    assert xdmf_number_type(np.zeros(3, dtype)) == expected


def test_a_computed_array_is_declared_as_numpy_types_the_expression():
    # offset + l2g, as the global_sync DG writers compute their elements
    offsets, l2g = np.zeros(3, "int64"), np.zeros((2, 3), "int32")
    assert xdmf_number_type(offsets, l2g) == {"DataType": "Int", "Precision": "8"}
    assert (offsets[0] + l2g).dtype == np.int64
    # a Python int does not widen the array
    assert xdmf_number_type(l2g, 5) == {"DataType": "Int", "Precision": "4"}


def test_a_dtype_with_no_xdmf_type_is_refused_not_guessed():
    with pytest.raises(YmfArchiveError):
        xdmf_number_type(np.zeros(2, "complex128"))


def test_topology_and_geometry_cannot_be_written_without_a_type():
    ar = Archiver.AR_base.__new__(Archiver.AR_base)
    ar.dataItemFormat, ar.hdfFile, ar.hdfFilename = "HDF", object(), "out.h5"
    grid = Archiver.ArchiveCollection("Mesh").new_grid(0.0, 0)
    with pytest.raises(TypeError):
        ar.write_topology(grid, "Triangle", 2, [2, 3], "elements", "elements")
    with pytest.raises(TypeError):
        ar.write_geometry(grid, [4, 3], "nodes", "nodes")
    item = ar.write_geometry(grid, [4, 3], "nodes", "nodes", dtype="float32")
    assert (item["data_type"], item["precision"]) == ("Float", 4)
    assert grid["geometry"]["data_item"] is item


# --------------------------------------------------------------------------
# through the real writers and a real archive
# --------------------------------------------------------------------------

NODES = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0]], dtype="float32")
ELEMENTS = np.array([[0, 1, 2], [1, 3, 2]], dtype="int64")


def fabricated_mesh():
    """A two-triangle mesh with every array in a type proteus cannot make yet."""
    mesh = MeshTools.Mesh()
    mesh.nodeArray = NODES.copy()
    mesh.elementNodesArray = ELEMENTS.copy()
    mesh.nElements_owned = mesh.nElements_global = 2
    mesh.nNodes_owned = mesh.nNodes_global = 4
    mesh.nNodes_element = 3
    mesh.nodeMaterialTypes = np.array([1, 1, 2, 2], dtype="int16")
    mesh.elementMaterialTypes = np.array([3, 4], dtype="uint8")
    mesh.globalMesh = None
    return mesh


class DGP1DofMap(object):
    """What writeMeshXdmf_DGP1Lagrange reads from a DG P1 dof map."""
    def __init__(self, mesh):
        self.l2g = np.arange(mesh.nElements_global * mesh.nNodes_element,
                             dtype="int64").reshape(mesh.nElements_global, -1)
        self.nDOF = self.l2g.size


@pytest.fixture
def archive(tmp_path):
    Comm.init()
    return Archiver.XdmfArchive(str(tmp_path), "dtypes", global_sync=False)


def written_arrays(path):
    with h5py.File(path, "r") as f:
        return {name: f[name][()] for name in f
                if isinstance(f[name], h5py.Dataset) and f[name].dtype.kind != "S"}


def items_of(domain):
    for collection in domain["time_collections"]:
        for step in collection["data"]:
            for grid in step.get("spatial_collection", [step]):
                yield grid["topology"]["data_item"]
                yield grid["geometry"]["data_item"]
                for attribute in grid["attributes"]:
                    yield attribute["data_item"]


def check_round_trip(ar, tmp_path):
    """Every DataItem declares its dataset's type, and reads back unchanged."""
    ar.sync()
    ar.close()
    arrays = written_arrays(str(tmp_path / "dtypes.h5"))
    for domain in (Archiver.readArchiveDomain("dtypes", str(tmp_path)),
                   read_ymf(str(tmp_path / "dtypes.ymf"))[0]):
        items = list(items_of(domain))
        assert items
        for item in items:
            dataset = item["data"].rsplit(":/", 1)[1]
            written = arrays[dataset]
            assert xdmf_number_type(written) == {"DataType": item["data_type"],
                                                 "Precision": str(item["precision"])}, dataset
        # the values copied into the document come back as the declared
        # type: a declaration that disagreed with the data would change them
        inlined = list(items_of(inline_domain(domain, str(tmp_path))))
        assert len(inlined) == len(items)
        for item, copied in zip(items, inlined):
            written = arrays[item["data"].rsplit(":/", 1)[1]]
            back = load_array(copied)
            assert back.dtype == written.dtype, item["data"]
            np.testing.assert_array_equal(back.reshape(written.shape), written)
    return arrays


def test_the_spatial_mesh_keeps_its_types(archive, tmp_path):
    mesh = fabricated_mesh()
    mesh.writeMeshXdmf(archive, "Spatial_Domain", init=True, meshChanged=True)
    arrays = check_round_trip(archive, tmp_path)
    assert arrays["nodes0Spatial_Domain0"].dtype == np.float32
    assert arrays["elements0Spatial_Domain0"].dtype == np.int64
    assert arrays["nodeMaterialTypes_p0_t0"].dtype == np.int16
    assert arrays["elementMaterialTypes_p0_t0"].dtype == np.uint8


def test_a_dg_mesh_keeps_its_types(archive, tmp_path):
    # the DG nodes are a temporary filled from mesh.nodeArray, and were
    # always float64; they now take the mesh's type
    mesh = fabricated_mesh()
    writer = Archiver.XdmfWriter()
    writer.writeMeshXdmf_DGP1Lagrange(archive, "u", mesh, 2, DGP1DofMap(mesh),
                                      None, init=True, meshChanged=True)
    arrays = check_round_trip(archive, tmp_path)
    assert arrays["elements0_dgp1_Lagrange0"].dtype == np.int64
    assert arrays["nodes0_dgp1_Lagrange0"].dtype == np.float32


def test_quadrature_points_keep_their_type(archive, tmp_path):
    # the point numbering is generated by the writer, the points are not
    mesh = fabricated_mesh()
    x = np.zeros((2, 4, 3), dtype="float32")
    writer = Archiver.XdmfWriter()
    writer.writeMeshXdmf_elementQuadrature(archive, mesh, 2, x,
                                           init=True, meshChanged=True)
    arrays = check_round_trip(archive, tmp_path)
    assert arrays["nodes0_elementQuadrature0"].dtype == np.float32
    assert (arrays["elements0_elementQuadrature0"].dtype
            == Archiver.GENERATED_INDEX_DTYPE)
