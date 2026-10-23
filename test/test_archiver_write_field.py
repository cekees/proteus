"""Tests for AR_base.write_field, the attribute-and-data-item chokepoint.

write_field replaces an eight-line block that was hand-written at ~50 call
sites. It used to build an XDMF ``Attribute``/``DataItem`` pair; it now
builds the ymf attribute dict the archive stores, and these tests pin that
dict -- the data reference, the declared dimensions and type, the sidecar
path in text mode -- so the conversion of each call site is a refactor and
not a rewrite.

They build a stub archive rather than a real XdmfArchive: the real one opens
an MPI-mode HDF5 file in its constructor, which is not what is under test
here.
"""

import numpy as np
import pytest

from proteus.Archiver import AR_base, ArchiveCollection


class StubComm(object):
    def __init__(self, rank=0):
        self._rank = rank

    def rank(self):
        return self._rank


class StubArchive(AR_base):
    """AR_base with just the attributes write_field touches.

    Bypasses __init__ deliberately -- see the module docstring.
    """

    def __init__(self, hdf=True, global_sync=False, rank=0, size=1):
        self.global_sync = global_sync
        self.size = size
        self.comm = StubComm(rank)
        self.hdfFile = object() if hdf else None
        self.hdfFilename = "out.h5"
        self.dataItemFormat = "HDF" if hdf else "XML"
        self.driver, self.inline = "mpio", False
        self.textDataDir = "out_Data"
        self.sync_calls = []
        self.async_calls = []
        self.saved = []

    def create_dataset_sync(self, name, offsets, data):
        self.sync_calls.append((name, np.asarray(offsets).copy(), np.asarray(data).copy()))

    def create_dataset_async(self, name, data):
        self.async_calls.append((name, np.asarray(data).copy()))


@pytest.fixture(autouse=True)
def _no_real_savetxt(monkeypatch):
    """Capture numpy.savetxt so the text path doesn't touch the filesystem."""
    import proteus.Archiver as archiver_module

    saved = []

    def fake_savetxt(path, arr, *a, **kw):
        saved.append((path, np.asarray(arr).copy()))

    monkeypatch.setattr(archiver_module.numpy, "savetxt", fake_savetxt)
    return saved


def grid():
    """A grid as write_grid starts one: nothing on it yet."""
    return ArchiveCollection("Mesh Spatial_Domain").new_grid(0.0, 0)


def the_attribute(g):
    (attr,) = g["attributes"]
    return attr


def the_item(g):
    return the_attribute(g)["data_item"]


# --------------------------------------------------------------------------
# equivalence with what ymf's own builders produce
# --------------------------------------------------------------------------


def test_write_field_builds_the_attribute_ymf_would():
    """The dict is the one ymf.archive.attribute/data_item_for build.

    That is what makes the archive one representation with one writer:
    a field written by proteus is indistinguishable from one described
    by hand with ymf's builders.
    """
    from ymf.archive import attribute, data_item_for

    dof = np.zeros(17, dtype="float64")
    g = grid()
    StubArchive().write_field(g, "u", dof, 3, dimensions=[17])
    expected = attribute("u", data_item_for(dof, data="out.h5:/u_p0_t3", dimensions=[17]),
                         center="Node", attribute_type="Scalar")
    assert the_attribute(g) == expected


def test_the_dataset_name_matches_the_old_convention():
    ar = StubArchive(rank=2)
    ar.write_field(grid(), "u", np.zeros(4), 7, dimensions=[4])
    (name, _), = ar.async_calls
    assert name == "u_p2_t7"


def test_the_hdf5_reference_matches_the_dataset_name():
    ar = StubArchive(rank=1)
    g = grid()
    ar.write_field(g, "vof", np.zeros(4), 2, dimensions=[4])
    assert the_item(g)["data"] == "out.h5:/vof_p1_t2"
    assert ar.async_calls[0][0] == "vof_p1_t2"


def test_the_item_returned_is_the_one_attached():
    g = grid()
    item = StubArchive().write_field(g, "u", np.zeros(4), 0)
    assert item is the_item(g)


def test_fields_accumulate_on_the_grid_in_order():
    g = grid()
    ar = StubArchive()
    ar.write_field(g, "p", np.zeros(4), 0)
    ar.write_field(g, "u", np.zeros(4), 0)
    assert [a["name"] for a in g["attributes"]] == ["p", "u"]


# --------------------------------------------------------------------------
# the attribute itself
# --------------------------------------------------------------------------


def test_defaults_are_node_centered_scalar():
    g = grid()
    StubArchive().write_field(g, "u", np.zeros(3), 0, dimensions=[3])
    attr = the_attribute(g)
    assert attr["attribute_type"] == "Scalar"
    assert attr["center"] == "Node"


@pytest.mark.parametrize("center", ["Node", "Cell", "Face"])
@pytest.mark.parametrize("rank", ["Scalar", "Vector", "Tensor"])
def test_center_and_rank_reach_the_attribute(center, rank):
    g = grid()
    StubArchive().write_field(g, "u", np.zeros(3), 0, center=center,
                              rank=rank, dimensions=[3])
    attr = the_attribute(g)
    assert (attr["center"], attr["attribute_type"]) == (center, rank)


def test_multi_dimensional_dimensions_are_kept():
    g = grid()
    StubArchive().write_field(g, "vel", np.zeros((5, 3)), 0, rank="Vector")
    assert list(the_item(g)["dimensions"]) == [5, 3]


def test_dimensions_default_to_the_array_shape():
    g = grid()
    StubArchive().write_field(g, "u", np.zeros(11), 0)
    assert list(the_item(g)["dimensions"]) == [11]


# --------------------------------------------------------------------------
# dtype-derived DataType/Precision -- a correctness fix over the old code
# --------------------------------------------------------------------------


def test_float64_is_described_as_eight_byte_float():
    g = grid()
    StubArchive().write_field(g, "u", np.zeros(3, dtype="float64"), 0)
    item = the_item(g)
    assert (item["data_type"], item["precision"]) == ("Float", 8)


def test_float32_is_described_as_four_byte_float():
    # The old call sites hardcoded Precision="8". A float32 field was
    # therefore described to consumers as 8-byte, which reads as garbage.
    g = grid()
    StubArchive().write_field(g, "u", np.zeros(3, dtype="float32"), 0)
    item = the_item(g)
    assert (item["data_type"], item["precision"]) == ("Float", 4)


def test_integer_arrays_are_described_as_int():
    g = grid()
    StubArchive().write_field(g, "mat", np.zeros(3, dtype="int32"), 0, center="Cell")
    item = the_item(g)
    assert (item["data_type"], item["precision"]) == ("Int", 4)


# --------------------------------------------------------------------------
# global_sync
# --------------------------------------------------------------------------


def test_global_sync_declares_the_global_size_not_the_slice_size():
    # The DataItem describes the assembled global array; each rank writes
    # only what it owns.
    ar = StubArchive(global_sync=True)
    g = grid()
    dof = np.arange(10, dtype="d")
    offsets = np.array([0, 4, 10])
    ar.write_field(g, "u", dof, 1, dimensions=[10], sync_offsets=offsets,
                   sync_data=dof[:4])
    assert list(the_item(g)["dimensions"]) == [10]
    (name, got_offsets, got_data), = ar.sync_calls
    assert name == "u_t1"          # no _p<rank> in the sync convention
    np.testing.assert_array_equal(got_offsets, offsets)
    np.testing.assert_array_equal(got_data, dof[:4])


def test_global_sync_uses_the_unranked_dataset_name():
    ar = StubArchive(global_sync=True, rank=3)
    g = grid()
    ar.write_field(g, "p", np.zeros(4), 5, dimensions=[4], sync_offsets=np.array([0, 4]),
                   sync_data=np.zeros(4))
    assert the_item(g)["data"] == "out.h5:/p_t5"
    assert not ar.async_calls


# --------------------------------------------------------------------------
# the text fallback -- and the bug class it makes inexpressible
# --------------------------------------------------------------------------


def test_text_mode_references_its_own_sidecar(_no_real_savetxt):
    ar = StubArchive(hdf=False)
    g = grid()
    ar.write_field(g, "u", np.zeros(4), 2)
    item = the_item(g)
    assert item["format"] == "XML"
    assert item["include"] == "./out_Data/u2.txt"
    assert "data" not in item


def test_text_mode_writes_the_sidecar_file(_no_real_savetxt):
    ar = StubArchive(hdf=False)
    dof = np.arange(4, dtype="d")
    ar.write_field(grid(), "u", dof, 2)
    (path, arr), = _no_real_savetxt
    assert path == "out_Data/u2.txt"
    np.testing.assert_array_equal(arr, dof)


def test_two_fields_in_text_mode_each_reference_their_own_sidecar(_no_real_savetxt):
    # This is the bug at Archiver.py:1475 made unrepresentable: there, the
    # include for a second field was attached to the first field's
    # DataItem, leaving one field with no data reference and the other with
    # two. The reference is now part of the data item write_field creates.
    ar = StubArchive(hdf=False)
    g = grid()
    ar.write_field(g, "u", np.zeros(4), 0)
    ar.write_field(g, "u_dof", np.zeros(4), 0, center="Face")
    items = [a["data_item"] for a in g["attributes"]]
    assert [item["include"] for item in items] == ["./out_Data/u0.txt",
                                                   "./out_Data/u_dof0.txt"]


def test_text_mode_with_global_sync_is_refused():
    # The old code asserted this too; keep it an assertion rather than
    # silently writing an archive with no data in it.
    ar = StubArchive(hdf=False, global_sync=True)
    with pytest.raises(AssertionError, match="text heavy data"):
        ar.write_field(grid(), "u", np.zeros(4), 0, dimensions=[4])


# --------------------------------------------------------------------------
# textDataDir is conditional on how the archive was opened
# --------------------------------------------------------------------------


def test_hdf_mode_never_touches_text_data_dir():
    """AR_base only sets textDataDir when useTextArchive=True.

    Regression test: an earlier version of write_field built the sidecar
    path unconditionally, which raised AttributeError on every HDF5 write.
    The stub above always defines textDataDir, so it did not catch that --
    hence a stub that behaves like a real HDF5 archive and lacks it.
    """
    ar = StubArchive()
    del ar.textDataDir
    g = grid()
    ar.write_field(g, "u", np.zeros(4), 0, dimensions=[4])
    assert the_item(g)["data"] == "out.h5:/u_p0_t0"


def test_hdf_mode_with_global_sync_never_touches_text_data_dir():
    ar = StubArchive(global_sync=True)
    del ar.textDataDir
    g = grid()
    ar.write_field(g, "u", np.zeros(4), 0, dimensions=[4],
                   sync_offsets=np.array([0, 4]), sync_data=np.zeros(4))
    assert the_item(g)["data"] == "out.h5:/u_t0"


# --------------------------------------------------------------------------
# dataset / text_stem overrides, and a reference without a write
# --------------------------------------------------------------------------


def test_an_explicit_dataset_name_overrides_the_convention():
    # writeFunctionXdmf_DGP2Lagrange predates the _p<rank>_t<tCount>
    # convention and must keep naming its dataset <name><tCount>, or
    # converting it would rename datasets inside existing archives.
    ar = StubArchive(rank=2)
    g = grid()
    ar.write_field(g, "u", np.zeros(4), 7, dimensions=[4], dataset="u7")
    assert the_item(g)["data"] == "out.h5:/u7"
    assert ar.async_calls[0][0] == "u7"


def test_an_explicit_dataset_name_also_applies_under_global_sync():
    ar = StubArchive(global_sync=True)
    g = grid()
    ar.write_field(g, "u", np.zeros(4), 7, dimensions=[4], dataset="u7",
                   sync_offsets=np.array([0, 4]), sync_data=np.zeros(4))
    assert ar.sync_calls[0][0] == "u7"


def test_text_stem_overrides_the_sidecar_filename(_no_real_savetxt):
    ar = StubArchive(hdf=False)
    g = grid()
    ar.write_field(g, "u", np.zeros(4), 3, text_stem="custom")
    assert the_item(g)["include"] == "./out_Data/custom.txt"
    assert _no_real_savetxt[0][0] == "out_Data/custom.txt"


def test_a_reference_can_be_attached_without_writing(_no_real_savetxt):
    # the mesh writers reference the local-to-global maps on every pass
    # but write them only when the mesh changed
    ar = StubArchive(rank=1)
    g = grid()
    ar.write_field(g, "NodeMapL2G", np.zeros(4, dtype="int32"), 2,
                   dataset="nodeMapL2G1Spatial_Domain2", write_data=False)
    assert the_item(g)["data"] == "out.h5:/nodeMapL2G1Spatial_Domain2"
    assert not ar.async_calls and not ar.sync_calls and not _no_real_savetxt


# --------------------------------------------------------------------------
# the declared-vs-actual dimension check, and where it must not apply
# --------------------------------------------------------------------------


def test_a_declared_size_that_disagrees_with_the_data_is_rejected_in_serial():
    """The check that caught the phi_s corruption must stay on in serial.

    A DataItem claiming more values than the dataset behind it produces an
    archive that opens and shows garbage. With one rank there is no reason
    for global and local to differ, so a disagreement is a real defect.
    """
    from ymf.archive import YmfArchiveError

    ar = StubArchive(global_sync=True, size=1)
    with pytest.raises(YmfArchiveError, match="dimensions"):
        ar.write_field(grid(), "phi_s", np.zeros(25), 0, dimensions=[81],
                       sync_offsets=np.array([0, 25]), sync_data=np.zeros(25))


def test_a_larger_declared_size_is_accepted_on_a_parallel_collective_write():
    """...and must be off once ranks genuinely hold slices.

    Under global_sync with several ranks the DataItem covers the assembled
    global array while this rank holds its own piece, so declared > local
    is correct by construction. Checking it here would reject every correct
    parallel write -- which it did, until the real MPI tests caught it.
    """
    ar = StubArchive(global_sync=True, size=4)
    g = grid()
    ar.write_field(g, "u", np.zeros(6), 0, dimensions=[26],
                   sync_offsets=np.array([0, 6, 13, 19, 26]),
                   sync_data=np.zeros(6))
    assert list(the_item(g)["dimensions"]) == [26]


def test_the_per_rank_path_keeps_the_check_at_any_size():
    # Without global_sync each rank's DataItem describes exactly its own
    # array, so a disagreement is a defect however many ranks there are.
    from ymf.archive import YmfArchiveError

    ar = StubArchive(global_sync=False, size=8)
    with pytest.raises(YmfArchiveError, match="dimensions"):
        ar.write_field(grid(), "u", np.zeros(10), 0, dimensions=[99])


# --------------------------------------------------------------------------
# the grid builders: write_grid, write_topology, write_geometry
# --------------------------------------------------------------------------


def test_write_grid_starts_the_collections_current_grid():
    ar = StubArchive()
    collection = ArchiveCollection("Mesh_c0p2_Lagrange")
    g = ar.write_grid(collection, "Grid_c0p2_Lagrange", 0.5, 3)
    assert collection.grid is g
    assert collection.is_current(0.5) and not collection.is_current(0.75)
    assert g["name"] == "Grid_c0p2_Lagrange"
    assert g["attributes"] == [] and g["topology"] is None


def test_the_collection_forgets_its_grid_when_the_step_is_stored():
    collection = ArchiveCollection("Mesh Spatial_Domain")
    collection.new_grid(0.0, 0)
    collection.clear()
    assert collection.grid is None and not collection.is_current(0.0)


def test_topology_and_geometry_reference_the_named_datasets():
    ar = StubArchive(global_sync=True)
    g = grid()
    topo = ar.write_topology(g, "Tri_6", 2, [2, 6], "elements_c0p2_Lagrange0",
                             "elements_c0p2_Lagrange0", dtype=np.zeros(1, "int32"))
    geo = ar.write_geometry(g, [9, 3], "nodes_c0p2_Lagrange0",
                            "nodes_c0p2_Lagrange0", dtype=np.zeros(1, "float64"))
    assert g["topology"]["type"] == "Tri_6"
    assert g["topology"]["number_of_elements"] == 2
    assert g["topology"]["data_item"] is topo
    assert (topo["data"], topo["data_type"], topo["precision"]) == \
        ("out.h5:/elements_c0p2_Lagrange0", "Int", 4)
    assert g["geometry"]["type"] == "XYZ" and g["geometry"]["data_item"] is geo
    assert (geo["data"], geo["data_type"], geo["precision"]) == \
        ("out.h5:/nodes_c0p2_Lagrange0", "Float", 8)
    # the builders describe; the caller writes the arrays
    assert not ar.sync_calls and not ar.async_calls


def test_a_polyvertex_topology_says_how_many_nodes_per_element():
    g = grid()
    StubArchive().write_topology(g, "Polyvertex", 4, [4, 8], "elements_q0", "elements_q0",
                                 nodes_per_element=8, dtype=np.dtype("int32"))
    assert g["topology"]["nodes_per_element"] == 8


def test_mesh_data_items_in_text_mode_reference_the_sidecar():
    ar = StubArchive(hdf=False)
    g = grid()
    ar.write_topology(g, "Triangle", 2, [2, 3], "elements0", "elementsSpatial_Domain",
                      dtype=np.dtype("int32"))
    assert g["topology"]["data_item"]["include"] == "./out_Data/elementsSpatial_Domain.txt"


def test_a_computed_connectivity_is_declared_as_numpy_types_the_expression():
    # offset + l2g, as the DG writers compute their elements
    g = grid()
    StubArchive().write_topology(g, "Triangle", 2, [2, 3], "e", "e",
                                 dtype=(np.zeros(1, "int64"), np.zeros(1, "int32")))
    assert g["topology"]["data_item"]["precision"] == 8


def test_a_grid_the_writers_build_is_a_grid_ymf_accepts():
    """The whole point: the dict goes into the archive as it is."""
    from ymf.archive import canonicalize_domain, new_domain, validate_domain

    ar = StubArchive()
    collection = ArchiveCollection("Mesh Spatial_Domain")
    g = ar.write_grid(collection, None, 0.0, 0)
    ar.write_topology(g, "Triangle", 2, [2, 3], "elements0", "elements",
                      dtype=np.dtype("int32"))
    ar.write_geometry(g, [4, 3], "nodes0", "nodes", dtype=np.dtype("float64"))
    ar.write_field(g, "u", np.zeros(4), 0)
    domain = new_domain("Mesh Spatial_Domain")
    domain["time_collections"][0]["data"].append(
        {"time": collection.time, "topology": g["topology"],
         "geometry": g["geometry"], "attributes": g["attributes"]})
    validate_domain(canonicalize_domain(domain))
