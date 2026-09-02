"""Tests for AR_base.write_field, the Attribute+DataItem chokepoint.

write_field replaces an eight-line block that was hand-written at ~50 call
sites. These tests pin its XML against the block it replaces, so the
conversion of each call site is a refactor and not a rewrite.

They build a stub archive rather than a real XdmfArchive: the real one opens
an MPI-mode HDF5 file in its constructor, which is not what is under test
here.
"""

from xml.etree.ElementTree import Element, tostring

import numpy as np
import pytest

from proteus.Archiver import AR_base


class StubComm(object):
    def __init__(self, rank=0):
        self._rank = rank

    def rank(self):
        return self._rank


class StubArchive(AR_base):
    """AR_base with just the attributes write_field touches.

    Bypasses __init__ deliberately -- see the module docstring.
    """

    def __init__(self, hdf=True, global_sync=False, rank=0):
        self.global_sync = global_sync
        self.comm = StubComm(rank)
        self.hdfFile = object() if hdf else None
        self.hdfFilename = "out.h5"
        self.dataItemFormat = "HDF" if hdf else "XML"
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
    return Element("Grid", {"GridType": "Uniform"})


# --------------------------------------------------------------------------
# equivalence with the block write_field replaces
# --------------------------------------------------------------------------


def reference_block(ar, grid_elem, name, dof, tCount, n_declared):
    """The hand-written block, verbatim in structure, for comparison.

    This mirrors what FemTools.writeFunctionXdmf did before the conversion:
    an Attribute, a DataItem with Format/DataType/Precision/Dimensions, and
    the HDF5 reference as the DataItem's text.
    """
    from xml.etree.ElementTree import SubElement

    attribute = SubElement(grid_elem, "Attribute",
                           {"Name": name,
                            "AttributeType": "Scalar",
                            "Center": "Node"})
    values = SubElement(attribute, "DataItem",
                        {"Format": ar.dataItemFormat,
                         "DataType": "Float",
                         "Precision": "8",
                         "Dimensions": "%i" % (n_declared,)})
    values.text = (ar.hdfFilename + ":/" + name + "_p"
                   + repr(ar.comm.rank()) + "_t{0:d}".format(tCount))
    return values


def test_write_field_emits_the_same_xml_as_the_block_it_replaces():
    dof = np.zeros(17, dtype="float64")

    old_ar = StubArchive()
    old_grid = grid()
    reference_block(old_ar, old_grid, "u", dof, 3, 17)

    new_ar = StubArchive()
    new_grid = grid()
    new_ar.write_field(new_grid, "u", dof, 3, dimensions=[17])

    assert tostring(new_grid) == tostring(old_grid)


def test_the_dataset_name_matches_the_old_convention():
    ar = StubArchive(rank=2)
    ar.write_field(grid(), "u", np.zeros(4), 7, dimensions=[4])
    (name, _), = ar.async_calls
    assert name == "u_p2_t7"


def test_the_hdf5_reference_matches_the_dataset_name():
    ar = StubArchive(rank=1)
    g = grid()
    ar.write_field(g, "vof", np.zeros(4), 2, dimensions=[4])
    item = g.find("Attribute").find("DataItem")
    assert item.text == "out.h5:/vof_p1_t2"
    assert ar.async_calls[0][0] == "vof_p1_t2"


# --------------------------------------------------------------------------
# the Attribute itself
# --------------------------------------------------------------------------


def test_defaults_are_node_centered_scalar():
    g = grid()
    ar = StubArchive()
    ar.write_field(g, "u", np.zeros(3), 0, dimensions=[3])
    attr = g.find("Attribute")
    assert attr.attrib["AttributeType"] == "Scalar"
    assert attr.attrib["Center"] == "Node"


@pytest.mark.parametrize("center", ["Node", "Cell", "Face"])
@pytest.mark.parametrize("rank", ["Scalar", "Vector", "Tensor"])
def test_center_and_rank_reach_the_attribute(center, rank):
    g = grid()
    StubArchive().write_field(g, "u", np.zeros(3), 0, center=center,
                              rank=rank, dimensions=[3])
    attr = g.find("Attribute")
    assert (attr.attrib["Center"], attr.attrib["AttributeType"]) == (center, rank)


def test_multi_dimensional_dimensions_are_space_separated():
    g = grid()
    StubArchive().write_field(g, "vel", np.zeros((5, 3)), 0, rank="Vector")
    assert g.find("Attribute").find("DataItem").attrib["Dimensions"] == "5 3"


def test_dimensions_default_to_the_array_shape():
    g = grid()
    StubArchive().write_field(g, "u", np.zeros(11), 0)
    assert g.find("Attribute").find("DataItem").attrib["Dimensions"] == "11"


# --------------------------------------------------------------------------
# dtype-derived DataType/Precision -- a correctness fix over the old code
# --------------------------------------------------------------------------


def test_float64_is_described_as_eight_byte_float():
    g = grid()
    StubArchive().write_field(g, "u", np.zeros(3, dtype="float64"), 0)
    item = g.find("Attribute").find("DataItem")
    assert (item.attrib["DataType"], item.attrib["Precision"]) == ("Float", "8")


def test_float32_is_described_as_four_byte_float():
    # The old call sites hardcoded Precision="8". A float32 field was
    # therefore described to consumers as 8-byte, which reads as garbage.
    g = grid()
    StubArchive().write_field(g, "u", np.zeros(3, dtype="float32"), 0)
    item = g.find("Attribute").find("DataItem")
    assert (item.attrib["DataType"], item.attrib["Precision"]) == ("Float", "4")


def test_integer_arrays_are_described_as_int():
    g = grid()
    StubArchive().write_field(g, "mat", np.zeros(3, dtype="int32"), 0, center="Cell")
    item = g.find("Attribute").find("DataItem")
    assert (item.attrib["DataType"], item.attrib["Precision"]) == ("Int", "4")


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
    assert g.find("Attribute").find("DataItem").attrib["Dimensions"] == "10"
    (name, got_offsets, got_data), = ar.sync_calls
    assert name == "u_t1"          # no _p<rank> in the sync convention
    np.testing.assert_array_equal(got_offsets, offsets)
    np.testing.assert_array_equal(got_data, dof[:4])


def test_global_sync_uses_the_unranked_dataset_name():
    ar = StubArchive(global_sync=True, rank=3)
    g = grid()
    ar.write_field(g, "p", np.zeros(4), 5, dimensions=[4], sync_offsets=np.array([0, 4]),
                   sync_data=np.zeros(4))
    assert g.find("Attribute").find("DataItem").text == "out.h5:/p_t5"
    assert not ar.async_calls


# --------------------------------------------------------------------------
# the text fallback -- and the bug class it makes inexpressible
# --------------------------------------------------------------------------


def test_text_mode_attaches_an_xi_include_to_its_own_data_item(_no_real_savetxt):
    ar = StubArchive(hdf=False)
    g = grid()
    ar.write_field(g, "u", np.zeros(4), 2)
    item = g.find("Attribute").find("DataItem")
    assert item.attrib["Format"] == "XML"
    (include,) = list(item)
    assert include.tag == "xi:include"
    assert include.attrib == {"parse": "text", "href": "./out_Data/u2.txt"}
    assert not (item.text or "").strip()


def test_text_mode_writes_the_sidecar_file(_no_real_savetxt):
    ar = StubArchive(hdf=False)
    dof = np.arange(4, dtype="d")
    ar.write_field(grid(), "u", dof, 2)
    (path, arr), = _no_real_savetxt
    assert path == "out_Data/u2.txt"
    np.testing.assert_array_equal(arr, dof)


def test_two_fields_in_text_mode_each_get_exactly_one_include(_no_real_savetxt):
    # This is the bug at Archiver.py:1475 made unrepresentable: there, the
    # include for a second field was attached to the first field's
    # DataItem, leaving one field with no data reference and the other with
    # two. write_field always attaches to the DataItem it just created.
    ar = StubArchive(hdf=False)
    g = grid()
    ar.write_field(g, "u", np.zeros(4), 0)
    ar.write_field(g, "u_dof", np.zeros(4), 0, center="Face")
    items = [a.find("DataItem") for a in g.findall("Attribute")]
    assert len(items) == 2
    hrefs = []
    for item in items:
        includes = [c for c in item if c.tag == "xi:include"]
        assert len(includes) == 1, "expected exactly one xi:include per DataItem"
        hrefs.append(includes[0].attrib["href"])
    assert hrefs == ["./out_Data/u0.txt", "./out_Data/u_dof0.txt"]


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
    assert g.find("Attribute").find("DataItem").text == "out.h5:/u_p0_t0"


def test_hdf_mode_with_global_sync_never_touches_text_data_dir():
    ar = StubArchive(global_sync=True)
    del ar.textDataDir
    g = grid()
    ar.write_field(g, "u", np.zeros(4), 0, dimensions=[4],
                   sync_offsets=np.array([0, 4]), sync_data=np.zeros(4))
    assert g.find("Attribute").find("DataItem").text == "out.h5:/u_t0"


# --------------------------------------------------------------------------
# dataset / text_stem overrides
# --------------------------------------------------------------------------


def test_an_explicit_dataset_name_overrides_the_convention():
    # writeFunctionXdmf_DGP2Lagrange predates the _p<rank>_t<tCount>
    # convention and must keep naming its dataset <name><tCount>, or
    # converting it would rename datasets inside existing archives.
    ar = StubArchive(rank=2)
    g = grid()
    ar.write_field(g, "u", np.zeros(4), 7, dimensions=[4], dataset="u7")
    assert g.find("Attribute").find("DataItem").text == "out.h5:/u7"
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
    include = list(g.find("Attribute").find("DataItem"))[0]
    assert include.attrib["href"] == "./out_Data/custom.txt"
    assert _no_real_savetxt[0][0] == "out_Data/custom.txt"
