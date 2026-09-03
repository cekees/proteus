"""Phase 3: reading the archive by key instead of by tree position.

Every reader used to walk the XDMF tree positionally -- the root's last
child as the Domain, that Domain's last child as the collection, a grid's
first child as its Time. Those positions are not guaranteed by XDMF:
``<Information>`` is legal ahead of ``<Domain>``, a real proteus archive
commonly carries two grid collections (the linear mesh and a quadratic
space), and nothing fixes the order of a grid's children. Each of these
tests puts the thing being looked for somewhere other than the position the
old code assumed, so a positional reader fails them.
"""

import os
import sys
from xml.etree.ElementTree import Element, ElementTree, SubElement, parse

import numpy as np
import pytest

h5py = pytest.importorskip("h5py")

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPTS = os.path.join(REPO_ROOT, "scripts")

TWO_COLLECTIONS = ("Mesh Spatial_Domain", "Mesh_c0p2_Lagrange")


def build_xmf(path, n_steps=2, collections=TWO_COLLECTIONS, rank=0,
              lead_with_information=False, time_last=False):
    """A per-rank .xmf shaped like proteus's, with optional awkwardness.

    ``lead_with_information`` puts an <Information> element ahead of
    <Domain>, which XDMF allows and which breaks ``getroot()[-1]``.
    ``time_last`` puts <Time> after Topology/Geometry, which breaks
    ``Grid[0]``.
    """
    root = Element("Xdmf", {"Version": "2.0"})
    if lead_with_information:
        SubElement(root, "Information", {"Name": "YMF", "Value": "x"})
    domain = SubElement(root, "Domain")
    for name in collections:
        coll = SubElement(domain, "Grid",
                          {"Name": name, "GridType": "Collection",
                           "CollectionType": "Temporal"})
        for tn in range(n_steps):
            grid = SubElement(coll, "Grid", {"GridType": "Uniform"})
            if not time_last:
                SubElement(grid, "Time", {"Value": "%e" % tn, "Name": str(tn)})
            topo = SubElement(grid, "Topology",
                              {"Type": "Triangle", "NumberOfElements": "2"})
            SubElement(topo, "DataItem",
                       {"Format": "HDF", "DataType": "Int",
                        "Dimensions": "2 3"}).text = "f.h5:/e%d_p%d" % (tn, rank)
            geo = SubElement(grid, "Geometry", {"Type": "XYZ"})
            SubElement(geo, "DataItem",
                       {"Format": "HDF", "DataType": "Float", "Precision": "8",
                        "Dimensions": "4 3"}).text = "f.h5:/n%d_p%d" % (tn, rank)
            attr = SubElement(grid, "Attribute",
                              {"Name": "nodeMaterialTypes",
                               "AttributeType": "Scalar", "Center": "Node"})
            SubElement(attr, "DataItem",
                       {"Format": "HDF", "DataType": "Int",
                        "Dimensions": "4"}).text = "f.h5:/nodeMat%d" % tn
            if time_last:
                SubElement(grid, "Time", {"Value": "%e" % tn, "Name": str(tn)})
    with open(path, "wb") as handle:
        ElementTree(root).write(handle)
    return path


# --------------------------------------------------------------------------
# MeshTools.findXMLgridElement
# --------------------------------------------------------------------------


def test_the_named_collection_is_found_not_the_last_one(tmp_path):
    """`Domain[-1]` would return the quadratic space, not the base mesh."""
    from proteus.MeshTools import findXMLgridElement

    path = build_xmf(str(tmp_path / "a.xmf"))
    grid = findXMLgridElement(parse(path), MeshTag="Spatial_Domain")
    # the base mesh's connectivity, not the c0p2 space's
    assert grid.find("Topology").find("DataItem").text.endswith("e1_p0")
    assert grid.attrib["GridType"] == "Uniform"


def test_an_information_element_before_domain_is_tolerated(tmp_path):
    """XDMF permits <Information*> ahead of <Grid+> inside <Xdmf>.

    `getroot()[-1]` happened to work only because nothing was written
    after <Domain>.
    """
    from proteus.MeshTools import findXMLgridElement

    path = build_xmf(str(tmp_path / "a.xmf"), lead_with_information=True)
    grid = findXMLgridElement(parse(path), MeshTag="Spatial_Domain")
    assert grid.attrib["GridType"] == "Uniform"


def test_an_unknown_mesh_tag_falls_back_to_the_first_collection(tmp_path):
    from proteus.MeshTools import findXMLgridElement

    path = build_xmf(str(tmp_path / "a.xmf"))
    grid = findXMLgridElement(parse(path), MeshTag="NoSuchMesh")
    assert grid.attrib["GridType"] == "Uniform"


def test_a_step_can_be_selected_by_index(tmp_path):
    from proteus.MeshTools import findXMLgridElement

    path = build_xmf(str(tmp_path / "a.xmf"), n_steps=3)
    first = findXMLgridElement(parse(path), id_in_collection=0)
    last = findXMLgridElement(parse(path), id_in_collection=-1)
    assert first.find("Topology").find("DataItem").text.endswith("e0_p0")
    assert last.find("Topology").find("DataItem").text.endswith("e2_p0")


def test_a_spatial_collection_step_resolves_to_a_uniform_grid(tmp_path):
    """A per-subdomain archive nests Grid(Spatial) > Grid(Uniform)."""
    from proteus.MeshTools import findXMLgridElement

    root = Element("Xdmf", {"Version": "2.0"})
    domain = SubElement(root, "Domain")
    coll = SubElement(domain, "Grid",
                      {"Name": "Mesh Spatial_Domain", "GridType": "Collection",
                       "CollectionType": "Temporal"})
    spatial = SubElement(coll, "Grid", {"GridType": "Collection",
                                        "CollectionType": "Spatial"})
    SubElement(spatial, "Time", {"Value": "0.0", "Name": "0"})
    for rank in range(2):
        g = SubElement(spatial, "Grid",
                       {"GridType": "Uniform", "Name": "p%d" % rank})
        topo = SubElement(g, "Topology",
                          {"Type": "Triangle", "NumberOfElements": "2"})
        SubElement(topo, "DataItem",
                   {"Format": "HDF", "DataType": "Int",
                    "Dimensions": "2 3"}).text = "f.h5:/e_p%d" % rank
    path = str(tmp_path / "s.xmf")
    with open(path, "wb") as handle:
        ElementTree(root).write(handle)

    grid = findXMLgridElement(parse(path))
    assert grid.attrib["GridType"] == "Uniform"
    assert grid.attrib["Name"] == "p0"


# --------------------------------------------------------------------------
# MeshTools.extractPropertiesFromXdmfGridNode and the DataItem reads
# --------------------------------------------------------------------------


def test_grid_properties_are_found_regardless_of_child_order(tmp_path):
    """`Grid[0]` assumed Time came first; nothing in XDMF says so."""
    from proteus.MeshTools import extractPropertiesFromXdmfGridNode, findXMLgridElement

    path = build_xmf(str(tmp_path / "a.xmf"), time_last=True)
    grid = findXMLgridElement(parse(path))
    topology, geometry, node_materials, element_materials = \
        extractPropertiesFromXdmfGridNode(grid)
    assert topology is not None and topology.tag == "Topology"
    assert geometry is not None and geometry.tag == "Geometry"
    assert node_materials is not None
    assert node_materials.attrib["Name"] == "nodeMaterialTypes"
    assert element_materials is None       # not written by this fixture


def test_a_dataitem_reference_resolves_to_its_dataset_name(tmp_path):
    from proteus.MeshTools import _dataitem_dataset, findXMLgridElement

    path = build_xmf(str(tmp_path / "a.xmf"))
    grid = findXMLgridElement(parse(path))
    assert _dataitem_dataset(grid.find("Geometry"), "Geometry") == "n1_p0"


def test_a_dataset_path_containing_a_colon_survives():
    """Splitting on ':' took the last colon-separated piece.

    A reference is <file>:/<dataset>, so splitting on ':/' keeps a dataset
    path that itself contains a colon intact.
    """
    from proteus.MeshTools import _dataitem_dataset

    geometry = Element("Geometry")
    SubElement(geometry, "DataItem").text = "out.h5:/group/od:d"
    assert _dataitem_dataset(geometry, "Geometry") == "group/od:d"


def test_a_missing_dataitem_is_reported(tmp_path):
    from proteus.MeshTools import _dataitem_dataset

    with pytest.raises(AssertionError, match="no <DataItem>"):
        _dataitem_dataset(Element("Geometry"), "Geometry")


# --------------------------------------------------------------------------
# scripts/gatherArchives.py
# --------------------------------------------------------------------------


@pytest.fixture
def gather_archives():
    sys.path.insert(0, SCRIPTS)
    try:
        import gatherArchives
    finally:
        sys.path.remove(SCRIPTS)
    return gatherArchives


def test_time_step_grids_uses_the_named_collection(tmp_path, gather_archives):
    path = build_xmf(str(tmp_path / "run0.xmf"), n_steps=3)
    assert len(gather_archives.timeStepGrids(parse(path))) == 3


def test_gather_opt_merges_every_rank_into_each_step(tmp_path, monkeypatch,
                                                     gather_archives):
    monkeypatch.chdir(tmp_path)
    size, n_steps = 3, 2
    for rank in range(size):
        build_xmf("run%d.xmf" % rank, n_steps=n_steps, rank=rank)
    gather_archives.gatherXDMFfilesOpt(size, "run", dataDir=".", addname="_optAll")

    steps = parse("run_optAll%d.xmf" % size).getroot().find("Domain") \
        .find("Grid").findall("Grid")
    assert len(steps) == n_steps
    for step in steps:
        assert len(step.findall("Grid")) == size


def test_gather_preserves_both_grid_collections(tmp_path, monkeypatch,
                                                gather_archives):
    """The old positional read reached only the last collection.

    A real archive with a quadratic space has two, so merging with
    `Domain[-1]` silently dropped the base mesh.
    """
    monkeypatch.chdir(tmp_path)
    size = 3
    for rank in range(size):
        build_xmf("run%d.xmf" % rank, rank=rank)
    gather_archives.gatherXDMFfiles(size, "run", dataDir=".", addname="_all")

    collections = parse("run_all%d.xmf" % size).getroot() \
        .find("Domain").findall("Grid")
    assert [c.attrib.get("Name") for c in collections] == list(TWO_COLLECTIONS)
    first_step = collections[0].findall("Grid")[0]
    assert len(first_step.findall("Grid")) == size


def test_gather_writes_text_not_bytes(tmp_path, monkeypatch, gather_archives):
    """Both gather functions wrote tostring() bytes to a text-mode file.

    That raised TypeError on every call, so neither had ever run on
    Python 3.
    """
    monkeypatch.chdir(tmp_path)
    for rank in range(2):
        build_xmf("run%d.xmf" % rank, rank=rank)
    gather_archives.gatherXDMFfiles(2, "run", dataDir=".", addname="_all")
    gather_archives.gatherXDMFfilesOpt(2, "run", dataDir=".", addname="_opt")
    for name in ("run_all2.xmf", "run_opt2.xmf"):
        assert parse(name).getroot().tag == "Xdmf"


# --------------------------------------------------------------------------
# hot start step selection
# --------------------------------------------------------------------------


class _StubArchive:
    """Just enough for archived_times."""

    def __init__(self, times):
        from proteus.Archiver import AR_base

        self.hdfFile = None
        self.archived_domain = {
            "TimeCollections": [
                {"Name": "Mesh Spatial_Domain",
                 "Data": [{"Time": t} for t in times]}
            ]
        }
        self.archived_times = AR_base.archived_times.__get__(self)


def select_step(times, hot_start_time):
    """The selection NumericalSolution performs, in isolation."""
    tCount = len(times) - 1
    while tCount > 0 and times[tCount] > hot_start_time:
        tCount -= 1
    return tCount


@pytest.mark.parametrize(
    "hot_start_time,expected",
    [(0.0, 0), (0.1, 1), (0.15, 1), (0.2, 2), (99.0, 3), (-1.0, 0)],
)
def test_hot_start_picks_the_latest_step_at_or_before_the_time(
        hot_start_time, expected):
    times = [0.0, 0.1, 0.2, 0.3]
    assert select_step(times, hot_start_time) == expected


def test_archived_times_reads_the_domain_not_the_xml_tree():
    times = [0.0, 0.25, 0.5]
    assert _StubArchive(times).archived_times == times


def test_archived_times_is_empty_before_the_archive_is_assembled():
    from proteus.Archiver import AR_base

    archive = AR_base.__new__(AR_base)
    archive.hdfFile = None
    archive.archived_domain = None
    assert archive.archived_times == []
