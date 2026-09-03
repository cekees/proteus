"""Reading archives written by proteus <= 1.9.x.

A campaign objective added after the fact: preserve the XDMF consumer
pathways. An existing archive must stay hot-startable, and the scripts must
keep working on HDF5 files that proteus 1.9.x wrote. This reverses the
kickoff decision that backward compatibility was not required -- 1.9.x is
about to be released off main, so its archives will be around.

The two layouts differ in ways that are not obvious, and proteus 1.9.x
recorded neither the format version, the collection names, nor which
layout it used:

* **global** -- one fragment per step, of shape ``(1,)``, and the ``<Time>``
  element is *inside* the ``<Grid>``. The dataset's ``Time`` attribute was
  never set on this path.
* **per-subdomain** -- one fragment per rank, shape ``(size,)``, with
  ``<Time>`` moved *out* of each grid before storing, and the time carried
  on the dataset's ``Time`` attribute instead.

So the layout is recovered from whether a fragment still has a ``<Time>``
child. Shape alone will not do it: both layouts give ``(1,)`` at one rank.

The fixtures here are synthesised rather than vendored because a real
archive is tens of megabytes. They were checked against one: an archive
produced by the parent commit's writer (main @81662ec7, i.e. 1.9.x) read
back with 2 steps, Triangle topology over 204800 elements and all four of
its fields, and validated as a ymf domain.
"""

import os
from xml.etree.ElementTree import Element, SubElement, tostring

import numpy as np
import pytest

h5py = pytest.importorskip("h5py")


def _grid_fragment(step, rank=None, with_time=True, n_elements=32, n_nodes=25):
    """One XDMF <Grid> fragment exactly as proteus <= 1.9.x stored it."""
    grid = Element("Grid", {"GridType": "Uniform"})
    if with_time:
        SubElement(grid, "Time",
                   {"Value": "%e" % (0.1 * step), "Name": str(step)})
    suffix = "" if rank is None else "_p%d" % rank
    topology = SubElement(grid, "Topology",
                          {"Type": "Triangle",
                           "NumberOfElements": str(n_elements)})
    SubElement(topology, "DataItem",
               {"Format": "HDF", "DataType": "Int",
                "Dimensions": "%d 3" % n_elements}).text = \
        "run.h5:/elements%s%d" % (suffix, step)
    geometry = SubElement(grid, "Geometry", {"Type": "XYZ"})
    SubElement(geometry, "DataItem",
               {"Format": "HDF", "DataType": "Float", "Precision": "8",
                "Dimensions": "%d 3" % n_nodes}).text = \
        "run.h5:/nodes%s%d" % (suffix, step)
    attribute = SubElement(grid, "Attribute",
                           {"Name": "u", "AttributeType": "Scalar",
                            "Center": "Node"})
    SubElement(attribute, "DataItem",
               {"Format": "HDF", "DataType": "Float", "Precision": "8",
                "Dimensions": str(n_nodes)}).text = \
        "run.h5:/u%s_t%d" % (suffix, step)
    return tostring(grid, encoding="utf-8")


def write_v1_archive(path, n_steps=3, n_ranks=None,
                     collections=("Mesh Spatial_Domain",)):
    """A version 1 archive: XML fragments, and no version attribute.

    ``n_ranks=None`` writes the global layout; an integer writes the
    per-subdomain layout with that many ranks.
    """
    with h5py.File(path, "w") as f:
        for name in collections:
            dataset_base = name.replace(" ", "_")
            for step in range(n_steps):
                if n_ranks is None:
                    payloads = [_grid_fragment(step, with_time=True)]
                else:
                    payloads = [_grid_fragment(step, rank=r, with_time=False)
                                for r in range(n_ranks)]
                width = max(len(p) for p in payloads)
                dataset = f.create_dataset(
                    "%s_%d" % (dataset_base, step),
                    shape=(len(payloads),), dtype="|S%d" % width)
                for j, payload in enumerate(payloads):
                    dataset[j] = payload
                if n_ranks is not None:
                    # only the per-subdomain path set this
                    dataset.attrs["Time"] = "%e" % (0.1 * step)
        # a little field data, so the file is not only metadata
        f["u0_t0"] = np.zeros(25, dtype="d")
    return path


def read_domain(path):
    from proteus.Archiver import readArchiveDomain

    base = os.path.basename(path)[:-len(".h5")]
    return readArchiveDomain(base, dataDir=os.path.dirname(path))


# --------------------------------------------------------------------------
# the version marker
# --------------------------------------------------------------------------


def test_an_archive_with_no_version_attribute_is_version_one(tmp_path):
    from proteus.Archiver import AR_base

    path = write_v1_archive(str(tmp_path / "run.h5"))
    archive = AR_base.__new__(AR_base)
    archive.hdfFilename = "run.h5"
    with h5py.File(path, "r") as f:
        archive.hdfFile = f
        assert archive._metadata_version() == 1


def test_a_future_version_is_still_refused(tmp_path):
    from proteus.Archiver import AR_base

    path = write_v1_archive(str(tmp_path / "run.h5"))
    with h5py.File(path, "a") as f:
        f.attrs[AR_base.METADATA_VERSION_ATTR] = 99
    archive = AR_base.__new__(AR_base)
    archive.hdfFilename = "run.h5"
    with h5py.File(path, "r") as f:
        archive.hdfFile = f
        with pytest.raises(ValueError, match="reads up to version"):
            archive._metadata_version()


# --------------------------------------------------------------------------
# reading both v1 layouts
# --------------------------------------------------------------------------


def test_a_global_v1_archive_reads_as_uniform_steps(tmp_path):
    domain = read_domain(write_v1_archive(str(tmp_path / "run.h5"), n_steps=3))
    (collection,) = domain["TimeCollections"]
    assert len(collection["Data"]) == 3
    for step in collection["Data"]:
        assert "SpatialCollection" not in step
        assert step["Topology"]["Type"] == "Triangle"


def test_a_per_subdomain_v1_archive_reads_as_spatial_steps(tmp_path):
    """The layout is inferred from the missing <Time>, not from the shape.

    Both layouts give shape (1,) at a single rank, so shape cannot decide
    it.
    """
    domain = read_domain(
        write_v1_archive(str(tmp_path / "run.h5"), n_steps=2, n_ranks=3))
    (collection,) = domain["TimeCollections"]
    for step in collection["Data"]:
        assert len(step["SpatialCollection"]) == 3


def test_a_single_rank_per_subdomain_v1_archive_is_not_mistaken_for_global(tmp_path):
    """The case shape alone gets wrong."""
    domain = read_domain(
        write_v1_archive(str(tmp_path / "run.h5"), n_steps=2, n_ranks=1))
    for step in domain["TimeCollections"][0]["Data"]:
        assert "SpatialCollection" in step, \
            "a one-rank per-subdomain archive was read as global"


@pytest.mark.parametrize("n_ranks", [None, 2])
def test_v1_times_are_recovered_from_wherever_they_were_kept(tmp_path, n_ranks):
    """Global kept the time inside the fragment; per-rank on the dataset."""
    domain = read_domain(
        write_v1_archive(str(tmp_path / "run.h5"), n_steps=4, n_ranks=n_ranks))
    times = [step["Time"] for step in domain["TimeCollections"][0]["Data"]]
    np.testing.assert_allclose(times, [0.0, 0.1, 0.2, 0.3])


def test_v1_collection_names_are_discovered(tmp_path):
    """1.9.x recorded no names, so they come from the file itself.

    Discovery confirms a candidate by what it is -- a 1-D array of byte
    strings -- rather than by matching its name, since a collection name
    can itself contain underscores and digits.
    """
    domain = read_domain(write_v1_archive(
        str(tmp_path / "run.h5"), n_steps=2,
        collections=("Mesh Spatial_Domain", "Mesh_c0p2_Lagrange")))
    names = [c["Name"] for c in domain["TimeCollections"]]
    # spaces became underscores in the dataset names 1.9.x wrote, so the
    # recovered name is the underscored form. Callers match on a substring
    # ("Spatial_Domain"), which still works.
    assert sorted(names) == ["Mesh_Spatial_Domain", "Mesh_c0p2_Lagrange"]


def test_field_data_is_not_mistaken_for_metadata(tmp_path):
    """Discovery must not pick up numeric datasets, however they are named."""
    path = write_v1_archive(str(tmp_path / "run.h5"), n_steps=2)
    with h5py.File(path, "a") as f:
        f["u_0"] = np.zeros(4, dtype="d")      # looks like <name>_<step>
        f["elements_3"] = np.zeros((2, 3), dtype="i")
    domain = read_domain(path)
    assert [c["Name"] for c in domain["TimeCollections"]] == \
        ["Mesh_Spatial_Domain"]


def test_a_v1_archive_validates_as_a_ymf_domain(tmp_path):
    from ymf.archive import validate_domain

    validate_domain(read_domain(
        write_v1_archive(str(tmp_path / "run.h5"), n_steps=2, n_ranks=2)))


# --------------------------------------------------------------------------
# the consumer pathways, on a v1 archive
# --------------------------------------------------------------------------


def test_hot_start_times_come_back_from_a_v1_archive(tmp_path):
    """archived_times is what the hot-start step selection consumes."""
    from proteus.Archiver import AR_base

    path = write_v1_archive(str(tmp_path / "run.h5"), n_steps=4)
    archive = AR_base.__new__(AR_base)
    archive.hdfFilename = "run.h5"
    archive.global_sync = True
    archive.archived_domain = None
    with h5py.File(path, "r") as f:
        archive.hdfFile = f
        np.testing.assert_allclose(archive.archived_times, [0.0, 0.1, 0.2, 0.3])


def test_gather_times_regenerates_a_document_from_a_v1_archive(tmp_path,
                                                               monkeypatch):
    """The script must keep working on 1.9.x output.

    And from the .h5 alone: the case it exists for is an archive whose
    .xmf is the missing part.
    """
    import importlib.machinery
    import importlib.util
    from xml.etree.ElementTree import parse

    monkeypatch.chdir(tmp_path)
    write_v1_archive("run.h5", n_steps=3)

    script = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "scripts", "gatherTimes")
    loader = importlib.machinery.SourceFileLoader("gatherTimes", script)
    module = importlib.util.module_from_spec(
        importlib.util.spec_from_loader("gatherTimes", loader))
    loader.exec_module(module)
    module.gatherTimes("run", tCount=-1)

    from ymf.archive import read_ymf

    domain, _ = read_ymf("run_complete.ymf")
    assert len(domain["TimeCollections"][0]["Data"]) == 3
    root = parse("run_complete.xmf").getroot()
    assert len(root.find("Domain").find("Grid").findall("Grid")) == 3


def test_a_v1_archive_converts_straight_to_xmf(tmp_path, monkeypatch):
    """1.9.x .h5 -> .ymf -> .xmf, the whole consumer path."""
    import importlib.machinery
    import importlib.util

    from ymf.cli import ymf2xmf

    monkeypatch.chdir(tmp_path)
    write_v1_archive("run.h5", n_steps=2, n_ranks=2)
    script = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "scripts", "gatherTimes")
    loader = importlib.machinery.SourceFileLoader("gatherTimes", script)
    module = importlib.util.module_from_spec(
        importlib.util.spec_from_loader("gatherTimes", loader))
    loader.exec_module(module)
    module.gatherTimes("run", tCount=-1)

    destination = ymf2xmf("run_complete.ymf", "from_v1.xmf")
    assert destination.exists()
