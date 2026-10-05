"""scripts/gatherTimes must rebuild a partial .xmf from the .h5 and leave a complete one alone."""
import os
import importlib.machinery
import importlib.util
import xml.etree.ElementTree as ET
import numpy as np
import pytest

h5py = pytest.importorskip("h5py")

SCRIPT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "scripts", "gatherTimes")
NFRAMES = 5
COLLECTION = "Mesh Spatial_Domain"


def load_gatherTimes():
    loader = importlib.machinery.SourceFileLoader("gatherTimes_script", SCRIPT)
    spec = importlib.util.spec_from_loader(loader.name, loader)
    module = importlib.util.module_from_spec(spec)
    loader.exec_module(module)
    return module.gatherTimes


def grid_xml(i):
    return ('<Grid GridType="Uniform"><Time Value="{0:e}" Name="{1}" />'
            '<Attribute Name="u" Center="Node"><DataItem>archive.h5:/u_t{1}</DataItem></Attribute>'
            '</Grid>').format(0.01 * i, i)


def write_archive(base, frames_in_xmf):
    """An .h5 holding NFRAMES grids, as the archiver stores them, and an .xmf holding frames_in_xmf"""
    with h5py.File(base + ".h5", "w") as f:
        for i in range(NFRAMES):
            f.create_dataset("Mesh_Spatial_Domain_{0}".format(i), data=np.array([grid_xml(i).encode()]))
    grids = "".join(grid_xml(i) for i in frames_in_xmf)
    with open(base + ".xmf", "w") as f:
        f.write('<Xdmf Version="2.0"><Domain>'
                '<Grid Name="{0}" GridType="Collection" CollectionType="Temporal">{1}</Grid>'
                '</Domain></Xdmf>'.format(COLLECTION, grids))


def frames(path):
    collection = ET.parse(path).getroot()[0][0]
    return [(g.find("Time").attrib["Name"], g.find("Time").attrib["Value"]) for g in collection]


def expected(n):
    return [(str(i), "{0:e}".format(0.01 * i)) for i in range(n)]


@pytest.fixture
def base(tmp_path):
    return str(tmp_path / "archive")


def test_partial_xmf_is_rebuilt(base):
    # a run killed at walltime leaves only the newest frame in the .xmf
    write_archive(base, [NFRAMES - 1])
    load_gatherTimes()(base, tCount=-1)
    assert frames(base + "_complete.xmf") == expected(NFRAMES)


def test_complete_xmf_is_left_alone(base):
    # the old code read the count from the FIRST grid's Name ("0") and cut this to one frame
    write_archive(base, range(NFRAMES))
    before = frames(base + ".xmf")
    load_gatherTimes()(base, tCount=-1)
    assert frames(base + "_complete.xmf") == before == expected(NFRAMES)


def test_explicit_count_does_not_duplicate(base):
    # the old code kept the existing grids when -t was given and appended them all again
    for present in ([NFRAMES - 1], range(NFRAMES)):
        write_archive(base, present)
        load_gatherTimes()(base, tCount=NFRAMES)
        assert frames(base + "_complete.xmf") == expected(NFRAMES)


def test_default_tCount_is_all_frames(base):
    write_archive(base, [NFRAMES - 1])
    load_gatherTimes()(base)
    assert frames(base + "_complete.xmf") == expected(NFRAMES)


def test_fewer_frames(base):
    write_archive(base, range(NFRAMES))
    load_gatherTimes()(base, tCount=2)
    assert frames(base + "_complete.xmf") == expected(2)


def test_more_frames_than_stored(base):
    write_archive(base, [NFRAMES - 1])
    with pytest.raises(ValueError, match="holds"):
        load_gatherTimes()(base, tCount=NFRAMES + 1)
