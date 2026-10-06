"""scripts/gatherTimes must rebuild a complete document from the .h5.

The case it exists for (#32): a run killed at walltime, whose .xmf holds
only the newest step while its .h5 holds them all. These archives come from
proteus <= 1.9.x, so the fixture is a version 1 archive; current-format
archives are covered in test_archive_parallel_metadata.py.
"""
import importlib.machinery
import importlib.util
import os
import xml.etree.ElementTree as ET

import pytest

h5py = pytest.importorskip("h5py")
pytest.importorskip("ymf")

from test_archive_v1_compat import write_v1_archive  # noqa: E402

SCRIPT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "scripts", "gatherTimes")
NFRAMES = 5


def load_gatherTimes():
    loader = importlib.machinery.SourceFileLoader("gatherTimes_script", SCRIPT)
    spec = importlib.util.spec_from_loader(loader.name, loader)
    module = importlib.util.module_from_spec(spec)
    loader.exec_module(module)
    return module.gatherTimes


def frames(path):
    """(Name, time) of each step in the first temporal collection."""
    collection = ET.parse(path).getroot().find("Domain").find("Grid")
    return [(g.find("Time").attrib["Name"], float(g.find("Time").attrib["Value"]))
            for g in collection.findall("Grid")]


def expected(n):
    # write_v1_archive stores step i at time 0.1 i
    return [(str(i), pytest.approx(0.1 * i)) for i in range(n)]


def write_xmf(path, steps):
    """The .xmf a run leaves behind: complete, or only its newest step."""
    grids = "".join('<Grid GridType="Uniform"><Time Value="%e" Name="%d"/></Grid>'
                    % (0.1 * i, i) for i in steps)
    with open(path, "w") as f:
        f.write('<Xdmf Version="2.0"><Domain><Grid Name="Mesh Spatial_Domain" '
                'GridType="Collection" CollectionType="Temporal">%s</Grid>'
                '</Domain></Xdmf>' % grids)


@pytest.fixture
def base(tmp_path):
    write_v1_archive(str(tmp_path / "archive.h5"), n_steps=NFRAMES)
    return str(tmp_path / "archive")


def run(base, **kwargs):
    load_gatherTimes()(os.path.basename(base), dataDir=os.path.dirname(base), **kwargs)


def test_partial_xmf_is_rebuilt(base):
    # a run killed at walltime leaves only the newest frame in the .xmf
    write_xmf(base + ".xmf", [NFRAMES - 1])
    run(base, tCount=-1)
    assert frames(base + "_complete.xmf") == expected(NFRAMES)


def test_complete_xmf_is_left_alone(base):
    # the old code read the count from the FIRST grid's Name ("0") and cut this to one frame
    write_xmf(base + ".xmf", range(NFRAMES))
    with open(base + ".xmf", "rb") as f:
        before = f.read()
    run(base, tCount=-1)
    assert frames(base + "_complete.xmf") == expected(NFRAMES)
    with open(base + ".xmf", "rb") as f:
        assert f.read() == before


def test_explicit_count_does_not_duplicate(base):
    # the old code kept the existing grids when -t was given and appended them all again
    for present in ([NFRAMES - 1], range(NFRAMES)):
        write_xmf(base + ".xmf", present)
        run(base, tCount=NFRAMES)
        assert frames(base + "_complete.xmf") == expected(NFRAMES)


def test_default_tCount_is_all_frames(base):
    write_xmf(base + ".xmf", [NFRAMES - 1])
    run(base)
    assert frames(base + "_complete.xmf") == expected(NFRAMES)


def test_no_xmf_is_needed_at_all(base):
    run(base)
    assert frames(base + "_complete.xmf") == expected(NFRAMES)


def test_fewer_frames(base):
    run(base, tCount=2)
    assert frames(base + "_complete.xmf") == expected(2)


def test_more_frames_than_stored(base):
    with pytest.raises(ValueError, match="holds"):
        run(base, tCount=NFRAMES + 1)
