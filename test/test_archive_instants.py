"""The archive document is one list of instants with the spaces inside each.

The writer keeps one temporal collection per finite-element space while it
runs (``ArchiveDomain``), and the HDF5 metadata is stored per collection per
step. What the *document* holds, live and at close, is instants: see
cekees/ymf docs/design-time-outside-space.md.
"""
import os

import numpy as np
import pytest

h5py = pytest.importorskip("h5py")
ymf_archive = pytest.importorskip("ymf.archive")
if getattr(ymf_archive, "ARCHIVE_FORMAT_VERSION", 1) < 2:
    pytest.skip("needs a ymf with archive format 2", allow_module_level=True)

from proteus import Archiver, Comm


def write_step(ar, name, t, tCount):
    collection = ar.domain.collection(name)
    grid = ar.write_grid(collection, None, t, tCount)
    ar.write_topology(grid, "Triangle", 2, [2, 3], "elements_%s_%d" % (name, tCount),
                      "elements%d" % tCount, dtype="i")
    ar.write_geometry(grid, [4, 3], "nodes_%s_%d" % (name, tCount),
                      "nodes%d" % tCount, dtype="d")
    ar.create_dataset_async("elements_%s_%d" % (name, tCount), data=np.zeros((2, 3), "i"))
    ar.create_dataset_async("nodes_%s_%d" % (name, tCount), data=np.zeros((4, 3), "d"))
    ar.write_field(grid, "u", np.zeros(4), tCount, dimensions=[4],
                   dataset="u_%s_t%d" % (name, tCount))


def run(tmp_path, steps, close=True):
    """steps: [(t, [space names archived at t]), ...]"""
    Comm.init()
    ar = Archiver.XdmfArchive(str(tmp_path), "inst", global_sync=False)
    for tCount, (t, names) in enumerate(steps):
        for name in names:
            write_step(ar, name, t, tCount)
        ar.sync()
    if close:
        ar.close()
    else:
        ar.hdfFile.flush()
    return ar


def read(tmp_path):
    return ymf_archive.read_ymf(str(tmp_path / "inst.ymf"))[0]


def test_the_document_is_instants_with_the_spaces_inside(tmp_path):
    run(tmp_path, [(0.0, ["Mesh", "Mesh_c0p2"]), (0.5, ["Mesh", "Mesh_c0p2"])])
    domain = read(tmp_path)
    assert [i["time"] for i in domain["times"]] == [0.0, 0.5]
    assert all(list(i["spaces"]) == ["Mesh", "Mesh_c0p2"] for i in domain["times"])


def test_a_live_step_is_one_record_whatever_the_number_of_spaces(tmp_path):
    run(tmp_path, [(0.0, ["Mesh", "Mesh_c0p2"]), (0.5, ["Mesh", "Mesh_c0p2"])], close=False)
    text = (tmp_path / "inst.ymf").read_text(encoding="utf-8")
    assert text.count("\n---\n") == 2
    assert text.count("\n...\n") == 2
    domain = read(tmp_path)
    assert [list(i["spaces"]) for i in domain["times"]] == [["Mesh", "Mesh_c0p2"]] * 2


def test_a_run_killed_inside_an_instant_loses_all_of_it(tmp_path):
    run(tmp_path, [(0.0, ["Mesh", "Mesh_c0p2"]), (0.5, ["Mesh", "Mesh_c0p2"])], close=False)
    path = tmp_path / "inst.ymf"
    text = path.read_text(encoding="utf-8")
    path.write_text(text[: text.rindex("Mesh_c0p2:")], encoding="utf-8")
    with pytest.warns(RuntimeWarning):
        domain = read(tmp_path)
    assert [i["time"] for i in domain["times"]] == [0.0]
    assert list(domain["times"][0]["spaces"]) == ["Mesh", "Mesh_c0p2"]


def test_a_space_first_archived_late_is_absent_from_the_early_instants(tmp_path):
    run(tmp_path, [(0.0, ["Mesh"]), (0.5, ["Mesh", "Mesh_c0p2"])])
    domain = read(tmp_path)
    assert [list(i["spaces"]) for i in domain["times"]] == [["Mesh"], ["Mesh", "Mesh_c0p2"]]
    # the document assembled from the .h5 agrees with the live one
    assembled = Archiver.readArchiveDomain("inst", str(tmp_path))
    assert [list(i["spaces"]) for i in assembled["times"]] == [["Mesh"], ["Mesh", "Mesh_c0p2"]]


def test_archived_times_are_the_instants(tmp_path):
    ar = run(tmp_path, [(0.0, ["Mesh", "Mesh_c0p2"]), (0.5, ["Mesh", "Mesh_c0p2"])])
    assert ar.archived_times == [0.0, 0.5]
