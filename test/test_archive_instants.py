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


def hot_start(tmp_path, steps, first_step, close=True):
    """Reopen the archive as a hot start from the step before ``first_step``
    and write ``steps`` after it, numbered as NumericalSolution numbers them
    (``continue_from`` the step it started from)."""
    Comm.init()
    ar = Archiver.XdmfArchive(str(tmp_path), "inst", global_sync=False, hotStart=True)
    ar.continue_from(first_step - 1)
    for tCount, (t, names) in enumerate(steps, start=first_step):
        for name in names:
            write_step(ar, name, t, tCount)
        ar.sync()
    if close:
        ar.close()
    return ar


def times(tmp_path):
    return [i["time"] for i in read(tmp_path)["times"]]


def test_a_hot_start_replaces_the_steps_after_the_one_it_starts_from(tmp_path):
    both = ["Mesh", "Mesh_c0p2"]
    run(tmp_path, [(0.0, both), (0.5, both), (1.0, both)])
    hot_start(tmp_path, [(0.75, both), (1.25, both)], first_step=2)
    assert times(tmp_path) == [0.0, 0.5, 0.75, 1.25]


def test_a_hot_start_that_ends_sooner_drops_the_old_tail(tmp_path):
    both = ["Mesh", "Mesh_c0p2"]
    run(tmp_path, [(0.0, both), (0.5, both), (1.0, both)])
    hot_start(tmp_path, [(0.25, both)], first_step=1)
    assert times(tmp_path) == [0.0, 0.25]


def test_a_space_first_archived_late_waits_for_its_first_instant(tmp_path):
    run(tmp_path, [(0.0, ["Mesh"]), (0.5, ["Mesh", "Mesh_c0p2"]), (1.0, ["Mesh", "Mesh_c0p2"])])
    domain = read(tmp_path)
    assert [list(i["spaces"]) for i in domain["times"]] == [
        ["Mesh"], ["Mesh", "Mesh_c0p2"], ["Mesh", "Mesh_c0p2"]]


def test_a_hot_start_after_a_late_space_keeps_both(tmp_path):
    run(tmp_path, [(0.0, ["Mesh"]), (0.5, ["Mesh", "Mesh_c0p2"]), (1.0, ["Mesh", "Mesh_c0p2"])])
    hot_start(tmp_path, [(1.5, ["Mesh", "Mesh_c0p2"])], first_step=3)
    domain = read(tmp_path)
    assert [i["time"] for i in domain["times"]] == [0.0, 0.5, 1.0, 1.5]
    assert [list(i["spaces"]) for i in domain["times"]][0] == ["Mesh"]
    assert all(len(i["spaces"]) == 2 for i in domain["times"][1:])
