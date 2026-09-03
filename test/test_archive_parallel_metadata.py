"""The archive's in-HDF5 grid metadata, including under real MPI.

Phase 2 of the YMF I/O campaign replaced the XML fragments that
``allGatherIncremental`` used to stash in the HDF5 file with YAML
describing ymf grid dicts, and made the final document be assembled from
those rather than from an accumulating ElementTree.

The parallel behaviour is the part that most needs a test and had none:
the dataset holding per-rank metadata is created collectively, so every
rank must agree on its width, and the width depends on data only some
ranks have. These tests run the real thing under ``mpiexec`` at several
rank counts, with deliberately different grid sizes per rank so that an
agreement bug shows up as a truncated payload rather than passing by
luck.

Run directly (``pytest``) they cover the serial path; the MPI cases
re-invoke themselves through ``mpiexec`` as subprocesses.
"""

import os
import shutil
import subprocess
import sys
import textwrap

import numpy as np
import pytest

h5py = pytest.importorskip("h5py")
pytest.importorskip("proteus.Archiver")

def _find_mpiexec():
    """The launcher that matches this interpreter's mpi4py, not just any.

    A mismatched launcher does not fail -- it starts N independent
    single-rank jobs, so a test asking for 3 ranks quietly exercises 1 and
    passes for the wrong reason. On this machine PATH resolves mpiexec to
    Homebrew's OpenMPI while mpi4py is built against conda's MPICH, which
    is exactly that situation. Prefer the launcher installed beside
    sys.executable.
    """
    beside = os.path.join(os.path.dirname(sys.executable), "mpiexec")
    if os.path.exists(beside):
        return beside
    return shutil.which("mpiexec")


MPIEXEC = _find_mpiexec()

# The body run inside each MPI rank. Kept as source text rather than a
# module so the rank count and mode can be varied without a fixture file
# per combination, and so a failure inside a rank surfaces as that rank's
# traceback on stderr.
RANK_SCRIPT = textwrap.dedent(
    '''
    import os, shutil, sys
    from xml.etree.ElementTree import SubElement
    import numpy as np

    from proteus import Comm
    Comm.init()
    comm = Comm.get()
    rank, size = comm.rank(), comm.size()

    global_sync = os.environ["GLOBAL_SYNC"] == "1"
    datadir = os.environ["DATADIR"]
    if rank == 0:
        shutil.rmtree(datadir, ignore_errors=True)
        os.makedirs(datadir, exist_ok=True)
    comm.barrier()

    from proteus import Archiver

    ar = Archiver.XdmfArchive(datadir, "mpitest", useGlobalXMF=True,
                              global_sync=global_sync)
    ar.domain = SubElement(ar.tree.getroot(), "Domain")

    # Sizes differ per rank on purpose: the metadata dataset is fixed-width
    # and created collectively, so if the width agreement is wrong the
    # longest rank's YAML is silently truncated.
    n_elements, n_nodes = 4 + rank, 6 + rank
    collection = SubElement(ar.domain, "Grid",
                            {"Name": "Mesh Spatial_Domain",
                             "GridType": "Collection",
                             "CollectionType": "Temporal"})

    # global_sync describes one assembled global array, so the metadata
    # carries global counts and each rank contributes only the slice it owns.
    comm_world = comm.comm.tompi4py()
    node_offsets = np.concatenate(
        ([0], np.cumsum(comm_world.allgather(n_nodes)))).astype("i")
    n_nodes_global = int(node_offsets[-1])
    n_elements_global = int(sum(comm_world.allgather(n_elements)))

    for tCount, t in enumerate([0.0, 0.5]):
        grid, _ = ar.write_grid(collection, "Grid_p%d" % rank, t, tCount)
        if global_sync:
            ar.write_topology(grid, "Triangle", n_elements_global,
                              [n_elements_global, 3],
                              "elements_t%d" % tCount, "elements%d" % tCount)
            ar.write_geometry(grid, [n_nodes_global, 3],
                              "nodes_t%d" % tCount, "nodes%d" % tCount)
            ar.write_field(grid, "u", np.arange(n_nodes, dtype="d"), tCount,
                           dimensions=[n_nodes_global],
                           sync_offsets=node_offsets,
                           sync_data=np.arange(n_nodes, dtype="d"))
        else:
            ar.write_topology(grid, "Triangle", n_elements, [n_elements, 3],
                              "elements_p%d_t%d" % (rank, tCount),
                              "elements%d" % tCount)
            ar.write_geometry(grid, [n_nodes, 3],
                              "nodes_p%d_t%d" % (rank, tCount),
                              "nodes%d" % tCount)
            ar.write_field(grid, "u", np.arange(n_nodes, dtype="d"), tCount,
                           dimensions=[n_nodes])
        ar.sync()

    ar.close()
    comm.barrier()
    if rank == 0:
        print("OBSERVED_SIZE=%d" % size)
    '''
)


def run_ranks(tmp_path, n_ranks, global_sync):
    """Run the rank script under mpiexec (or directly for one rank)."""
    datadir = str(tmp_path / "archive")
    env = dict(os.environ, GLOBAL_SYNC="1" if global_sync else "0",
               DATADIR=datadir)
    cmd = [sys.executable, "-c", RANK_SCRIPT]
    if n_ranks > 1:
        cmd = [MPIEXEC, "-n", str(n_ranks)] + cmd
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=300,
                            env=env)
    assert result.returncode == 0, (
        "rank script failed with %d ranks:\nSTDOUT:\n%s\nSTDERR:\n%s"
        % (n_ranks, result.stdout[-2000:], result.stderr[-4000:]))
    # Refuse to pass on a degraded run: a mismatched launcher yields N
    # independent single-rank jobs rather than an error.
    observed = [line for line in result.stdout.splitlines()
                if line.startswith("OBSERVED_SIZE=")]
    assert observed, (
        "rank 0 never reported its communicator size; stdout:\n%s"
        % result.stdout[-2000:])
    got = int(observed[0].split("=")[1])
    assert got == n_ranks, (
        "asked for %d ranks but the job ran with a communicator of %d -- the "
        "launcher %s does not match this interpreter's mpi4py, so this test "
        "would otherwise pass while exercising a single rank"
        % (n_ranks, got, MPIEXEC))
    return datadir


def read_domain(datadir):
    from ymf.archive import read_ymf

    return read_ymf(os.path.join(datadir, "mpitest.ymf"))[0]


# --------------------------------------------------------------------------
# serial
# --------------------------------------------------------------------------


def test_serial_per_rank_archive_is_written_as_ymf(tmp_path):
    domain = read_domain(run_ranks(tmp_path, 1, global_sync=False))
    (collection,) = domain["TimeCollections"]
    assert collection["Name"] == "Mesh Spatial_Domain"
    assert len(collection["Data"]) == 2
    assert [s["Time"] for s in collection["Data"]] == [0.0, 0.5]


def test_serial_global_sync_archive_has_uniform_steps(tmp_path):
    domain = read_domain(run_ranks(tmp_path, 1, global_sync=True))
    (collection,) = domain["TimeCollections"]
    for step in collection["Data"]:
        assert "SpatialCollection" not in step
        assert step["Topology"]["Type"] == "Triangle"


def test_metadata_is_yaml_not_xml(tmp_path):
    datadir = run_ranks(tmp_path, 1, global_sync=False)
    with h5py.File(os.path.join(datadir, "mpitest.h5"), "r") as f:
        (name,) = [k for k in f if k.startswith("Mesh_Spatial_Domain_0")]
        payload = f[name][0].decode("utf-8")
    assert not payload.lstrip().startswith("<"), "metadata is still XML"
    from ymf.archive import load_grid

    grid = load_grid(payload)
    assert grid["Topology"]["Type"] == "Triangle"


def test_the_metadata_format_version_is_recorded(tmp_path):
    from proteus.Archiver import AR_base

    datadir = run_ranks(tmp_path, 1, global_sync=False)
    with h5py.File(os.path.join(datadir, "mpitest.h5"), "r") as f:
        assert int(f.attrs[AR_base.METADATA_VERSION_ATTR]) == \
            AR_base.METADATA_FORMAT_VERSION


def test_an_archive_without_the_version_marker_is_refused(tmp_path):
    """A pre-YMF archive must fail with an explanation, not a parse error.

    Version 1 metadata was XML in datasets with the same names, so without
    this check the YAML loader would be handed XML and fail somewhere
    unhelpful.
    """
    from proteus.Archiver import AR_base

    datadir = run_ranks(tmp_path, 1, global_sync=False)
    path = os.path.join(datadir, "mpitest.h5")
    with h5py.File(path, "a") as f:
        del f.attrs[AR_base.METADATA_VERSION_ATTR]

    ar = AR_base.__new__(AR_base)
    ar.hdfFilename = "mpitest.h5"
    with h5py.File(path, "r") as f:
        ar.hdfFile = f
        with pytest.raises(ValueError, match="pre-YMF layout"):
            ar._check_metadata_version()


def test_a_future_version_marker_is_refused(tmp_path):
    from proteus.Archiver import AR_base

    datadir = run_ranks(tmp_path, 1, global_sync=False)
    path = os.path.join(datadir, "mpitest.h5")
    with h5py.File(path, "a") as f:
        f.attrs[AR_base.METADATA_VERSION_ATTR] = 99

    ar = AR_base.__new__(AR_base)
    ar.hdfFilename = "mpitest.h5"
    with h5py.File(path, "r") as f:
        ar.hdfFile = f
        with pytest.raises(ValueError, match="format version 99"):
            ar._check_metadata_version()


# --------------------------------------------------------------------------
# parallel
# --------------------------------------------------------------------------


@pytest.mark.skipif(MPIEXEC is None, reason="mpiexec not available")
@pytest.mark.parametrize("n_ranks", [2, 3])
def test_every_rank_contributes_a_grid_to_each_step(tmp_path, n_ranks):
    domain = read_domain(run_ranks(tmp_path, n_ranks, global_sync=False))
    (collection,) = domain["TimeCollections"]
    assert len(collection["Data"]) == 2
    for step in collection["Data"]:
        subs = step["SpatialCollection"]
        assert len(subs) == n_ranks
        assert [g["Name"] for g in subs] == [
            "Grid_p%d" % r for r in range(n_ranks)
        ]


@pytest.mark.skipif(MPIEXEC is None, reason="mpiexec not available")
@pytest.mark.parametrize("n_ranks", [2, 3])
def test_differently_sized_ranks_are_not_truncated(tmp_path, n_ranks):
    """The collective width agreement must fit the *longest* rank.

    Each rank writes n_elements = 4 + rank, so the YAML payloads differ in
    length. A width agreed from only one rank's view would truncate the
    others and their topology counts would come back wrong or unparseable.
    """
    domain = read_domain(run_ranks(tmp_path, n_ranks, global_sync=False))
    (collection,) = domain["TimeCollections"]
    for step in collection["Data"]:
        counts = [g["Topology"]["NumberOfElements"]
                  for g in step["SpatialCollection"]]
        assert counts == [4 + r for r in range(n_ranks)]


@pytest.mark.skipif(MPIEXEC is None, reason="mpiexec not available")
def test_parallel_archive_validates_as_a_ymf_document(tmp_path):
    from ymf.archive import validate_domain

    validate_domain(read_domain(run_ranks(tmp_path, 2, global_sync=False)))


@pytest.mark.skipif(MPIEXEC is None, reason="mpiexec not available")
def test_the_derived_xmf_holds_every_ranks_grid(tmp_path):
    """The .xmf must be derived from the same domain, not from a stale tree.

    Regression test: the legacy code wrote ``self.treeGlobal`` to this same
    file handle at the end of gatherAndWriteTimes. Once the grids stopped
    accumulating in that tree, that write truncated the file back to an
    empty temporal collection -- the .ymf was complete and the .xmf was a
    stub.
    """
    from xml.etree.ElementTree import parse

    datadir = run_ranks(tmp_path, 2, global_sync=False)
    root = parse(os.path.join(datadir, "mpitest.xmf")).getroot()
    temporal = root.find("Domain").find("Grid")
    assert temporal.attrib["CollectionType"] == "Temporal"
    steps = temporal.findall("Grid")
    assert len(steps) == 2, "expected one spatial collection per timestep"
    for step in steps:
        assert step.attrib["CollectionType"] == "Spatial"
        assert step.find("Time") is not None
        assert len(step.findall("Grid")) == 2


# --------------------------------------------------------------------------
# mode (a): global arrays, with the decomposition invisible to a consumer
# --------------------------------------------------------------------------


@pytest.mark.skipif(MPIEXEC is None, reason="mpiexec not available")
@pytest.mark.parametrize("n_ranks", [2, 3])
def test_global_sync_hides_the_decomposition(tmp_path, n_ranks):
    """The point of global_sync: a consumer sees one undivided mesh.

    Each rank writes only the nodes it owns into one collectively-sized
    global array, and the metadata describes that global array. So the
    archive must hold uniform steps -- no spatial collection, no per-rank
    grids -- carrying the global counts.
    """
    domain = read_domain(run_ranks(tmp_path, n_ranks, global_sync=True))
    (collection,) = domain["TimeCollections"]
    expected_nodes = sum(6 + r for r in range(n_ranks))
    expected_elements = sum(4 + r for r in range(n_ranks))
    for step in collection["Data"]:
        assert "SpatialCollection" not in step, \
            "the decomposition leaked into a global archive"
        assert step["Topology"]["NumberOfElements"] == expected_elements
        assert step["Geometry"]["DataItem"]["Dimensions"] == [expected_nodes, 3]
        (attr,) = step["Attributes"]
        assert attr["DataItem"]["Dimensions"] == [expected_nodes]


@pytest.mark.skipif(MPIEXEC is None, reason="mpiexec not available")
@pytest.mark.parametrize("n_ranks", [2, 3])
def test_global_sync_assembles_every_ranks_contribution(tmp_path, n_ranks):
    """The global array must actually contain every rank's values.

    A collective write with wrong offsets produces an array of the right
    shape with holes in it, which no amount of metadata checking catches.
    """
    datadir = run_ranks(tmp_path, n_ranks, global_sync=True)
    with h5py.File(os.path.join(datadir, "mpitest.h5"), "r") as f:
        u = f["u_t0"][:]
    expected = np.concatenate([np.arange(6 + r, dtype="d")
                               for r in range(n_ranks)])
    assert u.shape == expected.shape
    np.testing.assert_array_equal(u, expected)


@pytest.mark.skipif(MPIEXEC is None, reason="mpiexec not available")
def test_global_sync_datasets_carry_no_rank_in_their_names(tmp_path):
    # A consumer of a global archive should not need to know how many ranks
    # produced it.
    datadir = run_ranks(tmp_path, 2, global_sync=True)
    with h5py.File(os.path.join(datadir, "mpitest.h5"), "r") as f:
        names = [k for k in f if not k.startswith("Mesh_")]
    assert names, "no data datasets were written"
    assert not [n for n in names if "_p" in n], names


# --------------------------------------------------------------------------
# reading back: hot start supports exactly the two modes the archive writes
# --------------------------------------------------------------------------


def _archive_with_fields(tmp_path):
    """An archive holding one per-subdomain field and one global field."""
    path = str(tmp_path / "a.h5")
    with h5py.File(path, "w") as f:
        for r in range(4):
            f["u_p%d_t0" % r] = np.arange(3, dtype="d")
        f["v_t0"] = np.arange(12, dtype="d")
    return path


class _StubArchive:
    """Just what field_dataset touches."""

    def __init__(self, hdf, global_sync, rank, size):
        from proteus.Archiver import AR_base

        self.field_dataset = AR_base.field_dataset.__get__(self)
        self.hdfFile = hdf
        self.hdfFilename = "a.h5"
        self.global_sync = global_sync
        self.size = size

        class _Comm:
            def __init__(s, r):
                s._r = r

            def rank(s):
                return s._r

        self.comm = _Comm(rank)


def test_a_global_field_reads_back_at_any_task_count(tmp_path):
    path = _archive_with_fields(tmp_path)
    with h5py.File(path, "r") as f:
        for size in (1, 4, 9):
            ds = _StubArchive(f, True, 0, size).field_dataset("v", 0)
            assert ds.shape == (12,), "global mode must not depend on task count"


def test_a_per_subdomain_field_reads_back_at_the_matching_task_count(tmp_path):
    path = _archive_with_fields(tmp_path)
    with h5py.File(path, "r") as f:
        for rank in range(4):
            ds = _StubArchive(f, False, rank, 4).field_dataset("u", 0)
            assert ds.shape == (3,)


def test_a_per_subdomain_field_at_the_wrong_task_count_says_so(tmp_path):
    """The failure mode a user will actually hit, with a usable message.

    Rank i reads subdomain i, so a per-subdomain archive is only readable
    at the task count that wrote it. Without this the symptom is a bare
    KeyError on a mangled dataset name.
    """
    path = _archive_with_fields(tmp_path)
    with h5py.File(path, "r") as f:
        with pytest.raises(KeyError) as exc:
            _StubArchive(f, False, 5, 6).field_dataset("u", 0)
    message = str(exc.value)
    assert "same number of MPI tasks" in message
    assert "holds 4 subdomain" in message      # what the archive has
    assert "6 task" in message                 # what this run has


def test_reading_a_per_subdomain_archive_in_global_mode_says_so(tmp_path):
    path = _archive_with_fields(tmp_path)
    with h5py.File(path, "r") as f:
        with pytest.raises(KeyError, match="written per-subdomain"):
            _StubArchive(f, True, 0, 1).field_dataset("u", 0)


def test_the_pytables_hot_start_path_is_gone():
    """hdfFileGlb read fields with get_node, a PyTables method h5py lacks.

    It had been dead since the PyTables migration and silently so, because
    the file open was wrapped in a bare `except: pass`. Removed rather than
    repaired -- either supported write mode can be hot started from.
    """
    import inspect

    from proteus import Archiver, FemTools

    assert not hasattr(Archiver.AR_base, "hdfFileGlb")
    for module in (Archiver, FemTools):
        source = inspect.getsource(module)
        # the explanatory docstring may name it; live code must not use it
        for line in source.splitlines():
            stripped = line.strip()
            if stripped.startswith("#") or stripped.startswith("*"):
                continue
            assert "get_node(" not in stripped, line
            assert "ar.hdfFileGlb" not in stripped, line
