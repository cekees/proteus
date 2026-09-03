#!/usr/bin/env python
"""Regression tests for the one-dimensional Richards benchmarks.

Three benchmarks, one deck pair each, with the scheme chosen through proteus's
Context rather than through duplicated decks:

===============  =======================  ==========================  =========
case             deck directory           reference figure            schemes
===============  =======================  ==========================  =========
test_1           test_1/                  Celia et al. (1990) Fig 6b  all three
test_2_HYDRUS    test_2_HYDRUS/           HYDRUS-1D, 20 m column      no stab_0
test_3           test_3/                  Szymkiewicz (2009) Fig 6    no stab_0
===============  =======================  ==========================  =========

The scheme names are the (STABILIZATION_TYPE, FCT) pair each deck reads off its
Context options:

* ``stab_0``  ``STABILIZATION_TYPE=0``, ``FCT=False`` -- plain Galerkin
* ``stab_2``  ``STABILIZATION_TYPE=2``, ``FCT=False`` -- entropy viscosity
* ``FCT``     ``STABILIZATION_TYPE=2``, ``FCT=True``  -- entropy viscosity + FCT

``beta`` is 0 in all three decks, so the schemes differ by stabilization alone.
The same decks run standalone under parun, e.g.

    parun re_vgm_sand_10m_1d_p.py re_vgm_sand_10m_1d_c0p1_n.py \\
          -C "STABILIZATION_TYPE=0 FCT=False"

Each case has ONE comparison file, ``comparison_files/richards_<case>.csv``,
holding the node coordinate plus one column per (scheme, output frame).  To
(re)generate them::

    PROTEUS_SAVE_COMPARISON=1 python -m pytest test_richards.py

Saving merges: a run of a single scheme rewrites only that scheme's columns and
leaves the rest of the file alone, so ``-k stab_0`` regenerates just those.
Inspect the diff before committing -- a regenerated file silently blesses
whatever the code does today.
"""

import csv
import os
import sys
from pathlib import Path

import h5py
import numpy as np
import pytest

from proteus import Context, defaults
from proteus.iproteus import NumericalSolution, default_s, opts


HERE = Path(__file__).resolve().parent
COMPARISON_DIR = HERE / "comparison_files"

P_MODULE = "re_vgm_sand_10m_1d_p"
N_MODULE = "re_vgm_sand_10m_1d_c0p1_n"

SAVE_COMPARISON = os.environ.get("PROTEUS_SAVE_COMPARISON", "") not in (
    "",
    "0",
    "no",
    "false",
    "False",
)

# scheme -> the Context override string the decks parse.  Context.Options splits
# on single spaces and ast.literal_eval's each right-hand side, so no spaces
# around "=" and no quotes.
SCHEMES = {
    "stab_0": "STABILIZATION_TYPE=0 FCT=False",
    "stab_2": "STABILIZATION_TYPE=2 FCT=False",
    "FCT": "STABILIZATION_TYPE=2 FCT=True",
}
ALL_SCHEMES = ("stab_0", "stab_2", "FCT")
FCT_AND_STAB_2 = ("stab_2", "FCT")

CASE_DIRS = {
    "test_1": HERE / "test_1",
    "test_2_HYDRUS": HERE / "test_2_HYDRUS",
    "test_3": HERE / "test_3",
}

# Archive frames to compare.  Every deck writes nDTout=1000 outputs over its own
# T, so frame i is at t = i*T/1000.  The final frame alone is a weak regression
# target for test_2_HYDRUS -- by 48 h that column has wet through and relaxed
# onto psi = 0 everywhere -- so the moving front is sampled as well.
CASE_FRAMES = {
    "test_1": (500, 1000),  # T = 1 d:     12 h, 24 h (the Celia comparison time)
    "test_2_HYDRUS": (104, 229, 1000),  # T = 2 d: ~5 h, ~11 h, 48 h
    "test_3": (500, 1000),  # T = 0.125 d: 1.5 h, 3 h (the Fig. 6 time)
}

CASE_SCHEMES = {
    "test_1": ALL_SCHEMES,
    "test_2_HYDRUS": FCT_AND_STAB_2,
    "test_3": FCT_AND_STAB_2,
}

# The FCT limiter clips against a bound, so nodes sitting almost exactly on that
# bound can flip sides under a tiny cross-build arithmetic difference; the
# unlimited schemes reproduce to round-off.  Same reasoning as the
# STABILIZATION_TYPE=4 note in test/TADR/test_tadr.py.
DECIMALS = {"stab_0": 8, "stab_2": 8, "FCT": 8}

# The FCT cases used to be skipped on macOS ("not reproducible"), which was a
# misreading: the macOS columns were not a drifted answer at all but a frozen
# initial condition with -pcBarMax/alpha = -2985.07 m at the four boundary DOFs.
# calculateResidual read the never-written stack array fluxJacobian_un_un into
# TransportMatrixConsistentn, which is NaN on macOS and zero on Linux; the
# low-order schemes hid it (fmax(0.0, -NaN) is 0.0) and FCT did not (0.0 * NaN
# is NaN).  Fixed in Richards.h, so all three schemes run on every platform.


def _column(scheme, frame):
    return "{0}_t{1}".format(scheme, frame)


def _comparison_path(case):
    return COMPARISON_DIR / "richards_{0}.csv".format(case)


# How many mismatching nodes to list before truncating.  The meshes are 41, 101
# and 11 nodes, so this shows the whole story on test_1 and test_3 and the head
# of it on test_2_HYDRUS; the worst node is reported either way.
MAX_REPORTED = 20


def _assert_probes(case, scheme, name, x, expected, computed):
    """assert_almost_equal over the profile, naming the node behind every miss.

    numpy prints the flat array position and the first five misses in index
    order, which does not say where in the column the two runs parted or how
    far apart they ended up.  Each mismatch is probed against its mesh node and
    coordinate here, in node order, so the drift can be read straight off a CI
    log: on a wetting front the differences grow from round-off ahead of the
    front to O(0.1) at it, and that shape is the diagnosis.
    """
    # The threshold assert_almost_equal itself applies:
    # abs(desired - actual) < 1.5 * 10**(-decimal).
    tolerance = 1.5 * 10.0 ** (-DECIMALS[scheme])
    difference = np.abs(computed - expected)
    violations = np.nonzero(difference > tolerance)[0]
    if violations.size == 0:
        return

    worst = int(difference.argmax())
    lines = [
        "",
        "{0} {1}: {2} of {3} nodes differ by more than {4:g} (decimal={5})"
        .format(case, name, violations.size, difference.size, tolerance,
                DECIMALS[scheme]),
        "  {0:>4s} {1:>9s} {2:>21s} {3:>21s} {4:>12s}".format(
            "node", "x", "computed", "expected", "|difference|"),
    ]
    for node in violations[:MAX_REPORTED]:
        lines.append("  {0:4d} {1:9.5g} {2:21.15g} {3:21.15g} {4:12.3e}".format(
            int(node), x[node], computed[node], expected[node],
            difference[node]))
    if violations.size > MAX_REPORTED:
        lines.append("  ... {0} further nodes not listed".format(
            violations.size - MAX_REPORTED))
    lines.append("  worst |difference| {0:.3e} at node {1} (x={2:g})".format(
        difference[worst], worst, x[worst]))
    raise AssertionError("\n".join(lines))


def _read_comparison(case):
    """Return {column name: np.ndarray} for one case, or {} if not written yet."""
    path = _comparison_path(case)
    if not path.exists():
        return {}
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.reader(stream))
    header, body = rows[0], rows[1:]
    values = np.array([[float(v) for v in row] for row in body])
    return {name: values[:, i] for i, name in enumerate(header)}


def _write_comparison(case, columns):
    """Write one case's file, x first then (scheme, frame) in a fixed order."""
    COMPARISON_DIR.mkdir(exist_ok=True)
    names = ["x"] + [
        _column(scheme, frame)
        for scheme in CASE_SCHEMES[case]
        for frame in CASE_FRAMES[case]
        if _column(scheme, frame) in columns
    ]
    with _comparison_path(case).open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(names)
        for row in zip(*[columns[name] for name in names]):
            # %.17g round-trips a float64 exactly, so a regenerated file and the
            # run that produced it agree to the last bit.
            writer.writerow(["%.17g" % value for value in row])


def _load_case(case, scheme):
    """Load the physics/numerics decks for one benchmark at one scheme."""
    case_dir = str(CASE_DIRS[case])

    for module_name in (P_MODULE, N_MODULE):
        sys.modules.pop(module_name, None)

    saved_path = sys.path[:]
    # The decks call Context.Options at import, which reads this module global --
    # the same channel parun's -C writes.  defaults.load_source re-executes the
    # deck on every call and never caches it, so setting the string immediately
    # before the load is enough; no reload() and no ordering coupling between
    # parameterized cases.
    Context.contextOptionsString = SCHEMES[scheme]
    try:
        # load_physics appends the deck directory to the END of sys.path and then
        # drops it again with sys.path.remove(), which deletes the FIRST
        # occurrence -- ours -- so the insert has to be repeated per loader call.
        sys.path.insert(0, case_dir)
        physics = defaults.load_physics(P_MODULE, case_dir)

        # The numerics deck does "from re_vgm_sand_10m_1d_p import *".  Left to
        # the import machinery that resolves to test/richards's own opts-driven
        # re_vgm_sand_10m_1d_p.py (pytest puts the test's directory on the front
        # of sys.path), which does not define `galerkin` and describes a
        # different column.  Publishing the physics we just loaded under the deck
        # name makes the numerics deck read back exactly this case -- and reuse
        # the coefficients object built at the scheme we just selected.
        sys.modules[P_MODULE] = physics
        sys.path.insert(0, case_dir)
        numerics = defaults.load_numerics(N_MODULE, case_dir)
    finally:
        Context.contextOptionsString = None
        sys.path[:] = saved_path
        sys.modules.pop(P_MODULE, None)

    # The deck is the authority on what it built; if a scheme override silently
    # failed to apply, every later assertion would compare the wrong scheme.
    expected = dict(
        pair.split("=") for pair in SCHEMES[scheme].split(" ")
    )
    assert physics.coefficients.STABILIZATION_TYPE == int(
        expected["STABILIZATION_TYPE"]
    ), "{0}/{1}: deck built STABILIZATION_TYPE={2}".format(
        case, scheme, physics.coefficients.STABILIZATION_TYPE
    )
    assert bool(physics.coefficients.FCT) == (expected["FCT"] == "True"), (
        "{0}/{1}: deck built FCT={2}".format(case, scheme, physics.coefficients.FCT)
    )
    return physics, numerics


def _archive(output_dir, name):
    for candidate in (output_dir / (name + ".h5"), output_dir / (name + "global.h5")):
        if candidate.exists():
            return candidate
    raise AssertionError(
        "Richards run produced no archive for {0} in {1}: {2}".format(
            name, output_dir, sorted(p.name for p in output_dir.glob("*.h5"))
        )
    )


def _run_case(case, scheme, output_dir, monkeypatch):
    """Run one benchmark at one scheme and return the path to its archive."""
    monkeypatch.chdir(output_dir)
    physics, numerics = _load_case(case, scheme)

    system = defaults.System_base()
    system.name = physics.name = "richards_{0}_{1}".format(case, scheme)
    system.sList = [default_s]
    system.tnList = numerics.tnList

    opts.logLevel = 1
    opts.verbose = False
    opts.profile = False
    opts.gatherArchive = True

    numerical_solution = NumericalSolution.NS_base(
        system, [physics], [numerics], system.sList, opts
    )
    numerical_solution.calculateSolution(system.name)
    del numerical_solution

    return _archive(output_dir, system.name)


def _compare(case, scheme, archive):
    """Compare, or save, this scheme's columns of the case's comparison file."""
    with h5py.File(archive, "r") as actual:
        computed = {}
        for frame in CASE_FRAMES[case]:
            field = "pressure_head_t{0}".format(frame)
            assert field in actual, "{0} missing from {1}".format(field, archive.name)
            computed[_column(scheme, frame)] = actual[field][:]
        x = actual["nodesSpatial_Domain0"][:, 0]

    if SAVE_COMPARISON:
        columns = _read_comparison(case)
        if columns and columns["x"].shape != x.shape:
            columns = {}  # the mesh changed; the other schemes' columns are stale
        columns["x"] = x
        columns.update(computed)
        _write_comparison(case, columns)
        pytest.skip("wrote {0} columns of {1}".format(
            len(computed), _comparison_path(case).name))

    columns = _read_comparison(case)
    assert columns, (
        "missing comparison file {0}; regenerate with PROTEUS_SAVE_COMPARISON=1 "
        "python -m pytest {1}".format(_comparison_path(case), Path(__file__).name)
    )
    np.testing.assert_allclose(
        columns["x"], x, rtol=0, atol=1.0e-12,
        err_msg="{0}: the deck's mesh no longer matches the comparison file, so "
                "it has to be regenerated".format(case),
    )
    for name, values in computed.items():
        assert name in columns, (
            "{0} has no column {1}; regenerate with PROTEUS_SAVE_COMPARISON=1"
            .format(_comparison_path(case).name, name)
        )
        _assert_probes(case, scheme, name, x, columns[name], values)


class TestRichards:
    @pytest.mark.parametrize("scheme", CASE_SCHEMES["test_1"])
    def test_1(self, scheme, tmp_path, monkeypatch):
        archive = _run_case("test_1", scheme, tmp_path, monkeypatch)
        _compare("test_1", scheme, archive)

    @pytest.mark.parametrize("scheme", CASE_SCHEMES["test_2_HYDRUS"])
    def test_2_HYDRUS(self, scheme, tmp_path, monkeypatch):
        archive = _run_case("test_2_HYDRUS", scheme, tmp_path, monkeypatch)
        _compare("test_2_HYDRUS", scheme, archive)

    @pytest.mark.parametrize("scheme", CASE_SCHEMES["test_3"])
    def test_3(self, scheme, tmp_path, monkeypatch):
        archive = _run_case("test_3", scheme, tmp_path, monkeypatch)
        _compare("test_3", scheme, archive)
