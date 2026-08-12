# Handoff: follow-up work from the --download-proteus packaging session

Written 2026-07-29. This session got proteus building end-to-end via
PETSc's `--download-proteus` (see `git@gitlab.com:cekees/petsc.git`,
branch `download-proteus-support`, and that branch's `README_PROTEUS.md`),
found and fixed a batch of real bugs along the way (see `git log` on
`torino_narwhal` from `baf4e8ff` through `13d526b1`), and along the way
turned up four pieces of follow-up work that are each substantial enough
to be their own session. This note hands each of them off with enough
context to start cold.

## 1. Editable (`-e`) installs for developers

**Current state:** the development-build workflow documented in
`README_PROTEUS.md` uses `pip install --no-build-isolation --no-deps
--target=$PREFIX/lib .` — the same thing `proteus.py`'s own `Install()`
runs for `--download-proteus`. This works, but it's not editable: a `.py`
change needs the install command rerun (fast, since nothing recompiles)
to take effect, and there's a real footgun where running Python from
inside the checkout directory can silently shadow the installed copy with
stale files (`sys.path[0]=''` — see the "import-shadowing" warning in both
READMEs). Neither of these is what a `pip install -e .` workflow would
give you.

**Why it's not simply `pip install -e .` today:** proteus's non-PETSc-
downloaded-but-PETSc-built dependencies (h5py, mpi4py, petsc4py, and
proteus's own C/C++ extensions) currently only resolve via
`PYTHONPATH=$PREFIX/lib` pointing at the `--target=` install location —
they're not installed anywhere that's naturally on `sys.path` for a given
Python interpreter (e.g. the conda env's own site-packages). `pip`'s `-e`
and `--target` flags are mutually exclusive, so as long as those
dependencies live under `$PREFIX/lib`, an editable proteus install would
still need `PYTHONPATH=$PREFIX/lib` for everything *except* proteus
itself, which is a confusing half-measure.

**Where to start:**
- The cleanest fix is probably to stop using `--target=` for the
  PETSc-downloaded Python packages too, and instead have them install
  directly into the conda/mamba environment's own site-packages (i.e. the
  same Python that `--with-python-exec` points at). Check whether PETSc's
  `PythonPackage` base class (`config/BuildSystem/config/package.py`) or
  `numpy.py`/`h5py.py`/`mpi4py`/`petsc4py`'s own package definitions
  support that directly, or whether it needs a package.py change.
- If that works, `proteus.py`'s own `Install()` could then just run
  `pip install --no-build-isolation --no-deps -e .` for `--download-proteus`
  itself (still against its own fresh clone), and the development workflow
  in `README_PROTEUS.md` would become "run that same command against your
  own checkout" — no `--target=`, no `PYTHONPATH` juggling, no shadowing
  footgun.
- Worth checking early: does proteus's `setup.py`/`pyproject.toml` support
  editable installs cleanly at all (some Cython-heavy packages need
  `--no-build-isolation` plus care with `build_ext --inplace` for editable
  mode to actually find compiled extensions)? Validate on a small
  extension before assuming it'll work for the whole package.

## 2. Replace xtensor with a native proteus module

**Current state:** ~40 of proteus's C++ extensions (`ArgumentsDict.h`,
`RANS2P.h`, `RANS2P2D.h`, and everything that includes them —
see `get_xtensor_include()` in `setup.py`) `#include` xtensor/xtensor-python
headers directly. This pulls in three extra PETSc packages
(`xtl.py`, `xtensor.py`, `xtensor-python.py`) purely for their headers —
nothing in proteus links against a compiled xtensor library, it's a
header-only dependency used for its array-view type.

This has already caused real friction this session: `xtensor-python`
0.28.0 doesn't compile against pybind11 3.x (`xtensor-python.py` has to
pin `pybind11.version = '2.13.6'` to work around it), which is exactly the
kind of transitive-version-compatibility problem that motivates removing
the dependency rather than continuing to pin around it.

**The plan** (already noted in `proteus.py`'s own comments): replace
xtensor's array-view usage with a proteus-owned, PyArrayView-style header
built only on pybind11 and numpy — both already hard dependencies of
proteus regardless of xtensor. This drops `xtl`/`xtensor`/`xtensor-python`
from the dependency list entirely.

**Where to start:**
- Find every actual xtensor API surface used in `ArgumentsDict.h`/
  `RANS2P.h`/`RANS2P2D.h` (likely just `xt::pyarray<double>`-style views
  over numpy arrays passed from Python, plus whatever indexing/slicing
  operations proteus actually calls on them — audit rather than assume,
  xtensor's API surface is large but proteus's usage of it is probably
  narrow).
- Design the replacement header to cover exactly that usage, using
  pybind11's own `py::array_t<double>` (or a thin wrapper around it) —
  this is a much smaller surface to implement and maintain than
  reimplementing xtensor generally.
- This is a mechanical-but-widespread change (touches ~40 extensions'
  worth of `#include` and usage sites), so plan for a dedicated pass with
  a clear "compiles and every affected test still passes" checkpoint,
  rather than doing it incrementally alongside other work.
- Once done: remove `xtl.py`/`xtensor.py`/`xtensor-python.py` from
  `config/BuildSystem/config/packages/`, drop the corresponding
  `--download-*` flags from `configure_macos_arm64.sh` and
  `README_PROTEUS.md`, and drop `proteus.py`'s dependency on
  `xtensorpython`.

## 3. Re-enable skipped tests

A full-repo scan for `@pytest.mark.skip` (excluding a couple of already
commented-out, inactive ones) turned up 16 skipped tests across 9 files.
None of these were touched this session — they're a separate, pre-existing
backlog, grouped here by likely cause:

**High-confidence candidates for re-enabling now**, since the thing they
say is broken has since been fixed/validated this session:
- `test/TwoPhaseFlow/test_TwoPhaseFlow.py::test_damBreak_genPUMI`,
  `::test_damBreak_runPUMI` — skipped with reason `"PUMI is broken"`. This
  session found and fixed the actual PUMI/PCU bugs (see `torino_narwhal`
  commits on `ErrorResidualMethod.cpp`/`partitioning.cpp`, plus the
  scorec.py rpath fixes on the PETSc side) and confirmed the full
  MeshAdaptPUMI test suite passes (21/21). Worth trying these first.
- `test/test_mbd_chrono.py::testHangingCableANCF` and one more in the same
  file (no skip reason given) — uses `proteus.mbd.CouplingFSI` and
  `pychrono` directly. Chrono is now confirmed working this session
  (AddedMass tests pass). No stated reason for the skip, so it's not
  guaranteed to be chrono-availability related — investigate what
  actually fails before assuming it "just works" now.

**Needs investigation into a shared root cause** — six skips across five
files all give the identical reason `"need to redo after history
revision"`, which reads like a past git history rewrite (rebase/
filter-branch, or a large refactor) broke a shared assumption (moved
comparison-file paths, changed fixture/import structure, a renamed API)
and they were mass-skipped rather than fixed individually:
- `test/test_spatialtools.py` (3: `test_create_shapes`,
  `test_assemble_domain`, one more)
- `test/cylinder2D/ibm_rans2p/test_cylinder2D_ibm_rans2p.py`
- `test/cylinder2D/ibm_method/test_cylinder2D_ibm_rans3p.py`
- `test/cylinder2D/ibm_rans2p_3D/test_cylinder3D_ibm_rans2p.py`
- `test/FSI/test_FSI.py` (2)

Worth checking whether these all fail the same way (same error/exception)
before fixing them one at a time — if so, there's likely one shared fix
(e.g. a shared helper function or fixture that needs updating) rather than
five separate ones.

**Individually-reasoned skips, lower priority / need their own
investigation:**
- `test/CLSVOF/with_RANS2P/test_clsvof_with_rans2p.py` — `"Not
  reproducible on both python2 and python3"`. Python 2 has been dead for
  years; the original reason is almost certainly obsolete, but the test
  itself needs to actually be run and checked, not just un-skipped blindly.
- `test/LS_with_edgeBased_EV/MCorr/test_mcorr.py::test_mcorr` — `"results
  can't be reproduced reliably"`. This sounds like it could be the same
  class of tolerance/version-drift issue covered in
  `docs/test_tolerance_and_reliability_notes.md` — worth checking with
  that lens (does it fail by a small, tolerance-shaped amount, or does it
  actually produce qualitatively different results run-to-run?) before
  assuming either "just loosen the tolerance" or "this is a real flaky
  test" is the right framing.
- `test/ci/test_Isosurface.py` — skipped with no reason given at all.
  Needs a first look just to find out why.
- `test/SWFlow/test_SWFlow.py::test_obstacle_flow` — the one skip added
  *this* session (mesh shape mismatch from a different `triangle` CLI
  version). The user's own stated plan: store a fixed reference mesh via
  git-lfs instead of regenerating with `triangle` on every run, matching
  the pattern already used for other tests in this suite, then un-skip.

## 4. Test framework refactorization

Fully written up already in `docs/test_tolerance_and_reliability_notes.md`
— that note covers the inconsistent `assert_almost_equal(decimal=N)` vs.
`assert_allclose(atol, rtol)` conventions, the specific tolerance values
calibrated this session (with the two loosest ones flagged for a closer
look), and a five-point refactor plan (standardize on `assert_allclose`,
tie tolerance floors to double-precision reality, distinguish "should
reproduce tightly" from "solver-path-sensitive" tests explicitly,
regenerate-and-record rather than just loosen, and make full-suite runs
against fresh dependencies a periodic practice rather than an incident
response). Start there rather than re-deriving the plan from scratch.

One thing to add to that plan, informed by item 3 above: a
"refactorization" pass is also a natural place to build whatever shared
helper/fixture ends up fixing the six "need to redo after history
revision" skips, so that future comparison-test additions use one
consistent, well-tested comparison helper instead of five different
ad hoc `assert_almost_equal`/`assert_allclose`/`np.isclose` call sites
copy-pasted around the suite.
