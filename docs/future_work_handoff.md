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

## 2. Replace xtensor with a native proteus module — DONE

Done on the `remove-xtensor` branch. `proteus/pyarray.h` defines
`proteus::pyarray<T>`, a non-owning numpy-array view built on pybind11 and
the numpy C-API only, and every `xt::pyarray<T>` in the tree now uses it.
`xtl`/`xtensor`/`xtensor-python` are gone from proteus's dependency list
(`setup.py`'s `get_xtensor_include()` became `get_pybind_include_dirs()`,
and the `environment-*-dev.yml` files list `pybind11` directly instead of
getting it transitively from `xtensor-python`).

The audit that made this small: of xtensor's API, the 8,500-odd
`xt::pyarray` sites used only `data()`, `size()`, `shape(i)`, `operator[]`
and `operator()`. Five sites used xtensor *expressions* and were rewritten
as plain loops over `std::valarray` (`xt::where`/broadcast arithmetic in
`SW2DCV.h`, `xt::xarray` + `xt::amax` in `RANS2P.h`/`RANS2P2D.h`) — the
same idiom the surrounding code in those files already used.

Two things worth knowing about the replacement:

- `operator[](i)` and `operator()(i0, i1, ...)` reproduce xtensor's index
  alignment rule exactly (fewer indices than rank align with the *trailing*
  dimensions; more than rank drops the leading ones), which is what makes
  `arr[i]` a flat index and lets it agree with the `arr.data()[i]` the
  kernels use interchangeably on the same array.
- It deliberately does **not** reproduce xtensor's conversion behaviour.
  `xt::pyarray`'s caster used `PyArray_FromAny(..., NPY_ARRAY_FORCECAST)`,
  so it accepted any array-like — a tuple, a list, a float32 array where
  double was wanted, a strided slice — by silently making a converted copy.
  Harmless for an input array, silently wrong for an output one (the kernel
  writes into the copy). `proteus::pyarray` converts nothing. That
  immediately surfaced eleven real `argsDict[...] = arr,` typos on the
  Python side (a trailing comma makes the value a 1-tuple) in
  `RANS3PF.py`, `RANS2P_IB.py`, `PresInc.py`, `RANS3PSed.py`,
  `richards/ADR.py` and `richards/Richards.py`; all are fixed.

**Remaining, and it has an ordering constraint:** the PETSc fork
(`cekees/petsc`, branch `download-proteus-support`) still carries
`xtl.py`/`xtensor.py`/`xtensor-python.py`, `proteus.py`'s dependency on
`xtensorpython`, and the `--download-xtl --download-xtensor
--download-xtensor-python` flags in `configure_macos_arm64.sh` /
`README_PROTEUS.md`. Those edits are prepared but must not land until this
proteus change is merged, because `proteus.py` pins `self.gitcommit =
'main'` — dropping xtensor from the fork while proteus `main` still
includes xtensor headers would break `--download-proteus`. Removing
`xtensor-python.py` also drops that file's `self.pybind11.version =
'2.13.6'` pin, which is the only thing holding pybind11 back to 2.x in the
BuildSystem path.

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
