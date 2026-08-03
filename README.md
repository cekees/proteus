# Proteus: Computational Methods and Simulation Toolkit [![Build Status](https://travis-ci.com/cekees/proteus.svg?branch=main)](https://app.travis-ci.com/github/cekees/proteus) [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/erdc/proteus_tutorial/master?filepath=index.ipynb)  [![DOI](https://zenodo.org/badge/2212385.svg)](https://zenodo.org/badge/latestdoi/2212385)


Proteus (http://proteustoolkit.org) is a Python package for
rapidly developing computer models and numerical methods.

# Installation

## conda / mamba

```bash
mamba install proteus -c conda-forge
```

For a development installation, you want to install Proteus's dependencies and compile Proteus from source:

```bash
mamba env create -f environment-dev.yml #environment-dev-up.yml to try unpinned dependencies
mamba activate proteus-dev
pip install -v -e .
```

`environment-dev.yml` pins conda-forge builds of everything proteus links
against (PETSc, MPI, HDF5, SuperLU/SuperLU_DIST, METIS/ParMETIS, Chrono,
SCOREC, Triangle/TetGen, xtensor, ...), so this is the path to use if you
want the optional Chrono (multibody/FSI) and SCOREC (mesh adaptation)
support built in. It intentionally excludes the `defaults` channel
(`nodefaults`) -- conda-forge alone resolves everything here, and
`defaults` pulls in `repo.anaconda.com`, a commercial channel with its own
Terms of Service and registration-gated rate limits.

## pip

Proteus is not a pure-Python package: most of it is C/C++/Cython/Fortran
extensions linking against PETSc, MPI, HDF5, SuperLU/SuperLU_DIST,
METIS/ParMETIS, and BLAS/LAPACK. All of that can be provisioned with pip
alone (no conda, no system package manager beyond a C/C++/Fortran compiler
and `make`) using PETSc's own `--download-x` configure options, exposed to
its PyPI package via the `PETSC_CONFIGURE_OPTIONS` environment variable.
This builds everything from source, so expect it to take a while.

Chrono and SCOREC (proteus's optional multibody/FSI and mesh-adaptation
support) don't have a pip-installable path yet; the recipe below skips them
via `PROTEUS_SKIP_PUMI_CHRONO=1`. Use the conda/mamba or HPC paths above if
you need those.

```bash
python -m venv proteus-env && source proteus-env/bin/activate
pip install cython pybind11 wheel numpy mpi4py "cmake>=3.29"

# xtensor/xtl/xtensor-python aren't in upstream PETSc (proteus's plan is to
# drop this dependency; until then, install from our fork instead of PyPI's
# own `petsc` package, since only this fork's PETSc knows about them):
export PETSC_CONFIGURE_OPTIONS="--download-fblaslapack --download-superlu --download-superlu_dist --download-metis --download-parmetis --download-hdf5 --download-triangle --download-triangle-build-exec=1 --download-tetgen --download-tetgen-build-exec=1 --download-xtl --download-xtensor --download-xtensor-python"
pip install "petsc @ git+https://gitlab.com/cekees/petsc.git@download-proteus-support"
pip install petsc4py

# h5py's own PyPI wheels are serial-only; build against the parallel HDF5
# PETSc just downloaded and compiled above instead:
PETSC_DIR=$(python -c "import petsc; print(petsc.get_petsc_dir())")
# Work around a packaging quirk in the `petsc` wheel: its libhdf5.so is a
# linker script (`INPUT(libhdf5.so.NNN)`), not a real file, which trips up
# h5py's own HDF5 version/config introspection at build time.
real_hdf5=$(readlink -f "$PETSC_DIR"/lib/libhdf5.so.*.* 2>/dev/null | head -1)
ln -sf "$(basename "$real_hdf5")" "$PETSC_DIR/lib/libhdf5.so"
CC=mpicc HDF5_MPI=ON HDF5_DIR="$PETSC_DIR" pip install --no-binary h5py h5py

export PETSC_ARCH=""
export PROTEUS_PREFIX="$PETSC_DIR"
export PROTEUS_SKIP_PUMI_CHRONO=1
export PATH="$PETSC_DIR/bin:$PATH"
pip install --no-build-isolation --no-deps .
```

If your PETSc is already built (e.g. by an HPC site, or you don't need
xtensor and can use PyPI's own `petsc` package directly), skip straight to
the `petsc4py`/`h5py`/proteus steps with `PETSC_DIR` (and `PETSC_ARCH`, if
set) pointing at that install; `PETSC_CONFIGURE_OPTIONS` is only consulted
when `petsc4py`'s own install triggers a fresh PETSc build.

This path is new and less battle-tested than the conda/mamba and HPC
`--download-proteus` routes above -- if something here breaks, those are
the more mature fallbacks.

# HPC Installation

For installation on high performance environments, the recommended path is
PETSc's own build system, which most HPC sites and module systems already
support: `git@gitlab.com:cekees/petsc.git`, branch `download-proteus-support`,
adds a `--download-proteus` PETSc package (plus PETSc packages for proteus's
optional native dependencies: chrono, scorec, xtensor) so a single
`./configure` + `make` builds PETSc, proteus, and everything in between,
instead of the separate per-dependency manual build this section used to
describe.

```bash
git clone git@gitlab.com:cekees/petsc.git
cd petsc
git checkout download-proteus-support
CONDA_PREFIX=/path/to/your/conda/env PREFIX=/path/to/install ./configure_macos_arm64.sh   # macOS/arm64
make PETSC_DIR="$(pwd)" PETSC_ARCH=arch-darwin-download-proteus all
make PETSC_DIR="$(pwd)" PETSC_ARCH=arch-darwin-download-proteus install
```

See that branch's `README_PROTEUS.md` for prerequisites, post-install steps,
activating the resulting environment, running proteus's test suite, and —
if you have your own proteus checkout you're actively developing rather
than wanting `--download-proteus`'s own fresh clone — the development-build
workflow (build PETSc and proteus's dependencies without `--download-proteus`,
then `pip install` your own checkout against that prefix directly).

Only macOS/arm64 has been validated end-to-end so far; other platforms are
expected to need at most flag adjustments (MPI/BLAS locations) rather than
changes to the package definitions themselves, but that hasn't been
confirmed yet. If you're building on a platform other than macOS/arm64 and
hit trouble, that branch's `README_PROTEUS.md` and the comments in
`config/BuildSystem/config/packages/proteus.py`/`scorec.py` are the places
to start (in particular, a couple of post-install fixups in there are
known to be macOS/dyld-specific workarounds, not applicable elsewhere).

The package definitions themselves are plain PETSc packages and don't
depend on that specific fork -- if your site already has its own PETSc
checkout, copying `config/BuildSystem/config/packages/{proteus,chrono,scorec,xtl,xtensor,xtensor-python,numpy,h5py}.py`
into it works the same way.

See https://github.com/erdc/proteus/wiki/How-to-Build-Proteus for old information on building the entire stack by hand.

# Developer Information

The source code, wiki, and issue tracker are on GitHub at

https://github.com/erdc/proteus.
