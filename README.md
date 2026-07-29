# Proteus: Computational Methods and Simulation Toolkit [![Build Status](https://travis-ci.com/cekees/proteus.svg?branch=main)](https://app.travis-ci.com/github/cekees/proteus) [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/erdc/proteus_tutorial/master?filepath=index.ipynb)  [![DOI](https://zenodo.org/badge/2212385.svg)](https://zenodo.org/badge/latestdoi/2212385)


Proteus (http://proteustoolkit.org) is a Python package for
rapidly developing computer models and numerical methods.

# Installation



```bash
mamba install proteus -c conda-forge
```

For a development installation, you want to install Proteus's dependencies and compile Proteus from source:

```bash
mamba env create -f environment-dev.yml #environment-dev-up.yml to try unpinned dependencies
mamba activate proteus-dev
pip install -v -e .
```

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
