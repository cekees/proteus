"""Build configuration for cross-compiling proteus to emscripten-wasm32.

Selected with ``PROTEUS_ARCH=emscripten`` (see ``proteus/config/__init__.py``).

This targets the in-browser/JupyterLite build, which is deliberately a much
smaller slice of proteus than a normal install:

* **No PETSc, no SCOREC/PUMI, no Chrono, no DAETK, no METIS.** The
  serial, direct-solver path (``LinearSolvers.LU`` on SuperLU, proteus's
  own default per ``default_n.py``) needs none of them, and none of the
  C/C++ sources actually built here ``#include`` any PETSc header --
  those entries in ``setup.py``'s ``Extension()`` definitions are
  vestigial, so pointing them at empty lists here is enough to drop them
  without touching ``setup.py``.
* **BLAS/LAPACK come from f2cblaslapack**, not OpenBLAS or reference
  LAPACK. Both of those are compiled from real Fortran (flang on this
  toolchain) and so pass a hidden trailing character-length argument per
  CHARACTER dummy; wasm-ld's cross-module signature check and the wasm
  VM's own runtime import type-check both reject that mismatch where
  native ELF silently tolerates it. f2cblaslapack is transpiled to C, so
  there is no Fortran ABI to match at all. (PETSc's own official
  Emscripten instructions make the same choice --
  ``--download-f2cblaslapack=1``.)
* **Everything is linked statically.** A wasm side module that pulls in
  additional ``.so`` dependencies has to have them fetched and linked by
  the browser's dynamic loader at import time; keeping the dependencies
  static side-steps that entirely.

MPI is still present (mpi-serial, via mpi4py) because ``cmeshTools.pyx``
imports ``mpi4py.MPI`` at module scope -- but only at the *Python* level;
no C source here needs an MPI header, so the MPI include/lib lists are
empty too.
"""

import os
import sys
from os.path import join as pjoin

prefix = os.getenv('PROTEUS_PREFIX')
if not prefix:
    prefix = sys.exec_prefix

PROTEUS_PRELOAD_LIBS = []
PROTEUS_INCLUDE_DIR = pjoin(prefix, 'include')
PROTEUS_LIB_DIR = pjoin(prefix, 'lib')

PROTEUS_OPT = os.getenv('PROTEUS_OPT')
if not PROTEUS_OPT:
    PROTEUS_OPT = []
else:
    PROTEUS_OPT = PROTEUS_OPT.split()

# -mavx and friends are meaningless for wasm32; emcc has its own
# vectorisation flag (-msimd128), which the toolchain's own CFLAGS
# already set where it wants it.
PROTEUS_EXTRA_COMPILE_ARGS = ['-DF77_POST_UNDERSCORE',
                              '-DUSE_BLAS',
                              '-DCMRVEC_BOUNDS_CHECK',
                              '-DMV_VECTOR_BOUNDS_CHECK']
PROTEUS_EXTRA_LINK_ARGS = ['-L' + PROTEUS_LIB_DIR]

PROTEUS_EXTRA_FC_COMPILE_ARGS = []
PROTEUS_EXTRA_FC_LINK_ARGS = []

# --- BLAS / LAPACK (f2cblaslapack: transpiled to C, static) --------------
PROTEUS_BLAS_INCLUDE_DIR = PROTEUS_INCLUDE_DIR
PROTEUS_BLAS_LIB_DIR = PROTEUS_LIB_DIR
PROTEUS_BLAS_LIB = 'f2cblas'
PROTEUS_BLAS_H = r'"proteus_blas.h"'

PROTEUS_LAPACK_INCLUDE_DIR = PROTEUS_INCLUDE_DIR
PROTEUS_LAPACK_LIB_DIR = PROTEUS_LIB_DIR
PROTEUS_LAPACK_LIB = 'f2clapack'
PROTEUS_LAPACK_H = r'"proteus_lapack.h"'
PROTEUS_LAPACK_INTEGER = 'int'

# --- SuperLU (sequential, static, built with its own internal CBLAS) -----
PROTEUS_SUPERLU_INCLUDE_DIR = PROTEUS_INCLUDE_DIR
PROTEUS_SUPERLU_LIB_DIR = PROTEUS_LIB_DIR
PROTEUS_SUPERLU_LIB = 'superlu'
PROTEUS_SUPERLU_H = r'"slu_ddefs.h"'

# --- HDF5 ---------------------------------------------------------------
# None of the C/C++ sources built for this target actually call into HDF5;
# `hdf5` is only named because cmeshTools' Extension() hardcodes it in its
# `libraries` list. Point the search at $PREFIX so that -lhdf5 resolves.
PROTEUS_HDF5_INCLUDE_DIR = PROTEUS_INCLUDE_DIR
PROTEUS_HDF5_LIB_DIRS = [PROTEUS_LIB_DIR]
PROTEUS_HDF5_LIBS = []

# --- Everything deliberately not built for this target -------------------
PROTEUS_MPI_INCLUDE_DIRS = []
PROTEUS_MPI_LIB_DIRS = []
PROTEUS_MPI_LIBS = []

PROTEUS_PETSC_INCLUDE_DIRS = []
PROTEUS_PETSC_LIB_DIRS = []
PROTEUS_PETSC_LIBS = []
PROTEUS_PETSC_EXTRA_COMPILE_ARGS = []
PROTEUS_PETSC_EXTRA_LINK_ARGS = []

PROTEUS_SCOREC_INCLUDE_DIRS = []
PROTEUS_SCOREC_LIB_DIRS = []
PROTEUS_SCOREC_LIBS = []
PROTEUS_SCOREC_EXTRA_COMPILE_ARGS = []
PROTEUS_SCOREC_EXTRA_LINK_ARGS = []

PROTEUS_DAETK_INCLUDE_DIR = PROTEUS_INCLUDE_DIR
PROTEUS_DAETK_LIB_DIR = PROTEUS_LIB_DIR
PROTEUS_DAETK_LIB = 'daetk'
PROTEUS_DAETK_LIB_DIRS = [PROTEUS_LIB_DIR]

PROTEUS_CHRONO_INCLUDE_DIR = PROTEUS_INCLUDE_DIR
PROTEUS_CHRONO_LIB_DIR = PROTEUS_LIB_DIR
PROTEUS_CHRONO_CXX_FLAGS = []

PROTEUS_METIS_LIB_DIR = PROTEUS_LIB_DIR
# csmoothers and superluWrappers name PROTEUS_METIS_LIB in their `libraries`
# lists, but neither actually calls into METIS -- sequential SuperLU doesn't
# need it (it's only an optional ordering backend), and there is no METIS in
# this target's dependency set. Point it at libm, which is already being
# linked anyway: default.py uses exactly this 'm' placeholder for
# PROTEUS_BLAS_LIB/PROTEUS_LAPACK_LIB when it can't find a real library, so
# it's the established way to say "nothing to link here" in these configs.
PROTEUS_METIS_LIB = 'm'
