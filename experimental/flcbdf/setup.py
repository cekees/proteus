"""Build the flcbdfWrappers extension (FLCBDF time integration from Daetk).

Experimental. Uses the Daetk sources in ./daetk with PETSc vectors, and the
PETSc and MPI of the active environment (CONDA_PREFIX, or PETSC_DIR/PETSC_ARCH).

    cd experimental/flcbdf && python setup.py build_ext --inplace

then put this directory on PYTHONPATH (proteus.TimeIntegration falls back to
a top-level `flcbdfWrappers`).
"""
import glob
import os

import numpy
from Cython.Build import cythonize
from setuptools import Extension, setup

here = os.path.dirname(os.path.abspath(__file__))
if os.environ.get("PETSC_DIR"):
    petsc = os.environ["PETSC_DIR"]
    arch = os.environ.get("PETSC_ARCH", "")
    prefix = os.path.join(petsc, arch) if arch else petsc
    includes = [os.path.join(petsc, "include"), os.path.join(prefix, "include")]
    libdirs = [os.path.join(prefix, "lib")]
else:
    prefix = os.environ["CONDA_PREFIX"]
    includes = [os.path.join(prefix, "include")]
    libdirs = [os.path.join(prefix, "lib")]

sources = [os.path.join("daetk", os.path.basename(f))
           for f in sorted(glob.glob(os.path.join(here, "daetk", "*.cpp")))]

ext = Extension(
    "flcbdfWrappers",
    ["flcbdfWrappers.pyx"] + sources,
    language="c++",
    depends=["flcbdfWrappers.pxd"] + glob.glob(os.path.join(here, "daetk", "*.h")),
    define_macros=[("F77_POST_UNDERSCORE", 1), ("USE_BLAS", 1) if not os.environ.get("FLCBDF_NOBLAS") else ("NO_BLAS_DEFINE", 1),
                   ("CMRVEC_BOUNDS_CHECK", 1), ("MV_VECTOR_BOUNDS_CHECK", 1),
                   ("PETSCVEC_BOUNDS_CHECK", 1),
                   ("FLCBDF_VERBOSE", 1) if os.environ.get("FLCBDF_VERBOSE") else ("FLCBDF_QUIET", 1),
                   ("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")],
    include_dirs=["daetk", numpy.get_include()] + includes,
    library_dirs=libdirs,
    libraries=["petsc", "mpi"],
    extra_compile_args=["-std=c++20", "-Wno-deprecated-declarations"]
    + (["-O0", "-g"] if os.environ.get("FLCBDF_DEBUG") else ["-O2"]),
    extra_link_args=["-Wl,-rpath," + d for d in libdirs],
)
setup(name="flcbdfWrappers", ext_modules=cythonize([ext], language_level=3))
