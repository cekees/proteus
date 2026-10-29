# FLCBDF time integration (experimental)

The fixed-leading-coefficient BDF integrator of Daetk (github.com/erdc/daetk, MIT,
copyright C. Kees) absorbed into proteus on PETSc vectors, so `TimeIntegration.FLCBDF`
and `StepControl.FLCBDF_controller(_sys)` have an extension to run on again. The
`flcbdfWrappers` extension was removed from `setup.py` and nothing could run it.

* `daetk/` the closure of `FLCBDF_lite` (23 sources, headers, PETE expression templates).
  Changes from upstream: `register` removed (C++17), six functions that fell off the end
  without returning now return, debug prints behind `FLCBDF_VERBOSE`.
* `flcbdfWrappers.pyx/.pxd` the Cython wrapper. Changes: imports petsc4py (PETSc must be
  initialized), uses a non-owning `Sys` (the old constructor called `PetscFinalize` on first
  dealloc), py3 string handling for the data file name, `predicted()` for diagnostics.
* `setup.py` builds it against the active environment's PETSc and MPI.
* `ode_check.py` y' = -y: step and order adaptation without proteus models.

```
cd experimental/flcbdf && CONDA_PREFIX=... CC=mpicc CXX=mpicxx python setup.py build_ext --inplace
PYTHONPATH=experimental/flcbdf python experimental/flcbdf/ode_check.py
```

`proteus.TimeIntegration.FLCBDF` imports `flcbdfWrappers` from the package or, failing
that, from the top level (this directory on `PYTHONPATH`).

The wrapper keeps the numpy buffers passed to its first `calculate_yprime` call: update
arrays in place.

Status: see the commit message / handoff; the vortex2D case does not yet run to completion.
