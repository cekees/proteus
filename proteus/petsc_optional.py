"""
petsc4py when it is installed, and a clear error when a PETSc feature is used without it.

Proteus's own solvers -- the multilevel ones in particular -- need only MPI for
their parallel vectors (LinearAlgebraTools.ParVec_mpi4py), so a run that does
not ask for a PETSc solver should not need petsc4py. Modules import PETSc from
here instead of from petsc4py. When petsc4py is missing, PETSc is a stand-in
whose Vec and Mat are plain base classes (so the classes deriving from them can
still be defined) and on which any other attribute raises ImportError naming
what was asked for.

The PETSc C library is a separate matter: the mesh partitioner links it and
initializes it itself when petsc4py has not (see cpartitioning).
"""
try:
    from petsc4py import PETSc
    HAVE_PETSC4PY = True
except ImportError as _error:
    HAVE_PETSC4PY = False
    _reason = str(_error)

    class _MissingPETSc(object):
        """Stands in for petsc4py.PETSc when petsc4py cannot be imported."""
        Vec = object
        Mat = object

        def __getattr__(self, name):
            raise ImportError(
                "PETSc.%s needs petsc4py, which could not be imported (%s). Proteus's "
                "own solvers (LU, Jacobi, GaussSeidel, StarILU, MGM, NI) run without it; "
                "KSP_petsc4py and petsc_MG do not." % (name, _reason))

    PETSc = _MissingPETSc()
