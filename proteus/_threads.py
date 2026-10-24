"""One thread per MPI rank, unless the user asked for something else.

Proteus runs MPI-only on a node: its legacy kernels are limited by
instruction issue on one core, so a rank cannot use a second core
(ON_NODE_PERFORMANCE_BASELINE.md, section 6.2).  The numerical libraries
that sit under it do not know that.  OpenBLAS starts one thread per core
in every process (16 per rank on a 16-core Mac), so a node running one rank
per core is oversubscribed by the number of cores, and a rank waiting in a
BLAS call is slowed by every other rank's threads.

Under MPI with more than one rank, the thread-count variables of the
libraries numpy and scipy bring in are therefore given the value 1,
**unless the user already set them**.  Without MPI (or with one rank) nothing
is touched: a serial run is free to use the node's threads.

``OPENBLAS_NUM_THREADS`` alone is not enough: an OpenBLAS built on OpenMP
(conda-forge's, on macOS and on Linux) takes its thread count from
``OMP_NUM_THREADS`` and ignores ``OPENBLAS_NUM_THREADS=1`` (measured: 13.8
cores busy in one process with only the latter set, one with the former).  All
the variables are set.

The variables are read when the library is loaded, which is when numpy is
imported, so this has to happen before that.  ``import proteus`` does it
for a launch whose size the launcher announces in the environment.  If numpy
was imported first, or the launcher announces nothing, ``proteus.Comm`` -- which
knows the real size -- sets the variables and, when threadpoolctl is installed,
also limits the libraries that are already loaded.
"""
import os
import sys

#: variable -> threadpoolctl ``user_api`` of the library it controls
THREAD_VARIABLES = {
    "OMP_NUM_THREADS": "openmp",
    "OPENBLAS_NUM_THREADS": "blas",
    "MKL_NUM_THREADS": "blas",
    "VECLIB_MAXIMUM_THREADS": "blas",
}

#: variables in which launchers announce the number of ranks in the job
LAUNCHER_SIZE_VARIABLES = (
    "OMPI_COMM_WORLD_SIZE",   # Open MPI (mpiexec, mpirun)
    "PMI_SIZE",               # MPICH/Hydra, Slurm pmi2, Cray
    "PMIX_SIZE",
    "SLURM_STEP_NUM_TASKS",   # srun
)

#: the variables this process has defaulted (the user set none of them)
defaulted = []
#: whether numpy was already loaded when they were defaulted
numpy_was_loaded = False


def launched_size(environ=None):
    """The number of ranks the launcher announces, or 1 if it announces none."""
    environ = os.environ if environ is None else environ
    size = 1
    for name in LAUNCHER_SIZE_VARIABLES:
        try:
            size = max(size, int(environ.get(name, "1")))
        except ValueError:
            pass
    return size


def apply_defaults(size, environ=None):
    """Give every thread variable the user left unset the value 1, if size > 1.

    Returns the names set by this call.
    """
    global numpy_was_loaded
    environ = os.environ if environ is None else environ
    if size <= 1:
        return []
    names = [n for n in THREAD_VARIABLES if n not in environ]
    for name in names:
        environ[name] = "1"
    if names:
        defaulted.extend(names)
        numpy_was_loaded = numpy_was_loaded or "numpy" in sys.modules
    return names


def limit_loaded_libraries():
    """Apply the defaulted variables to libraries that were loaded before them.

    Returns the ``user_api`` kinds limited; empty if there was nothing to do or
    threadpoolctl is not installed.
    """
    if not (defaulted and numpy_was_loaded):
        return []
    try:
        import threadpoolctl
    except ImportError:
        return []
    apis = sorted({THREAD_VARIABLES[n] for n in defaulted})
    for api in apis:
        threadpoolctl.threadpool_limits(limits=1, user_api=api)
    return apis


def after_mpi_init(size):
    """Called by proteus.Comm once the number of ranks is known.

    Returns a message to log when libraries already loaded could not be limited.
    """
    apply_defaults(size)
    if not (defaulted and numpy_was_loaded):
        return None
    if limit_loaded_libraries():
        return None
    return ("numpy was imported before proteus, so its BLAS may use more than one thread per MPI rank. "
            "Import proteus first, set OMP_NUM_THREADS=1 and OPENBLAS_NUM_THREADS=1, or install threadpoolctl.")
