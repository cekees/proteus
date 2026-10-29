"""Imported first by the MPI helpers in this directory; not collected.

An uncaught exception on one rank leaves the others waiting in a collective, so the launch
runs to its timeout and reports a hang with no traceback. This prints the traceback and
aborts the whole job at once, and, if PROTEUS_TEST_DUMP_AFTER is set (retrying_run does, at
80% of the launch's timeout), dumps every rank's Python stack to stderr that many seconds
in, so a real deadlock says where each rank is stuck.
"""
import faulthandler
import os
import sys
import traceback


def _abort_job(exc_type, exc, tb):
    traceback.print_exception(exc_type, exc, tb)
    sys.stderr.flush()
    mpi = sys.modules.get('mpi4py.MPI')
    if mpi is not None and mpi.Is_initialized() and not mpi.Is_finalized():
        mpi.COMM_WORLD.Abort(1)
    sys.exit(1)


def install():
    sys.excepthook = _abort_job
    after = os.environ.get('PROTEUS_TEST_DUMP_AFTER')
    if after:
        faulthandler.dump_traceback_later(float(after), exit=False, file=sys.stderr)


install()
