"""Helper run under mpiexec by test_multigrid.py; not collected by pytest.

Builds a genuinely ghosted ParVec by hand on two ranks, duplicates its layout
onto a second array with ParVec_petsc4py.duplicate_with_array, and checks that
the duplicate really is ghosted -- that a forward scatter brings the owner's
value into the ghost slot. NI relies on this to give its coarse-level work
vectors the right parallel layout.
"""
import sys

import numpy as np

from proteus import Comm

comm = Comm.init()
from proteus.LinearAlgebraTools import ParVec_petsc4py

rank, size = comm.rank(), comm.size()
assert size == 2, "this helper expects exactly 2 ranks"

n, N, nghosts = 4, 8, 1
if rank == 0:
    subdomain2global = np.array([0, 1, 2, 3, 4], dtype='i')  # ghost is rank 1's
else:
    subdomain2global = np.array([4, 5, 6, 7, 3], dtype='i')  # ghost is rank 0's

source = ParVec_petsc4py(np.zeros(n + nghosts, dtype='d'), 1, n, N, nghosts,
                         subdomain2global)

duplicate_array = np.zeros(n + nghosts, dtype='d')
duplicate = source.duplicate_with_array(duplicate_array)

assert duplicate.getLocalSize() == source.getLocalSize()
assert duplicate.getSize() == source.getSize()
assert duplicate.nghosts == nghosts

duplicate_array[:n] = subdomain2global[:n] + 100.0
duplicate_array[n:] = -1.0           # deliberately wrong before the scatter
duplicate.scatter_forward_insert()

expected = subdomain2global[n] + 100.0
got = duplicate_array[n]
assert abs(got - expected) < 1.0e-12, (
    "rank %i: ghost entry is %r, expected %r -- the duplicate is not sharing "
    "the source's ghost layout" % (rank, got, expected))

print("GHOST_DUPLICATE_OK rank=%i" % rank)
sys.stdout.flush()
