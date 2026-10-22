#!/usr/bin/env python
"""SparseMatFromDict with rows that have no entries.

The row pointer used to be built by stepping one row at a time, so a gap in the row numbers shifted every
later row up one place and the trailing pointers stayed at zero (negative row lengths). A global transfer
operator has no empty rows; a rank's piece of one does -- ghost rows whose coarse parents are not on the
rank -- and at 64 ranks on a discontinuous space the trailing ones tripped a "negative repeats" error.
"""
import numpy as np
import pytest

from proteus import LinearAlgebraTools as LAT


@pytest.mark.parametrize('entries,nr,expected', [
    ({(1, 0): 1.0, (1, 3): 2.0, (4, 2): 3.0}, 7, [0, 0, 2, 2, 2, 3, 3, 3]),   # leading, middle, trailing gaps
    ({(0, 1): 1.0, (1, 0): 2.0, (2, 2): 3.0}, 3, [0, 1, 2, 3]),                # none
    ({(5, 0): 1.0}, 6, [0, 0, 0, 0, 0, 0, 1]),                                  # a single row at the end
    ({}, 4, [0, 0, 0, 0, 0]),                                                   # nothing at all
])
def test_row_pointer_has_one_entry_per_row_whatever_the_gaps(entries, nr, expected):
    matrix, _ = LAT.SparseMatFromDict(nr, 5, entries)
    rowptr, colind, values = matrix.getCSRrepresentation()
    assert rowptr.tolist() == expected
    assert np.all(np.diff(rowptr) >= 0)
    # every entry is where it was put
    for (i, j), v in entries.items():
        row = slice(rowptr[i], rowptr[i + 1])
        assert v in values[row][colind[row] == j]


def test_an_entry_outside_the_rows_is_refused():
    with pytest.raises(AssertionError, match='outside'):
        LAT.SparseMatFromDict(2, 2, {(2, 0): 1.0})
