#!/usr/bin/env python


def read_from_hdf5(hdfFile,label,dof_map=None):
    """
    Just grab the array stored in the node with label label and return it
    If dof_map is not none, use this to map values in the array
    If dof_map is not none, this determines shape of the output array
    """
    assert hdfFile is not None, "requires hdf5 for heavy data"
    # h5py, not PyTables: every caller passes archive.hdfFile, which is an
    # h5py.File. get_node().read() is the PyTables spelling and raised
    # AttributeError here from the PyTables-to-h5py migration onwards.
    vals = hdfFile[label][:]
    if dof_map is not None:
        dof = vals[dof_map]
    else:
        dof = vals

    return dof