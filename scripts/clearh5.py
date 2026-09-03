#!/usr/bin/env python
"""Truncate an archive to the first N timesteps.

For when archiving collected results past the point a run is worth keeping,
or the tail of a run is bad and the earlier steps are still wanted.

Reads the ``.ymf`` archive and drops the steps at or after ``tCount`` from
its domain, then copies across only the HDF5 datasets those remaining steps
reference. What is kept is decided by *what the archive says it references*
rather than by pattern-matching digits out of dataset names, which is what
the previous version did (``int(re.search(r'\\d+', name).group()) < tCount``
on every key in the file -- that reads the wrong number out of any name
with more than one digit group, and throws on any name with none).

The previous version also had two outright bugs: it removed grids from
``Domain[0]`` regardless of which collection they belonged to, so
truncating an archive with a quadratic space corrupted it; and it wrote the
resulting tree to a text-mode file, which raises TypeError on Python 3, so
it could not have run at all.
"""
import os
import re


def clearh5(filename, dataDir='.', addname="_clean", tCount=None,
            global_sync=True):
    """Write ``<filename><addname>.ymf`` and ``.h5`` holding steps < tCount.

    ``global_sync`` is accepted for command-line compatibility and unused:
    which datasets to keep comes from the archive's own references, so it
    does not matter whether they are global arrays or per-subdomain ones.
    """
    import h5py

    from ymf.archive import read_ymf, write_ymf

    if tCount is None or tCount < 0:
        raise SystemExit("clearh5: give a positive --tCount to truncate at")

    source_base = os.path.join(dataDir, filename)
    domain, extra = read_ymf(source_base + ".ymf")

    kept_datasets = set()
    for collection in domain.get("TimeCollections", []):
        steps = collection["Data"][:tCount]
        dropped = len(collection["Data"]) - len(steps)
        if dropped:
            print("%s: dropping %d step(s) past %d"
                  % (collection["Name"], dropped, tCount))
        collection["Data"] = steps
        for step in steps:
            for grid in step.get("SpatialCollection", [step]):
                for node in (grid["Topology"], grid["Geometry"]):
                    kept_datasets.add(_dataset_of(node["DataItem"]))
                for attribute in grid.get("Attributes", []):
                    kept_datasets.add(_dataset_of(attribute["DataItem"]))
    kept_datasets.discard(None)

    out_base = os.path.join(dataDir, filename + addname)
    write_ymf(domain, out_base + ".ymf", extra=extra)
    print("wrote %s.ymf" % (out_base,))

    with h5py.File(source_base + ".h5", "r") as source, \
            h5py.File(out_base + ".h5", "w") as destination:
        for name in kept_datasets:
            if name in source:
                destination.copy(source[name], name)
            else:
                print("  warning: %r is referenced by a kept step but is not "
                      "in %s.h5" % (name, source_base))
        # carry the archive's own attributes across, so the copy is still a
        # readable archive rather than a bag of datasets
        for key, value in source.attrs.items():
            destination.attrs[key] = value
    print("wrote %s.h5 with %d dataset(s)" % (out_base, len(kept_datasets)))


def _dataset_of(data_item):
    """The HDF5 dataset a DataItem references, or None for a text sidecar."""
    reference = data_item.get("Data")
    if reference is None:
        return None
    return reference.split(":/")[-1]


if __name__ == '__main__':
    from optparse import OptionParser
    usage = ""
    parser = OptionParser(usage=usage)
    parser.add_option("-f","--filebase",
                      help="base name for storage files",
                      action="store",
                      type="string",
                      dest="filebase",
                      default="simulation")

    parser.add_option("-t","--tCount",
                      help="number of time steps",
                      action="store",
                      type="int",
                      dest="tCount",
                      default="-1")
    parser.add_option("-n","--no-global-sync",
                      help="accepted for compatibility; has no effect",
                      action="store_true",
                      dest="not_global_sync",
                      default=False)

    (opts,args) = parser.parse_args()

    clearh5(opts.filebase,tCount = opts.tCount,
            global_sync = not opts.not_global_sync)
