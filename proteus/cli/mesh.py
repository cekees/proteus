"""Make for meshes: run the mesh generator only when its output is out of date.

Shared by ``pro msh``, which only meshes, and ``pro run``, which meshes
when needed and then runs.

A mesh made here gets a manifest, ``<prefix>.mesh.ymf`` next to the mesh
files (a ymf document, so ymf core is all it needs). The manifest records
a key over what determines the mesh -- the generator, its command line and
the contents of its input files (the .poly or .geo) -- plus a checksum of
every file the generator wrote. The mesh is current when the recomputed
key matches and every file is still the one that was written.

Mesh generation is serial and runs on one process; the caller decides
which (rank 0 in ``pro run``).
"""
import hashlib
import json
import os

__all__ = ["SUFFIX", "manifest_path", "mesh_key", "state", "make"]

SUFFIX = ".mesh.ymf"


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def manifest_path(plan):
    return plan["prefix"] + SUFFIX


def mesh_key(plan):
    """sha256 over the generator, its command line and its inputs' contents."""
    record = {
        "generator": plan["generator"],
        "command": plan["command"],
        "inputs": {os.path.basename(p): _sha256(p) for p in plan["inputs"]},
    }
    return hashlib.sha256(
        json.dumps(record, sort_keys=True).encode("utf-8")).hexdigest()


def state(plan):
    """``("current" | "stale" | "absent", reason)`` for a mesh plan from
    :func:`proteus.MeshTools.meshFiles`."""
    missing_inputs = [p for p in plan["inputs"] if not os.path.exists(p)]
    if missing_inputs:
        return "stale", "input missing: " + ", ".join(
            os.path.basename(p) for p in missing_inputs)
    path = manifest_path(plan)
    if not os.path.exists(path):
        return "absent", "no " + os.path.basename(path)
    from ymf.archive import read_document
    record = read_document(path).get("mesh") or {}
    if record.get("key") != mesh_key(plan):
        return "stale", "the generator's inputs or command changed"
    directory = os.path.dirname(path)
    for name, digest in (record.get("files") or {}).items():
        f = os.path.join(directory, name)
        if not os.path.exists(f):
            return "stale", name + " is missing"
        if _sha256(f) != digest:
            return "stale", name + " changed since it was generated"
    return "current", ""


def make(plan, domain, meshOptions):
    """Run the generator for ``plan`` and write its manifest."""
    from proteus import MeshTools, __version__
    from ymf.archive import write_document
    # Remove what the last run wrote first, and its record. The gmsh path
    # reuses an existing .msh rather than remeshing, so a stale one would
    # otherwise survive a change to the .geo.
    for f in plan["outputs"] + plan["optional"] + [manifest_path(plan)]:
        if os.path.exists(f):
            os.remove(f)
    MeshTools.writeMeshFiles(domain, meshOptions)
    absent = [p for p in plan["outputs"] if not os.path.exists(p)]
    if absent:
        raise RuntimeError("{0} did not write {1}".format(
            plan["generator"], ", ".join(os.path.basename(p) for p in absent)))
    written = plan["outputs"] + [p for p in plan["optional"] if os.path.exists(p)]
    write_document(manifest_path(plan), {"mesh": {
        "generator": plan["generator"],
        "command": plan["command"],
        "key": mesh_key(plan),
        "inputs": {os.path.basename(p): _sha256(p) for p in plan["inputs"]},
        "files": {os.path.basename(p): _sha256(p) for p in written},
        "proteus": __version__,
    }})
