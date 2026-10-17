"""Run a helper script with petsc4py made unimportable; not collected by pytest.

Usage: without_petsc4py.py HELPER [ARGS...]

Installs an import hook that refuses petsc4py, then runs HELPER as __main__
with ARGS. Proteus must then take its PETSc-free parallel path -- mpi4py
halo exchanges, local transfer products, a redundant SuperLU coarse solve --
and any code that still reaches for petsc4py fails loudly instead of quietly
using it. The PETSc C library may still be loaded by the mesh partitioner.
"""
import importlib.abc
import os
import runpy
import sys


class _BlockPetsc4py(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == 'petsc4py' or fullname.startswith('petsc4py.'):
            raise ImportError("petsc4py is blocked for this run (%s)" % fullname)
        return None


sys.meta_path.insert(0, _BlockPetsc4py())
assert 'petsc4py' not in sys.modules, "petsc4py was imported before the block was installed"
helper = sys.argv[1]
sys.argv = sys.argv[1:]
sys.path.insert(0, os.path.dirname(os.path.abspath(helper)))
runpy.run_path(helper, run_name='__main__')
assert not any(m == 'petsc4py' or m.startswith('petsc4py.') for m in sys.modules), \
    "petsc4py was imported despite the block"
