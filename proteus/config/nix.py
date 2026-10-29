# Nix (and NixOS, nix-shell, nix develop): selected with PROTEUS_ARCH=nix.
#
# default.py assumes one prefix holding every dependency's include/ and lib/, which is how
# conda, spack and a PETSc --prefix install lay things out. Nix gives each package, and each
# output of a package (out, dev, lib), a store path of its own, so there is no such prefix.
# What there is, is a compiler wrapper that already adds -isystem and -L for every build input
# (NIX_CFLAGS_COMPILE, NIX_LDFLAGS), so the compile and the link find their headers and
# libraries without help. What the wrapper cannot do is tell proteus's own configuration where
# a package lives, and setup.py reads files from those places while it configures: PETSc's
# petscvariables for BLAS/LAPACK, the SuperLU header to pick the wrapper variant, the PUMI
# libraries to see which partitioners they carry, ChronoConfig.cmake for Chrono's compile flags.
#
# So find each package among the store paths the wrapper was given, and name it through the
# <PKG>_DIR variables default.py's get_flags() already honours. Nothing here is Nix-specific
# beyond where the search list comes from.
import os
import shlex

_search = []
for _var, _flag in (('NIX_CFLAGS_COMPILE', '-isystem'), ('NIX_LDFLAGS', '-L')):
    _words = shlex.split(os.environ.get(_var, ''))
    for _i, _word in enumerate(_words):
        if _word == _flag and _i + 1 < len(_words):
            _path = _words[_i + 1]
        elif _word.startswith(_flag) and _word != _flag:
            _path = _word[len(_flag):]
        else:
            continue
        # /nix/store/<hash>-<name>/include -> /nix/store/<hash>-<name>
        _root = os.path.dirname(_path) if os.path.basename(_path) in ('include', 'lib', 'lib64') else _path
        if _root not in _search:
            _search.append(_root)


def _find(*markers):
    """First store path (of those given to the compiler wrapper) holding every marker file."""
    for root in _search:
        if all(os.path.exists(os.path.join(root, m)) for m in markers):
            return root
    return None


# <PKG>_DIR, the file that says it is the right store path. A path is only taken when the
# caller has not already set one, so an explicit PETSC_DIR=... still wins.
for _pkg, _markers in (
        ('PETSC', ('include/petscconf.h', 'lib/petsc/conf/petscvariables')),
        ('SUPERLU', ('include/slu_ddefs.h',)),
        ('MPI', ('include/mpi.h',)),
        ('SCOREC', ('include/apf.h',)),
        ('CHRONO', ('include/chrono/ChConfig.h',)),
        ('METIS', ('include/metis.h',))):
    _root = _find(*_markers)
    if _root and not os.environ.get(_pkg + '_DIR'):
        os.environ[_pkg + '_DIR'] = _root

# Nix has no PROTEUS_PREFIX; without one default.py falls back to sys.exec_prefix, the Python
# environment, which holds none of this. Point it at somewhere harmless that exists.
os.environ.setdefault('PROTEUS_PREFIX', os.environ.get('out', os.environ.get('PETSC_DIR', '')) or os.getcwd())

from .default import *  # noqa: E402,F401,F403
