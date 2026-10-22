"""Deprecated: start Proteus for interactive use. Use :mod:`proteus.runtime`.

Importing this module starts MPI and PETSc with no PETSc options, opens
``proteus.log`` at log level 1, and re-exports ``from proteus import *``
together with ``opts``, ``comm``, ``log`` and ``reload``. ``opts`` is one
:class:`~proteus.runtime.RunOptions` shared by everything that imports it.

The same thing with :mod:`proteus.runtime`, without the shared state::

    from proteus import runtime
    comm = runtime.init()            # PETSc options go here
    opts = runtime.RunOptions()      # a fresh one for each run

This module used to copy parun's option parser and parse an empty command
line; ``opts`` held parun's defaults. It now holds the options
:class:`~proteus.NumericalSolution.NS_base` reads, with the same defaults,
and ``hotStartTime``, which the parser copy never declared.
"""
from importlib import reload
import os
import sys
import inspect
from warnings import *
import warnings

import proteus
from proteus import Profiling, Comm, runtime

warnings.warn("proteus.iproteus is deprecated; use proteus.runtime "
              "(runtime.init() and runtime.RunOptions())",
              DeprecationWarning, stacklevel=2)

try:
    from IPython.display import display, Markdown
except Exception:
    pass


def display_src(python_object):
    display(Markdown("```Python\n"+inspect.getsource(python_object)+"\n```\n"))


def load_macros(filename, verbose=False):
    with open(filename, 'r') as macros:
        markdown = ""
        for line in macros.readlines():
            if len(line.strip('\n')) and not line.startswith('%'):
                markdown += "${0}$\n".format(line.strip('\n'))
        if verbose:
            print(markdown)
        display(Markdown(markdown))


opts = runtime.RunOptions()
log = Profiling.logEvent

comm = runtime.init()
Profiling.procID = comm.rank()
Profiling.openLog("proteus.log", 1)

# blanket import statements go below here, once petsc4py is initialized
from proteus import *
try:
    from proteusGraphical import *
except Exception:
    pass

# parun's --probDir default: model modules are importable from the working
# directory
if '.' not in sys.path:
    sys.path.insert(0, '.')

pList = []
nList = []
sList = []
