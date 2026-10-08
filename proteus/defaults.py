import sys, os, copy, inspect, dataclasses
from . import (TransportCoefficients,
               Transport,
               default_p,
               TimeIntegration,
               Quadrature,
               FemTools,
               SubgridError,
               ShockCapturing,
               NumericalFlux,
               NonlinearSolvers,
               LinearAlgebraTools,
               LinearSolvers,
               clapack,
               StepControl,
               AuxiliaryVariables,
               MeshTools,
               default_n,
               SplitOperator,
               default_so)

from .Archiver import ArchiveFlags
from .Profiling import logEvent

import sys
import importlib.util
import importlib.machinery

def load_source(modname, filename):
    loader = importlib.machinery.SourceFileLoader(modname, filename)
    spec = importlib.util.spec_from_file_location(modname, filename, loader=loader)
    module = importlib.util.module_from_spec(spec)
    # The module is always executed and not cached in sys.modules.
    # Uncomment the following line to cache the module.
    # sys.modules[module.__name__] = module
    loader.exec_module(module)
    return module

# ---------------------------------------------------------------------------
# Loading a model set: an so module and its p and n modules
#
# Two things are wanted at once, and plain imports give only one of them:
#
# - Every load starts fresh. A run's modules are executed again, from fresh
#   defaults, even when the same process (a pytest session) has loaded them
#   before; nothing a previous run executed leaks into this one.
#
# - Within one run, the modules are ordinary modules. A numerics module that
#   does `from my_p import coefficients` gets the very object the physics
#   object holds, so what numerics does to physics -- stabilization adding
#   a coupling to the coefficients' stencil, physics-based preconditioning
#   reading them -- happens to the physics that is solved.
#
# load_source alone gives the first and breaks the second: the numerics
# module's import executes the p file a second time (and caches that copy
# in sys.modules for the rest of the process, so it leaks too). So the
# loaders share a model set: the modules of the files in a model directory
# are imported normally, registered under their names, while a load runs;
# kept with the set between loads of the same run; and taken out of
# sys.modules again afterwards, with whatever was there before put back.
# A set starts at load_system, or when a module already in the set is
# loaded again (that is a new run).
# ---------------------------------------------------------------------------

class _ModelSet(object):
    """The modules of one run, and the physics objects made from them."""

    def __init__(self):
        self.modules = {}   # absolute file name -> module
        self.physics = []   # [module, physics object, {key: value as snapshotted}]


_model_set = _ModelSet()


def _new_model_set():
    global _model_set
    _model_set = _ModelSet()
    return _model_set


def _module_file(path, name):
    return os.path.abspath(os.path.join(path, name + ".py"))


def _load_module(name, path):
    """Import ``path/name.py`` as module ``name`` within the current model set.

    The modules of ``path`` already in the set are visible under their names
    while it runs; everything else from ``path`` is executed fresh. Returns
    the module. sys.path and sys.modules are restored afterwards.
    """
    path = os.path.abspath(path)
    model_set = _model_set
    if _module_file(path, name) in model_set.modules:
        model_set = _new_model_set()          # loaded again: a new run
    local = set(f[:-3] for f in os.listdir(path) if f.endswith(".py"))
    saved = dict((n, sys.modules.pop(n)) for n in local if n in sys.modules)
    for n in local:
        module = model_set.modules.get(_module_file(path, n))
        if module is not None:
            sys.modules[n] = module
    sys.path.insert(0, path)
    importlib.invalidate_caches()
    try:
        module = importlib.import_module(name)
    finally:
        sys.path.remove(path)
        for n in local:
            loaded = sys.modules.pop(n, None)
            if loaded is not None and getattr(loaded, "__file__", None) and \
               os.path.abspath(loaded.__file__) == _module_file(path, n):
                model_set.modules[_module_file(path, n)] = loaded
        sys.modules.update(saved)
    return module


def _resync_physics():
    """Carry what later modules rebound in a physics module into its object.

    A numerics module may rebind a name in a physics module (``my_p.T = 1``)
    after the physics object was made. Only names whose binding changed since
    the object was made are copied, so a caller's own edits to the object
    (setting its name, say) are not undone.
    """
    for entry in _model_set.physics:
        module, physics_object, snapshot = entry
        for k, v in module.__dict__.items():
            if k in physics_excluded_keys:
                continue
            if k not in snapshot or snapshot[k] is not v:
                physics_object.__dict__[k] = v
                snapshot[k] = v


if sys.version_info.major < 3:  # Python 2?
    # Using exec avoids a SyntaxError in Python 3.
    exec("""def reraise(exc_type, exc_value, exc_traceback=None):
                raise exc_type, exc_value, exc_traceback""")
else:
    def reraise(exc_type, exc_value, exc_traceback=None):
        if exc_value is None:
            exc_value = exc_type()
        if exc_value.__traceback__ is not exc_traceback:
            raise exc_value.with_traceback(exc_traceback)
        raise exc_value

physics_default_keys = []
physics_excluded_keys = []

for k in (set(dir(default_p)) -
          (set(dir(FemTools))|
           set(dir(MeshTools))|
           set(dir(TransportCoefficients))|
           set(dir(Transport)))):
    if (k[:2] != '__' and k not in ['FemTools',
                                    'MeshTools',
                                    'TransportCoefficients',
                                    'Transport']):
        physics_default_keys.append(k)
    else:
        physics_excluded_keys.append(k)

_Physics_base = dataclasses.make_dataclass('Physics_base',
                                           [(k,
                                             type(default_p.__dict__[k]),
                                             dataclasses.field(default_factory= lambda x=default_p.__dict__[k]: x))
                                            for k in physics_default_keys])
class Physics_base(_Physics_base):
    __frozen = False

    def __init__(self, **args):
        super(Physics_base,self).__init__(**args)
        for k in set(physics_default_keys) - set(args.keys()):
            v = default_p.__dict__[k]
            if not inspect.isclass(v):
                try:
                    self.__dict__[k] = copy.deepcopy(v)
                except:
                    pass

    def __getitem__(self, key):
        return self.__dict__[key]

    def __setitem__(self, key, val):
        self.__setattr__(key, val)

    def __setattr__(self, key, val):
        if self.__frozen and not hasattr(self, key):
            raise TypeError("{key} is not an option".format(key=key))
        object.__setattr__(self, key, val)

    def _freeze(self):
        self.__frozen = True

    def _unfreeze(self):
        self.__frozen = False

    def addOption(self, name, value):
        if self.__frozen is True:
            frozen = True
            self.__frozen = False
        self.__setattr__(name, value)
        if frozen is True:
            self._freeze()
                
def reset_default_p():
    for k,v in Physics_base().__dict__.items():
        default_p.__dict__[k] = v

def load_physics(pModule, path='.'):
    """A Physics_base from module ``pModule`` in ``path``, freshly executed.

    It joins the current model set (see _load_module), so a numerics module
    loaded after it shares its module, and its coefficients.
    """
    reset_default_p()
    p = _load_module(pModule, path)
    physics_object = Physics_base()
    snapshot = {}
    for k,v in p.__dict__.items():
        if k not in physics_excluded_keys:
            physics_object.__dict__[k] = v
            snapshot[k] = v
    _model_set.physics.append([p, physics_object, snapshot])
    return physics_object

numerics_default_keys = []
numerics_excluded_keys = []

for k in (set(dir(default_n)) -
          (set(dir(TimeIntegration))|
           set(dir(Quadrature))|
           set(dir(FemTools))|
           set(dir(SubgridError))|
           set(dir(ShockCapturing))|
           set(dir(NumericalFlux))|
           set(dir(NonlinearSolvers))|
           set(dir(LinearAlgebraTools))|
           set(dir(LinearSolvers))|
           set(dir(clapack))|
           set(dir(StepControl))|
           set(dir(AuxiliaryVariables))|
           set(dir(MeshTools)))):
    if (k[:2] != '__' and k not in ['TimeIntegration',
                                    'Quadrature',
                                    'FemTools',
                                    'SubgridError',
                                    'ShockCapturing',
                                    'NumericalFlux',
                                    'NonlinearSolvers',
                                    'LinearAlgebraTools',
                                    'LinearSolvers',
                                    'clapack',
                                    'StepControl',
                                    'AuxiliaryVariables',
                                    'MeshTools']):
        numerics_default_keys.append(k)
    else:
        numerics_excluded_keys.append(k)

_Numerics_base = dataclasses.make_dataclass('Numerics_base',
                                            [(k,type(default_n.__dict__[k]), dataclasses.field(default_factory= lambda x=default_n.__dict__[k]: x))
                                             for k in numerics_default_keys])
                                            
class Numerics_base(_Numerics_base):
    __frozen = False

    def __init__(self, **args):
        super(Numerics_base,self).__init__(**args)
        for k in set(numerics_default_keys) - set(args.keys()):
            v = default_n.__dict__[k]
            if not inspect.isclass(v):
                try:
                    self.__dict__[k] = copy.deepcopy(default_n.__dict__[k])
                except:
                    pass

    def __getitem__(self, key):
        return self.__dict__[key]

    def __setitem__(self, key, val):
        self.__setattr__(key, val)

    def __setattr__(self, key, val):
        if self.__frozen and not hasattr(self, key):
            raise TypeError("{key} is not an option".format(key=key))
        object.__setattr__(self, key, val)

    def _freeze(self):
        self.__frozen = True

    def _unfreeze(self):
        self.__frozen = False

    def addOption(self, name, value):
        if self.__frozen is True:
            frozen = True
            self.__frozen = False
        self.__setattr__(name, value)
        if frozen is True:
            self._freeze()

def reset_default_n():
    for k,v in Numerics_base().__dict__.items():
        default_n.__dict__[k] = v

def load_numerics(nModule, path='.'):
    """A Numerics_base from module ``nModule`` in ``path``, freshly executed.

    It joins the current model set, so its imports of physics modules loaded
    for this run get those modules -- and what it does to them (a subgrid
    error class adding to the coefficients' stencil, say) reaches the
    physics objects.
    """
    reset_default_n()
    n = _load_module(nModule, path)
    _resync_physics()
    numerics_object = Numerics_base()
    for k,v in n.__dict__.items():
        if k not in numerics_excluded_keys:
            numerics_object.__dict__[k] = v
    return numerics_object

system_default_keys = []
system_excluded_keys = []

for k in (set(dir(default_so)) -
          (set(dir(SplitOperator))|
           set(dir(ArchiveFlags)))):
    if (k[:2] != '__' and k not in ['SplitOperator',
                                    'ArchiveFlags']):
        system_default_keys.append(k)
    else:
        system_excluded_keys.append(k)

_System_base = dataclasses.make_dataclass('System_base',
                                          [(k,type(default_so.__dict__[k]),dataclasses.field(default_factory= lambda x=default_so.__dict__[k]: x))
                                           for k in system_default_keys])

class System_base(_System_base):
    def __init__(self, **args):
        super(System_base,self).__init__(**args)
        for k in set(system_default_keys) - set(args.keys()):
            v = default_so.__dict__[k]
            if not inspect.isclass(v):
                try:
                    self.__dict__[k] = copy.deepcopy(default_so.__dict__[k])
                except:
                    pass
def reset_default_so():
    for k,v in System_base().__dict__.items():
        default_so.__dict__[k] = v

def load_system(soModule, path='.'):
    """A System_base from module ``soModule`` in ``path``; starts a model set."""
    reset_default_so()
    _new_model_set()
    so = _load_module(soModule, path)
    system_object = System_base()
    for k,v in so.__dict__.items():
        if k not in system_excluded_keys:
            system_object.__dict__[k] = v
    return system_object


def load_models(soModule, path='.'):
    """Load an so module and every p and n module it names, as one model set.

    Returns ``(so, pList, nList)``. Entries of ``so.pnList`` that are
    already objects are passed through. A physics object without a name is
    named after its module, as parun does.
    """
    so = load_system(soModule, path)
    pList, nList = [], []
    for pModule, nModule in so.pnList:
        if isinstance(pModule, Physics_base):
            pList.append(pModule)
            nList.append(nModule)
            continue
        pList.append(load_physics(pModule, path))
        if pList[-1].name is None:
            pList[-1].name = pModule
        nList.append(load_numerics(nModule, path))
    if so.name is None:
        so.name = soModule
    return so, pList, nList
