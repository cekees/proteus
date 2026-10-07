"""Test-suite wide fixtures."""
import sys

import pytest


@pytest.fixture(autouse=True)
def _restore_petsc_options():
    """Put PETSc's options database back as it was before each test.

    The database is a process-wide singleton: an option a test sets (directly,
    or through a numerics module that writes it at import) otherwise stays set
    for every later test in the process, and a test's outcome can depend on
    which tests ran before it. CI runs each test in a fork (--forked), which
    hides this; a plain local pytest run does not.

    Only when petsc4py is already loaded: importing it here would initialize
    PETSc before a test could hand it its own arguments.
    """
    if "petsc4py.PETSc" not in sys.modules:
        yield
        return
    from petsc4py import PETSc
    database = PETSc.Options()
    before = database.getAll()
    yield
    after = database.getAll()
    for name in after:
        if name not in before:
            database.delValue(name)
    for name, value in before.items():
        if after.get(name) != value:
            database.setValue(name, value)


@pytest.fixture(autouse=True)
def _restore_proteus_process_state():
    """Put proteus.iproteus.opts, the default_p/n/so modules, proteus.Context and
    the working directory back after each test.

    opts is a process-wide object that tests import by reference and set
    (opts.hotStart, opts.dataDir, ...); a value one test sets otherwise holds
    for every test after it.
    """
    import os
    cwd = os.getcwd()
    module = sys.modules.get("proteus.iproteus")
    saved = dict(vars(module.opts)) if module is not None else None
    # default_p/default_n/default_so are modules model files star-import their
    # defaults from; a test that assigns to one (POD did: numerics = default_n,
    # then numerics.subgridError = ...) changes every later model's defaults
    defaults_modules = dict((name, dict(vars(sys.modules[name])))
                            for name in ("proteus.default_p", "proteus.default_n",
                                         "proteus.default_so") if name in sys.modules)
    context = sys.modules.get("proteus.Context")
    saved_context = ((context.context, context.contextOptionsString)
                     if context is not None else None)
    yield
    if saved is not None:
        vars(module.opts).clear()
        vars(module.opts).update(saved)
    for name, saved_vars in defaults_modules.items():
        namespace = vars(sys.modules[name])
        for key in list(namespace):
            if key not in saved_vars:
                del namespace[key]
        namespace.update(saved_vars)
    if saved_context is not None:
        # Context.contextOptionsString (parun -C) is read by every model module
        # that calls Context.Options; one a test leaves set reconfigures every
        # later model (IFEM's refinement=N renamed MCorr's output).
        context.context, context.contextOptionsString = saved_context
    os.chdir(cwd)
