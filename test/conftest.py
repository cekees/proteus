"""Test-suite wide fixtures."""
import os
import sys

import pytest

_TEST_ROOT = os.path.dirname(os.path.abspath(__file__)) + os.sep


def pytest_configure(config):
    """Under pytest-xdist, run each worker in a directory of its own.

    Tests write proteus.log, <so.name>.h5, matrix dumps and other files under
    fixed names relative to the working directory. Workers sharing one
    directory read back each other's files, or ones another worker has just
    deleted. Run as CI does (python -m pytest -n auto --dist loadgroup from
    an output directory), each worker gets <that directory>/<worker id>.

    A file whose tests read what an earlier test in it wrote marks itself
    pytestmark = pytest.mark.xdist_group(...), so --dist loadgroup keeps it
    on one worker, in order.
    """
    config.addinivalue_line(
        "markers", "xdist_group(name): run on one pytest-xdist worker, in order")
    worker = getattr(config, "workerinput", {}).get("workerid")
    if worker:
        path = os.path.join(os.getcwd(), worker)
        os.makedirs(path, exist_ok=True)
        os.chdir(path)

# Model modules a test imports at collection can call Comm.get(), which starts
# PETSc with Comm.argv, by default all of sys.argv: pytest's own options would
# reach PETSc. Start it with none instead, as importing proteus.iproteus did.
# (This sets a list; it does not start PETSc. A test that wants options of
# its own still sets Comm.argv before anything starts it.)
from proteus import Comm, runtime  # noqa: E402
Comm.argv = runtime.petsc_argv()


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


@pytest.fixture
def opts():
    """A fresh proteus.runtime.RunOptions, the options NS_base reads, with
    Proteus started (proteus.runtime.start: PETSc with no options, and
    proteus.log open in the working directory).

    It replaces ``from proteus.iproteus import opts``, which was one object
    shared by every test that imported it.
    """
    from proteus import runtime
    runtime.start()
    return runtime.RunOptions()


@pytest.fixture(autouse=True)
def _restore_proteus_process_state():
    """Put proteus.iproteus.opts, the default_p/n/so modules, proteus.Context,
    sys.modules (for modules under test/) and the working directory back after
    each test.

    opts is a process-wide object that tests import by reference and set
    (opts.hotStart, opts.dataDir, ...); a value one test sets otherwise holds
    for every test after it.
    """
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
    modules_before = set(sys.modules)
    yield
    # Model modules a test imports by bare name from its own directory
    # (twp_navier_stokes_p, cylinder, ...) share names across test directories;
    # left in sys.modules, the next test that imports that name from its own
    # directory gets this one instead (cylinder2D/ibm_rans2p got
    # conforming_rans3p's twp_navier_stokes_p: no RANS2P). Such modules
    # imported during the test go again. Package-relative ones
    # (HotStart_3P.NS_hotstart_so) have unique names and stay: tests keep and
    # reload them.
    for name in set(sys.modules) - modules_before:
        if "." in name:
            continue
        added = sys.modules.get(name)
        filename = getattr(added, "__file__", None) or ""
        if os.path.abspath(filename).startswith(_TEST_ROOT):
            del sys.modules[name]
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
