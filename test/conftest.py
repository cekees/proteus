"""Test-suite wide fixtures."""
import os
import subprocess
import sys
import time

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


# mpiexec itself sometimes fails to start on a CI runner (the macOS Intel runner with Open MPI 5,
# mostly). Either PMIx cannot find a network interface for a moment ("ptl_tool: problems getting
# address for index 0 (kernel index -1)", exit status 213), or a rank's MPI_Init cannot reach the
# others ("It looks like MPI_INIT failed for some reason ... PML add procs failed --> Returned
# "Not found" (-13)", exit status 14). Every launch in that second fails, whatever it was testing,
# and no rank has run anything of the job when it happens, so launching it again is safe; any
# other failure is left alone. The messages are Open MPI's, so only its launches are ever retried.
MPI_LAUNCH_FAILURES = ('ptl_tool: problems getting address', 'It looks like MPI_INIT failed')


def mpi_launch_never_started(result):
    """True for a ``subprocess.CompletedProcess`` whose mpiexec failed before any rank ran."""
    err = getattr(result, 'stderr', None)
    if isinstance(err, bytes):
        err = err.decode(errors='replace')
    return (result.returncode != 0 and isinstance(err, str)
            and any(text in err for text in MPI_LAUNCH_FAILURES))


# A helper that has finished its work can still die at interpreter exit -- a SIGSEGV or SIGABRT
# in the teardown of PETSc, MPI or an extension, after its last line of output -- and on the
# macOS Intel runner it does, intermittently (test_parallel_space_multigrid's helper printed its
# complete RESULT line and then "exited on signal 11"). Such a launch is run once more; if the
# second attempt also dies, the test fails as it would have. Every relaunch is reported as a
# RuntimeWarning naming the command and the signal, so the count stays visible in the summary
# rather than vanishing into a green run, and the launched interpreters run with
# PYTHONFAULTHANDLER=1 so a crash prints the Python stack it happened in.
CRASH_SIGNALS = ('SIGSEGV', 'SIGABRT', 'SIGBUS', 'SIGILL', 'SIGFPE')
CRASH_MESSAGES = ('exited on signal', 'Segmentation fault', 'Bus error', 'Abort trap')


def died_by_signal(result):
    """The signal name a ``subprocess.CompletedProcess`` died by, or None."""
    import signal
    rc = result.returncode
    number = -rc if rc < 0 else rc - 128 if 128 < rc < 160 else None
    if number is not None:
        try:
            name = signal.Signals(number).name
        except ValueError:
            name = None
        if name in CRASH_SIGNALS:
            return name
    err = getattr(result, 'stderr', None)
    if isinstance(err, bytes):
        err = err.decode(errors='replace')
    if rc != 0 and isinstance(err, str) and any(text in err for text in CRASH_MESSAGES):
        return 'a signal (see stderr)'
    return None


def scaled_timeout(timeout):
    """``timeout`` scaled for the machine, by PROTEUS_TEST_TIMEOUT_SCALE.

    The timeouts the tests give their subprocesses are sized for an idle workstation. A CI
    runner that is starved -- the 4-core macOS Intel runner took 2 h 15 min for a suite that
    takes 15 min on Linux, and a 4-rank launch inside it hit its 300 s budget -- needs more,
    and a timeout that fires there reports a hang that was not one. The scale is a number, or
    ``auto``: the one-minute load average per core, clamped to [1, 4], read at launch.
    """
    if timeout is None:
        return None
    setting = os.environ.get('PROTEUS_TEST_TIMEOUT_SCALE', '1').strip().lower()
    if setting == 'auto':
        try:
            load = os.getloadavg()[0] / max(os.cpu_count() or 1, 1)
        except (OSError, AttributeError):
            load = 1.0
        scale = min(max(load, 1.0), 4.0)
    else:
        try:
            scale = float(setting)
        except ValueError:
            scale = 1.0
    return timeout * scale


def launched_environment(env):
    """The environment a test's subprocess runs in: the caller's, with faulthandler on."""
    env = dict(os.environ if env is None else env)
    env.setdefault('PYTHONFAULTHANDLER', '1')
    return env


@pytest.fixture
def launch_policy():
    """The launch policy's pieces, for the tests of it (a conftest is not importable under
    --import-mode=importlib, so they reach them through this fixture)."""
    import types
    return types.SimpleNamespace(scaled_timeout=scaled_timeout, died_by_signal=died_by_signal,
                                 launched_environment=launched_environment, os=os)


@pytest.fixture(autouse=True)
def _retry_an_mpi_launch_that_never_started(monkeypatch):
    """Every ``subprocess.run`` from a test, hardened for a shared runner:

    * launched again, up to three times in all, if an mpiexec never started
      (MPI_LAUNCH_FAILURES); the pause before a relaunch is PROTEUS_TEST_MPI_RETRY_DELAY
      seconds (default 3), doubled before the last attempt;
    * launched once more, with a RuntimeWarning, if it died by a signal (died_by_signal);
    * its ``timeout`` scaled by PROTEUS_TEST_TIMEOUT_SCALE (scaled_timeout);
    * run with PYTHONFAULTHANDLER=1 (launched_environment).
    """
    import warnings
    run = subprocess.run

    def run_with_retry(cmd, *args, **kwargs):
        if 'timeout' in kwargs:
            kwargs['timeout'] = scaled_timeout(kwargs['timeout'])
        kwargs['env'] = launched_environment(kwargs.get('env'))
        crashed_once = False
        attempt = 0
        while True:
            result = run(cmd, *args, **kwargs)
            if mpi_launch_never_started(result) and attempt < 2:
                time.sleep(float(os.environ.get('PROTEUS_TEST_MPI_RETRY_DELAY', 3.0)) * (attempt + 1))
                attempt += 1
                continue
            signal_name = died_by_signal(result)
            if signal_name is not None and not crashed_once:
                crashed_once = True
                warnings.warn(
                    'relaunching %s: it died by %s (exit status %d); the launched interpreter '
                    'ran with PYTHONFAULTHANDLER=1, so its stderr holds the Python stack if the '
                    'crash was in Python: %s'
                    % (' '.join(str(c) for c in cmd), signal_name, result.returncode,
                       (result.stderr if isinstance(result.stderr, str) else '')[-1500:]),
                    RuntimeWarning, stacklevel=2)
                time.sleep(float(os.environ.get('PROTEUS_TEST_MPI_RETRY_DELAY', 3.0)))
                continue
            return result
    monkeypatch.setattr(subprocess, 'run', run_with_retry)


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
