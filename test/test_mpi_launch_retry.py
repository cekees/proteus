"""The relaunch of an mpiexec that never started (test/conftest.py).

A fake launcher stands in for mpiexec: it counts its calls in a file, fails the first
``fail`` of them with the stderr text given, and succeeds after that. The retry in the
root conftest wraps every ``subprocess.run`` from a test, so calling it here goes
through the retry.
"""
import os
import shutil
import subprocess
import sys
import textwrap
import time

import pytest

OMPI_MPI_INIT_FAILED = textwrap.dedent('''\
    --------------------------------------------------------------------------
    It looks like MPI_INIT failed for some reason; your parallel process is
    likely to abort.  There are many reasons that a parallel process can
    fail during MPI_INIT; some of which are due to configuration or environment
    problems.  This failure appears to be an internal failure; here's some
    additional information (which may only be relevant to an Open MPI
    developer):

      PML add procs failed
      --> Returned "Not found" (-13) instead of "Success" (0)
    --------------------------------------------------------------------------
    [host:00000] *** An error occurred in MPI_Init_thread
    [host:00000] *** MPI_ERRORS_ARE_FATAL (processes in this communicator will now abort,
    prterun detected that one or more processes exited with non-zero status,
    thus causing the job to be terminated. The first process to do so was:

       Process name: [prterun-host-58199@1,1]
       Exit code:    14
    --------------------------------------------------------------------------
    ''')
OMPI_PTL_TOOL = ('[host:58199] ptl_tool: problems getting address for index 0 (kernel index -1)\n')
SOMETHING_ELSE = 'Traceback (most recent call last):\n  ...\nAssertionError: the job itself failed\n'

FAKE_LAUNCHER = textwrap.dedent('''\
    import os, sys
    counter, fail, message, status = sys.argv[1], int(sys.argv[2]), sys.argv[3], int(sys.argv[4])
    calls = int(open(counter).read()) + 1 if os.path.exists(counter) else 1
    with open(counter, 'w') as f:
        f.write(str(calls))
    if calls <= fail:
        sys.stderr.write(open(message).read())
        sys.exit(status)
    print('RESULT started on call', calls)
    ''')


@pytest.fixture
def launch(tmp_path, monkeypatch):
    monkeypatch.setenv('PROTEUS_TEST_MPI_RETRY_DELAY', '0')
    script = tmp_path / 'fake_mpiexec.py'
    script.write_text(FAKE_LAUNCHER)
    counter = tmp_path / 'calls'

    def _launch(fail, text, status, **kwargs):
        message = tmp_path / 'message'
        message.write_text(text)
        result = subprocess.run([sys.executable, str(script), str(counter), str(fail), str(message),
                                 str(status)], capture_output=True, text=True, **kwargs)
        return result, int(counter.read_text())
    return _launch


def test_an_mpi_init_failure_is_launched_again(launch):
    result, calls = launch(2, OMPI_MPI_INIT_FAILED, 14)
    assert result.returncode == 0 and 'started on call 3' in result.stdout
    assert calls == 3


def test_a_pmix_address_failure_is_launched_again(launch):
    result, calls = launch(1, OMPI_PTL_TOOL, 213)
    assert result.returncode == 0 and 'started on call 2' in result.stdout
    assert calls == 2


def test_three_attempts_at_most(launch):
    result, calls = launch(10, OMPI_MPI_INIT_FAILED, 14)
    assert result.returncode == 14
    assert 'MPI_INIT failed' in result.stderr
    assert calls == 3


def test_a_failure_of_the_job_itself_is_not_launched_again(launch):
    result, calls = launch(10, SOMETHING_ELSE, 1)
    assert result.returncode == 1
    assert calls == 1


def test_a_launch_whose_stderr_is_not_captured_is_not_launched_again(tmp_path, monkeypatch):
    monkeypatch.setenv('PROTEUS_TEST_MPI_RETRY_DELAY', '0')
    script = tmp_path / 'fake_mpiexec.py'
    script.write_text(FAKE_LAUNCHER)
    counter, message = tmp_path / 'calls', tmp_path / 'message'
    message.write_text(OMPI_MPI_INIT_FAILED)
    result = subprocess.run([sys.executable, str(script), str(counter), '10', str(message), '14'],
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    assert result.returncode == 14
    assert int(counter.read_text()) == 1


# --------------------------------------------------------------------------
# the rest of the launch policy: a crash at exit, the timeout scale, faulthandler
# --------------------------------------------------------------------------

CRASHING_LAUNCHER = textwrap.dedent('''\
    import os, signal, sys
    counter, fail = sys.argv[1], int(sys.argv[2])
    calls = int(open(counter).read()) + 1 if os.path.exists(counter) else 1
    with open(counter, 'w') as f:
        f.write(str(calls))
    print('RESULT finished on call', calls, flush=True)
    if calls <= fail:
        # the work is done and printed; the interpreter dies on the way out
        os.kill(os.getpid(), signal.SIGSEGV)
    ''')


@pytest.fixture
def crashing_launch(tmp_path, monkeypatch):
    monkeypatch.setenv('PROTEUS_TEST_MPI_RETRY_DELAY', '0')
    script = tmp_path / 'crashing.py'
    script.write_text(CRASHING_LAUNCHER)
    counter = tmp_path / 'calls'

    def _launch(fail):
        result = subprocess.run([sys.executable, str(script), str(counter), str(fail)],
                                capture_output=True, text=True)
        return result, int(counter.read_text())
    return _launch


def test_a_crash_at_exit_is_launched_once_more_and_reported(crashing_launch):
    with pytest.warns(RuntimeWarning, match='died by SIGSEGV'):
        result, calls = crashing_launch(1)
    assert result.returncode == 0 and 'finished on call 2' in result.stdout
    assert calls == 2


def test_a_crash_that_repeats_fails(crashing_launch):
    with pytest.warns(RuntimeWarning, match='died by SIGSEGV'):
        result, calls = crashing_launch(10)
    assert result.returncode != 0 and calls == 2


def test_a_launchers_signal_report_counts_as_a_crash(launch_policy):
    died_by_signal = launch_policy.died_by_signal
    fake = subprocess.CompletedProcess(['mpiexec'], 139, '',
                                       'prterun noticed that process rank 0 exited on signal 11 '
                                       '(Segmentation fault: 11).')
    assert died_by_signal(fake) is not None
    assert died_by_signal(subprocess.CompletedProcess(['x'], -11, '', '')) == 'SIGSEGV'
    assert died_by_signal(subprocess.CompletedProcess(['x'], 1, '', 'AssertionError')) is None
    assert died_by_signal(subprocess.CompletedProcess(['x'], 0, '', '')) is None


def test_the_timeout_is_scaled_for_a_slow_runner(monkeypatch, launch_policy):
    # Not by running something slow against a short timeout: that depends on how long an
    # interpreter takes to start, which on a loaded runner is the very thing that varies
    # (it failed there with 0.3 s of margin). The wrapper is handed a stand-in for
    # subprocess.run and asked what timeout it passed on.
    monkeypatch.setenv('PROTEUS_TEST_TIMEOUT_SCALE', '3')
    seen = {}

    def stand_in(cmd, *args, **kwargs):
        seen.update(kwargs)
        return subprocess.CompletedProcess(cmd, 0, b'', b'')

    launch_policy.retrying_run(stand_in)(['x'], timeout=0.5)
    assert seen['timeout'] == 1.5
    seen.clear()
    launch_policy.retrying_run(stand_in)(['x'])
    assert 'timeout' not in seen          # none given, none invented


def test_the_scale_is_at_least_one_and_at_most_four(monkeypatch, launch_policy):
    scaled_timeout = launch_policy.scaled_timeout
    monkeypatch.setenv('PROTEUS_TEST_TIMEOUT_SCALE', 'auto')
    monkeypatch.setattr(launch_policy.os, 'getloadavg', lambda: (0.0, 0.0, 0.0))
    assert scaled_timeout(10) == 10
    monkeypatch.setattr(launch_policy.os, 'getloadavg', lambda: (1e6, 0.0, 0.0))
    assert scaled_timeout(10) == 40
    monkeypatch.setenv('PROTEUS_TEST_TIMEOUT_SCALE', 'not a number')
    assert scaled_timeout(10) == 10
    assert scaled_timeout(None) is None


def test_the_launched_interpreter_runs_with_faulthandler(monkeypatch):
    monkeypatch.delenv('PYTHONFAULTHANDLER', raising=False)
    probe = [sys.executable, '-c', 'import os; print(os.environ.get("PYTHONFAULTHANDLER"), '
                                   'os.environ.get("MARKER"))']
    assert subprocess.run(probe, capture_output=True, text=True).stdout.split() == ['1', 'None']
    # a caller's own environment is kept, and gains the setting
    result = subprocess.run(probe, capture_output=True, text=True,
                            env={'PATH': os.environ.get('PATH', ''), 'MARKER': 'kept'})
    assert result.stdout.split() == ['1', 'kept']


def test_a_rank_that_raises_aborts_the_whole_job(tmp_path):
    """An uncaught exception on rank 1 must end a 3-rank job at once, with its traceback,
    not leave ranks 0 and 2 in a barrier until the launch times out."""
    helper = tmp_path / 'helper.py'
    helper.write_text(textwrap.dedent('''\
        import sys
        sys.path.insert(0, %r)
        import mpi_failfast
        from mpi4py import MPI
        if MPI.COMM_WORLD.rank == 1:
            raise ValueError('rank one failed')
        MPI.COMM_WORLD.Barrier()
        ''' % os.path.join(os.path.dirname(os.path.abspath(__file__)), 'multilevel')))
    mpiexec = shutil.which('mpiexec')
    if mpiexec is None:
        pytest.skip('no mpiexec')
    start = time.time()
    result = subprocess.run([mpiexec, '-n', '3', sys.executable, str(helper)],
                            capture_output=True, text=True, timeout=60)
    assert result.returncode != 0
    assert 'ValueError: rank one failed' in result.stderr
    assert time.time() - start < 45


def test_the_launch_asks_its_helpers_to_dump_stacks_before_the_timeout(launch_policy):
    seen = {}

    def run(cmd, *args, **kwargs):
        seen.update(kwargs['env'])
        return subprocess.CompletedProcess(cmd, 0, '', '')

    launch_policy.retrying_run(run)(['x'], timeout=100)
    assert seen['PROTEUS_TEST_DUMP_AFTER'] == '80'
