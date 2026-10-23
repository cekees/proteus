"""The relaunch of an mpiexec that never started (test/conftest.py).

A fake launcher stands in for mpiexec: it counts its calls in a file, fails the first
``fail`` of them with the stderr text given, and succeeds after that. The retry in the
root conftest wraps every ``subprocess.run`` from a test, so calling it here goes
through the retry.
"""
import subprocess
import sys
import textwrap

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
