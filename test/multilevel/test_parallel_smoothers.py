#!/usr/bin/env python
"""proteus's own smoothers, running in parallel.

They update the unknowns a rank owns and subtract the correction from the
whole local vector, ghosts included -- but a ghost entry's correction belongs
to the rank that owns it, so the values left behind are wrong.  The next
residual reads them, because the subdomain matrix couples owned rows to ghost
columns.  Refreshing the ghosts from their owners after each update is what
makes the smoothers parallel; it is also all that is needed, given node-based
partitioning makes the subdomain matrix correct for the rows a rank owns.

Two further things had to change for this to run at all: ``par_du`` was built
un-ghosted (KSP only ever reads its owned range, so it never needed them), and
Transport asserted outright that the linear solver must be ``KSP_petsc4py`` in
parallel.

Damped Jacobi is pointwise, so with the ghosts current its iterates are
exactly the serial ones and the solutions must agree to solver tolerance.
Gauss-Seidel becomes processor-block Gauss-Seidel -- a different iteration,
which need only reach the same solution.
"""
import os
import shutil
import subprocess
import sys

import pytest


def _mpiexec():
    bindir = os.path.dirname(sys.executable)
    for name in ('mpiexec', 'mpirun'):
        candidate = os.path.join(bindir, name)
        if os.path.exists(candidate):
            return candidate
    return shutil.which('mpiexec') or shutil.which('mpirun')


def _run(smoother, n_ranks):
    mpiexec = _mpiexec()
    if mpiexec is None:
        pytest.skip('no MPI launcher available')
    helper = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          'parallel_smoother_mpi.py')
    env = dict(os.environ)
    env['PYTHONPATH'] = os.pathsep.join(
        [p for p in sys.path if p] + [env.get('PYTHONPATH', '')]).strip(os.pathsep)
    result = subprocess.run([mpiexec, '-n', str(n_ranks), sys.executable, helper, smoother],
                            capture_output=True, text=True, env=env)
    assert result.returncode == 0, result.stdout + result.stderr
    for line in result.stdout.splitlines():
        if line.startswith('RESULT'):
            return float(line.split('checksum=')[1].split()[0])
    raise AssertionError(result.stdout + result.stderr)


@pytest.mark.parametrize('smoother', ['Jacobi', 'GaussSeidel'])
@pytest.mark.parametrize('n_ranks', [2, 3])
def test_smoother_reaches_the_serial_solution(smoother, n_ranks):
    serial = _run(smoother, 1)
    parallel = _run(smoother, n_ranks)
    assert abs(parallel - serial) < 1.0e-8 * abs(serial), (
        "%s on %i ranks gave %.12e against %.12e in serial"
        % (smoother, n_ranks, parallel, serial))


@pytest.mark.parametrize('smoother', ['StarILU', 'StarBILU'])
def test_schwarz_smoothers_are_refused_in_parallel(smoother):
    """Their local solves span rows the rank does not own.

    The block inverted around a node near a partition boundary includes rows
    whose entries in the subdomain matrix are incomplete, so the iteration
    converges to the wrong fixed point rather than just converging differently
    -- two ranks reach 7.0144 where serial reaches 7.0345, and stay there at
    5000 iterations.  Silently returning that would be far worse than refusing.
    """
    mpiexec = _mpiexec()
    if mpiexec is None:
        pytest.skip('no MPI launcher available')
    helper = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          'parallel_smoother_mpi.py')
    env = dict(os.environ)
    env['PYTHONPATH'] = os.pathsep.join(
        [p for p in sys.path if p] + [env.get('PYTHONPATH', '')]).strip(os.pathsep)
    result = subprocess.run([mpiexec, '-n', '2', sys.executable, helper, smoother],
                            capture_output=True, text=True, env=env)
    assert result.returncode != 0
    assert 'cannot be used in parallel' in result.stdout + result.stderr


def test_the_schwarz_smoother_is_untouched_in_serial():
    """Refusing it in parallel must not disturb the serial path.

    Only StarILU is checked: StarBILU returns a zero solution even in serial
    on this problem, which is its own question and not one this change touches.
    """
    assert _run('StarILU', 1) == pytest.approx(_run('Jacobi', 1), rel=1.0e-8)
