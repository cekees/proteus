#!/usr/bin/env python
"""proteus's own geometric multigrid, in parallel.

Two things stood between MGM and a parallel run, and the first one is what this
file pins.

The coarse solve. MGM factors the coarsest level's matrix with LU. In parallel
that matrix is a rank's *subdomain* matrix, so the factorization solves each
rank's block on its own and drops the coupling between ranks -- a block-Jacobi
coarse correction instead of the exact one. Nothing fails; the answer just
depends on how many ranks there are, from the eighth digit on this problem. The
parallel coarse solver gathers the coarse system and factors it in full (PETSc's
redundant preconditioner), which is exact.

That exactness is testable without a tolerance argument. Jacobi smoothing is
pointwise and the transfer operators are exact, so with an exact coarse solve
the parallel cycle *is* the serial cycle, and the fine-level checksum must agree
at every rank count to roundoff. Only the coarse solve can break that, which
makes it a sharp detector: the block-Jacobi version is off by ~1e-9 relative.

The smoothers that become processor-block variants in parallel (Gauss-Seidel,
StarILU) are not rank-independent per cycle, so they are held to the weaker
property that matters -- converging to the serial solution.

StarILU turned up a second defect. Between ranks its sweep is additive, and
undamped that is not convergent for every partition: as a plain linear solver on
the 81-node mesh it diverges on 3 ranks (checksum 2e15) while converging on 1, 2
and 4. The parallel default is therefore damped; see starILURelaxation.
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


def _run(n_levels, smoother, n_ranks, cycles=3):
    mpiexec = _mpiexec()
    if mpiexec is None:
        pytest.skip('no MPI launcher available')
    helper = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          'mgm_parallel_mpi.py')
    env = dict(os.environ)
    env['PYTHONPATH'] = os.pathsep.join(
        [p for p in sys.path if p] + [env.get('PYTHONPATH', '')]).strip(os.pathsep)
    result = subprocess.run([mpiexec, '-n', str(n_ranks), sys.executable, helper,
                             str(n_levels), smoother, str(cycles)],
                            capture_output=True, text=True, env=env)
    assert result.returncode == 0, result.stdout + result.stderr
    for line in result.stdout.splitlines():
        if line.startswith('RESULT'):
            return float(line.split('checksum=')[1].split()[0])
    raise AssertionError(result.stdout + result.stderr)


@pytest.mark.parametrize('n_levels', [2, 3, 4])
@pytest.mark.parametrize('n_ranks', [2, 3])
def test_jacobi_cycle_is_rank_independent(n_levels, n_ranks):
    serial = _run(n_levels, 'Jacobi', 1)
    parallel = _run(n_levels, 'Jacobi', n_ranks)
    assert abs(parallel - serial) < 1.0e-11 * abs(serial), (
        "%i levels on %i ranks gave %.14e against %.14e in serial: a parallel "
        "cycle with an exact coarse solve is the serial cycle"
        % (n_levels, n_ranks, parallel, serial))


@pytest.mark.parametrize('smoother', ['GaussSeidel', 'StarILU'])
@pytest.mark.parametrize('n_ranks', [2, 3, 4])
def test_block_smoothers_converge_to_the_serial_solution(smoother, n_ranks):
    serial = _run(3, smoother, 1, cycles=30)
    parallel = _run(3, smoother, n_ranks, cycles=30)
    assert abs(parallel - serial) < 1.0e-9 * abs(serial), (
        "%s MGM on %i ranks gave %.12e against %.12e in serial"
        % (smoother, n_ranks, parallel, serial))
