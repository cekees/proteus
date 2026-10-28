#!/usr/bin/env python
"""PCMG on discontinuous and non-conforming P1: the smoother matters, point Jacobi is the wrong default.

C0 P1 takes 12-13 CG iterations under PCMG with Chebyshev(2) over point Jacobi. The same smoother on
discontinuous P1 (SIPG) needs 39-44 and on non-conforming P1 29-32: the unknowns come in element
blocks (DG) or edge-neighbour clusters (NC) that a diagonal smoother ignores. What reduces them,
measured at 4 and 8 ranks (mg_scaling_mpi.py, ``smoother=``):

* DG: block Jacobi with ILU(0) on each rank's block, 18-19 iterations and 1.7x less linear-solve time
  (ASM with overlap 1 is about the same; the LU variants reach 13-17 iterations but need 27 s to set up);
* NC: Chebyshev(4) over Jacobi or one SOR sweep per application, 15-16 iterations, 1.7x less time.

What is pinned here is the iteration reduction, so that a change that costs it is seen.
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


def _iterations(space, smoother, n_ranks=4, nn=5):
    mpiexec = _mpiexec()
    if mpiexec is None:
        pytest.skip('no MPI launcher available')
    helper = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'mg_scaling_mpi.py')
    env = dict(os.environ)
    env['PYTHONPATH'] = os.pathsep.join(
        [p for p in sys.path if p] + [env.get('PYTHONPATH', '')]).strip(os.pathsep)
    result = subprocess.run([mpiexec, '-n', str(n_ranks), sys.executable, helper, 'tet', str(nn), '3', 'pcmg_cg',
                             'space=' + space, 'smoother=' + smoother],
                            capture_output=True, text=True, env=env, timeout=300)
    assert result.returncode == 0, result.stdout + result.stderr
    line = [l for l in result.stdout.splitlines() if l.startswith('RESULT')][0]
    got = dict(item.split('=', 1) for item in line.split() if '=' in item)
    assert int(got['reason']) > 0, line
    return int(got['its'])


@pytest.mark.parametrize('space,tuned,bound', [('dg', 'bjacobi_ilu', 0.6), ('nc', 'cheb4', 0.6), ('nc', 'sor', 0.6)])
def test_the_tuned_smoother_cuts_the_iterations(space, tuned, bound):
    point = _iterations(space, 'jacobi')
    block = _iterations(space, tuned)
    assert block <= bound * point, (space, tuned, block, point)
