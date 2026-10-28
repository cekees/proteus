"""The on-node baseline scripts keep running: the ceilings in quick mode, the kernel baseline on
one rank and a tiny mesh. They are measurement tools, not regression tests; what this pins is that
they still start, account for every pass, and write the JSON the campaign documents read."""
import json
import os
import subprocess
import sys

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))


def run(script, args, cwd):
    proc = subprocess.run([sys.executable, os.path.join(HERE, script)] + args, cwd=str(cwd),
                          capture_output=True, text=True, timeout=600)
    assert proc.returncode == 0, proc.stdout[-3000:] + proc.stderr[-3000:]
    return proc.stdout


def test_machine_ceilings_quick(tmp_path):
    out = run('machine_ceilings.py', ['quick=1', 'out=ceilings.json'], tmp_path)
    assert 'triad GB/s' in out and 'FMA GFLOP/s' in out
    with open(tmp_path / 'ceilings.json') as f:
        c = json.load(f)
    m = c['measurements']['proteus']
    assert m['triad']['1'][0]['per_proc_min_gbps'] > 0
    assert m['fma']['1']['per_proc_min_gflops'] > 0


@pytest.mark.parametrize('kind', ['hex', 'tet'])
def test_kernel_baseline_one_rank(kind, tmp_path):
    run('machine_ceilings.py', ['quick=1', 'out=ceilings.json'], tmp_path)
    out = run('kernel_baseline_mpi.py', [kind, '5', 'repeat=2', 'ceilings=ceilings.json', 'out=baseline.json'],
              tmp_path)
    result, = [line for line in out.splitlines() if line.startswith('RESULT ')]
    assert 'ns_per_element_residual=' in result
    assert 'CEILING stream_per_rank=' in out
    with open(tmp_path / 'baseline.json') as f:
        r = json.load(f)['results'][0]
    assert r['kind'] == kind and r['ranks'] == 1
    assert r['t_residual'] > 0 and r['t_jacobian'] > 0
    passes = r['kernel']['passes']
    # the assembly made C passes, attributed to the phases they ran in
    assert any(k.startswith('residual/element residual | femIntegrals.') for k in passes)
    assert any(k.startswith('jacobian/element jacobian | femIntegrals.') for k in passes)
    assert all(v['bytes'] >= 0 and v['calls'] > 0 for v in passes.values())
    # a phase's C time never exceeds its inclusive time
    phases = r['kernel']['phases']
    for path, v in phases.items():
        c_time = sum(p['seconds'] for k, p in passes.items()
                     if k.split(' | ')[0] == path or k.split(' | ')[0].startswith(path + '/'))
        assert c_time <= v['seconds'] * 1.01 + 1e-6, path
