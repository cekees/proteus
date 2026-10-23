"""The rank-slot limiter of conftest.py: concurrent mpiexec launches never hold more than the slot count.

Two files cannot import conftest portably, so this runs a small driver that loads it by path and
has a pool of processes ask for slots at once, recording how many are held at any moment.
"""
import os
import subprocess
import sys
import textwrap

CONFTEST = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'conftest.py')

DRIVER = textwrap.dedent('''
    import importlib.util, multiprocessing as mp, os, sys, time
    os.environ['PROTEUS_TEST_MPI_SLOTS'] = '4'
    spec = importlib.util.spec_from_file_location('conftest_under_test', sys.argv[1])
    c = importlib.util.module_from_spec(spec); spec.loader.exec_module(c)
    log = sys.argv[2]

    def job(ranks):
        with c._rank_slots(ranks):
            open(log, 'a').write('+%d\\n' % min(ranks, 4))
            time.sleep(0.2)
            open(log, 'a').write('-%d\\n' % min(ranks, 4))

    if __name__ == '__main__':
        with mp.Pool(10) as p:
            p.map(job, [4, 1, 2, 1, 3, 2, 1, 4, 2, 2, 1, 1, 6, 1, 1, 2] * 2)
        held = peak = 0
        for line in open(log):
            held += int(line); peak = max(peak, held)
        print('PEAK', peak, 'END', held)
        assert (c._ranks(['/x/mpiexec', '-n', '3', 'python']), c._ranks(['python']), c._ranks(['mpiexec', 'a'])) == (3, None, 1)
''')


def test_never_more_than_the_slots_are_held(tmp_path):
    log = tmp_path / 'log'
    driver = tmp_path / 'driver.py'  # a file, not -c: multiprocessing's spawn start method re-imports it
    driver.write_text(DRIVER)
    result = subprocess.run([sys.executable, str(driver), CONFTEST, str(log)],
                            capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, result.stdout + result.stderr
    assert 'PEAK 4 END 0' in result.stdout, result.stdout
