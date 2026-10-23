"""What this node can do: memory bandwidth and FMA rate, measured with proteus's own compiler and flags.

The on-node baseline (C30) reports every kernel as a fraction of a ceiling, and the ceiling has to
be measured on the same machine, by the same compiler, at the same flags proteus's extensions are
built with -- the theoretical peak of the part is not what a C loop compiled at ``-O2`` with no
``-march`` reaches. Two tiny C programs are compiled and run:

* **triad**: STREAM's ``a[i] = b[i] + s*c[i]`` (24 bytes per element), at array sizes from L1
  to DRAM, so the result is a bandwidth ladder (GB/s at each working-set size), not one number;
* **fma**: independent ``acc = acc*x + y`` chains on registers, which is the floating-point
  throughput this compiler at these flags gets out of one core (2 flops per FMA); the best over
  several chain counts, since how each count compiles varies with the flags.

Both are run on one process and on P processes at once (P = all cores, and half of them), because
proteus is MPI-only on node: the aggregate bandwidth P ranks share is the ceiling an assembly
pass running on every rank sees, and it is much less than P times one core.

Usage: machine_ceilings.py [flags=<extra compile flags>] [procs=P[,P..]] [out=ceilings.json] [quick=1]

With ``flags=`` the ceilings are measured a second time at proteus's flags plus those (for example
``flags=-O3 -mcpu=native``), to show what a flag change alone would buy. Writes one JSON file the
kernel baseline reads (default ``ceilings_<hostname>.json`` in the current directory), and prints
a table. ``quick=1`` (the test) runs one process over the three smallest sizes in a few seconds;
its numbers are not a ceiling.
"""
import json
import os
import platform
import shlex
import subprocess
import sys
import sysconfig
import tempfile
import time

TRIAD_C = r"""
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
static double now(void){struct timespec t; clock_gettime(CLOCK_MONOTONIC,&t); return t.tv_sec+1e-9*t.tv_nsec;}
int main(int argc, char** argv){
  long n = atol(argv[1]); int reps = atoi(argv[2]); double s = 3.0;
  /* a timed block is `inner` sweeps, so that it lasts well over the clock's resolution
     (microseconds on macOS) even when the arrays fit in L1 */
  long inner = 20000000L / n; if (inner < 1) inner = 1;
  double *a = malloc(n*sizeof(double)), *b = malloc(n*sizeof(double)), *c = malloc(n*sizeof(double));
  for (long i=0;i<n;i++){a[i]=1.0;b[i]=2.0;c[i]=0.5;}
  double best = 1e30;
  for (int r=0;r<reps;r++){
    double t = now();
    for (long j=0;j<inner;j++){
      for (long i=0;i<n;i++) a[i] = b[i] + s*c[i];
      s += a[n/2]*1e-12;   /* keep the sweep from being hoisted */
    }
    t = now()-t; if (t<best) best=t;
  }
  printf("%.9g\n", 24.0*n*inner/best/1e9);
  return 0;
}
"""

FMA_C = r"""
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
static double now(void){struct timespec t; clock_gettime(CLOCK_MONOTONIC,&t); return t.tv_sec+1e-9*t.tv_nsec;}
#ifndef CHAINS
#define CHAINS 64
#endif
int main(int argc, char** argv){
  long iters = atol(argv[1]);
  double acc[CHAINS], x = 1.0000001, y = 1e-9;
  for (int k=0;k<CHAINS;k++) acc[k] = 1.0 + 1e-3*k;
  double t = now();
  for (long i=0;i<iters;i++){
    for (int k=0;k<CHAINS;k++) acc[k] = acc[k]*x + y;
  }
  t = now()-t;
  double sum = 0; for (int k=0;k<CHAINS;k++) sum += acc[k];
  printf("%.9g %.3g\n", 2.0*CHAINS*iters/t/1e9, sum);
  return 0;
}
"""

#: per-array sizes of the triad ladder, bytes (3 arrays per process)
LADDER = [16 << 10, 128 << 10, 1 << 20, 8 << 20, 64 << 20, 512 << 20]


def proteus_flags():
    """The compiler and flags the extensions are built with: sysconfig's, plus PROTEUS_OPT."""
    # sysconfig's CC can carry flags ("gcc -pthread -B ..." in conda's Linux builds)
    cc = shlex.split(os.environ.get('CC') or sysconfig.get_config_var('CC') or 'cc')
    raw = shlex.split(sysconfig.get_config_var('CFLAGS') or '')
    flags, skip = [], False
    for f in raw:
        if skip:
            skip = False
        elif f in ('-isystem', '-I', '-include'):
            skip = True
        elif f.startswith(('-I', '-isystem', '-fdebug-prefix-map', '-W', '-D')) or f == '-fPIC':
            pass
        elif f not in flags:
            flags.append(f)
    flags += shlex.split(os.environ.get('PROTEUS_OPT', ''))
    return cc, flags


def compile_(cc, flags, source, name, tmp, defines=()):
    src = os.path.join(tmp, name + '.c')
    exe = os.path.join(tmp, name)
    with open(src, 'w') as f:
        f.write(source)
    subprocess.run(cc + flags + list(defines) + [src, '-o', exe], check=True)
    return exe


def run_concurrently(exe, args, procs):
    """Start procs copies at once; return each one's first printed number."""
    ps = [subprocess.Popen([exe] + [str(a) for a in args], stdout=subprocess.PIPE, text=True)
          for _ in range(procs)]
    outs = [p.communicate()[0] for p in ps]
    if any(p.returncode for p in ps):
        raise RuntimeError('%s failed' % exe)
    return [float(o.split()[0]) for o in outs]


def triad_ladder(exe, procs, quick=False):
    ladder = []
    for size in LADDER[:3] if quick else LADDER:
        n = size // 8
        # timed blocks of ~2e7 element updates each; enough of them for ~0.2 s per size
        reps = 3 if quick else max(5, min(50, int(2e8 // max(n, 2e7 // 50))))
        rates = run_concurrently(exe, [n, reps], procs)
        ladder.append({'bytes_per_array': size, 'procs': procs, 'per_proc_min_gbps': min(rates),
                       'per_proc_max_gbps': max(rates), 'aggregate_gbps': sum(rates)})
    return ladder


#: independent accumulator chains in the FMA loop. How many a core needs to cover FMA latency x
#: pipes x lanes differs by core, and how the compiler unrolls, vectorizes or spills each count
#: differs by flags (one clang got 47 GFLOP/s at 64 chains and 12 at 32), so every count is run
#: and the best is the ceiling: the most this compiler gets out of an FMA loop at these flags.
CHAIN_COUNTS = (8, 16, 32, 64, 128)


def fma(exes, procs, quick=False):
    best = None
    for chains, exe in exes.items():
        rates = run_concurrently(exe, [20_000_000 if quick else 200_000_000], procs)
        r = {'procs': procs, 'chains': chains, 'per_proc_min_gflops': min(rates),
             'per_proc_max_gflops': max(rates), 'aggregate_gflops': sum(rates)}
        if best is None or r['per_proc_min_gflops'] > best['per_proc_min_gflops']:
            best = r
    return best


def measure(cc, flags, procs_list, tmp, tag, quick=False):
    triad = compile_(cc, flags, TRIAD_C, 'triad_' + tag, tmp)
    fmas = {n: compile_(cc, flags, FMA_C, 'fma%d_%s' % (n, tag), tmp, ['-DCHAINS=%d' % n])
            for n in (CHAIN_COUNTS[2:3] if quick else CHAIN_COUNTS)}
    return {'flags': ' '.join(flags),
            'triad': {str(p): triad_ladder(triad, p, quick) for p in procs_list},
            'fma': {str(p): fma(fmas, p, quick) for p in procs_list}}


def cpu_name():
    if sys.platform == 'darwin':
        try:
            return subprocess.run(['sysctl', '-n', 'machdep.cpu.brand_string'], capture_output=True,
                                  text=True).stdout.strip()
        except OSError:
            pass
    try:
        with open('/proc/cpuinfo') as f:
            for line in f:
                if line.startswith('model name'):
                    return line.split(':', 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or platform.machine()


def human(size):
    for unit in ('B', 'KB', 'MB', 'GB'):
        if size < 1024:
            return '%d %s' % (size, unit)
        size //= 1024
    return '%d TB' % size


def report(result):
    for tag, m in result['measurements'].items():
        print('\n== %s flags: %s' % (tag, m['flags']))
        procs = sorted(m['triad'], key=int)
        print('%-22s' % 'triad GB/s (per proc)' + ''.join('%14s' % ('P=%s' % p) for p in procs))
        for i, size in enumerate(LADDER[:len(m['triad'][procs[0]])]):
            row = '%-22s' % ('3 x %s' % human(size))
            for p in procs:
                row += '%14.1f' % m['triad'][p][i]['per_proc_min_gbps']
            print(row)
        row = '%-22s' % 'triad DRAM aggregate'
        for p in procs:
            row += '%14.1f' % m['triad'][p][-1]['aggregate_gbps']
        print(row)
        row = '%-22s' % 'FMA GFLOP/s per proc'
        for p in procs:
            row += '%9.1f (%3d)' % (m['fma'][p]['per_proc_min_gflops'], m['fma'][p]['chains'])
        print(row + '   (chains)')
        row = '%-22s' % 'FMA GFLOP/s aggregate'
        for p in procs:
            row += '%14.1f' % m['fma'][p]['aggregate_gflops']
        print(row)


def main(argv):
    extra = None
    procs_list = None
    out = None
    quick = False
    for arg in argv:
        key, _, value = arg.partition('=')
        if key == 'flags':
            extra = shlex.split(value)
        elif key == 'procs':
            procs_list = [int(p) for p in value.split(',')]
        elif key == 'out':
            out = value
        elif key == 'quick':
            quick = value not in ('', '0', 'no', 'false')
        else:
            raise SystemExit(__doc__)
    cores = os.cpu_count() or 1
    if procs_list is None:
        procs_list = [1] if quick else sorted({1, max(1, cores // 2), cores})
    cc, flags = proteus_flags()
    version = subprocess.run(cc[:1] + ['--version'], capture_output=True, text=True).stdout.splitlines()[0]
    result = {'host': platform.node(), 'cpu': cpu_name(), 'cores': cores, 'compiler': version,
              'cc': ' '.join(cc),
              'date': time.strftime('%Y-%m-%d'), 'measurements': {}}
    with tempfile.TemporaryDirectory() as tmp:
        result['measurements']['proteus'] = measure(cc, flags, procs_list, tmp, 'proteus', quick)
        if extra:
            result['measurements']['proteus+' + ' '.join(extra)] = measure(cc, flags + extra, procs_list,
                                                                             tmp, 'extra', quick)
    print('%s: %s, %d cores, %s' % (result['host'], result['cpu'], cores, version))
    report(result)
    out = out or 'ceilings_%s.json' % result['host'].split('.')[0]
    with open(out, 'w') as f:
        json.dump(result, f, indent=1)
    print('\nwrote', out)


if __name__ == '__main__':
    main(sys.argv[1:])
