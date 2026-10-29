"""Hot start measurement (C17) for an SSP Runge-Kutta + edge-based entropy viscosity time
integrator (RKEV): the linear advection of a Gaussian, periodic, on test/periodic's
la_periodicGauss deck with VOS3P.RKEV of order 3 (three stages per step), 11 x 11 nodes.

    python run_rkev.py OUTDIR MODE [N]

MODE    full   run 0 -> T
        stop   run 0 -> t_N (N user steps) and exit; archive left in OUTDIR
        hot    hotstart from OUTDIR's archive at t_N, then run on to T

Compare the archives with compare.py (the field is `u`).
"""
import os
import sys

here = os.path.dirname(os.path.abspath(__file__))
outdir, mode = sys.argv[1], sys.argv[2]
N = int(sys.argv[3]) if len(sys.argv) > 3 else 20
os.makedirs(outdir, exist_ok=True)
os.chdir(outdir)
sys.path.insert(0, here)

import rkev_gauss_p as p  # noqa: E402
import rkev_gauss_n as n  # noqa: E402
from proteus import NumericalSolution, default_s, default_so as so  # noqa: E402
from proteus.Archiver import ArchiveFlags  # noqa: E402
from proteus.runtime import RunOptions  # noqa: E402

so.name = "hs_rkev"
so.pnList = [("rkev_gauss_p", "rkev_gauss_n")]
so.archiveFlag = ArchiveFlags.EVERY_USER_STEP
so.systemStepExact = True
DT = p.T / float(p.nDTout)
so.tnList = [i * DT for i in range(p.nDTout + 1)]
opts = RunOptions(gatherArchive=True)
if mode == "stop":
    so.tnList = [i * DT for i in range(N + 1)]
elif mode in ("hot", "full"):
    if os.environ.get("HS_ENDN"):
        so.tnList = [i * DT for i in range(int(os.environ["HS_ENDN"]) + 1)]
if mode == "hot":
    opts.hotStart = True
    opts.hotStartTime = N * DT
ns = NumericalSolution.NS_base(so, [p], [n], [default_s], opts)
ns.calculateSolution("hs_rkev")
print("DONE", mode, "N", N)
