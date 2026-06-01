"""
Split-operator driver for the FluidFlower start_again case.

    pnList[0] = ("flow_p",      "flow_n")        -> mphase_co2 (p_w, S_n)
    pnList[1] = ("transport_p", "transport_n")   -> TADR (c)

Cross-references (model indices follow pnList order):
    flow_p.coefficients.density_model = 1  -> mphase_co2 reads TADR density
    transport_p.coefficients.V_model  = 0  -> TADR reads mphase_co2 velocity
    Stage-3b dissolution: flow_p k_d == transport_p k_d (0.02), c_sat == 1.0

Sequential_MinModelStep: mphase_co2 advances first each dt so TADR sees the
fresh density-corrected brine velocity and the new gas saturation.
"""
from proteus.default_so import *

from flow_p import *
from flow_n import *
from transport_p import *
from transport_n import *


pnList = [ ("flow_p",      "flow_n"),
            ("transport_p", "transport_n"),]

systemStepControllerType = Sequential_MinModelStep
systemStepExact = True
name = "CCS_so"
needEBQ_GLOBAL = False
needEBQ        = False

# Time horizon = benchmark 120 h (5-day monitoring) in (m, h) units.
# Injection ends at t = 5 (= 5 h).  nDTout = 720 -> 10-minute reporting
# cadence as required by the benchmark (Nordbotten 3.1).
T = 48.0 # 120.0
nDTout = 960 #480 int(720/5*2)
DT     = T / nDTout
tnList = [0.0, 1.0e-8, 2.0e-8] + [i * DT for i in range(1, nDTout + 1)]

archiveFlag    = ArchiveFlags.EVERY_USER_STEP
useOneMesh     = True
useOneArchive  = True
