from proteus.default_so import *

from flow_p import *
from flow_n import *
from tranport_p import *
from tranport_n import *

pnList = [("flow_p", "flow_n"), ("tranport_p", "tranport_n")]

systemStepControllerType = Sequential_MinModelStep
systemStepExact = True

name = "ls_CCS_so"

needEBQ_GLOBAL = False
needEBQ = False

T = 600.0  # days (Liu et al. simulation time; fingers appear by ~50 d)
nDTout = 801
DT = T / nDTout
# Time list is specified directly in days.
tnList = [0.0, 1.0e-8, 2.0e-8] + [i * DT for i in range(1, nDTout + 2)]

archiveFlag = ArchiveFlags.EVERY_USER_STEP
useOneMesh = True
useOneArchive = True
