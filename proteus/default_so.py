"""
The default values for so-files describing split operator formulations
"""
try:
    from importlib import reload
except:
    pass

from .SplitOperator import *

name = None

pnList = []

systemStepControllerType = Sequential_MinModelStep

systemStepExact = True

useOneMesh = True

tnList = None

needEBQ_GLOBAL = False

needEBQ = False

modelSpinUpList = []

useOneArchive=True#False

fastArchive = False

sList = []

from .Archiver import ArchiveFlags

archiveFlag = ArchiveFlags.EVERY_USER_STEP
#CEK CHANGED DEFAULT FROM EVERY_SEQUENCE_STEP

dt_system_fixed = None
"""A system-wide wide time step used by SplitOperator objects"""

skipSpinupOnHotstart = False
"""Use True if one wants to skip the spinup step when HotStart begins"""
