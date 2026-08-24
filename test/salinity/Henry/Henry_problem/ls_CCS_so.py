from proteus.default_so import *

from flow_p import *
from flow_n import *
from tranport_p import *
from tranport_n import *
#from thelper_tadr_n import *

pnList = [ ("flow_p","flow_n"), #,
         ("tranport_p","tranport_n")]
#pnList = [("CCS_p","CCS_n")]
#         ("thelper_tadr_p","thelper_tadr_n")]

systemStepControllerType = Sequential_MinModelStep

#systemStepControllerType = Sequential_MinAdaptiveModelStep

#systemStepControllerType = Sequential_MinFLCBDFModelStep 
#Sequential_MinAdaptiveModelStep

systemStepExact = True

name="ls_CCS_so"

needEBQ_GLOBAL  = False
needEBQ = False

T=  1000.0 #0003
nDTout = 401
DT = T/nDTout 
tnList = [0.0,1.0e-8, 2.0e-8]+[i*DT for i  in range(1,nDTout+2)]


#nDTout= 201
archiveFlag = ArchiveFlags.EVERY_USER_STEP
#archiveFlag = ArchiveFlags.EVERY_MODEL_STEP
DT = T/float(nDTout)
#tnList=[0.,1E-6]+[float(n)*T/float(nDTout) for n in range(1,nDTout+1)]
#tnList = [i*DT for i  in range(nDTout+1)]
#cek hard coded steps for article snapshots
#tnList = [0.0,4.0,8.0]
useOneArchive = True

