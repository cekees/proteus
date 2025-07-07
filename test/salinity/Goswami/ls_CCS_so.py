from proteus.default_so import *
from CCS_p import *
from CCS_n import *
from thelper_tadr import *
from thelper_tadr_p import *
from thelper_tadr_n import *

pnList = [ ("CCS_p","CCS_n"), #,
         ("thelper_tadr_p","thelper_tadr_n")]
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

T=  1.0 #0003
nDTout = 201
DT = T/nDTout 
tnList = [0.0,1.0e-8]+[i*DT for i  in range(1,nDTout+1)]


#nDTout= 201
archiveFlag = ArchiveFlags.EVERY_USER_STEP
#archiveFlag = ArchiveFlags.EVERY_MODEL_STEP
DT = T/float(nDTout)
#tnList=[0.,1E-6]+[float(n)*T/float(nDTout) for n in range(1,nDTout+1)]
#tnList = [i*DT for i  in range(nDTout+1)]
#cek hard coded steps for article snapshots
#tnList = [0.0,4.0,8.0]
useOneArchive = True

