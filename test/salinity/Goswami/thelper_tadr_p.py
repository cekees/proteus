from __future__ import absolute_import
from builtins import object
import numpy as np
from proteus import *
from proteus.default_p import *
from proteus.ctransportCoefficients import smoothedHeaviside
from math import *
try:
    from .thelper_tadr import *
except:
    from thelper_tadr import *
#from gw_cl_domain import *

LevelModelType = TADR.LevelModel
logEvent = Profiling.logEvent
name=soname
a0= 0.001
def a(x):
    return np.array([[a0,0.0],[0.0,a0]])
aOfX = {0:a}
coefficients = MyCoefficients(
    aOfX,
    V_model=0,  
    checkMass=checkMass,
    FCT=ct.FCT,
    LUMPED_MASS_MATRIX=ct.LUMPED_MASS_MATRIX, 
    STABILIZATION_TYPE=ct.STABILIZATION_TYPE,
    diagonal_conductivity= True, 
    ENTROPY_TYPE=ct.ENTROPY_TYPE, 
    cE=ct.cE, cK=ct.cK, physicalDiffusion=ct.physicalDiffusion) 
coefficients.variableNames=['u']

# Define model list for TADR simulation
#modelList = [recoeffs, coefficients]



# ##################
# # VELOCITY FIELD #
# ##################
# def velx(X,t):
#     if ct.problem in [0,2]:
#         return 0.5
#     else:        
#         return 5.0 #-2*pi*(X[1]-0.5)

# def vely(X,t):
#     if ct.problem in [0,2]:
#         return 0.5
#     else:
#         return -5.0 # 2*pi*(X[0]-0.5)

# velocityFieldAsFunction={0:velx,
#                          1:vely}


# Density Relation
rho_fw =  998.2#1000 # density of freshwater
rho_sw =1026.10 #  density of saltwater- corresponds to conc=37.1kg/m3
mf_fw = 0.0 # saline mass fraction, freshwater
mf_sw = 1.0# .0357 # saline mass fraction, saltwater
epsilon =(rho_sw - rho_fw)/rho_fw 
drho_dmf =epsilon*rho_fw #700. #epsilon*rho_fw # density change with mass fraction





#####################
# INITIAL CONDITION #
#####################
###########################
# Defining Initial Conditions Functions
# Initially will start at all freshwater
# and then go to a steady state
###########################

class constantIC:
    def __init__(self,cval=0.0):
        self.cval = cval
    def uOfXT(self,x,t):
        if (x[0]<=0.15 and x[1]<= (-0.8667 * x[0] + 0.13)):
            return 1.0
        else:
            return self.cval
    def uOfX(self,x):
        if (x[0]<=0.15 and x[1]<= (-0.8667 * x[0] + 0.13)):
            return 1.0
        else:
            return self.cval
#        return self.cval
initialConditions ={0:constantIC(mf_fw)} 
analyticalSolution = {0:constantIC(mf_fw)} 



#######################
# BOUNDARY CONDITIONS #
#######################



###########################
# Setting Concentration and Mass Flux Boundary Conditions
# (Dirichlet, Neumann and Mixed)
###########################

# coupled transport
def getConcDirichletBCs(x,tag):
    if abs(x[0]-L[0])<1.e-12:     
        return lambda x,t: mf_fw 
    elif (abs(x[0])<1.e-8 and x[1]<=.255):
        return lambda x,t: mf_sw


dirichletConditions ={0:getConcDirichletBCs} 


#dirichletConditions = {0:getDBC}

def getzeroMassDiffusiveFluxBCs(x,tag):
    return None

   
def getzeroMassAdvectiveFluxBCs(x,tag):
    return None

fluxBoundaryConditions = {0:'setFlow'}
advectiveFluxBoundaryConditions ={0:getzeroMassDiffusiveFluxBCs}
diffusiveFluxBoundaryConditions ={0:{0:getzeroMassDiffusiveFluxBCs}}#{0:{0:getzeroBCs}}#{0:{0:getzeroMassDiffusiveFluxBCs}}

