from proteus import *
from proteus.default_p import *
from proteus.ctransportCoefficients import smoothedHeaviside
from math import *
import numpy as np
#from scipy.special import erf
try:
    from .thelper_tadr import *
except:
    from thelper_tadr import *

LevelModelType = TADR.LevelModel
logEvent = Profiling.logEvent
name=soname
a0= 0.00
def a(x):
    #if nd == 1:
    #    return np.array([[a0]])
    return np.array([[a0,0.0],[0.0,a0]])
aOfX = {0:a}
nd=ct.nd

coefficients = MyCoefficients(
    aOfX,
    checkMass=checkMass,
    forceStrongConditions=False,
    FCT=ct.FCT,
    LUMPED_MASS_MATRIX=ct.LUMPED_MASS_MATRIX, 
    STABILIZATION_TYPE=ct.STABILIZATION_TYPE, 
    ENTROPY_TYPE='LOG', #ct.ENTROPY_TYPE, 
    cE=ct.cE, cK=ct.cK, physicalDiffusion=ct.physicalDiffusion,
    Dm=ct.D, alpha_L=0.0, alpha_T=0.0) 
coefficients.variableNames=['u']

##################
# VELOCITY FIELD #
##################
def velx(X,t):
    return ct.v

def vely(X,t):
    return 0.0

velocityFieldAsFunction = {0: velx}
if nd > 1:
    velocityFieldAsFunction[1] = vely

#####################
# INITIAL CONDITION #
#####################
class init_cond(object):
    def __init__(self,L):
        self.C0 = ct.C0
        self.x0 = ct.x0
        self.sigma0 = ct.sigma0
    def uOfXT(self,x,t):
        dx = x[0] - self.x0
        return self.C0*np.exp(-0.5*(dx/self.sigma0)**2)

class analytical_gaussian_solution(object):
    def __init__(self):
        self.C0 = ct.C0
        self.v = ct.v
        self.D = ct.D
        self.x0 = ct.x0
        self.sigma0 = ct.sigma0
    def uOfXT(self,x,t):
        var_t = self.sigma0**2 + 2.0*self.D*t
        amp = self.C0*np.sqrt((self.sigma0**2)/var_t)
        dx = x[0] - (self.x0 + self.v*t)
        return amp*np.exp(-0.5*(dx*dx)/var_t)

initialConditions  = {0:init_cond(L)}
analyticalSolution = {0:analytical_gaussian_solution()}

#######################
# BOUNDARY CONDITIONS #
#######################
def getDBC(x,flag):
    return None
dirichletConditions = {0:getDBC}

def zeroadv(x,flag):
    None
advectiveFluxBoundaryConditions =  {0:zeroadv}

fluxBoundaryConditions = {0:'outFlow'}
diffusiveFluxBoundaryConditions = {0:{}}
