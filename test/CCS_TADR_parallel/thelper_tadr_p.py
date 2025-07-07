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

LevelModelType = TADR.LevelModel
logEvent = Profiling.logEvent
name=soname
a0= 0.01
def a(x):
    return np.array([[a0,0.0],[0.0,a0]])
aOfX = {0:a}

nd=ct.nd
coefficients = MyCoefficients(
    aOfX,
    checkMass=checkMass,
    FCT=ct.FCT,
    LUMPED_MASS_MATRIX=ct.LUMPED_MASS_MATRIX, 
    STABILIZATION_TYPE=ct.STABILIZATION_TYPE,
    diagonal_conductivity= True, 
    ENTROPY_TYPE=ct.ENTROPY_TYPE, 
    cE=ct.cE, cK=ct.cK, physicalDiffusion=ct.physicalDiffusion) 
coefficients.variableNames=['u']

##################
# VELOCITY FIELD #
##################
def velx(X,t):
    if ct.problem in [0,2]:
        return 0.5
    else:        
        return 5.0 #-2*pi*(X[1]-0.5)

def vely(X,t):
    if ct.problem in [0,2]:
        return 0.5
    else:
        return -5.0 # 2*pi*(X[0]-0.5)

velocityFieldAsFunction={0:velx,
                         1:vely}

#####################
# INITIAL CONDITION #
#####################
class init_cond(object):
    def __init__(self,L):
        self.point_x = 1.4   # X-coordinate of the small point
        self.point_y = 1.2  # Y-coordinate of the small point
        self.radius = 0.1  # Radius of the small point region    
        #self.radius = 0.15
        #self.xc=0.5
        #self.yc=0.75
        #self.xc2=0.5
        #self.yc2=0.25
        #self.xc3=0.25
        #self.yc3=0.5
    def uOfXT(self,x,t):
        if ct.problem==0:
            return 0.0
            # if (x[0] >= 0.1 and x[0] <= 0.4):
            #     if x[1]>= 0.0 and x[1]<= 1.0:
            #         return 0.5
            # else:
            # 	return 0.0
        elif ct.problem==1: 
            distance = math.sqrt((x[0] - self.point_x)**2 + (x[1] - self.point_y)**2)
            if distance <= self.radius:
                return 2.0  # Value of the small point region
            else:
                return 0.0  # Value elsewhere
            # r  = math.sqrt((x[0]-self.xc )**2 + (x[1]-self.yc )**2)
            # r2 = math.sqrt((x[0]-self.xc2)**2 + (x[1]-self.yc2)**2)
            # r3 = math.sqrt((x[0]-self.xc3)**2 + (x[1]-self.yc3)**2)
            # slit = x[0] > self.xc-0.025 and x[0] < self.xc+0.025 and x[1] < self.yc+0.1125
            # if (r <= self.radius and slit == False):
            #     return 1.
            # elif (r <= self.radius and slit == True):
            #     return 0.
            # elif (r2 <= self.radius):
            #     return (1.0 - r2/self.radius)
            # elif (r3 <= self.radius):
            #     return (1.0 + math.cos(math.pi*r3/self.radius))/4.0
            # else:
            #     return 0.0
        elif ct.problem == 2:
            if ct.nd == 1:
                a = 0.15
                b = 1.0
                if fabs(x[0] - 0.5) <= a:
                    return b/a*math.sqrt(a**2 - (x[0] - 0.5)**2)
                else:
                    return 0.0
        # elif ct.problem==3:
        #     if (x[0] >= 0.0 and x[0] <= 0.1):
        #         return 1.0
        #     else:
        #         return 0.0

initialConditions  = {0:init_cond(L)}
analyticalSolution = {0:init_cond(L)}

#######################
# BOUNDARY CONDITIONS #
#######################


def getDBC(x,flag):
     if ct.problem in [0,2]:      
         if flag == domain.boundaryTags['left']:
             return lambda x,t:1.0
         if flag == domain.boundaryTags['top']:
             return lambda x,t:0.0
         if flag == domain.boundaryTags['bottom']:
             return lambda x,t:1.0
         if flag == domain.boundaryTags['right']:
             return lambda x,t:0.0
     #else:
     #    return lambda x,t: 0.0
    
dirichletConditions = {0:getDBC}

#Periodic Boundary Conditions
#eps=1.0e-8
#if ct.problem in [0,2]:
#    def getPDBC(x,flag):
#        if x[0] <= eps or x[0] >= (L[0]-eps):
#            return np.array([0.0,x[1],0.0])
#    periodicDirichletConditions = {0:getPDBC}

def zeroadv(x,flag):
    return None

def zerodiff(x,flag):
    return None

advectiveFluxBoundaryConditions =  {0:zeroadv}

fluxBoundaryConditions = {0:'outFlow'}
diffusiveFluxBoundaryConditions = {0:{0:zerodiff}}
