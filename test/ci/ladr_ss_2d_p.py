from proteus import *
from proteus.default_p import *
from proteus import ADR, Domain, AnalyticalSolutions
import numpy
import math

LevelModelType = ADR.LevelModel

"""
Linear advection-diffusion-reaction at equilibrium in 2D.
"""

## \page Tests Test Problems 
# \ref ladr_ss_2d_p.py "Linear advection-diffusion-reaction at steady state"
#

##\ingroup test
#\file ladr_ss_2d_p.py
#
#\brief Linear advection-diffusion-reaction at equilibrium in 1D.
#\todo finish ladr_ss_2d_p.py doc

nd = 2
L=(2.0,2.0)#,2.0)
x0 = (-1.0,-1.0)#,-1.0)
domain = Domain.RectangularDomain(L=L,x=x0,name="adr",units="m")
#a0=0.01
a0 = 1.0
#b0=1.0
b0 = 0.0
A0_1c={0:numpy.array([[a0,0],[0,a0]])}
B0_1c={0:numpy.array([0.0,b0])}
C0_1c={0:0.0}
M0_1c={0:0.0}

#ans = AnalyticalSolutions.LinearAD_SteadyState(b=B0_1c[0][1],a=A0_1c[0][0,0])
class LevequeLiExample1(AnalyticalSolutions.SteadyState):
    def __init__(self):
        super(LevequeLiExample1, self).__init__()
    def uOfX(self, x):
        r = (x[0]**2 + x[1]**2)**0.5
        if r <= 0.5:
            return 1.0
        else:
            return 1.0 + math.log(2*r)
ans = LevequeLiExample1()
analyticalSolution = {0:ans}
initialConditions = None

def a(x):
    return numpy.array([[a0,0.0],[0.0,a0]])
def f(x):
    return 0.0

aOfX = {0:a}; fOfX = {0:f}

center = (0.5,0.5)
radius = 0.45
center = (0.0,0.0)
radius = 0.5

def embeddedBoundary_sdf(x,t):
    xr = x[0] - center[0]
    yr = x[1] - center[1]
    r = math.sqrt(xr**2 + yr**2)
    if r > 1.0e-16:
        n = (-xr/r,-yr/r,0.)
    else:
        n = (1.0,0.0,0.0)
    sdf = radius - r
    if -1.0e-16 < sdf < 1.0e-16:
        print(sdf,x[0],x[1])
    return sdf,n

def embeddedBoundary_u(x,t):
    return ans.uOfX(x)

""" coefficients = ADR.Coefficients(aOfX=aOfX,fOfX=fOfX,velocity=B0_1c[0],nc=1,nd=nd,forceStrongDirichlet=False,
                                embeddedBoundary=True,
                                embeddedBoundary_sdf=embeddedBoundary_sdf,
                                embeddedBoundary_u=embeddedBoundary_u) """

coefficients = ADR.Coefficients(aOfX=aOfX,fOfX=fOfX,velocity=B0_1c[0],nc=1,nd=nd,forceStrongDirichlet=False,
                                immersedBoundary=True,
                                immersedBoundary_sdf=embeddedBoundary_sdf,
                                immersedBoundary_u=embeddedBoundary_u,
                                immersedBoundary_penalty=0.0,
                                immersedBoundary_ghost_penalty=0.0)

def getDBC(x,flag):
    if flag in [domain.boundaryTags['left'], domain.boundaryTags['right'], 
                domain.boundaryTags['bottom'], domain.boundaryTags['top']]:
        return lambda x,t: ans.uOfX(x) 
    
dirichletConditions = {0:getDBC}

fluxBoundaryConditions = {0:'noFlow'}

def getFlux(x,flag):
    pass

advectiveFluxBoundaryConditions =  {0:getFlux}

diffusiveFluxBoundaryConditions = {0:{0:getFlux}}