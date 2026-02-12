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

opts = Context.Options([
    ("test", 1.0, "which test to run (default is 1.0)"),
    ("unstructured", False, "use unstructured mesh (default is structured)"),
    ("skew", 0.0001, "skew the domain when using unstructured mesh"),
    ("refinement", 0, "number of times to refine the mesh (default is 0)"),
    ("qOrder", 6, "order of quadrature (default is 6)"),
    ("usePetsc", False, "Use PETSc linear solvers"),
    ("name", "ladr_ss_2d", "name of the test (default is ladr_ss_2d)"),
])

name = opts.name
nd = 2
L=(2.0,2.0)#,2.0)
if opts.unstructured:
    L=(2.0,2.0+opts.skew)#,2.0)#throw off rectangular domain
x0 = (-1.0,-1.0)#,-1.0)
domainR = Domain.RectangularDomain(L=L,x=x0,name="adr",units="m")
domainR.writePoly("ladr_ss_2d_p")
domainUS = Domain.PlanarStraightLineGraphDomain("ladr_ss_2d_p")
domainUS.boundaryTags = domainR.boundaryTags
domain = domainR
if opts.unstructured:
    domain = domainUS    
a0=1.0
if opts.test == 2.0:
    a0 = 10.0
elif opts.test ==2.1:
    a0 = -3.0
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
        
class LevequeLiExample2(AnalyticalSolutions.SteadyState):
    def __init__(self):
        super(LevequeLiExample2, self).__init__()
    def uOfX(self, x):
        b=a0
        C=0.1
        r = (x[0]**2 + x[1]**2)**0.5
        if r <= 0.5:
            return r**2
        else:
            return (1 - 1/(8*b) - 1/b)/4 + ((r**4)/2 + r**2)/b + C*math.log(2*r)/b
        
class LevequeLiExample3(AnalyticalSolutions.SteadyState):
    def __init__(self):
        super(LevequeLiExample3, self).__init__()
    def uOfX(self, x):
        import math
        r = (x[0]**2 + x[1]**2)**0.5
        if r <= 0.5:
            return math.exp(x[0])*math.cos(x[1])
        else:
            return 0.0
        
class LevequeLiExample4(AnalyticalSolutions.SteadyState):
    def __init__(self):
        super(LevequeLiExample4, self).__init__()
    def uOfX(self, x):
        import math
        r = (x[0]**2 + x[1]**2)**0.5
        if r <= 0.5:
            return x[0]**2 - x[1]**2
        else:
            return 0.0

class LevequeLiExample4l(AnalyticalSolutions.SteadyState):
    def __init__(self):
        super(LevequeLiExample4l, self).__init__()
    def uOfX(self, x):
        r = (x[0]**2 + x[1]**2)**0.5
        if r <= 0.5:
            return x[0] - x[1]
        else:
            return 0.0

class PWC(AnalyticalSolutions.SteadyState):
    def __init__(self):
        super(PWC, self).__init__()
    def uOfX(self, x):
        r = (x[0]**2 + x[1]**2)**0.5
        if r <= 0.5:
            return 1.0
        else:
            return 0.0
        
class PWL(AnalyticalSolutions.SteadyState):
    def __init__(self):
        super(PWL, self).__init__()
    def uOfX(self, x):
        r = (x[0]**2 + x[1]**2)**0.5
        if r <= 0.5:
            return x[0] + x[1] + 1.0
        else:
            return x[0] + x[1]
        
class PWQ(AnalyticalSolutions.SteadyState):
    def __init__(self):
        super(PWQ, self).__init__()
    def uOfX(self, x):
        r = (x[0]**2 + x[1]**2)**0.5
        if r <= 0.5:
            return x[0]**2 + x[1]**2 + 1.0
        else:
            return x[0]**2 + x[1]**2

class PWCubic(AnalyticalSolutions.SteadyState):
    def __init__(self):
        super(PWCubic, self).__init__()
    def uOfX(self, x):
        r = (x[0]**2 + x[1]**2)**0.5
        if r <= 0.5:
            return x[0]**3 + x[1]**3 + 1.0
        else:
            return x[0]**3 + x[1]**3

class JiEtal14Example1(AnalyticalSolutions.SteadyState):
    def __init__(self, betaMinus=1.0, betaPlus=1000.0):
        self.betaMinus = betaMinus
        self.betaPlus = betaPlus
        super(JiEtal14Example1, self).__init__()
    def uOfX(self, x):
        r = (x[0]**2 + x[1]**2)**0.5
        if r <= 0.5:
            return r**3/self.betaMinus
        else:
            return r**3/self.betaPlus + (1.0/self.betaMinus-1.0/self.betaPlus)*0.5**3

class trigSolution(AnalyticalSolutions.SteadyState):
    def __init__(self):
        super(trigSolution, self).__init__()
    def uOfX(self, x):
        return math.sin(math.pi*x[0])*math.sin(math.pi*x[1])

if opts.test == 1.0:
    ans = LevequeLiExample1()
elif opts.test == 2.0 or opts.test == 2.1:
    ans = LevequeLiExample2()
elif opts.test == 3.0:
    ans = LevequeLiExample3()
elif opts.test == 4.0:
    ans = LevequeLiExample4()
elif opts.test == 4.1:
    ans = LevequeLiExample4l()
elif opts.test == 5.0:
    ans = PWC()
elif opts.test == 6.0:
    ans = PWL()
elif opts.test == 7.0:
    ans = PWQ()
elif opts.test == 8.0:
    ans = JiEtal14Example1(betaMinus=1.0, betaPlus=1000.0)
elif opts.test == 9.0:
    ans = PWCubic()
elif opts.test == 10.0:
    ans = trigSolution()
else:
    assert False, "Unknown test %s" % opts.test

analyticalSolution = {0:ans}
initialConditions = None

center = (0.0,0.0)
radius = 0.5

# Interface material parameters (diffusion on each side of the interface).
# a(x) below uses these as the source of truth.
mua = 1.0   # inner (r <= radius)
mub = 1.0   # outer (r >  radius)
jf  = 0.0
if opts.test == 2.0:
    mua = radius**2 + 1  # = 1.25, inner diffusion at the interface
    mub = a0             # = 10.0
elif opts.test == 2.1:
    mua = radius**2 + 1  # = 1.25
    mub = a0             # = -3.0
elif opts.test == 8.0:
    mub = 1000.0

if opts.test == 2.0 or opts.test == 2.1:
    # Leveque & Li 1994, Example 2 — inner diffusion varies with position
    def a(x):
        if (x[0]**2 + x[1]**2) <= radius**2:
            _a = x[0]**2 + x[1]**2 + 1
        else:
            _a = mub
        return numpy.array([[_a,0.0],[0.0,_a]])
    def f(x):
        return -(8*(x[0]**2 + x[1]**2) + 4)

if opts.test in [1.0, 3.0, 4.0, 4.1, 5.0, 6.0]:
    # Leveque & Li 1994, Example 1, 3 & 4, PWC, PWL
    def a(x):
        return numpy.array([[mua,0.0],[0.0,mua]])
    def f(x):
        return 0.0
elif opts.test == 7.0:
    # PWQ
    def a(x):
        return numpy.array([[mua,0.0],[0.0,mua]])
    def f(x):
        return -4.0
elif opts.test == 8.0:
    # Ji et al 2014, Example 1 — piecewise constant diffusion
    def a(x):
        if (x[0]**2 + x[1]**2) <= radius**2:
            return numpy.array([[mua,0.0],[0.0,mua]])
        else:
            return numpy.array([[mub,0.0],[0.0,mub]])
    def f(x):
        return -9.0*(x[0]**2 + x[1]**2)**0.5
elif opts.test == 9.0:
    # PWCubic
    def a(x):
        return numpy.array([[mua,0.0],[0.0,mua]])
    def f(x):
        return -6.0*(x[0] + x[1])
elif opts.test == 10.0:
    # trig solution
    def a(x):
        return numpy.array([[mua,0.0],[0.0,mua]])
    def f(x):
        return 2.0*math.pi**2*math.sin(math.pi*x[0])*math.sin(math.pi*x[1])

aOfX = {0:a}; fOfX = {0:f}

def embeddedBoundary_sdf(x,t):
    xr = x[0] - center[0]
    yr = x[1] - center[1]
    r = (xr**2 + yr**2)**0.5
    if r > 1.0e-16:
        n = (xr/r,yr/r,0.)
    else:
        n = (1.0,0.0,0.0)
    sdf = r - radius
    return sdf,n

#n = (1.0/2.0**0.5,-1.0/2.0**0.5,0.0)
#def embeddedBoundary_sdf(x,t):
#    n = (1.0/2.0**0.5,-1.0/2.0**0.5,0.0)
#    sdf = x[0] - x[1] - 1.0
#    return sdf,n

def embeddedBoundary_u(x,t):
    return ans.uOfX(x)

""" coefficients = ADR.Coefficients(aOfX=aOfX,fOfX=fOfX,velocity=B0_1c[0],nc=1,nd=nd,forceStrongDirichlet=False,
                                embeddedBoundary=True,
                                embeddedBoundary_sdf=embeddedBoundary_sdf,
                                embeddedBoundary_u=embeddedBoundary_u) """

coefficients = ADR.Coefficients(aOfX=aOfX,fOfX=fOfX,velocity=B0_1c[0],nc=1,nd=nd,
                                forceStrongDirichlet=True,
                                mua=mua,
                                mub=mub,
                                jf=jf,
                                immersedBoundary=True,
                                immersedBoundary_sdf=embeddedBoundary_sdf,
                                immersedBoundary_u=embeddedBoundary_u,
                                immersedBoundary_penalty=0.0,
                                immersedBoundary_ghost_penalty=0.0,
                                test = opts.test)

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
