"""test_1 -- Celia et al. (1990) Fig. 6(b): infiltration into a 1 m sand column.

Dirichlet head at both ends (psi = -0.75 m at the top, -10 m at the bottom),
uniform psi = -10 m initially, T = 1 d.
"""
from proteus import *
from proteus.default_p import *
from proteus.richards import Richards
from proteus import Context

# Scheme selection.  The defaults are the FCT scheme; the other two are
#   Galerkin              -C "STABILIZATION_TYPE=0 FCT=False"
#   entropy viscosity     -C "STABILIZATION_TYPE=2 FCT=False"
# test_richards.py drives the same three through Context.contextOptionsString.
ct = Context.Options([
    ("STABILIZATION_TYPE", 2, "0: Galerkin (no stabilization), 2: entropy viscosity"),
    ("FCT", True, "apply the flux-corrected-transport limiter"),
], mutable=True)

nd = 1

#1 m column, x[0] measured upward from the bottom
L=(1.0,1.0,1.0)

analyticalSolution = None

viscosity     = 8.9e-4  #kg/(m*s)
density       = 998.2   #kg/m^3
gravity       = 9.8     #m/s^2
beta          = 0.0     #same for every scheme, so the comparison is controlled
m_per_s_by_m_per_d = 1.1574074e-5
permeability  = (7.967*m_per_s_by_m_per_d)*viscosity/(gravity*density)  #m^2
thetaS        = 0.368   #-
thetaR        = 0.102   #-
mvg_alpha     = 3.35    #1/m
mvg_n         = 2.00
mvg_m         = 1.0 - 1.0/mvg_n
lengthScale   = 1.0     #m
timeScale     = 1.0     #d
#make non-dimensional
dimensionless_conductivity  = (timeScale*density*gravity*permeability/(viscosity*lengthScale))/m_per_s_by_m_per_d
dimensionless_density  = 1.0
dimensionless_gravity  = numpy.array([-1.0,
                                       0.0,
                                       0.0])
dimensionless_alpha    = mvg_alpha*lengthScale

nMediaTypes  = 1
alphaVGtypes = numpy.zeros((nMediaTypes+1,),'d')
nVGtypes     = numpy.zeros((nMediaTypes+1,),'d')
thetaStypes  = numpy.zeros((nMediaTypes+1,),'d')
thetaRtypes  = numpy.zeros((nMediaTypes+1,),'d')
thetaSRtypes = numpy.zeros((nMediaTypes+1,),'d')
KsTypes      = numpy.zeros((nMediaTypes+1,1),'d')

for i in range(nMediaTypes+1):
    alphaVGtypes[i] = mvg_alpha
    nVGtypes[i]     = mvg_n
    thetaStypes[i]  = thetaS
    thetaRtypes[i]  = thetaR
    thetaSRtypes[i] = thetaStypes[i] - thetaRtypes[i]
    KsTypes[i,:]    = [dimensionless_conductivity]  #m/d

LevelModelType = Richards.LevelModel
coefficients = Richards.Coefficients(nd,
                                     KsTypes,
                                     nVGtypes,
                                     alphaVGtypes,
                                     thetaRtypes,
                                     thetaSRtypes,
                                     gravity=dimensionless_gravity,
                                     density=dimensionless_density,
                                     beta=beta,
                                     diagonal_conductivity=True,
                                     STABILIZATION_TYPE=ct.STABILIZATION_TYPE,
                                     ENTROPY_TYPE=1,
                                     LUMPED_MASS_MATRIX=False,
                                     FCT=ct.FCT,
                                     MONOLITHIC=False,
                                     num_fct_iter=1,
                                     # FOR ENTROPY VISCOSITY
                                     cE=1.0,
                                     uL=0.0,
                                     uR=1.0,
                                     # FOR ARTIFICIAL COMPRESSION
                                     cK=1.0,
                                     # OUTPUT quantDOFs
                                     outputQuantDOFs=False)

pondingPressure = -0.75   #m, imposed at the top face
bottomPressure  = -10.0   #m, imposed at the bottom face and the uniform IC

def getDBC_Richards_Shock(x,flag):
    if x[0] == L[0]:
        return lambda x,t: pondingPressure
    if x[0] == 0.0:
        return lambda x,t: bottomPressure

dirichletConditions = {0:getDBC_Richards_Shock}

class ShockIC_Richards:
    def uOfXT(self,x,t):
        f = getDBC_Richards_Shock(x,0)
        if f:
            return f(x,t)
        else:
            return bottomPressure

initialConditions  = {0:ShockIC_Richards()}

fluxBoundaryConditions = {0:'outFlow'}

def flux(x,flag):
    return None

advectiveFluxBoundaryConditions =  {0:flux}

diffusiveFluxBoundaryConditions = {0:{}}

T = 24*1/24/timeScale   #1 d
