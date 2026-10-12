"""1D heat equation, u_t - u_xx = 0, with homogeneous Dirichlet conditions."""
import numpy
from proteus.default_p import *
from proteus.TransportCoefficients import LinearVADR_ConstantCoefficients

nd = 1
L = (1.0, 1.0, 1.0)
T = 0.01

coefficients = LinearVADR_ConstantCoefficients(nc=1,
                                               M=[1.0],
                                               A=[numpy.eye(1)],
                                               B=[numpy.zeros((1,))],
                                               C=[0.0])

def getDBC(x, flag):
    if x[0] in (0.0, L[0]):
        return lambda x, t: 0.0

dirichletConditions = {0: getDBC}

class IC:
    def uOfXT(self, x, t):
        return numpy.sin(numpy.pi*x[0])

initialConditions = {0: IC()}
advectiveFluxBoundaryConditions = {}
diffusiveFluxBoundaryConditions = {0: {}}
