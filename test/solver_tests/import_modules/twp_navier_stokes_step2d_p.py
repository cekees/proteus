"""
Incompressible Navier-Stokes flow around a cylinder in 2D.
"""
from proteus import *
from proteus.default_p import *
from proteus import defaults
defaults.reset_default_p()
import sys
try:
    from . import step2d
except:
    import step2d
reload(step2d)
try:
    from .step2d import *
except:
    from step2d import *

from proteus.mprans import RANS2P

from proteus import Context
ct = Context.get()

bcsTimeDependent = False
LevelModelType = RANS2P.LevelModel
LS_model = None

def signedDistance(x):
    y_val = 0.5 - 1./72.*(x[0] +1)**2
    return x[1] - y_val

phase_func = lambda x : signedDistance(x)

coefficients = RANS2P.Coefficients(epsFact=epsFact_viscosity,
                                   rho_0=rho_0,
                                   nu_0=nu_0,
                                   rho_1=rho_1,
                                   nu_1=nu_1,
                                   g=g,
                                   nd=nd,
                                   LS_model=None,
                                   epsFact_density=epsFact_density,
                                   stokes=False,
                                   useVF=useVF,
                                   useRBLES=0.0,
                                   useMetrics=1.0,
                                   forceStrongDirichlet=ns_forceStrongDirichlet,
                                   NONCONSERVATIVE_FORM=1.0,
                                   MOMENTUM_SGE=1.0,
                                   PRESSURE_SGE=1.0,
                                   VELOCITY_SGE=1.0,
                                   PRESSURE_PROJECTION_STABILIZATION=0.0,
                                   phaseFunction=phase_func)

class uTrue(object):
    def __init__(self):
        pass
    def uOfX(self,x):
        return 1.
    def uOfXT(self,x,t):
        return 4.0*x[1]*(1-x[1])

class vTrue(object):
    def __init__(self):
        pass
    def vOfX(self,x):
        return 0.0
    def uOfXT(self,x,t):
        return self.vOfX(x)

boundary_condition_type = ct.opts.boundary_condition_type # 'ns' or 'fs'

def getDBC_p(x,flag):
    if flag in [boundaryTags['right']]:
        return lambda x,t: 0.0
    else:
        pass

if boundary_condition_type =='fs':
    def getDBC_u(x,flag):
        if flag in [boundaryTags['left']]:
            return lambda x,t : uTrue().uOfXT(x,t)
        elif flag in [boundaryTags['top']]:
            pass
        elif flag in [boundaryTags['bottom']]:
            if x[0] == 0.:
                return lambda x,t: 0.
            else:
                pass
        else:
            pass

elif boundary_condition_type == 'fs_weak':
    def getDBC_u(x,flag):
        if flag in [boundaryTags['left']]:
            return lambda x,t : uTrue().uOfXT(x,t)
        elif flag in [boundaryTags['right']]:
            return lambda x,t : 0.0
        else:
            pass

elif boundary_condition_type == 'ns':
    def getDBC_u(x,flag):
        if flag in [boundaryTags['left']]:
            return lambda x,t : uTrue().uOfXT(x,t)
        elif flag in [boundaryTags['top']]:
            return lambda x,t : 0.0
        elif flag in [boundaryTags['bottom']]:
            return lambda x,t : 0.0
        else:
            pass

if boundary_condition_type == 'fs':
    def getDBC_v(x,flag):
        if flag in [boundaryTags['left']]:
            return lambda x,t: 0.
        elif flag in [boundaryTags['top']]:
            return lambda x,t: 0.
        elif flag in [boundaryTags['bottom']]:
            if x[0] == 0.:
                pass
            else:
                return lambda x,t: 0.    
        else:
            pass

elif boundary_condition_type == 'fs_weak':
    def getDBC_v(x,flag):
        if flag in [boundaryTags['left']]:
            return lambda x,t : 0.
        elif flag in [boundaryTags['right']]:
            return lambda x,t : 0.0
        else:
            pass

elif boundary_condition_type == 'ns':
    def getDBC_v(x,flag):
        if flag in [boundaryTags['left']]:
            return lambda x,t: 0.
        elif flag in [boundaryTags['top'], boundaryTags['bottom']]:
            return lambda x,t: 0.    
        else:
            pass

dirichletConditions = {0:getDBC_p,
                       1:getDBC_u,
                       2:getDBC_v}

def getAFBC_p(x,flag):
    if flag in [boundaryTags['left']]:
        return lambda x,t: -uTrue().uOfXT(x,t)
    if flag in [boundaryTags['top'], boundaryTags['bottom']]:
        return lambda x,t: 0.
    else:
        pass

def getAFBC_u(x,flag):
    if flag in [boundaryTags['top'], boundaryTags['bottom']]:
        return lambda x,t: 0.

def getAFBC_v(x,flag):
    if flag in [boundaryTags['top'], boundaryTags['bottom']]:
        return lambda x,t: 0.

def getDFBC_p(x,flag):
    pass

if boundary_condition_type == 'fs' or boundary_condition_type == 'fs_weak':
    def getDFBC_u(x,flag):
        if flag in [boundaryTags['top'], boundaryTags['bottom'],boundaryTags['right']]:
            return lambda x,t: 0.0
        else:
            pass

if boundary_condition_type == 'ns':
    def getDFBC_u(x,flag):
        if flag in [boundaryTags['right']]:
            return lambda x,t: 0.0
        else:
            pass

if boundary_condition_type == 'fs' or boundary_condition_type == 'fs_weak':
    def getDFBC_v(x,flag):
        if flag in [boundaryTags['bottom'],boundaryTags['top'],boundaryTags['right']]:
            return lambda x,t: 0.0
        else:
            pass

if boundary_condition_type == 'ns':
    def getDFBC_v(x,flag):
        if flag in [boundaryTags['right']]:
            return lambda x,t: 0.0
        else:
            pass


advectiveFluxBoundaryConditions =  {0:getAFBC_p,
                                    1:getAFBC_u,
                                    2:getAFBC_v}

diffusiveFluxBoundaryConditions = {0:{},
                                   1:{1:getDFBC_u},
                                   2:{2:getDFBC_v}}

if boundary_condition_type == 'fs':
    diffusiveFluxNormalBoundaryConditions = {0:{},
                                             1:{},
                                             2:{}}

elif boundary_condition_type == 'ns':
    diffusiveFluxNormalBoundaryConditions = {0:{},
                                             1:{},
                                             2:{}}

class Steady_p(object):
    def __init__(self):
        pass
    def uOfXT(self,x,t):
        return 0.0

class Steady_u(object):
    def __init__(self):
        pass
    def uOfXT(self,x,t):
        return 0.0

class Steady_v(object):
    def __init__(self):
        pass
    def uOfXT(self,x,t):
        return 0.0


initialConditions = {0:Steady_p(),
                     1:Steady_u(),
                     2:Steady_v()}
