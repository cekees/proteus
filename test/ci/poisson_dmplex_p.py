from proteus import *
from proteus.default_p import *
from proteus import Context, Comm
from petsc4py import PETSc
comm = Comm.get()

"""
Inhomogeneous Poisson's equation, - Delta(u) = f(x), on unit domain [0,1]x[0,1]x[0,1]

The MMS solution we will use is 
u = sin(2 * pi * x[d]); d in [0, nDimensions-1]

Using this MMS, we get the following source term.

f = 4 * pi^2 * sin(2 * pi * x[d]); d in [0, nDimensions-1]
"""

##\page Tests Test Problems
# "Poisson's equation, -div(grad u) = f(x), on unit domain [0,1]x[0,1]x[0,1]"
#

##\ingroup test
#\file poisson_3d_p.py
#
#\brief Heterogenous Poisson's equations in 3D unit domain [0,1]x[0,1]x[0,1]

#----------------------------------------------------
# Domain - mesh - quadrature
#----------------------------------------------------
# Use the following PETSc options to run this test:
# -dm_plex_dim 3 -dm_refine_volume_limit_pre 0.5 -dm_view ascii -dm_plex_separate_marker 1 -dm_distribute 0 -ksp_type cg -pc_type gamg

name = "poisson_plex"
plexMesh = PETSc.DMPlex()
plexMesh.create(comm.comm)
plexMesh.setFromOptions()
nd = plexMesh.getDimension()
domain = Domain.DMPlexDomain(plexMesh)
domain.MeshOptions.use_plex = True

boundaryTags = { 'bottom': 1, 'front':2, 'right':3, 'back': 4, 'left':5, 'top':6, 'obstacle':7}
    
restrictFineSolutionToAllMeshes=False
parallelPartitioningType = MeshTools.MeshParallelPartitioningTypes.node
domain.MeshOptions.nLayersOfOverlapForParallel = 0

# if opts.usePlex:
#steady-state so no initial conditions
initialConditions = None
#use sparse diffusion representation
sd=True
#identity tensor for defining analytical heterogeneity functions
Ident = numpy.zeros((nd,nd),'d')
Ident[0,0]=1.0; Ident[1,1] = 1.0; Ident[2,2]=1.0

#for computing exact 'Darcy' velocity
class velEx(object):
    def __init__(self,duex,aex):
        self.duex = duex
        self.aex = aex
    def uOfX(self,X):
        du = self.duex.duOfX(X)
        A  = numpy.reshape(self.aex(X),(3,3))
        return -numpy.dot(A,du)
    def uOfXT(self,X,T):
        return self.uOfX(X)


##################################################
#define coefficients a(x)=[a_{ij}] i,j=0,2, right hand side f(x)  and analytical solution u(x)
#u = x*x + y*y + z*z, a_00 = x + 5, a_11 = y + 5.0 + a_22 = z + 10.0
#f = -2*x -2*(5+x) -2*y-2*(5+y) -2*z-2*(10+z)
#

# def a5(x):
#     return numpy.array([[1.0, 0.0, 0.0],[0.0, 1.0, 0.0],[0.0, 0.0, 1.0]],'d')
# def f5(x):
#     return 4.0 * numpy.pi**2 * ( numpy.sin(2.0 * numpy.pi * x[0]) + numpy.sin(2.0 * numpy.pi * x[1]) + numpy.sin(2.0 * numpy.pi * x[2]) )
# #'manufactured' analytical solution
# class u5Ex(object):
#     def __init__(self):
#         pass
#     def uOfX(self,x):
#         return numpy.sin(2.0 * numpy.pi * x[0]) + numpy.sin(2.0 * numpy.pi * x[1]) + numpy.sin(2.0 * numpy.pi * x[2])
#     def uOfXT(self,X,T):
#         return self.uOfX(X)
#     def duOfX(self,X):
#         du = 2.0*numpy.reshape(X[0:3],(3,))
#         return du
#     def duOfXT(self,X,T):
#         return self.duOfX(X)
def a5(x):
    return numpy.array([[x[0] + 5.0,0.0,0.0],[0.0,x[1] + 5.0,0.0],[0.0,0.0,x[2]+10.0]],'d')
def f5(x):
    return -2.0*x[0] -2*(5.+x[0]) -2.*x[1]-2.*(5.+x[1]) -2.*x[2]-2.*(10+x[2])
#'manufactured' analytical solution
class u5Ex(object):
    def __init__(self):
        pass
    def uOfX(self,x):
        return x[0]**2+x[1]**2+x[2]**2
    def uOfXT(self,X,T):
        return self.uOfX(X)
    def duOfX(self,X):
        du = 2.0*numpy.reshape(X[0:3],(3,))
        return du
    def duOfXT(self,X,T):
        return self.duOfX(X)

# #dirichlet boundary condition functions on (x=0,y,z), (x,y=0,z), (x,y=1,z), (x,y,z=0), (x,y,z=1)
# def getDBC5(x,flag):
#     if flag in [boundaryTags['bottom'],boundaryTags['top'],boundaryTags['front'],boundaryTags['back'],boundaryTags['left'],boundaryTags['right']]:
#         return lambda x,t: u5Ex().uOfXT(x,t)
# def getAdvFluxBC5(x,flag):
#     pass
# #specify flux on (x=1,y,z)
# def getDiffFluxBC5(x,flag):
#     pass
#     # if flag == boundaryTags['right']:
#     #     n = numpy.zeros((nd,),'d'); n[0]=1.0
#     #     return lambda x,t: numpy.dot(velEx(u5Ex(),a5).uOfXT(x,t),n)
#     # elif flag == 0:
#     #     return lambda x,t: 0.0

#dirichlet boundary condition functions on (x=0,y,z), (x,y=0,z), (x,y=1,z), (x,y,z=0), (x,y,z=1)
def getDBC5(x,flag):
    if flag in [boundaryTags['bottom'],boundaryTags['top'],boundaryTags['front'],boundaryTags['back'],boundaryTags['left']]:
        return lambda x,t: u5Ex().uOfXT(x,t)
def getAdvFluxBC5(x,flag):
    pass
#specify flux on (x=1,y,z)
def getDiffFluxBC5(x,flag):
    if flag == boundaryTags['right']:
        n = numpy.zeros((nd,),'d'); n[0]=1.0
        return lambda x,t: numpy.dot(velEx(u5Ex(),a5).uOfXT(x,t),n)
    elif flag == 0:
        return lambda x,t: 0.0

#dirichlet boundary condition functions on (x=0,y,z), (x,y=0,z), (x,y=1,z), (x,y,z=0), (x,y,z=1)
# def getDBC5(x,flag):
#     if x[0] in [0.0] or x[1] in [0.0,1.0] or x[2] in [0.0,1.0]:
#         return lambda x,t: u5Ex().uOfXT(x,t)
# def getAdvFluxBC5(x,flag):
#     pass
# #specify flux on (x=1,y,z)
# def getDiffFluxBC5(x,flag):
#     if x[0] == 1.0:
#         n = numpy.zeros((nd,),'d'); n[0]=1.0
#         return lambda x,t: numpy.dot(velEx(u5Ex(),a5).uOfXT(x,t),n)
#     if not (x[0] in [0.0] or x[1] in [0.0,1.0] or x[2] in [0.0,1.0]):
#         return lambda x,t: 0.0

#store a,f in dictionaries since coefficients class allows for one entry per component
aOfX = {0:a5}; fOfX = {0:f5}

#one component
nc = 1
#load analytical solution, dirichlet conditions, flux boundary conditions into the expected variables
analyticalSolution = {0:u5Ex()}
analyticalSolutionVelocity = {0:velEx(analyticalSolution[0],aOfX[0])}
#
dirichletConditions = {0:getDBC5}
advectiveFluxBoundaryConditions =  {0:getAdvFluxBC5}
diffusiveFluxBoundaryConditions = {0:{0:getDiffFluxBC5}}
fluxBoundaryConditions = {0:'setFlow'} #options are 'setFlow','noFlow','mixedFlow'


#equation coefficient names
coefficients = TransportCoefficients.PoissonEquationCoefficients(aOfX,fOfX,nc,nd)
#
coefficients.variableNames=['u0']
