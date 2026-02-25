from __future__ import division
from builtins import range
#from past.utils import old_div
import proteus
from .cmphase_co2 import *
import numpy as np
from proteus.Transport import OneLevelTransport
from proteus.Transport import TC_base, NonlinearEquation, logEvent, memory
from proteus.Transport import FluxBoundaryConditions, Comm, cfemIntegrals
from proteus.Transport import DOFBoundaryConditions, Quadrature
from proteus.mprans import cArgumentsDict
from proteus.LinearAlgebraTools import SparseMat
from proteus import TimeIntegration
from proteus.NonlinearSolvers import ExplicitLumpedMassMatrixForRichards
#from proteus.NonlinearSolvers import Newton


class ThetaScheme(TimeIntegration.BackwardEuler):
    def __init__(self,transport,integrateInterpolationPoints=False):
        self.transport=transport
        TimeIntegration.BackwardEuler.__init__(self,transport, integrateInterpolationPoints)
    def updateTimeHistory(self,resetFromDOF=False):
        TimeIntegration.BackwardEuler.updateTimeHistory(self,resetFromDOF)
        self.transport.u_dof_old_water[:] = self.u[0] #water
        self.transport.u_dof_old_air[:] = self.u[1] # air
        
        # self.transport.u_dof_old_ci[0][:] = self.u[0] #water
        # self.transport.u_dof_old_ci[1][:] = self.u[1] # air
            
class RKEV(TimeIntegration.SSP):
    from proteus import TimeIntegration
    """
    Wrapper for SSPRK time integration using EV

    ... more to come ...
    """

    def __init__(self, transport, timeOrder=1, runCFL=0.1, integrateInterpolationPoints=False):
        TimeIntegration.SSP.__init__(self, transport, integrateInterpolationPoints=integrateInterpolationPoints)
        self.runCFL = runCFL
        self.dtLast = None
        self.isAdaptive = True
        # About the cfl
        assert transport.coefficients.STABILIZATION_TYPE > 0, "SSP method just works for edge based EV methods; i.e., STABILIZATION_TYPE>0"
        assert hasattr(transport, 'edge_based_cfl'), "No edge based cfl defined"
        self.cfl = transport.edge_based_cfl
        # Stuff particular for SSP
        self.timeOrder = timeOrder  # order of approximation
        self.nStages = timeOrder  # number of stages total
        self.lstage = 0  # last stage completed
        # storage vectors
        self.u_dof_last = {}
        # per component stage values, list with array at each stage
        self.u_dof_stage = {}
        for ci in range(self.nc):
            if ('m', ci) in transport.q:
                self.u_dof_last[ci] = transport.u[ci].dof.copy()
                self.u_dof_stage[ci] = []
                for k in range(self.nStages + 1):
                    self.u_dof_stage[ci].append(transport.u[ci].dof.copy())
                    #print()
        

    # def set_dt(self, DTSET):
    #    self.dt = DTSET #  don't update t
    def choose_dt(self):
        comm = Comm.get()
        maxCFL = 1.0e-6
        maxCFL = max(maxCFL, comm.globalMax(self.cfl.max()))
        self.dt = self.runCFL/ maxCFL
        if self.dtLast is None:
            self.dtLast = self.dt
        self.t = self.tLast + self.dt
        self.substeps = [self.t for i in range(self.nStages)]  # Manuel is ignoring different time step levels for now
        
        

    def initialize_dt(self, t0, tOut, q):
        """
        Modify self.dt
        """
        self.tLast = t0
        self.choose_dt()
        self.t = t0 + self.dt

    def setCoefficients(self):
        """
        beta are all 1's here
        mwf not used right now
        """
        self.alpha = np.zeros((self.nStages, self.nStages), 'd')
        self.dcoefs = np.zeros((self.nStages), 'd')

    def updateStage(self):
        """
        Need to switch to use coefficients
        """
        self.lstage += 1
        assert self.timeOrder in [1, 2, 3]
        assert self.lstage > 0 and self.lstage <= self.timeOrder
        if self.timeOrder == 3:
            if self.lstage == 1:
                logEvent("First stage of SSP33 method", level=4)
                for ci in range(self.nc):
                    self.u_dof_stage[ci][self.lstage][:] = self.transport.u[ci].dof
                    # update u_dof_old
                    self.transport.u_dof_old[:] = self.u_dof_stage[ci][self.lstage]
            elif self.lstage == 2:
                logEvent("Second stage of SSP33 method", level=4)
                for ci in range(self.nc):
                    self.u_dof_stage[ci][self.lstage][:] = self.transport.u[ci].dof
                    self.u_dof_stage[ci][self.lstage] *= 1./ 4.
                    self.u_dof_stage[ci][self.lstage] += 3. / 4. * self.u_dof_last[ci]
                    # Update u_dof_old
                    self.transport.u_dof_old[:] = self.u_dof_stage[ci][self.lstage]
            elif self.lstage == 3:
                logEvent("Third stage of SSP33 method", level=4)
                for ci in range(self.nc):
                    self.u_dof_stage[ci][self.lstage][:] = self.transport.u[ci].dof
                    self.u_dof_stage[ci][self.lstage] *= 2.0/ 3.0
                    self.u_dof_stage[ci][self.lstage] += 1.0 / 3.0 * self.u_dof_last[ci]
                    # update u_dof_old
                    self.transport.u_dof_old[:] = self.u_dof_last[ci]
                    # update solution to u[0].dof
                    self.transport.u[ci].dof[:] = self.u_dof_stage[ci][self.lstage]
        elif self.timeOrder == 2:
            if self.lstage == 1:
                logEvent("First stage of SSP22 method", level=4)
                for ci in range(self.nc):
                    self.u_dof_stage[ci][self.lstage][:] = self.transport.u[ci].dof
                    # Update u_dof_old
                    self.transport.u_dof_old[:] = self.transport.u[ci].dof
            elif self.lstage == 2:
                logEvent("Second stage of SSP22 method", level=4)
                for ci in range(self.nc):
                    self.u_dof_stage[ci][self.lstage][:] = self.transport.u[ci].dof
                    self.u_dof_stage[ci][self.lstage][:] *= 1. / 2.
                    self.u_dof_stage[ci][self.lstage][:] += 1. / 2. * self.u_dof_last[ci]
                    # update u_dof_old
                    self.transport.u_dof_old[:] = self.u_dof_last[ci]
                    # update solution to u[0].dof
                    self.transport.u[ci].dof[:] = self.u_dof_stage[ci][self.lstage]
        else:
            assert self.timeOrder == 1
            for ci in range(self.nc):
                self.u_dof_stage[ci][self.lstage][:] = self.transport.u[ci].dof[:]
                self.transport.u_dof_old[:] = self.transport.u[ci].dof
                

    def initializeTimeHistory(self, resetFromDOF=True):
        """
        Push necessary information into time history arrays
        """
        for ci in range(self.nc):
            self.u_dof_last[ci][:] = self.transport.u[ci].dof[:]
            for k in range(self.nStages):
                self.u_dof_stage[ci][k][:] = self.transport.u[ci].dof[:]

    def updateTimeHistory(self, resetFromDOF=False):
        """
        assumes successful step has been taken
        """

        self.t = self.tLast + self.dt
        for ci in range(self.nc):
            self.u_dof_last[ci][:] = self.transport.u[ci].dof[:]
            for k in range(self.nStages):
                self.u_dof_stage[ci][k][:] = self.transport.u[ci].dof[:]
        self.lstage = 0
        self.dtLast = self.dt
        self.tLast = self.t

    def generateSubsteps(self, tList):
        """
        create list of substeps over time values given in tList. These correspond to stages
        """
        self.substeps = []
        tLast = self.tLast
        for t in tList:
            dttmp = t - tLast
            self.substeps.extend([tLast + dttmp for i in range(self.nStages)])
            tLast = t

    def resetOrder(self, order):
        """
        initialize data structures for stage updges
        """
        self.timeOrder = order  # order of approximation
        self.nStages = order  # number of stages total
        self.lstage = 0  # last stage completed
        # storage vectors
        # per component stage values, list with array at each stage
        self.u_dof_stage = {}
        for ci in range(self.nc):
            if ('m', ci) in self.transport.q:
                self.u_dof_stage[ci] = []
                for k in range(self.nStages + 1):
                    self.u_dof_stage[ci].append(self.transport.u[ci].dof.copy())
        self.substeps = [self.t for i in range(self.nStages)]

    def setFromOptions(self, nOptions):
        """
        allow classes to set various numerical parameters
        """
        if 'runCFL' in dir(nOptions):
            self.runCFL = nOptions.runCFL
        flags = ['timeOrder']
        for flag in flags:
            if flag in dir(nOptions):
                val = getattr(nOptions, flag)
                setattr(self, flag, val)
                if flag == 'timeOrder':
                    self.resetOrder(self.timeOrder)

class Coefficients(proteus.TransportCoefficients.TC_base):
    """
    version of Re where element material type id's used in evals
    """
    from proteus.ctransportCoefficients import conservativeHeadRichardsMualemVanGenuchtenHetEvaluateV2
    from proteus.ctransportCoefficients import conservativeHeadRichardsMualemVanGenuchten_sd_het
    def __init__(self,
                 nd,
                 Ksw_types,
                 vgm_n_types,
                 vgm_alpha_types,
                 thetaR_types,
                 thetaSR_types,
                 gravity,
                 density_water,
                 density_air,
                 beta_water,
                 beta_air,
                 diagonal_conductivity=True,
                 getSeepageFace=None,
                # FOR EDGE BASED EV
                 STABILIZATION_TYPE='Implicit_FCT',
                 PSK_TYPE=0, # PSK_TYPE=0: VG , PSK_TYPE=1: Brooks-Corey
                 ENTROPY_TYPE=2,  # logarithmic
                 LUMPED_MASS_MATRIX=False,
                 MONOLITHIC=False,
                 VMS=0.0,
                 SC=0.0,
                 FCT=True,
                 num_fct_iter=1,
                 # FOR ENTROPY VISCOSITY
                 cE=1.0,
                 uL=0.0,
                 uR=1.0,
                 # FOR ARTIFICIAL COMPRESSION
                 cK=1.0,
                 # OUTPUT quantDOFs
                 outputQuantDOFs=False,
                  ):
        self.VMS=VMS
        self.SC=SC
        self.anb_seepage_flux= 0.00
        #self.anb_seepage_flux_n =0.0
        variableNames=['pressure_head water' , 'pressure head air']
        nc=2
        mass={0:{0:'nonlinear'}, 1:{1:'nonlinear'}}
        advection={0:{0:'nonlinear'}, 1:{1:'nonlinear'}}
        diffusion={0:{0:{0:'nonlinear'}},  1:{1:{1:'nonlinear'}}}
        potential={0:{0:'u'}, 1:{1:'u'}}
        reaction={0:{0:'linear'}, 1:{1:'linear'} }
        hamiltonian={}
        self.getSeepageFace=getSeepageFace
        self.gravity=gravity
        self.rho_water = density_water
        self.rho_air = density_air
        self.beta_water=beta_water
        self.beta_air=beta_air
        self.vgm_n_types = vgm_n_types
        self.vgm_alpha_types = vgm_alpha_types
        self.thetaR_types    = thetaR_types
        self.thetaSR_types   = thetaSR_types
        self.elementMaterialTypes = None
        self.exteriorElementBoundaryTypes  = None
        self.materialTypes_q    = None
        self.materialTypes_ebq  = None
        self.materialTypes_ebqe  = None
        self.nd = nd
        self.nMaterialTypes = len(thetaR_types)
        self.q = {}; self.ebqe = {}; self.ebq = {}; self.ebq_global={}
        #try to allow some flexibility in input of permeability/conductivity tensor
        self.diagonal_conductivity = diagonal_conductivity
        self.Ksw_types_in = Ksw_types
        if self.diagonal_conductivity:
            sparseDiffusionTensors = {(0,0):(np.arange(self.nd+1,dtype='i'),
                                             np.arange(self.nd,dtype='i'))}

            assert len(Ksw_types.shape) in [1,2], "if diagonal conductivity true then Ksw_types scalar or vector of diagonal entries"
            #allow scalar input Ks
            if len(Ksw_types.shape)==1:
                self.Ksw_types = np.zeros((self.nMaterialTypes,self.nd),'d')
                for I in range(self.nd):
                    self.Ksw_types[:,I] = Ksw_types
            else:
                self.Ksw_types = Ksw_types
        else: #full
            sparseDiffusionTensors = {(0,0):(np.arange(self.nd**2+1,step=self.nd,dtype='i'),
                                             np.array([list(range(self.nd)) for row in range(self.nd)],dtype='i'))}
            assert len(Ksw_types.shape) in [1,2], "if full tensor conductivity true then Ksw_types scalar or 'flattened' row-major representation of entries"
            if len(Ksw_types.shape)==1:
                self.Ksw_types = np.zeros((self.nMaterialTypes,self.nd**2),'d')
                for I in range(self.nd):
                    self.Ksw_types[:,I*self.nd+I] = Ksw_types
            else:
                assert Ksw_types.shape[1] == self.nd**2
                self.Ksw_types = Ksw_types

        stabilization_types = {"Galerkin":0, 
                               "EV_Stab":1, 
                               "EntropyViscosity":2, 
                               "Implicit_FCT":3}
        try:
            if isinstance(STABILIZATION_TYPE, int):
                STABILIZATION_TYPE = [key for key, value in stabilization_types.items() if value == STABILIZATION_TYPE][0]
            
            self.STABILIZATION_TYPE = stabilization_types[STABILIZATION_TYPE]
        except:
            raise ValueError("STABILIZATION_TYPE must be one of "+str(stabilization_types.keys())+" not "+STABILIZATION_TYPE)
        
        
        # EDGE BASED (AND ENTROPY) VISCOSITY
        self.LUMPED_MASS_MATRIX = LUMPED_MASS_MATRIX
        self.MONOLITHIC = MONOLITHIC
        #self.STABILIZATION_TYPE = STABILIZATION_TYPE
        self.PSK_TYPE = PSK_TYPE
        self.ENTROPY_TYPE = ENTROPY_TYPE
        self.FCT = FCT
        self.num_fct_iter=num_fct_iter
        self.uL = uL
        self.uR = uR
        self.cK = cK
        self.forceStrongConditions = False
        self.cE = cE
        self.outputQuantDOFs = outputQuantDOFs
        #For seepage anb
        self.model = None 

        TC_base.__init__(self,
                         nc,
                         mass,
                         advection,
                         diffusion,
                         potential,
                         reaction,
                         hamiltonian,
                         variableNames,
                         sparseDiffusionTensors = sparseDiffusionTensors,
                         useSparseDiffusion = True)
        

    def initializeMesh(self,mesh):
        from proteus.SubsurfaceTransportCoefficients import BlockHeterogeneousCoefficients
        self.elementMaterialTypes,self.exteriorElementBoundaryTypes,self.elementBoundaryTypes = BlockHeterogeneousCoefficients(mesh).initializeMaterialTypes()
        #want element boundary material types for evaluating heterogeneity
        #not boundary conditions
        self.isSeepageFace = np.zeros((mesh.nExteriorElementBoundaries_global),'i')
        if self.getSeepageFace != None:
            for ebNE in range(mesh.nExteriorElementBoundaries_global):
                #mwf missing ebNE-->ebN?
                ebN = mesh.exteriorElementBoundariesArray[ebNE]
                #print "eb flag",mesh.elementBoundaryMaterialTypes[ebN]
            
                #print self.getSeepageFace(mesh.elementBoundaryMaterialTypes[ebN])
                self.isSeepageFace[ebNE] = self.getSeepageFace(mesh.elementBoundaryMaterialTypes[ebN])
        #print (self.isSeepageFace)
    def initializeElementQuadrature(self,t,cq):
        self.materialTypes_q = self.elementMaterialTypes
        self.q_shape = cq[('u',0)].shape
        for ci in range(getattr(self, 'nc', 1)):                    
            assert cq[('u', ci)].shape == self.q_shape, \
                "All components must share the same (element, qpoint) shape"
        #self.anb_seepage_flux= anb_seepage_flux
        #print("The seepage is ", anb_seepage_flux)
#        cq['Ks'] = np.zeros(self.q_shape,'d')
#        for k in range(self.q_shape[1]):
#            cq['Ks'][:,k] = self.Ksw_types[self.elementMaterialTypes,0]
        self.q[('vol_frac',0)] = np.zeros(self.q_shape,'d')
    def initializeElementBoundaryQuadrature(self,t,cebq,cebq_global):
        self.materialTypes_ebq = np.zeros(cebq[('u',0)].shape[0:2],'i')
        self.ebq_shape = cebq[('u',0)].shape
        for ci in range(getattr(self, 'nc', 1)):                     # 2PH
            assert cebq[('u', ci)].shape == self.ebq_shape, \
            "All components must share the same (elem, ebLocal, qpoint) shape"
        for ebN_local in range(self.ebq_shape[1]):
            self.materialTypes_ebq[:,ebN_local] = self.elementMaterialTypes
        self.ebq[('vol_frac',0)] = np.zeros(self.ebq_shape,'d')

    def initializeGlobalExteriorElementBoundaryQuadrature(self,t,cebqe):
        self.materialTypes_ebqe = self.exteriorElementBoundaryTypes
        self.ebqe_shape = cebqe[('u',0)].shape
        self.ebqe[('vol_frac',0)] = np.zeros(self.ebqe_shape,'d')
        #
    

    def evaluate(self,t,c):
        if c[('u',0)].shape == self.q_shape:
            materialTypes = self.materialTypes_q
            vol_frac = self.q[('vol_frac',0)]
        elif c[('u',0)].shape == self.ebqe_shape:
            materialTypes = self.materialTypes_ebqe
            vol_frac = self.ebqe[('vol_frac',0)]
        elif c[('u',0)].shape == self.ebq_shape:
            materialTypes = self.materialTypes_ebq
            vol_frac = self.ebq[('vol_frac',0)]
        else:
            assert False, "no materialType found to match c[('u',0)].shape= %s " % c[('u',0)].shape
        self.conservativeHeadRichardsMualemVanGenuchten_sd_het(self.sdInfo[(0,0)][0],
                                                               self.sdInfo[(0,0)][1],
                                                               materialTypes,
                                                               self.rho,
                                                               self.beta,
                                                               self.gravity,
                                                               self.vgm_alpha_types,
                                                               self.vgm_n_types,
                                                               self.thetaR_types,
                                                               self.thetaSR_types,
                                                               self.Ksw_types,
                                                               c[('u',0)],
                                                               c[('m',0)],
                                                               c[('dm',0,0)],
                                                               c[('f',0)],
                                                               c[('df',0,0)],
                                                               c[('a',0,0)],
                                                               c[('da',0,0,0)],
                                                               vol_frac)
         # Log grad(u) for debugging
        if ('grad(u)', 0) in c:
            logEvent(f"Richards grad(u): mean={c[('grad(u)', 0)].mean()}, min={c[('grad(u)', 0)].min()}, max={c[('grad(u)', 0)].max()}")
        else:
            logEvent("Warning: grad(u) is not available in Richards coefficients.")
        
        # Add logging for grad(u)
        # print "Picard---------------------------------------------------------------"
        # c[('df',0,0)][:] = 0.0
        # c[('da',0,0,0)][:] = 0.0
#         self.conservativeHeadRichardsMualemVanGenuchtenHetEvaluateV2(materialTypes,
#                                                                      self.rho,
#                                                                      self.beta,
#                                                                      self.gravity,
#                                                                      self.vgm_alpha_types,
#                                                                      self.vgm_n_types,
#                                                                      self.thetaR_types,
#                                                                      self.thetaSR_types,
#                                                                      self.Ksw_types,
#                                                                      c[('u',0)],
#                                                                      c[('m',0)],
#                                                                      c[('dm',0,0)],
#                                                                      c[('f',0)],
#                                                                      c[('df',0,0)],
#                                                                      c[('a',0,0)],
#                                                                      c[('da',0,0,0)])
        #mwf debug
        if (np.isnan(c[('da',0,0,0)]).any() or
            np.isnan(c[('a',0,0)]).any() or
            np.isnan(c[('df',0,0)]).any() or
            np.isnan(c[('f',0)]).any() or
            np.isnan(c[('u',0)]).any() or
            np.isnan(c[('m',0)]).any() or
            np.isnan(c[('dm',0,0)]).any()):
            import pdb
            pdb.set_trace()        
class LevelModel(proteus.Transport.OneLevelTransport):
    nCalls=0
    def __init__(self,
                 uDict,
                 phiDict,
                 testSpaceDict,
                 matType,
                 dofBoundaryConditionsDict,
                 dofBoundaryConditionsSetterDict,
                 coefficients,
                 elementQuadrature,
                 elementBoundaryQuadrature,
                 fluxBoundaryConditionsDict=None,
                 advectiveFluxBoundaryConditionsSetterDict=None,
                 diffusiveFluxBoundaryConditionsSetterDictDict=None,
                 stressTraceBoundaryConditionsSetterDict=None,
                 stabilization=None,
                 shockCapturing=None,
                 conservativeFluxDict=None,
                 numericalFluxType=None,
                 TimeIntegrationClass=None,
                 massLumping=False,
                 reactionLumping=False,
                 options=None,
                 name='defaultName',
                 reuse_trial_and_test_quadrature=True,
                 sd = True,
                 movingDomain=False,
                 bdyNullSpace=False):
        self.bdyNullSpace=bdyNullSpace
        #
        #set the objects describing the method and boundary conditions
        #
        self.movingDomain=movingDomain
        self.tLast_mesh=None
        #
        self.name=name
        self.sd=sd
        self.Hess=False
        self.lowmem=True
        self.timeTerm=True#allow turning off  the  time derivative
        #self.lowmem=False
        self.testIsTrial=True
        self.phiTrialIsTrial=True
        self.u = uDict
        self.ua = {}#analytical solutions
        self.phi  = phiDict
        self.dphi={}
        self.matType = matType

        assert coefficients.nc in (1, 2), "LevelModel expects 1 or 2 components"
        if coefficients.nc == 2:
            assert 0 in self.u and 1 in self.u, "uDict must have {0:water, 1:air}"
        #mwf try to reuse test and trial information across components if spaces are the same
        self.reuse_test_trial_quadrature = reuse_trial_and_test_quadrature#True#False
        if self.reuse_test_trial_quadrature:
            for ci in range(1,coefficients.nc):
                assert self.u[ci].femSpace.__class__.__name__ == self.u[0].femSpace.__class__.__name__, "to reuse_test_trial_quad all femSpaces must be the same!"
        self.u_dof_old = None
        
        self.u_dof_old_water = None  # to be filled on first getResidual call
        self.u_dof_old_air   = None  # idem
        ## Simplicial Mesh
        self.mesh = self.u[0].femSpace.mesh #assume the same mesh for  all components for now
        self.testSpace = testSpaceDict
        self.dirichletConditions = dofBoundaryConditionsDict
        self.dirichletNodeSetList=None #explicit Dirichlet  conditions for now, no Dirichlet BC constraints
        self.coefficients = coefficients
        self.coefficients.initializeMesh(self.mesh)
        self.nc = self.coefficients.nc
        self.stabilization = stabilization
        self.shockCapturing = shockCapturing
        self.conservativeFlux = conservativeFluxDict #no velocity post-processing for now
        self.fluxBoundaryConditions=fluxBoundaryConditionsDict
        self.advectiveFluxBoundaryConditionsSetterDict=advectiveFluxBoundaryConditionsSetterDict
        self.diffusiveFluxBoundaryConditionsSetterDictDict = diffusiveFluxBoundaryConditionsSetterDictDict
        #determine whether  the stabilization term is nonlinear
        self.stabilizationIsNonlinear = False
        #anb add 
        self.anb_seepage_flux= 0.0
        self.coefficients.model = self 

        #cek come back
        if self.stabilization != None:
            for ci in range(self.nc):
                if ci in coefficients.mass:
                    for flag in list(coefficients.mass[ci].values()):
                        if flag == 'nonlinear':
                            self.stabilizationIsNonlinear=True
                if  ci in coefficients.advection:
                    for  flag  in list(coefficients.advection[ci].values()):
                        if flag == 'nonlinear':
                            self.stabilizationIsNonlinear=True
                if  ci in coefficients.diffusion:
                    for diffusionDict in list(coefficients.diffusion[ci].values()):
                        for  flag  in list(diffusionDict.values()):
                            if flag != 'constant':
                                self.stabilizationIsNonlinear=True
                if  ci in coefficients.potential:
                    for flag in list(coefficients.potential[ci].values()):
                        if  flag == 'nonlinear':
                            self.stabilizationIsNonlinear=True
                if ci in coefficients.reaction:
                    for flag in list(coefficients.reaction[ci].values()):
                        if  flag == 'nonlinear':
                            self.stabilizationIsNonlinear=True
                if ci in coefficients.hamiltonian:
                    for flag in list(coefficients.hamiltonian[ci].values()):
                        if  flag == 'nonlinear':
                            self.stabilizationIsNonlinear=True
        #determine if we need element boundary storage
        self.elementBoundaryIntegrals = {}
        for ci  in range(self.nc):
            self.elementBoundaryIntegrals[ci] = ((self.conservativeFlux != None) or 
                                                 (numericalFluxType != None) or
                                                 (self.fluxBoundaryConditions[ci] == 'outFlow') or
                                                 (self.fluxBoundaryConditions[ci] == 'mixedFlow') or
                                                 (self.fluxBoundaryConditions[ci] == 'setFlow'))
        #
        #calculate some dimensions
        #
        self.nSpace_global    = self.u[0].femSpace.nSpace_global #assume same space dim for all variables
        # self.nDOF_trial_element     = [u_j.femSpace.max_nDOF_element for  u_j in list(self.u.values())]
        # self.nDOF_phi_trial_element     = [phi_k.femSpace.max_nDOF_element for  phi_k in list(self.phi.values())]
        # self.n_phi_ip_element = [phi_k.femSpace.referenceFiniteElement.interpolationConditions.nQuadraturePoints for  phi_k in list(self.phi.values())]
        # self.nDOF_test_element     = [femSpace.max_nDOF_element for femSpace in list(self.testSpace.values())]
        # self.nFreeDOF_global  = [dc.nFreeDOF_global for dc in list(self.dirichletConditions.values())]
        
        self.nDOF_trial_element = [self.u[ci].femSpace.max_nDOF_element for ci in range(self.nc)]
        self.nDOF_phi_trial_element = [self.phi[ci].femSpace.max_nDOF_element for ci in range(self.nc)]
        self.n_phi_ip_element = [self.phi[ci].femSpace.referenceFiniteElement.interpolationConditions.nQuadraturePoints for ci in range(self.nc)]
        self.nDOF_test_element = [self.testSpace[ci].max_nDOF_element for ci in range(self.nc)]
        self.nFreeDOF_global = [self.dirichletConditions[ci].nFreeDOF_global for ci in range(self.nc)]
        
        
        
        self.nVDOF_element    = sum(self.nDOF_trial_element)
        self.nFreeVDOF_global = sum(self.nFreeDOF_global)
        #
        NonlinearEquation.__init__(self,self.nFreeVDOF_global)
        #
        #build the quadrature point dictionaries from the input (this
        #is just for convenience so that the input doesn't have to be
        #complete)
        #
        elementQuadratureDict={}
        elemQuadIsDict = isinstance(elementQuadrature,dict)
        if elemQuadIsDict: #set terms manually
            for I in self.coefficients.elementIntegralKeys:
                if I in elementQuadrature:
                    elementQuadratureDict[I] = elementQuadrature[I]
                    
                else:
                    elementQuadratureDict[I] = elementQuadrature['default']
        else:
            for I in self.coefficients.elementIntegralKeys:
                elementQuadratureDict[I] = elementQuadrature
        if self.stabilization != None:
            for I in self.coefficients.elementIntegralKeys:
                if elemQuadIsDict:
                    if I in elementQuadrature:
                        elementQuadratureDict[('stab',)+I[1:]] = elementQuadrature[I]
                    else:
                        elementQuadratureDict[('stab',)+I[1:]] = elementQuadrature['default']
                else:
                    elementQuadratureDict[('stab',)+I[1:]] = elementQuadrature
        if self.shockCapturing != None:
            for ci in self.shockCapturing.components:
                if elemQuadIsDict:
                    if ('numDiff',ci,ci) in elementQuadrature:
                        elementQuadratureDict[('numDiff',ci,ci)] = elementQuadrature[('numDiff',ci,ci)]

                    else:
                        elementQuadratureDict[('numDiff',ci,ci)] = elementQuadrature['default']
                else:
                    elementQuadratureDict[('numDiff',ci,ci)] = elementQuadrature
        if massLumping:
            for ci in list(self.coefficients.mass.keys()):
                elementQuadratureDict[('m',ci)] = Quadrature.SimplexLobattoQuadrature(self.nSpace_global,1)
            for I in self.coefficients.elementIntegralKeys:
                elementQuadratureDict[('stab',)+I[1:]] = Quadrature.SimplexLobattoQuadrature(self.nSpace_global,1)
        if reactionLumping:
            for ci in list(self.coefficients.mass.keys()):
                elementQuadratureDict[('r',ci)] = Quadrature.SimplexLobattoQuadrature(self.nSpace_global,1)
            for I in self.coefficients.elementIntegralKeys:
                elementQuadratureDict[('stab',)+I[1:]] = Quadrature.SimplexLobattoQuadrature(self.nSpace_global,1)
        elementBoundaryQuadratureDict={}
        if isinstance(elementBoundaryQuadrature,dict): #set terms manually
            for I in self.coefficients.elementBoundaryIntegralKeys:
                if I in elementBoundaryQuadrature:
                    elementBoundaryQuadratureDict[I] = elementBoundaryQuadrature[I]
                else:
                    elementBoundaryQuadratureDict[I] = elementBoundaryQuadrature['default']
        else:
            for I in self.coefficients.elementBoundaryIntegralKeys:
                elementBoundaryQuadratureDict[I] = elementBoundaryQuadrature
        #
        # find the union of all element quadrature points and
        # build a quadrature rule for each integral that has a
        # weight at each point in the union
        #mwf include tag telling me which indices are which quadrature rule?
        (self.elementQuadraturePoints,self.elementQuadratureWeights,
         self.elementQuadratureRuleIndeces) = Quadrature.buildUnion(elementQuadratureDict)
        self.nQuadraturePoints_element = self.elementQuadraturePoints.shape[0]
        self.nQuadraturePoints_global = self.nQuadraturePoints_element*self.mesh.nElements_global
        #
        #Repeat the same thing for the element boundary quadrature
        #
        (self.elementBoundaryQuadraturePoints,
         self.elementBoundaryQuadratureWeights,
         self.elementBoundaryQuadratureRuleIndeces) = Quadrature.buildUnion(elementBoundaryQuadratureDict)
        self.nElementBoundaryQuadraturePoints_elementBoundary = self.elementBoundaryQuadraturePoints.shape[0]
        self.nElementBoundaryQuadraturePoints_global = (self.mesh.nElements_global*
                                                        self.mesh.nElementBoundaries_element*
                                                        self.nElementBoundaryQuadraturePoints_elementBoundary)

        #
        #storage dictionaries
        self.scalars_element = set()
        #
        #simplified allocations for test==trial and also check if space is mixed or not
        #
        self.q={}
        self.ebq={}
        self.ebq_global={}
        self.ebqe={}
        self.phi_ip={}
        self.edge_based_cfl = np.zeros(self.u[0].dof.shape)+100
        #mesh
        self.q['x'] = np.zeros((self.mesh.nElements_global,self.nQuadraturePoints_element,3),'d')
        self.q[('dV_u', 0)] = (1.0/ self.mesh.nElements_global) * np.ones((self.mesh.nElements_global, self.nQuadraturePoints_element), 'd')
        self.ebqe['x'] = np.zeros((self.mesh.nExteriorElementBoundaries_global,self.nElementBoundaryQuadraturePoints_elementBoundary,3),'d')
        
        for ci in range(self.nc):
            self.q[('u',ci)] = np.zeros((self.mesh.nElements_global,self.nQuadraturePoints_element),'d')
            self.q[('grad(u)',ci)] = np.zeros((self.mesh.nElements_global,self.nQuadraturePoints_element,self.nSpace_global),'d')
            self.q[('grad(phi)',ci)] = self.q[('u',ci)]
            self.q[('dphi',ci,ci)] = np.zeros((self.mesh.nElements_global,self.nQuadraturePoints_element,),'d')
            self.q[('da',ci,ci,ci)] = np.zeros((self.mesh.nElements_global,self.nQuadraturePoints_element,),'d')
            self.q[('grad(u_v)',ci)] = np.zeros((self.mesh.nElements_global,self.nQuadraturePoints_element,self.nSpace_global),'d')
            self.q[('dV_u', ci)] = (1.0/ self.mesh.nElements_global) * np.ones((self.mesh.nElements_global, self.nQuadraturePoints_element), 'd')    
            self.q['velocity',ci] = np.zeros((self.mesh.nElements_global,self.nQuadraturePoints_element,self.nSpace_global),'d')
            self.q[('m',ci)] = self.q[('u',ci)].copy()
            self.q[('mt',ci)] = self.q[('u',ci)].copy()
            self.q[('m_last',ci)] = self.q[('u',ci)].copy()
            self.q[('m_tmp',ci)] = self.q[('u',ci)].copy()
            self.q[('cfl',ci)] = np.zeros((self.mesh.nElements_global,self.nQuadraturePoints_element),'d')
            self.q[('numDiff',ci,ci)] =  np.zeros((self.mesh.nElements_global,self.nQuadraturePoints_element),'d')
            self.numDiff_star = self.q[('numDiff',ci,ci)]
            self.q[('numDiff_last',ci,ci)] =  np.zeros((self.mesh.nElements_global,self.nQuadraturePoints_element),'d')
            self.ebqe[('u',ci)] = np.zeros((self.mesh.nExteriorElementBoundaries_global,self.nElementBoundaryQuadraturePoints_elementBoundary),'d')
            self.ebqe[('grad(u)',ci)] = np.zeros((self.mesh.nExteriorElementBoundaries_global,self.nElementBoundaryQuadraturePoints_elementBoundary,self.nSpace_global),'d')
            self.ebqe['velocity', ci] = np.zeros((self.mesh.nExteriorElementBoundaries_global,self.nElementBoundaryQuadraturePoints_elementBoundary,self.nSpace_global),'d')
            self.ebqe[('advectiveFlux_bc_flag',ci)] = np.zeros((self.mesh.nExteriorElementBoundaries_global,self.nElementBoundaryQuadraturePoints_elementBoundary),'i')
            self.ebqe[('advectiveFlux_bc',ci)] = np.zeros((self.mesh.nExteriorElementBoundaries_global,self.nElementBoundaryQuadraturePoints_elementBoundary),'d')
            self.ebqe[('advectiveFlux',ci)] = np.zeros((self.mesh.nExteriorElementBoundaries_global,self.nElementBoundaryQuadraturePoints_elementBoundary),'d')
            self.ebqe[('penalty')] = np.zeros((self.mesh.nExteriorElementBoundaries_global,self.nElementBoundaryQuadraturePoints_elementBoundary),'d')

        self.points_elementBoundaryQuadrature= set()
        self.scalars_elementBoundaryQuadrature= set([('u',ci) for ci in range(self.nc)])
        self.vectors_elementBoundaryQuadrature= set()
        self.tensors_elementBoundaryQuadrature= set()
        self.inflowBoundaryBC = {}
        self.inflowBoundaryBC_values = {}
        self.inflowFlux = {}
        for cj in range(self.nc):
            self.inflowBoundaryBC[cj] = np.zeros((self.mesh.nExteriorElementBoundaries_global,),'i')
            self.inflowBoundaryBC_values[cj] = np.zeros((self.mesh.nExteriorElementBoundaries_global,self.nDOF_trial_element[cj]),'d')
            self.inflowFlux[cj] = np.zeros((self.mesh.nExteriorElementBoundaries_global,self.nElementBoundaryQuadraturePoints_elementBoundary),'d')
        self.internalNodes = set(range(self.mesh.nNodes_global))
        #identify the internal nodes this is ought to be in mesh
        ##\todo move this to mesh
        for ebNE in range(self.mesh.nExteriorElementBoundaries_global):
            ebN = self.mesh.exteriorElementBoundariesArray[ebNE]
            eN_global   = self.mesh.elementBoundaryElementsArray[ebN,0]
            ebN_element  = self.mesh.elementBoundaryLocalElementBoundariesArray[ebN,0]
            for i in range(self.mesh.nNodes_element):
                if i != ebN_element:
                    I = self.mesh.elementNodesArray[eN_global,i]
                    self.internalNodes -= set([I])
        self.nNodes_internal = len(self.internalNodes)
        self.internalNodesArray=np.zeros((self.nNodes_internal,),'i')
        for nI,n in enumerate(self.internalNodes):
            self.internalNodesArray[nI]=n
        #
        del self.internalNodes
        self.internalNodes = None
        logEvent("Updating local to global mappings",2)
        self.updateLocal2Global()
        logEvent("Building time integration object",2)
        logEvent(memory("inflowBC, internalNodes,updateLocal2Global","OneLevelTransport"),level=4)
        #mwf for interpolating subgrid error for gradients etc
        if self.stabilization and self.stabilization.usesGradientStabilization:
            self.timeIntegration = TimeIntegrationClass(self,integrateInterpolationPoints=True)
        else:
             self.timeIntegration = TimeIntegrationClass(self)

        if options != None:
            self.timeIntegration.setFromOptions(options)
        logEvent(memory("TimeIntegration","OneLevelTransport"),level=4)
        logEvent("Calculating numerical quadrature formulas",2)
        self.calculateQuadrature()
        #lay out components/equations contiguously for now
        self.offset = [0]
        for ci in range(1,self.nc):
            self.offset += [self.offset[ci-1]+self.nFreeDOF_global[ci-1]]
        self.stride = [1 for ci in range(self.nc)]
        #use contiguous layout of components for parallel, requires weak DBC's
        # mql. Some ASSERTS to restrict the combination of the methods
        if self.coefficients.STABILIZATION_TYPE > 0:
            pass
            #assert self.timeIntegration.isSSP == True, "If STABILIZATION_TYPE>0, use RKEV timeIntegration within VOF model"
            #cond = 'levelNonlinearSolver' in dir(options) and (options.levelNonlinearSolver ==
            #                                                   ExplicitLumpedMassMatrixForRichards or options.levelNonlinearSolver == ExplicitConsistentMassMatrixForRichards)
            #assert cond, "If STABILIZATION_TYPE>0, use levelNonlinearSolver=ExplicitLumpedMassMatrixForRichards or ExplicitConsistentMassMatrixForRichards"
        try:
            if 'levelNonlinearSolver' in dir(options) and options.levelNonlinearSolver == ExplicitLumpedMassMatrixForRichards:
                assert self.coefficients.LUMPED_MASS_MATRIX, "If levelNonlinearSolver=ExplicitLumpedMassMatrix, use LUMPED_MASS_MATRIX=True"
        except:
            pass
        if self.coefficients.LUMPED_MASS_MATRIX == True:
            cond = self.coefficients.STABILIZATION_TYPE == 2
            assert cond, "Use lumped mass matrix just with: STABILIZATION_TYPE=2 (smoothness based stab.)"
            cond = 'levelNonlinearSolver' in dir(options) and options.levelNonlinearSolver == ExplicitLumpedMassMatrixForRichards
            assert cond, "Use levelNonlinearSolver=ExplicitLumpedMassMatrixForRichards when the mass matrix is lumped"
        
        #################################################################
        ####################ARNOB_FCT_EDIT###############################
        #################################################################
        #if not self.coefficients.LUMPED_MASS_MATRIX and self.coefficients.STABILIZATION_TYPE == 2:
        #    cond = 'levelNonlinearSolver' in dir(options) and options.levelNonlinearSolver == Newton
        
        #if self.coefficients.FCT == True:
        #    cond = self.coefficients.STABILIZATION_TYPE = 3, "Use FCT just with STABILIZATION_TYPE=3; i.e., edge based stabilization"
        
        if self.coefficients.FCT:
            valid_stabilization_types = {1, 2}  # Only allow FCT for STABILIZATION_TYPE 1 (EV_Stab) and 2 (EntropyViscosity)
            if self.coefficients.STABILIZATION_TYPE not in valid_stabilization_types:
                raise ValueError("Use FCT only with STABILIZATION_TYPE 1 (EV_Stab) or 2 (EntropyViscosity).")       
        if self.coefficients.FCT == True:
            cond = self.coefficients.STABILIZATION_TYPE > 0, "Use FCT just with STABILIZATION_TYPE>0; i.e., edge based stabilization"
        # # END OF ASSERTS

        # cek adding empty data member for low order numerical viscosity structures here for now
        # self.ML = None  # lumped mass matrix
        # self.MC_global = None  # consistent mass matrix
        self.cterm_global = None
        self.cterm_global_transpose = None
        self.ML                         = {ci: None for ci in range(self.nc)}   # lumped mass (CSR or diag) per ci
        self.MC_global                  = {ci: None for ci in range(self.nc)}   # consistent mass per ci
        self.MC_a                      = {ci: None for ci in range(self.nc)}   # consistent mass per ci
        #self.cterm_global               = {ci: None for ci in range(self.nc)}
        #self.cterm_transpose_global     = {ci: None for ci in range(self.nc)}
        # dL_global and dC_global are not the full matrices but just the CSR arrays containing the non zero entries
        self.residualComputed=False #TMP
        self.dLow                       = {ci: None for ci in range(self.nc)}
        self.fluxMatrix                 = {ci: None for ci in range(self.nc)}
        #self.mDotLow                    = {ci: None for ci in range(self.nc)}
        #self.mLow                       = {ci: None for ci in range(self.nc)}
        self.dt_times_dC_minus_dL       = {ci: None for ci in range(self.nc)}
        self.min_m_bc                   = {ci: None for ci in range(self.nc)}
        self.max_m_bc                   = {ci: None for ci in range(self.nc)}
        # Aux quantity at DOFs to be filled by optimized code (MQL)
        self.quantDOFs          = {ci: np.zeros(self.u[ci].dof.shape, 'd') for ci in range(self.nc)}
        self.mLow               = {ci: np.zeros(self.u[ci].dof.shape, 'd') for ci in range(self.nc)}
        self.mHigh              = {ci: np.zeros(self.u[ci].dof.shape, 'd') for ci in range(self.nc)}
        self.mDotLow            = {ci: np.zeros(self.u[ci].dof.shape, 'd') for ci in range(self.nc)}
        self.mDotHigh           = {ci: np.zeros(self.u[ci].dof.shape, 'd') for ci in range(self.nc)}
        self.fluxCorrection     = {ci: np.zeros(self.u[ci].dof.shape, 'd') for ci in range(self.nc)}
        self.mn                 = {ci: np.zeros(self.u[ci].dof.shape, 'd') for ci in range(self.nc)}
        self.anb_seepage_flux_n = {ci: np.zeros(self.u[ci].dof.shape, 'd') for ci in range(self.nc)}

        comm = Comm.get()
        self.comm=comm
        if comm.size() > 1:
            assert numericalFluxType != None and numericalFluxType.useWeakDirichletConditions,"You must use a numerical flux to apply weak boundary conditions for parallel runs"
            self.offset = [0]
            for ci in range(1,self.nc):
                self.offset += [ci]
            self.stride = [self.nc for ci in range(self.nc)]
        #
        logEvent(memory("stride+offset","OneLevelTransport"),level=4)
        
        if numericalFluxType != None:
            if options is None or options.periodicDirichletConditions is None:
                self.numericalFlux = numericalFluxType(self,
                                                       dofBoundaryConditionsSetterDict,
                                                       advectiveFluxBoundaryConditionsSetterDict,
                                                       diffusiveFluxBoundaryConditionsSetterDictDict)
            else:
                self.numericalFlux = numericalFluxType(self,
                                                       dofBoundaryConditionsSetterDict,
                                                       advectiveFluxBoundaryConditionsSetterDict,
                                                       diffusiveFluxBoundaryConditionsSetterDictDict,
                                                       options.periodicDirichletConditions)
        else:
            self.numericalFlux = None
        #set penalty terms
        #cek todo move into numerical flux initialization
        if 'penalty' in self.ebq_global:
            for ebN in range(self.mesh.nElementBoundaries_global):
                for k in range(self.nElementBoundaryQuadraturePoints_elementBoundary):
                    self.ebq_global['penalty'][ebN,k] = self.numericalFlux.penalty_constant/(self.mesh.elementBoundaryDiametersArray[ebN]**self.numericalFlux.penalty_power)
        #penalty term
        #cek move  to Numerical flux initialization
        if 'penalty' in self.ebqe:
            for ebNE in range(self.mesh.nExteriorElementBoundaries_global):
                ebN = self.mesh.exteriorElementBoundariesArray[ebNE]
                for k in range(self.nElementBoundaryQuadraturePoints_elementBoundary):
                    self.ebqe['penalty'][ebNE,k] = self.numericalFlux.penalty_constant/self.mesh.elementBoundaryDiametersArray[ebN]**self.numericalFlux.penalty_power
        logEvent(memory("numericalFlux","OneLevelTransport"),level=4)
        self.elementEffectiveDiametersArray  = self.mesh.elementInnerDiametersArray
        #use post processing tools to get conservative fluxes, None by default
        from proteus import PostProcessingTools
        self.velocityPostProcessor = PostProcessingTools.VelocityPostProcessingChooser(self)  
        logEvent(memory("velocity postprocessor","OneLevelTransport"),level=4)
        #helper for writing out data storage
        from proteus import Archiver
        self.elementQuadratureDictionaryWriter = Archiver.XdmfWriter()
        self.elementBoundaryQuadratureDictionaryWriter = Archiver.XdmfWriter()
        self.exteriorElementBoundaryQuadratureDictionaryWriter = Archiver.XdmfWriter()
        #TODO get rid of this
        for ci,fbcObject  in list(self.fluxBoundaryConditionsObjectsDict.items()):
            self.ebqe[('advectiveFlux_bc_flag',ci)] = np.zeros(self.ebqe[('advectiveFlux_bc',ci)].shape,'i')
            for t,g in list(fbcObject.advectiveFluxBoundaryConditionsDict.items()):
                if ci in self.coefficients.advection:
                    self.ebqe[('advectiveFlux_bc',ci)][t[0],t[1]] = g(self.ebqe[('x')][t[0],t[1]],self.timeIntegration.t)
                    self.ebqe[('advectiveFlux_bc_flag',ci)][t[0],t[1]] = 1

        if hasattr(self.numericalFlux,'setDirichletValues'):
            self.numericalFlux.setDirichletValues(self.ebqe)
        if not hasattr(self.numericalFlux,'isDOFBoundary'):
            self.numericalFlux.isDOFBoundary = {ci: np.zeros(self.ebqe[('u', ci)].shape, 'i') for ci in range(self.nc)}
            #self.numericalFlux.isDOFBoundary = {0:np.zeros(self.ebqe[('u',0)].shape,'i')}
        if not hasattr(self.numericalFlux,'ebqe'):
            self.numericalFlux.ebqe = {('u', ci): np.zeros(self.ebqe[('u', ci)].shape, 'd') for ci in range(self.nc)}
#            self.numericalFlux.ebqe = {('u',0):np.zeros(self.ebqe[('u',0)].shape,'d')}
        
        for ci in range(self.nc):
            self.numericalFlux.ebqe[('u', ci)] = self.ebqe[('u', ci)]
        #TODO how to handle redistancing calls for calculateCoefficients,calculateElementResidual etc
        self.globalResidualDummy = None
        compKernelFlag=0
        self.delta_x_ij=None
        ## Adding some assertion to ensure they are ensuring same mesh as I am using same C++ object
        reference_map  = self.u[0].femSpace.elementMaps.localFunctionSpace.dim
        reference_tri  = self.u[0].femSpace.referenceFiniteElement.localFunctionSpace.dim
        reference_test = self.testSpace[0].referenceFiniteElement.localFunctionSpace.dim
        for ci in range(1, self.nc):
            assert self.u[ci].femSpace.elementMaps.localFunctionSpace.dim == reference_map,  \
                f"mesh map DOFs differ for ci={ci}: {self.u[ci].femSpace.elementMaps.localFunctionSpace.dim} != {reference_map}"
            assert self.u[ci].femSpace.referenceFiniteElement.localFunctionSpace.dim == reference_tri, \
                f"trial DOFs differ for ci={ci}: {self.u[ci].femSpace.referenceFiniteElement.localFunctionSpace.dim} != {reference_tri}"
            assert self.testSpace[ci].referenceFiniteElement.localFunctionSpace.dim == reference_test, \
                f"test DOFs differ for ci={ci}: {self.testSpace[ci].referenceFiniteElement.localFunctionSpace.dim} != {reference_test}"
        self.mphase_co2 = cMphase_co2_base(self.nSpace_global,
                             self.nQuadraturePoints_element,
                             self.u[0].femSpace.elementMaps.localFunctionSpace.dim,
                             self.u[0].femSpace.referenceFiniteElement.localFunctionSpace.dim,
                             self.testSpace[0].referenceFiniteElement.localFunctionSpace.dim,
                             self.nElementBoundaryQuadraturePoints_elementBoundary,
                             compKernelFlag)
       
        if self.movingDomain:
            self.MOVING_DOMAIN=1.0
        else:
            self.MOVING_DOMAIN=0.0
        #cek hack
        self.movingDomain=False
        self.MOVING_DOMAIN=0.0
        if self.mesh.nodeVelocityArray is None:
            self.mesh.nodeVelocityArray = np.zeros(self.mesh.nodeArray.shape,'d')
        self.dirichletConditionsForceDOF = {}
        if self.coefficients.forceStrongConditions:
            for cj in range(self.nc):
                self.dirichletConditionsForceDOF[cj] = DOFBoundaryConditions(self.u[cj].femSpace,dofBoundaryConditionsSetterDict[cj],weakDirichletConditions=False)
    def FCTStep(self):
#        ci=0 # water
        for ci in range(self.nc):
            rowptr, colind, MassMatrix = self.MC_global[ci].getCSRrepresentation()
            limited_solution = np.zeros((len(rowptr) - 1),'d')
            argsDict = cArgumentsDict.ArgumentsDict()
            argsDict["bc_mask"] = self.bc_mask
            argsDict["NNZ"] = self.nnz 
            argsDict["numDOFs"] = len(rowptr) - 1  # num of DOFs
            argsDict["dt"] = self.timeIntegration.dt
            argsDict["ML"] = self.ML[ci]
            argsDict["mn"] = self.mn[ci]
            argsDict["mHigh"] = self.mHigh[ci]
            argsDict["mLow"] = self.mLow[ci]
            argsDict["mDotHigh"] = self.mDotHigh[ci]
            argsDict["fluxCorrection"] = self.fluxCorrection[ci]
            argsDict["mDotLow"] = self.mDotLow[ci]
            argsDict["limited_solution"] = limited_solution
            argsDict["csrRowIndeces_DofLoops"] = rowptr
            argsDict["csrColumnOffsets_DofLoops"] = colind
            argsDict["MC"] = MassMatrix
            argsDict["dt_times_fH_minus_fL"] = self.dt_times_dC_minus_dL[ci]
            argsDict["min_m_bc"] = self.min_m_bc[ci]
            argsDict["max_m_bc"] = self.max_m_bc[ci]
            argsDict["LUMPED_MASS_MATRIX"] = self.coefficients.LUMPED_MASS_MATRIX
            argsDict["MONOLITHIC"] =0#cek hack self.coefficients.MONOLITHIC
            argsDict["anb_seepage_flux_n"]= self.anb_seepage_flux_n[ci]
            argsDict["elementMaterialTypes"] = self.mesh.elementMaterialTypes,
    #        self.richards.FCTStep(argsDict)
            self.mphase_co2.FCTStep(argsDict)
            old_dof = self.u[0].dof.copy()
            self.invert(u=limited_solution, ulow=self.u[ci].dof)
            #print("FCT - low",np.linalg.norm(self.u[0].dof- old_dof))
            self.timeIntegration.u[ci][:] = self.u[ci].dof

    def kth_FCT_step(self):
        #import pdb
        #pdb.set_trace()
        rowptr, colind, MassMatrix = self.MC_global.getCSRrepresentation()        
        limitedFlux = np.zeros(self.nnz)
        limited_solution = np.zeros((len(rowptr) - 1),'d')
        #limited_solution[:] = self.timeIntegration.u_dof_stage[0][self.timeIntegration.lstage]
        fromFreeToGlobal=0 #direction copying
        cfemIntegrals.copyBetweenFreeUnknownsAndGlobalUnknowns(fromFreeToGlobal,
                                                               self.offset[0],
                                                               self.stride[0],
                                                               self.dirichletConditions[0].global2freeGlobal_global_dofs,
                                                               self.dirichletConditions[0].global2freeGlobal_free_dofs,
                                                               limited_solution,
                                                               self.timeintegration.u_dof_stage[0][self.timeIntegration.lstage])
                                                               #self.timeintegration.u_dof_stage[0][self.timeIntegration.lstage])


        #self.richards.kth_FCT_step(
        self.mphase_co2.kth_FCT_step(
            self.timeIntegration.dt,
            self.coefficients.num_fct_iter,
            self.nnz,  # number of non zero entries
            len(rowptr) - 1,  # number of DOFs
            MassMatrix,
            self.ML,  # Lumped mass matrix
            self.u_dof_old,
            limited_solution,
            self.mDotLow,
            self.mLow,
            self.dLow,
            self.fluxMatrix,
            limitedFlux,
            rowptr,
            colind)
        #import pdb
        #pdb.set_trace()

        self.timeIntegration.u[:] = limited_solution
        #self.u[0].dof[:] = limited_solution
        fromFreeToGlobal=1 #direction copying
        cfemIntegrals.copyBetweenFreeUnknownsAndGlobalUnknowns(fromFreeToGlobal,
                                                               self.offset[0],
                                                               self.stride[0],
                                                               self.dirichletConditions[0].global2freeGlobal_global_dofs,
                                                               self.dirichletConditions[0].global2freeGlobal_free_dofs,
                                                               self.u[0].dof,
                                                               limited_solution)
    def calculateCoefficients(self):
        pass
    def calculateElementResidual(self):
        if self.globalResidualDummy != None:
            for ci in range(self.nc):
                self.getResidual(self.u[ci].dof,self.globalResidualDummy)
#            self.getResidual(self.u[0].dof,self.globalResidualDummy)
    def getResidual(self,u,r):
        import pdb
        import copy
        #print("[PY] getResidual start")
        """
        Calculate the element residuals and add in to the global residual
        """
        cfemIntegrals.zeroJacobian_CSR(self.nNonzerosInJacobian,
                                    self.jacobian)
        
        if self.u_dof_old_water is None:
            # Pass initial condition to u_dof_old
            self.u_dof_old_water = np.copy(self.u[0].dof)

        if self.u_dof_old_air is None:
            # Pass initial condition to u_dof_old
            self.u_dof_old_air = np.copy(self.u[1].dof)
        rowptr, colind, nzval = self.jacobian.getCSRrepresentation()
        nnz = nzval.shape[-1]  # number of non-zero entries in sparse matrix
        r.fill(0.0)
        ########################
        ### COMPUTE C MATRIX ###
        ########################
        #for ci in range(self.nc):
        if self.cterm_global is None:
            # since we only need cterm_global to persist, we can drop the other self.'s
            self.cterm = {}
            self.cterm_a = {}
            self.cterm_global = {}
            self.cterm_transpose = {}
            self.cterm_a_transpose = {}
            self.cterm_global_transpose = {}
            rowptr, colind, nzval = self.jacobian.getCSRrepresentation()
            nnz = nzval.shape[-1]  # number of non-zero entries in sparse matrix
            #di = self.q[('grad(u)', 0)].copy()  # direction of derivative
            # JACOBIANS (FOR ELEMENT TRANSFORMATION)
            self.q[('J')] = np.zeros((self.mesh.nElements_global,
                                    self.nQuadraturePoints_element,
                                    self.nSpace_global,
                                    self.nSpace_global),
                                    'd')
            self.q[('inverse(J)')] = np.zeros((self.mesh.nElements_global,
                                            self.nQuadraturePoints_element,
                                            self.nSpace_global,
                                            self.nSpace_global),
                                            'd')
            self.q[('det(J)')] = np.zeros((self.mesh.nElements_global,
                                        self.nQuadraturePoints_element),
                                        'd')
            
            for ci in range(self.nc):
                self.u[ci].femSpace.elementMaps.getJacobianValues(self.elementQuadraturePoints,
                                                                self.q['J'],
                                                                self.q['inverse(J)'],
                                                                self.q['det(J)'])
                self.q['abs(det(J))'] = np.abs(self.q['det(J)'])
                # SHAPE FUNCTIONS
                self.q[('w', ci)] = np.zeros((self.mesh.nElements_global,
                                             self.nQuadraturePoints_element,
                                             self.nDOF_test_element[ci]),
                                            'd')
                self.q[('w*dV_m', ci)] = self.q[('w', ci)].copy()
                self.u[ci].femSpace.getBasisValues(self.elementQuadraturePoints, self.q[('w', ci)])
                cfemIntegrals.calculateWeightedShape(self.elementQuadratureWeights[('u', ci)],
                                                    self.q['abs(det(J))'],
                                                    self.q[('w', ci)],
                                                    self.q[('w*dV_m', ci)])
                # GRADIENT OF TEST FUNCTIONS
                self.q[('grad(w)', ci)] = np.zeros((self.mesh.nElements_global,
                                            self.nQuadraturePoints_element,
                                            self.nDOF_test_element[ci],
                                            self.nSpace_global),
                                            'd')
                self.u[ci].femSpace.getBasisGradientValues(self.elementQuadraturePoints,
                                                    self.q['inverse(J)'],
                                                    self.q[('grad(w)', ci)])
                self.q[('grad(w)*dV_f', ci)] = np.zeros((self.mesh.nElements_global,
                                                    self.nQuadraturePoints_element,
                                                    self.nDOF_test_element[0],
                                                    self.nSpace_global),
                                                      'd')
                cfemIntegrals.calculateWeightedShapeGradients(self.elementQuadratureWeights[('u', ci)],
                                                        self.q['abs(det(J))'],
                                                        self.q[('grad(w)', ci)],
                                                        self.q[('grad(w)*dV_f', ci)])
                ##########################
                ### LUMPED MASS MATRIX ###
                ##########################
                # assume a linear mass term
                dm = np.ones(self.q[('u', ci)].shape, 'd')
                elementMassMatrix = np.zeros((self.mesh.nElements_global,
                                            self.nDOF_test_element[ci],
                                            self.nDOF_trial_element[ci]), 'd')
                cfemIntegrals.updateMassJacobian_weak_lowmem(dm,
                                                            self.q[('w', ci)],
                                                            self.q[('w*dV_m', ci)],
                                                            elementMassMatrix)
                
                #self.MC_a[ci] = np.zeros(nnz_ci, dtype='d')
                #self.MC_a = nzval.copy() #make room for two phase
                rowptr_ci, colind_ci, _ = self.jacobian.getCSRrepresentation()
                nnz_ci = int(rowptr_ci[-1])
                self.MC_a[ci] = np.zeros(nnz_ci, dtype='d') 
                self.MC_global[ci] = SparseMat(self.nFreeVDOF_global,
                                            self.nFreeVDOF_global,
                                            nnz_ci,#make room for two phase
                                            self.MC_a[ci], #make room for two phase
                                            colind_ci, #make room for two phase
                                            rowptr_ci)#make room for two phase
                cfemIntegrals.zeroJacobian_CSR(nnz_ci, self.MC_global[ci])
                
                cfemIntegrals.updateGlobalJacobianFromElementJacobian_CSR(self.l2g[ci]['nFreeDOF'],
                                                                        self.l2g[ci]['freeLocal'],
                                                                        self.l2g[ci]['nFreeDOF'],
                                                                        self.l2g[ci]['freeLocal'],
                                                                        self.csrRowIndeces[(ci, ci)],
                                                                        self.csrColumnOffsets[(ci, ci)],
                                                                        elementMassMatrix,
                                                                        self.MC_global[ci])

                # E = self.mesh.nElements_global
                # nt = self.nDOF_test_element[0]
                # nr = self.nDOF_trial_element[0]
                # assert self.l2g[ci]['freeLocal'].size == E*nt
                # assert self.csrRowIndeces[(ci,ci)].size == E*nt
                # assert self.csrColumnOffsets[(ci,ci)].size == E*nt*nr
                # assert elementMassMatrix.size == E*nt*nr

                
                
                rowptr_M, _, MCvals = self.MC_global[0].getCSRrepresentation()
                self.ML[ci] = np.zeros((self.nFreeDOF_global[ci],), 'd')
                # for k, irow in enumerate(grows_ci):
                #     self.ML[ci][k] = MCvals[rowptr_M[irow]:rowptr_M[irow+1]].sum()

                for i in range(self.nFreeDOF_global[ci]):
                    self.ML[ci][i] = MCvals[rowptr_M[i]:rowptr_M[i + 1]].sum()

                from mpi4py import MPI
                comm = MPI.COMM_WORLD

                # if comm.size >1:    
                #     local_trace = float(self.ML[ci].sum())
                #     global_trace = comm.allreduce(local_trace, op=MPI.SUM)
                #     # in parallel, the trace of the mass matrix is the sum of the local traces
                #     if comm.rank ==0:
                #         np.testing.assert_almost_equal(global_trace,
                #                                       self.mesh.volume,
                #                                       err_msg=f"Trace of lumped mass matrix should be the domain volume, ci={ci}", verbose=True)
                # else:
                #     np.testing.assert_almost_equal(self.ML[ci].sum(),
                #                             self.mesh.volume,
                #                             err_msg=f"Trace of lumped mass matrix should be the domain volume, ci={ci}", verbose=True)

                if ci not in self.cterm:
                    self.cterm[ci] = {}
                    self.cterm_a[ci] = {}
                    self.cterm_global[ci] = {}
                    self.cterm_transpose[ci] = {}
                    self.cterm_a_transpose[ci] = {}
                    self.cterm_global_transpose[ci] = {} 

                for d in range(self.nSpace_global):  # spatial dimensions
                    # C matrices
                    self.cterm[ci][d] = np.zeros((self.mesh.nElements_global,
                                            self.nDOF_test_element[ci],
                                            self.nDOF_trial_element[ci]), 'd')
                    self.cterm_a[ci][d] = np.zeros(nnz_ci, dtype="d") #nzval.copy()
                    #self.cterm_a[d] = np.zeros(nzval.size)
                    self.cterm_global[ci][d] = SparseMat(self.nFreeDOF_global[ci],
                                                    self.nFreeDOF_global[ci],
                                                    nnz_ci,
                                                    self.cterm_a[ci][d],
                                                    colind_ci, #make room for two phase
                                                    rowptr_ci) #make room for two phase
                    cfemIntegrals.zeroJacobian_CSR(nnz_ci, self.cterm_global[ci][d])
                    di = self.q[('grad(u)', ci)].copy()  # direction of derivative
                    di[:] = 0.0
                    di[..., d] = 1.0
                    cfemIntegrals.updateHamiltonianJacobian_weak_lowmem(di,
                                                                        self.q[('grad(w)*dV_f', ci)],
                                                                        self.q[('w', ci)],
                                                                        self.cterm[ci][d])  # int[(di*grad(wj))*wi*dV]
                    cfemIntegrals.updateGlobalJacobianFromElementJacobian_CSR(self.l2g[ci]['nFreeDOF'],
                                                                            self.l2g[ci]['freeLocal'],
                                                                            self.l2g[ci]['nFreeDOF'],
                                                                            self.l2g[ci]['freeLocal'],
                                                                            self.csrRowIndeces[(ci, ci)],
                                                                            self.csrColumnOffsets[(ci, ci)],
                                                                            self.cterm[ci][d],
                                                                            self.cterm_global[ci][d])
                    # C Transpose matrices
                    self.cterm_transpose[ci][d] = np.zeros((self.mesh.nElements_global,
                                                        self.nDOF_test_element[ci],
                                                        self.nDOF_trial_element[ci]), 'd')
                    self.cterm_a_transpose[ci][d] = np.zeros(nnz_ci, dtype="d") # nzval.copy()
                    self.cterm_global_transpose[ci][d] = SparseMat(self.nFreeVDOF_global, #self.nFreeDOF_global[ci],
                                                                    self.nFreeVDOF_global, #self.nFreeDOF_global[ci]
                                                                    nnz_ci,
                                                                    self.cterm_a_transpose[ci][d],
                                                                    colind_ci,
                                                                    rowptr_ci)
                    cfemIntegrals.zeroJacobian_CSR(nnz_ci, self.cterm_global_transpose[ci][d]) #self.nnz>> nnz
                    di[:] = 0.0
                    di[..., d] = -1.0
                    cfemIntegrals.updateAdvectionJacobian_weak_lowmem(di,
                                                                    self.q[('w', ci)],
                                                                    self.q[('grad(w)*dV_f', ci)],
                                                                    self.cterm_transpose[ci][d])  # -int[(-di*grad(wi))*wj*dV]
                    cfemIntegrals.updateGlobalJacobianFromElementJacobian_CSR(self.l2g[ci]['nFreeDOF'],
                                                                              self.l2g[ci]['freeLocal'],
                                                                              self.l2g[ci]['nFreeDOF'],
                                                                              self.l2g[ci]['freeLocal'],
                                                                              self.csrRowIndeces[(ci, ci)],
                                                                              self.csrColumnOffsets[(ci, ci)],
                                                                              self.cterm_transpose[ci][d],
                                                                              self.cterm_global_transpose[ci][d])


                rowptr_ci, colind_ci, Cx = self.cterm_global[ci][0].getCSRrepresentation()
                if (self.nSpace_global == 2):
#                    rowptr, colind, Cy = self.cterm_global[1].getCSRrepresentation()
                    _, _, Cy = self.cterm_global[ci][1].getCSRrepresentation()
                else:
                    Cy = np.zeros(Cx.shape, 'd')
                if (self.nSpace_global == 3):
                    #rowptr, colind, Cz = self.cterm_global[2].getCSRrepresentation()
                    _, _, Cz = self.cterm_global[ci][2].getCSRrepresentation()
                else:
                    Cz = np.zeros(Cx.shape, 'd')
                _, _, CTx = self.cterm_global_transpose[ci][0].getCSRrepresentation()
                if (self.nSpace_global == 2):
                    _, _, CTy = self.cterm_global_transpose[ci][1].getCSRrepresentation()
                else:
                    CTy = np.zeros(CTx.shape, 'd')
                if (self.nSpace_global == 3):
                    _, _, CTz = self.cterm_global_transpose[ci][2].getCSRrepresentation()
                else:
                    CTz = np.zeros(CTx.shape, 'd')

                # nnz-length arrays (match the CSR 'data' for this phase)
                self.dLow[ci]                 = np.zeros_like(Cx)                 # low-order limiter diag
                self.fluxMatrix[ci]           = np.zeros_like(Cx)                 # flux matrix (per ci)
                self.dt_times_dC_minus_dL[ci] = np.zeros_like(Cx)                 # dt*(dC) - dL cache

                # row count = number of free DOFs for this phase
                nFree = len(rowptr_ci) - 1
                self.min_m_bc[ci] = np.full(nFree,  1.0e10, dtype='d')            # large +inf sentinel
                self.max_m_bc[ci] = np.full(nFree, -1.0e10, dtype='d')            # large -inf sentinel

            # This is dummy. I just care about the csr structure of the sparse matrix
            # self.dLow = np.zeros(Cx.shape, 'd')
            # self.fluxMatrix = np.zeros(Cx.shape, 'd')
            # self.dt_times_dC_minus_dL = np.zeros(Cx.shape, 'd')
            # nFree = len(rowptr)-1
            # self.min_m_bc = np.ones(nFree, 'd')
            # self.min_m_bc *= 1.0e10
            # self.max_m_bc = np.ones(nFree, 'd')
            # self.max_m_bc *= -1.0e10


            #
            # cek end computationa of cterm_global
            #
            # cek showing mquezada an example of using cterm_global sparse matrix
            # calculation y = c*x where x==1
            # direction=0
            #rowptr, colind, c = self.cterm_global[direction].getCSRrepresentation()
            #y = np.zeros((self.nFreeDOF_global[0],),'d')
            #x = np.ones((self.nFreeDOF_global[0],),'d')
            # ij=0
            # for i in range(self.nFreeDOF_global[0]):
            #    for offset in range(rowptr[i],rowptr[i+1]):
            #        j = colind[offset]
            #        y[i] += c[ij]*x[j]
            #        ij+=1
            #Load the unknowns into the finite element dof
        self.timeIntegration.calculateCoefs()
        self.timeIntegration.calculateU(u)
        self.setUnknowns(self.timeIntegration.u)
        #cek can put in logic to skip of BC's don't depend on t or u
        #Dirichlet boundary conditions
        
        self.numericalFlux.setDirichletValues(self.ebqe)
        #flux boundary conditions
        #cek hack, just using advective flux for flux BC for now
        for ci in range(self.nc):
            for t,g in list(self.fluxBoundaryConditionsObjectsDict[ci].advectiveFluxBoundaryConditionsDict.items()):
                self.ebqe[('advectiveFlux_bc',ci)][t[0],t[1]] = g(self.ebqe[('x')][t[0],t[1]],self.timeIntegration.t)
                self.ebqe[('advectiveFlux_bc_flag',ci)][t[0],t[1]] = 1
            # for t,g in self.fluxBoundaryConditionsObjectsDict[0].diffusiveFluxBoundaryConditionsDict.iteritems():
            #     self.ebqe[('diffusiveFlux_bc',0)][t[0],t[1]] = g(self.ebqe[('x')][t[0],t[1]],self.timeIntegration.t)
            #     self.ebqe[('diffusiveFlux_bc_flag',0)][t[0],t[1]] = 1
            #self.shockCapturing.lag=True
        if not hasattr(self, 'bc_mask'):
            self.bc_mask = {}
        for ci in range(self.nc):
            self.bc_mask[ci] = np.ones_like(self.u[ci].dof)
        #    self.bc_mask = np.ones_like(self.u[0].dof)    
        
        if getattr(self.coefficients, 'forceStrongConditions', False):
    # maintain per-phase old DOFs
            if not hasattr(self, 'u_dof_old_ci'):
                self.u_dof_old_ci = {ci: np.copy(self.u[ci].dof) for ci in range(self.nc)}
            for ci in range(self.nc):
                if self.coefficients.forceStrongConditions:
                    self.bc_mask[ci] = np.ones_like(self.u[ci].dof)
                    for cj in range(len(self.dirichletConditionsForceDOF)):
                        for dofN,g in list(self.dirichletConditionsForceDOF[cj].DOFBoundaryConditionsDict.items()):
                            self.u[cj].dof[dofN] = g(self.dirichletConditionsForceDOF[cj].DOFBoundaryPointDict[dofN],self.timeIntegration.t)
                            self.u_dof_old_ci[ci][dofN] = self.u[cj].dof[dofN]
                            self.bc_mask[ci][dofN] = 0.0
        degree_polynomial = 1
        try:
            degree_polynomial = max(getattr(self.u[ci].femSpace, 'order', 1) for ci in range(self.nc))
            #degree_polynomial = self.u[0].femSpace.order
        except:
            pass


        for ci in range(self.nc):
            # Re-extract the per-phase C/CT *values* and CSR structure (if not already kept above)
            rowptr_ci, colind_ci, Cx = self.cterm_global[ci][0].getCSRrepresentation()              # 2PH CHANGE
            if (self.nSpace_global == 2):
                _, _, Cy = self.cterm_global[ci][1].getCSRrepresentation()                    # 2PH CHANGE
            else:
                Cy = np.zeros(Cx.shape, 'd')
            if (self.nSpace_global == 3):
                _, _, Cz = self.cterm_global[ci][2].getCSRrepresentation()                    # 2PH CHANGE
            else:
                Cz = np.zeros(Cx.shape, 'd')
            _, _, CTx = self.cterm_global_transpose[ci][0].getCSRrepresentation()             # 2PH CHANGE
            if (self.nSpace_global == 2):
                _, _, CTy = self.cterm_global_transpose[ci][1].getCSRrepresentation()         # 2PH CHANGE
            else:
                CTy = np.zeros(CTx.shape, 'd')
            if (self.nSpace_global == 3):
                _, _, CTz = self.cterm_global_transpose[ci][2].getCSRrepresentation()         # 2PH CHANGE
            else:
                CTz = np.zeros(CTx.shape, 'd')

            argsDict = cArgumentsDict.ArgumentsDict()
            argsDict["dt"] = self.timeIntegration.dt
            argsDict["Theta"] = 1.0
            argsDict["mesh_trial_ref"] = self.u[0].femSpace.elementMaps.psi
            argsDict["mesh_dof"] = self.mesh.nodeArray
            argsDict["mesh_velocity_dof"] = self.mesh.nodeVelocityArray
            argsDict["MOVING_DOMAIN"] = self.MOVING_DOMAIN
            argsDict["mesh_l2g"] = self.mesh.elementNodesArray
            ######################################################
            argsDict["rho_water"] = self.coefficients.rho_water
            argsDict["beta_water"] = self.coefficients.beta_water

            argsDict["rho_air"] = self.coefficients.rho_air
            argsDict["beta_air"] = self.coefficients.beta_air

            #####################################################
            argsDict["gravity"] = self.coefficients.gravity
            argsDict["alpha"] = self.coefficients.vgm_alpha_types
            argsDict["n"] = self.coefficients.vgm_n_types
            argsDict["thetaR"] = self.coefficients.thetaR_types
            argsDict["thetaSR"] = self.coefficients.thetaSR_types
            argsDict["KWs"] = self.coefficients.Ksw_types
            argsDict["useMetrics"] = 0.0
            argsDict["alphaBDF"] = self.timeIntegration.alpha_bdf
            argsDict["lag_shockCapturing"] = 0
            argsDict["shockCapturingDiffusion"] = self.coefficients.SC
            argsDict["VMS"] = self.coefficients.VMS
            argsDict["sc_uref"] = 1.0
            argsDict["sc_alpha"] = 2.0
            argsDict["nElements_global"] = self.mesh.nElements_global
            argsDict["ebqe_penalty_ext"] = self.ebqe['penalty']
            argsDict["elementMaterialTypes"] = self.mesh.elementMaterialTypes
            argsDict["isSeepageFace"] = self.coefficients.isSeepageFace
            argsDict["elementDiameter"] = self.mesh.elementDiametersArray
            argsDict["degree_polynomial"] = degree_polynomial
            argsDict["bc_mask"] = self.bc_mask[ci]
            argsDict["dV_ref"] = self.elementQuadratureWeights[('u',ci)] #stab[0,2]
            argsDict["u_trial_ref"] = self.u[0].femSpace.psi
            argsDict["u_grad_trial_ref"] = self.u[0].femSpace.grad_psi
            argsDict["u_test_ref"] = self.u[0].femSpace.psi
            argsDict["u_grad_test_ref"] = self.u[0].femSpace.grad_psi
            argsDict["mesh_grad_trial_ref"] = self.u[0].femSpace.elementMaps.grad_psi
            argsDict["mesh_trial_trace_ref"] = self.u[0].femSpace.elementMaps.psi_trace
            argsDict["mesh_grad_trial_trace_ref"] = self.u[0].femSpace.elementMaps.grad_psi_trace
            argsDict["dS_ref"] = self.elementBoundaryQuadratureWeights[('u',ci)] #stab[0,2]
            argsDict["u_trial_trace_ref"] = self.u[0].femSpace.psi_trace
            argsDict["u_grad_trial_trace_ref"] = self.u[0].femSpace.grad_psi_trace
            argsDict["u_test_trace_ref"] = self.u[0].femSpace.psi_trace
            argsDict["u_grad_test_trace_ref"] = self.u[0].femSpace.grad_psi_trace
            argsDict["normal_ref"] = self.u[0].femSpace.elementMaps.boundaryNormals
            argsDict["boundaryJac_ref"] = self.u[0].femSpace.elementMaps.boundaryJacobians            
            
            argsDict["a_rowptr"] = self.coefficients.sdInfo[(0,0)][0]
            argsDict["a_colind"] = self.coefficients.sdInfo[(0,0)][1]

            #argsDict["a_rowptr"] = self.coefficients.sdInfo[(ci,ci)][0]
            #argsDict["a_colind"] = self.coefficients.sdInfo[(ci,ci)][1]

            argsDict["u_l2g"] = self.u[ci].femSpace.dofMap.l2g #okay
            argsDict["r_l2g"] = self.l2g[ci]['freeGlobal']

            argsDict["u_l2g_water"] = self.u[0].femSpace.dofMap.l2g #okay
            argsDict["r_l2g_water"] = self.l2g[0]['freeGlobal']

            argsDict["u_l2g_air"] = self.u[1].femSpace.dofMap.l2g #okay
            argsDict["r_l2g_air"] = self.l2g[1]['freeGlobal']
            assert np.array_equal(self.u[0].femSpace.dofMap.l2g,
                      self.u[1].femSpace.dofMap.l2g), "u_l2g differs between phases"
            #assert np.array_equal(self.l2g[0]['freeGlobal'], self.l2g[1]['freeGlobal']), "r_l2g differs between phases"
            #######################################################
            argsDict["u_dof_water"] = self.u[0].dof
#            argsDict["u_dof_old_water"] = self.u_dof_old_ci[0]
            argsDict["u_dof_old_water"] = self.u_dof_old_water   
            argsDict["u_dof_air"] = self.u[1].dof
            argsDict["u_dof_old_air"] = self.u_dof_old_air
            argsDict["u_dof"] = self.u[ci].dof
            if ci==0:
                argsDict["u_dof_old"] = self.u_dof_old_water
            if ci==1:
                argsDict["u_dof_old"] = self.u_dof_old_air

            argsDict["phase"]= ci
            ###########################################################
            argsDict["velocity"] = self.q['velocity',ci]
            argsDict["q_m"] = self.timeIntegration.m_tmp[ci]
            argsDict["q_u"] = self.q[('u',ci)]
            argsDict["q_dV"] = self.q[('dV_u',ci)]
            argsDict["q_m_betaBDF"] = self.timeIntegration.beta_bdf[ci]
            argsDict["cfl"] = self.q[('cfl',ci)]
            argsDict["q_numDiff_u"] = self.q[('numDiff',ci,ci)]
            #argsDict["q_numDiff_u_last"] = self.q[('numDiff_last',0,0)]
#            argsDict["q_numDiff_u_last"] = self.numDiff_star
            argsDict["q_numDiff_u_last"] = self.q[('numDiff_last', ci, ci)]
            argsDict["offset_u"] = self.offset[ci]
            argsDict["stride_u"] = self.stride[ci]
            argsDict["ebqe_velocity_ext"] = self.ebqe['velocity',ci]
            
            argsDict["isDOFBoundary_u"] = self.numericalFlux.isDOFBoundary[ci]

            argsDict["isDOFBoundary_u_water"] = self.numericalFlux.isDOFBoundary[0]
            argsDict["isDOFBoundary_u_air"] = self.numericalFlux.isDOFBoundary[1]


            argsDict["ebqe_bc_u_ext"] = self.numericalFlux.ebqe[('u',ci)]
            argsDict["ebqe_bc_u_ext_water"] = self.numericalFlux.ebqe[('u',0)]
            argsDict["ebqe_bc_u_ext_air"] = self.numericalFlux.ebqe[('u',1)]


            argsDict["isFluxBoundary_u"] = self.ebqe[('advectiveFlux_bc_flag',ci)]
            argsDict["ebqe_bc_flux_ext"] = self.ebqe[('advectiveFlux_bc',ci)]
            argsDict["ebqe_phi"] = self.ebqe[('u',ci)]
            argsDict["epsFact"] = 0.0
            argsDict["ebqe_u"] = self.ebqe[('u',ci)]
            argsDict["ebqe_flux"] = self.ebqe[('advectiveFlux',ci)]

            argsDict["globalResidual"] = r
            argsDict["nExteriorElementBoundaries_global"] = self.mesh.nExteriorElementBoundaries_global
            argsDict["exteriorElementBoundariesArray"] = self.mesh.exteriorElementBoundariesArray
            argsDict["elementBoundaryElementsArray"] = self.mesh.elementBoundaryElementsArray
            argsDict["elementBoundaryLocalElementBoundariesArray"] = self.mesh.elementBoundaryLocalElementBoundariesArray

        
            argsDict['STABILIZATION_TYPE'] = self.coefficients.STABILIZATION_TYPE
            argsDict['PSK_TYPE'] = self.coefficients.PSK_TYPE
            
            # ENTROPY VISCOSITY and ARTIFICIAL COMRPESSION
            argsDict["cE"] = self.coefficients.cE
            argsDict["cK"] = self.coefficients.cK
            # PARAMETERS FOR LOG BASED ENTROPY FUNCTION
            argsDict["uL"] = self.coefficients.uL
            argsDict["uR"] = self.coefficients.uR
            # PARAMETERS FOR EDGE VISCOSITY
            argsDict["numDOFs"] = len(rowptr_ci) - 1  # num of DOFs
            argsDict["NNZ"] = Cx.size #self.nnz 
            #argsDict["Cx"] = len(Cx)  # num of non-zero entries in the sparsity pattern
            argsDict["csrRowIndeces_DofLoops"] = rowptr_ci  # Row indices for Sparsity Pattern (convenient for DOF loops)
            argsDict["csrColumnOffsets_DofLoops"] = colind_ci  # Column indices for Sparsity Pattern (convenient for DOF loops)
            argsDict["csrRowIndeces_CellLoops"] = self.csrRowIndeces[(ci, ci)]  # row indices (convenient for element loops)
            argsDict["csrColumnOffsets_CellLoops"] = self.csrColumnOffsets[(ci, ci)]  # column indices (convenient for element loops)
            argsDict["csrColumnOffsets_eb_CellLoops"] = self.csrColumnOffsets_eb[(ci, ci)]  # indices for boundary terms
            argsDict["globalJacobian"] = self.jacobian.getCSRrepresentation()[2]
            # C matrices
            argsDict["Cx"] = Cx
            argsDict["Cy"] = Cy
            argsDict["Cz"] = Cz
            argsDict["CTx"] = CTx
            argsDict["CTy"] = CTy
            argsDict["CTz"] = CTz
            argsDict["ML"] = self.ML[ci]
            argsDict["delta_x_ij"] = self.delta_x_ij
            
            # PARAMETERS FOR 1st or 2nd ORDER MPP METHOD
            argsDict["LUMPED_MASS_MATRIX"] = self.coefficients.LUMPED_MASS_MATRIX
            argsDict["STABILIZATTION_TYPE"] = self.coefficients.STABILIZATION_TYPE
            argsDict["PSK_TYPE"] = self.coefficients.PSK_TYPE
            
            argsDict["ENTROPY_TYPE"] = self.coefficients.ENTROPY_TYPE
            # FLUX CORRECTED TRANSPORT
            argsDict["dLow"] = self.dLow[ci]
            argsDict["fluxMatrix"] = self.fluxMatrix[ci]
            argsDict["mDotLow"] = self.mDotLow[ci]
            argsDict["mDotHigh"] = self.mDotHigh[ci]
            argsDict["fluxCorrection"] = self.fluxCorrection[ci]
            limited_solution = np.zeros((len(rowptr_ci) - 1),'d')
            argsDict["limited_solution"] = limited_solution
            argsDict["MONOLITHIC"] =0
            argsDict["mLow"] = self.mLow[ci]
            argsDict["dt_times_fH_minus_fL"] = self.dt_times_dC_minus_dL[ci]
            argsDict["min_m_bc"] = self.min_m_bc[ci]
            argsDict["max_m_bc"] = self.max_m_bc[ci]
            argsDict["quantDOFs"] = self.quantDOFs[ci]
            argsDict["mn"] = self.mn[ci]
            argsDict["anb_seepage_flux_n"]= self.anb_seepage_flux_n[ci]
    ######################################################################################
            argsDict["pn"] = self.u[ci].dof
            argsDict["mHigh"] = self.mHigh[ci]

            rowptr_ci2, colind_ci2, MassMatrix = self.MC_global[ci].getCSRrepresentation()
            argsDict["MassMatrix"] = MassMatrix
            argsDict["MC"] = MassMatrix
            
            #argsDict["MC"] = self.MC_a
            
    ######################################################################################        
            #argsDict["anb_seepage_flux"] = self.coefficients.anb_seepage_flux
            argsDict["anb_seepage_flux"] = self.anb_seepage_flux
            argsDict["q_velocity"] = self.q[('grad(u_v)', ci)]
            argsDict["csrRowIndeces_u_u"] = self.csrRowIndeces[(ci,ci)]
            argsDict["csrColumnOffsets_u_u"] = self.csrColumnOffsets[(ci,ci)]
            argsDict["csrColumnOffsets_eb_u_u"] = self.csrColumnOffsets_eb[(ci,ci)]
            #argsDict["q_grad_psi"] = self.q[('velocity', 0)]
            
            
            from mpi4py import MPI
            comm = MPI.COMM_WORLD
            rank = comm.Get_rank()

            seepage_flux_value = np.sum(self.anb_seepage_flux_n[ci])
            if seepage_flux_value > 0.0:
            # Each processor writes its own flux with its rank
                with open("seepage_flux_try.txt", "a") as f:
                    f.write(f"Rank {rank}:, {self.timeIntegration.t:.6f}, {seepage_flux_value:.8f}\n")
            if (self.coefficients.STABILIZATION_TYPE == 0):  # SUPG
                self.calculateResidual = self.mphase_co2.calculateResidual
                self.calculateJacobian = self.mphase_co2.calculateJacobian
            else:
                self.calculateResidual = self.mphase_co2.calculateResidual_entropy_viscosity
                self.calculateJacobian = self.mphase_co2.calculateMassMatrix
            
            if self.delta_x_ij is None:
                #self.delta_x_ij = -np.ones((self.nNonzerosInJacobian*3,),'d')
                self.delta_x_ij = -np.ones((Cx.shape[0] * 3,), 'd')
            self.calculateResidual(argsDict)
            


        #self.q[('mt',0)][:] =self.timeIntegration.m_tmp[0]
        #self.q[('mt',0)] *= self.timeIntegration.alpha_bdf
        #self.q[('mt',0)] += self.timeIntegration.beta_bdf[0]
        #self.timeIntegration.calculateElementCoefficients(self.q)
        if self.coefficients.forceStrongConditions:#
            for cj in range(len(self.dirichletConditionsForceDOF)):#
                for dofN,g in list(self.dirichletConditionsForceDOF[cj].DOFBoundaryConditionsDict.items()):
                    r[self.offset[cj]+self.stride[cj]*dofN] = 0
        if self.stabilization:
            self.stabilization.accumulateSubgridMassHistory(self.q)
        logEvent("Global residual",level=9,data=r)
        self.nonlinear_function_evaluations += 1
        if self.globalResidualDummy is None:
            self.globalResidualDummy = np.zeros(r.shape,'d')

    def invert(self, u, r=None, ulow=None):
        """
        Unified invert method that handles both standard and FCT modes
        using self.coefficients.FCT as the switching flag.
        """
        import numpy as np
        import copy
        for ci in range(self.nc):
            self.mHigh[ci][:] = u[ci].dof   
            rowptr, colind, nzval = self.jacobian.getCSRrepresentation()
            nnz = nzval.shape[-1]
        
            rowptr, colind, Cx = self.cterm_global[ci][0].getCSRrepresentation()
            Cy = self.cterm_global[ci][1].getCSRrepresentation()[2] if self.nSpace_global >= 2 else np.zeros(Cx.shape, 'd')
            Cz = self.cterm_global[ci][2].getCSRrepresentation()[2] if self.nSpace_global == 3 else np.zeros(Cx.shape, 'd')
        
            rowptr, colind, CTx = self.cterm_global_transpose[ci][0].getCSRrepresentation()
            CTy = self.cterm_global_transpose[ci][1].getCSRrepresentation()[2] if self.nSpace_global >= 2 else np.zeros(CTx.shape, 'd')
            CTz = self.cterm_global_transpose[ci][2].getCSRrepresentation()[2] if self.nSpace_global == 3 else np.zeros(CTx.shape, 'd')
        
            degree_polynomial = getattr(self.u[ci].femSpace, "order", 1)
        
            if self.delta_x_ij is None:
                self.delta_x_ij = -np.ones((self.nNonzerosInJacobian * 3,), 'd')
        
            argsDict = cArgumentsDict.ArgumentsDict()
            argsDict["dt"] = self.timeIntegration.dt
            argsDict["mesh_trial_ref"] = self.u[ci].femSpace.elementMaps.psi
            argsDict["mesh_grad_trial_ref"] = self.u[ci].femSpace.elementMaps.grad_psi
            argsDict["mesh_dof"] = self.mesh.nodeArray
            argsDict["mesh_velocity_dof"] = self.mesh.nodeVelocityArray
            argsDict["MOVING_DOMAIN"] = self.MOVING_DOMAIN
            argsDict["mesh_l2g"] = self.mesh.elementNodesArray
            argsDict["dV_ref"] = self.elementQuadratureWeights[('u',ci)]
            argsDict["u_trial_ref"] = self.u[ci].femSpace.psi
            argsDict["u_grad_trial_ref"] = self.u[ci].femSpace.grad_psi
            argsDict["u_test_ref"] = self.u[ci].femSpace.psi
            argsDict["u_grad_test_ref"] = self.u[ci].femSpace.grad_psi
            argsDict["mesh_trial_trace_ref"] = self.u[ci].femSpace.elementMaps.psi_trace
            argsDict["mesh_grad_trial_trace_ref"] = self.u[ci].femSpace.elementMaps.grad_psi_trace
            argsDict["dS_ref"] = self.elementBoundaryQuadratureWeights[('u',ci)]
            argsDict["u_trial_trace_ref"] = self.u[ci].femSpace.psi_trace
            argsDict["u_grad_trial_trace_ref"] = self.u[ci].femSpace.grad_psi_trace
            argsDict["u_test_trace_ref"] = self.u[ci].femSpace.psi_trace
            argsDict["u_grad_test_trace_ref"] = self.u[ci].femSpace.grad_psi_trace
            argsDict["normal_ref"] = self.u[ci].femSpace.elementMaps.boundaryNormals
            argsDict["boundaryJac_ref"] = self.u[ci].femSpace.elementMaps.boundaryJacobians
            argsDict["nElements_global"] = self.mesh.nElements_global
            argsDict["ebqe_penalty_ext"] = self.ebqe['penalty']
            argsDict["elementMaterialTypes"] = self.mesh.elementMaterialTypes
            argsDict["isSeepageFace"] = self.coefficients.isSeepageFace
            argsDict["a_rowptr"] = self.coefficients.sdInfo[(0,0)][0]
            argsDict["a_colind"] = self.coefficients.sdInfo[(0,0)][1]
            
            argsDict["rho_water"] = self.coefficients.rho_water
            argsDict["beta_water"] = self.coefficients.beta_water
            argsDict["rho_air"] = self.coefficients.rho_air
            argsDict["beta_air"] = self.coefficients.beta_air
            

            argsDict["gravity"] = self.coefficients.gravity
            argsDict["alpha"] = self.coefficients.vgm_alpha_types
            argsDict["n"] = self.coefficients.vgm_n_types
            argsDict["thetaR"] = self.coefficients.thetaR_types
            argsDict["thetaSR"] = self.coefficients.thetaSR_types
            argsDict["KWs"] = self.coefficients.Ksw_types
            argsDict["useMetrics"] = 0.0
            argsDict["alphaBDF"] = self.timeIntegration.alpha_bdf
            argsDict["lag_shockCapturing"] = 0
            argsDict["shockCapturingDiffusion"] = self.coefficients.SC
            argsDict["sc_uref"] = 1.0
            argsDict["sc_alpha"] = 2.0
            argsDict["u_l2g"] = self.u[0].femSpace.dofMap.l2g
            argsDict["r_l2g"] = self.l2g[0]['freeGlobal']
            argsDict["elementDiameter"] = self.mesh.elementDiametersArray
            argsDict["degree_polynomial"] = degree_polynomial
#            argsDict["u_dof"] = self.u[ci].dof
#            argsDict["u_dof_old"] = self.u[ci].dof

            argsDict["u_dof_water"] = self.u[0].dof
            argsDict["u_dof_old_water"] = self.u_dof_old_ci[0]    
            argsDict["u_dof_air"] = self.u[1].dof
            argsDict["u_dof_old_air"] = self.u_dof_old_ci[1]

            argsDict["velocity"] = self.q['velocity', ci]
            argsDict["q_m"] = self.timeIntegration.m_tmp[ci]
            argsDict["q_u"] = self.q[('u',ci)]
            argsDict["q_dV"] = self.q[('dV_u',ci)]
            argsDict["q_m_betaBDF"] = self.timeIntegration.beta_bdf[ci]
            argsDict["cfl"] = self.q[('cfl',ci)]
            argsDict["q_numDiff_u"] = self.q[('numDiff',ci,ci)]
            argsDict["q_numDiff_u_last"] = self.q[('numDiff_last',ci,ci)]
            argsDict["q_numDiff_u_last"] = self.numDiff_star
            argsDict["offset_u"] = self.offset[ci]
            argsDict["stride_u"] = self.stride[ci]
            argsDict["nExteriorElementBoundaries_global"] = self.mesh.nExteriorElementBoundaries_global
            argsDict["exteriorElementBoundariesArray"] = self.mesh.exteriorElementBoundariesArray
            argsDict["elementBoundaryElementsArray"] = self.mesh.elementBoundaryElementsArray
            argsDict["elementBoundaryLocalElementBoundariesArray"] = self.mesh.elementBoundaryLocalElementBoundariesArray
            argsDict["ebqe_velocity_ext"] = self.ebqe['velocity',ci]
            argsDict["isDOFBoundary_u"] = self.numericalFlux.isDOFBoundary[ci]
            argsDict["ebqe_bc_u_ext"] = self.numericalFlux.ebqe[('u',ci)]
            argsDict["isFluxBoundary_u"] = self.ebqe[('advectiveFlux_bc_flag',ci)]
            argsDict["ebqe_bc_flux_ext"] = self.ebqe[('advectiveFlux_bc',ci)]
            argsDict["ebqe_phi"] = self.ebqe[('u',ci)]
            argsDict["epsFact"] = 0.0
            argsDict["ebqe_u"] = self.ebqe[('u',ci)]
            argsDict["ebqe_flux"] = self.ebqe[('advectiveFlux',ci)]

            #argsDict["STABILIZATION_TYPE"] = self.coefficients.STABILIZATION_TYPE
            argsDict["PSK_TYPE"] = self.coefficients.PSK_TYPE
            
            argsDict["cE"] = self.coefficients.cE
            argsDict["cK"] = self.coefficients.cK
            argsDict["uL"] = self.coefficients.uL
            argsDict["uR"] = self.coefficients.uR
            argsDict["numDOFs"] = len(rowptr) - 1
            argsDict["NNZ"] = self.nnz
            argsDict["csrRowIndeces_DofLoops"] = rowptr
            argsDict["csrColumnOffsets_DofLoops"] = colind
            argsDict["csrRowIndeces_CellLoops"] = self.csrRowIndeces[(ci, ci)]
            argsDict["csrColumnOffsets_CellLoops"] = self.csrColumnOffsets[(ci, ci)]
            argsDict["csrColumnOffsets_eb_CellLoops"] = self.csrColumnOffsets_eb[(ci, ci)]
            argsDict["Cx"] = Cx
            argsDict["Cy"] = Cy
            argsDict["Cz"] = Cz
            argsDict["CTx"] = CTx
            argsDict["CTy"] = CTy
            argsDict["CTz"] = CTz
            argsDict["ML"] = self.ML[ci]
            argsDict["delta_x_ij"] = self.delta_x_ij
            argsDict["LUMPED_MASS_MATRIX"] = self.coefficients.LUMPED_MASS_MATRIX
            argsDict["ENTROPY_TYPE"] = self.coefficients.ENTROPY_TYPE
            argsDict["dLow"] = self.dLow[ci]
            argsDict["fluxMatrix"] = self.fluxMatrix[ci]
            argsDict["mDotLow"] = self.mDotLow[ci]
            argsDict["dt_times_fH_minus_fL"] = self.dt_times_dC_minus_dL[ci]
            argsDict["min_m_bc"] = self.min_m_bc[ci]
            argsDict["max_m_bc"] = self.max_m_bc[ci]
            argsDict["quantDOFs"] = self.quantDOFs[ci]
            argsDict["mn"] = self.mn[ci]
            argsDict["anb_seepage_flux"] = self.coefficients.anb_seepage_flux
            argsDict["limited_solution"] = u[ci]
            argsDict["mLow"] = self.u[ci].dof
    #        self.richards.invert(argsDict)
            self.mphase_co2.invert(argsDict)
     
    def getJacobian(self,jacobian):
        if (self.coefficients.STABILIZATION_TYPE == 0):  # SUPG
            cfemIntegrals.zeroJacobian_CSR(self.nNonzerosInJacobian,
                                           jacobian)
        degree_polynomial = 1
        try:
            degree_polynomial = self.u[0].femSpace.order
        except:
            pass
        if self.delta_x_ij is None:
            self.delta_x_ij = -np.ones((self.nNonzerosInJacobian*3,),'d')
        for ci in range(self.nc):
            argsDict = cArgumentsDict.ArgumentsDict()
            argsDict["dt"] = self.timeIntegration.dt
            argsDict["mesh_trial_ref"] = self.u[0].femSpace.elementMaps.psi
            argsDict["mesh_grad_trial_ref"] = self.u[0].femSpace.elementMaps.grad_psi
            argsDict["mesh_dof"] = self.mesh.nodeArray
            argsDict["mesh_velocity_dof"] = self.mesh.nodeVelocityArray
            argsDict["MOVING_DOMAIN"] = self.MOVING_DOMAIN
            argsDict["mesh_l2g"] = self.mesh.elementNodesArray
            argsDict["dV_ref"] = self.elementQuadratureWeights[('u',ci)]
            argsDict["u_trial_ref"] = self.u[0].femSpace.psi
            argsDict["u_grad_trial_ref"] = self.u[0].femSpace.grad_psi
            argsDict["u_test_ref"] = self.u[0].femSpace.psi
            argsDict["u_grad_test_ref"] = self.u[0].femSpace.grad_psi
            argsDict["mesh_trial_trace_ref"] = self.u[0].femSpace.elementMaps.psi_trace
            argsDict["mesh_grad_trial_trace_ref"] = self.u[0].femSpace.elementMaps.grad_psi_trace
            argsDict["dS_ref"] = self.elementBoundaryQuadratureWeights[('u',ci)]
            argsDict["u_trial_trace_ref"] = self.u[0].femSpace.psi_trace
            argsDict["u_grad_trial_trace_ref"] = self.u[0].femSpace.grad_psi_trace
            argsDict["u_test_trace_ref"] = self.u[0].femSpace.psi_trace
            argsDict["u_grad_test_trace_ref"] = self.u[0].femSpace.grad_psi_trace
            argsDict["normal_ref"] = self.u[0].femSpace.elementMaps.boundaryNormals
            argsDict["boundaryJac_ref"] = self.u[0].femSpace.elementMaps.boundaryJacobians
            argsDict["nElements_global"] = self.mesh.nElements_global
            argsDict["ebqe_penalty_ext"] = self.ebqe['penalty']
            argsDict["elementMaterialTypes"] = self.mesh.elementMaterialTypes,
            argsDict["isSeepageFace"] = self.coefficients.isSeepageFace
            argsDict["a_rowptr"] = self.coefficients.sdInfo[(0,0)][0]
            argsDict["a_colind"] = self.coefficients.sdInfo[(0,0)][1]
            
            argsDict["rho_water"] = self.coefficients.rho_water
            argsDict["beta_water"] = self.coefficients.beta_water
            argsDict["rho_air"] = self.coefficients.rho_air
            argsDict["beta_air"] = self.coefficients.beta_air


            
            argsDict["gravity"] = self.coefficients.gravity
            argsDict["alpha"] = self.coefficients.vgm_alpha_types
            argsDict["n"] = self.coefficients.vgm_n_types
            argsDict["thetaR"] = self.coefficients.thetaR_types
            argsDict["thetaSR"] = self.coefficients.thetaSR_types
            argsDict["KWs"] = self.coefficients.Ksw_types
            argsDict["useMetrics"] = 0.0
            argsDict["alphaBDF"] = self.timeIntegration.alpha_bdf
            argsDict["lag_shockCapturing"] = 0
            argsDict["shockCapturingDiffusion"] = 0.1
            argsDict["u_l2g"] = self.u[ci].femSpace.dofMap.l2g
            argsDict["r_l2g"] = self.l2g[ci]['freeGlobal']
            argsDict["elementDiameter"] = self.mesh.elementDiametersArray
            argsDict["degree_polynomial"] = degree_polynomial
            
            argsDict["u_dof_water"] = self.u[0].dof
            argsDict["u_dof_air"] = self.u[1].dof

            argsDict["phase"] = ci
            argsDict["PSK_TYPE"] = self.coefficients.PSK_TYPE
            
            

            
            argsDict["velocity"] = self.q['velocity', ci]
            argsDict["q_m_betaBDF"] = self.timeIntegration.beta_bdf[ci]
            argsDict["cfl"] = self.q[('cfl',ci)]
            argsDict["q_numDiff_u"] = self.q[('numDiff',ci,ci)]
            argsDict["q_numDiff_u_last"] = self.q[('numDiff',ci,ci)]
            argsDict["csrRowIndeces_u_u"] = self.csrRowIndeces[(ci,ci)]
            argsDict["csrColumnOffsets_u_u"] = self.csrColumnOffsets[(ci,ci)]
            argsDict["globalJacobian"] = jacobian.getCSRrepresentation()[2]
            argsDict["delta_x_ij"] = self.delta_x_ij
            argsDict["nExteriorElementBoundaries_global"] = self.mesh.nExteriorElementBoundaries_global
            argsDict["exteriorElementBoundariesArray"] = self.mesh.exteriorElementBoundariesArray
            argsDict["elementBoundaryElementsArray"] = self.mesh.elementBoundaryElementsArray
            argsDict["elementBoundaryLocalElementBoundariesArray"] = self.mesh.elementBoundaryLocalElementBoundariesArray
            argsDict["ebqe_velocity_ext"] = self.ebqe['velocity',ci]
            argsDict["isDOFBoundary_u"] = self.numericalFlux.isDOFBoundary[ci]
            argsDict["ebqe_bc_u_ext"] = self.numericalFlux.ebqe[('u',ci)]
            argsDict["isFluxBoundary_u"] = self.ebqe[('advectiveFlux_bc_flag',ci)]
            argsDict["ebqe_bc_flux_ext"] = self.ebqe[('advectiveFlux_bc',ci)]
            argsDict["csrColumnOffsets_eb_u_u"] = self.csrColumnOffsets_eb[(ci,ci)]
            argsDict["LUMPED_MASS_MATRIX"] = self.coefficients.LUMPED_MASS_MATRIX
            argsDict["VMS"] = self.coefficients.VMS
            #argsDict["anb_seepage_flux"] = self.coefficients.anb_seepage_flux

            self.calculateJacobian(argsDict)    
            if self.coefficients.forceStrongConditions:
                #for cj in range(self.nc):
                for dofN in list(self.dirichletConditionsForceDOF[ci].DOFBoundaryConditionsDict.keys()):
                    global_dofN = self.offset[ci]+self.stride[ci]*dofN
                    self.nzval[np.where(self.colind == global_dofN)] = 0.0 #column
                    self.nzval[self.rowptr[global_dofN]:self.rowptr[global_dofN+1]] = 0.0 #row
                    zeroRow=True
                    for i in range(self.rowptr[global_dofN],self.rowptr[global_dofN+1]):#row
                        if (self.colind[i] == global_dofN):
                            self.nzval[i] = 1.0
                            zeroRow = False
                    if zeroRow:
                        raise RuntimeError("Jacobian has a zero row because sparse matrix has no diagonal entry at row "+repr(global_dofN)+". You probably need add diagonal mass or reaction term")
            #scaling = 1.0#probably want to add some scaling to match non-dirichlet diagonals in linear system 
            #for cj in range(self.nc):
            #    for dofN in list(self.dirichletConditionsForceDOF[cj].DOFBoundaryConditionsDict.keys()):
            #        global_dofN = self.offset[cj]+self.stride[cj]*dofN
            #        for i in range(self.rowptr[global_dofN],self.rowptr[global_dofN+1]):
            #            if (self.colind[i] == global_dofN):
            #                self.nzval[i] = scaling
            #            else:
            #                self.nzval[i] = 0.0
        logEvent("Jacobian ",level=10,data=jacobian)
        #mwf decide if this is reasonable for solver statistics
        self.nonlinear_function_jacobian_evaluations += 1
        return jacobian
    def calculateElementQuadrature(self):
        """
        Calculate the physical location and weights of the quadrature rules
        and the shape information at the quadrature points.

        This function should be called only when the mesh changes.
        """
        #self.u[0].femSpace.elementMaps.getValues(self.elementQuadraturePoints,
        #                                         self.q['x'])
        self.u[0].femSpace.elementMaps.getBasisValuesRef(self.elementQuadraturePoints)
        self.u[0].femSpace.elementMaps.getBasisGradientValuesRef(self.elementQuadraturePoints)
        self.u[0].femSpace.getBasisValuesRef(self.elementQuadraturePoints)
        self.u[0].femSpace.getBasisGradientValuesRef(self.elementQuadraturePoints)
        self.coefficients.initializeElementQuadrature(self.timeIntegration.t,self.q)
        if self.stabilization != None:
            self.stabilization.initializeElementQuadrature(self.mesh,self.timeIntegration.t,self.q)
            self.stabilization.initializeTimeIntegration(self.timeIntegration)
        if self.shockCapturing != None:
            self.shockCapturing.initializeElementQuadrature(self.mesh,self.timeIntegration.t,self.q)
    def calculateElementBoundaryQuadrature(self):
        pass
    def calculateExteriorElementBoundaryQuadrature(self):
        """
        Calculate the physical location and weights of the quadrature rules
        and the shape information at the quadrature points on global element boundaries.

        This function should be called only when the mesh changes.
        """
        #
        #get physical locations of element boundary quadrature points
        #
        #assume all components live on the same mesh
        self.u[0].femSpace.elementMaps.getBasisValuesTraceRef(self.elementBoundaryQuadraturePoints)
        self.u[0].femSpace.elementMaps.getBasisGradientValuesTraceRef(self.elementBoundaryQuadraturePoints)
        self.u[0].femSpace.getBasisValuesTraceRef(self.elementBoundaryQuadraturePoints)
        self.u[0].femSpace.getBasisGradientValuesTraceRef(self.elementBoundaryQuadraturePoints)
        self.u[0].femSpace.elementMaps.getValuesGlobalExteriorTrace(self.elementBoundaryQuadraturePoints,
                                                                    self.ebqe['x'])
        self.fluxBoundaryConditionsObjectsDict = dict([(cj,FluxBoundaryConditions(self.mesh,
                                                                                  self.nElementBoundaryQuadraturePoints_elementBoundary,
                                                                                  self.ebqe[('x')],
                                                                                  getAdvectiveFluxBoundaryConditions=self.advectiveFluxBoundaryConditionsSetterDict[cj],
                                                                                  getDiffusiveFluxBoundaryConditions=self.diffusiveFluxBoundaryConditionsSetterDictDict[cj]))
                                                       for cj in list(self.advectiveFluxBoundaryConditionsSetterDict.keys())])
        self.coefficients.initializeGlobalExteriorElementBoundaryQuadrature(self.timeIntegration.t,self.ebqe)
        #argsDict = cArgumentsDict.ArgumentsDict()
        #argsDict["anb_seepage_flux"] = self.coefficients.anb_seepage_flux
        #print("Hi", self.coefficients.anb_seepage_flux)

        #print("The seepage is ", anb_seepage_flux)
    def estimate_mt(self):
        pass
    def calculateSolutionAtQuadrature(self):
        pass
    def calculateAuxiliaryQuantitiesAfterStep(self):
        pass
    #def postStep(self, t, firstStep=False):
    #    with open('seepage_flux_nnnnk', "a") as f:
    #        f.write("\n Time"+ ",\t" +"Seepage\n")
    #        f.write(repr(t)+ ",\t" +repr(self.coefficients.anb_seepage_flux))

    

#argsDict["anb_seepage_flux"] = self.coefficients.anb_seepage_flux        
#anb_seepage_flux= self.coefficients.anb_seepage_flux
#print("Hi",anb_seepage_flux)


#print("Hello from the python file", self.coefficients.anb_seepage_flux)
