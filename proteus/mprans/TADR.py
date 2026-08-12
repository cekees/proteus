# A type of -*- python -*- file
"""
An optimized Advection-Diffusion-Reaction  transport module
"""
import numpy as np
from proteus import cfemIntegrals, Quadrature, Norms, Comm, TimeIntegration
from proteus.NonlinearSolvers import NonlinearEquation
from proteus.FemTools import (DOFBoundaryConditions,
                              FluxBoundaryConditions,
                              C0_AffineLinearOnSimplexWithNodalBasis)

from proteus.Comm import globalMax
from proteus.Profiling import (memory, logEvent)
from proteus.Transport import OneLevelTransport
from proteus.TransportCoefficients import TC_base
from proteus.SubgridError import SGE_base
from proteus.ShockCapturing import ShockCapturing_base
from proteus.NumericalFlux import Advection_DiagonalUpwind_Diffusion_IIPG_exterior
from proteus.LinearAlgebraTools import SparseMat
from proteus.NonlinearSolvers import ExplicitLumpedMassMatrix,ExplicitConsistentMassMatrixForVOF,TwoStageNewton
from proteus.mprans.cTADR import cTADR_base

from . import cArgumentsDict

class SubgridError(SGE_base):
    def __init__(self, coefficients, nd):
        SGE_base.__init__(self, coefficients, nd, lag=False)

    def initializeElementQuadrature(self, mesh, t, cq):
        pass

    def updateSubgridErrorHistory(self, initializationPhase=False):
        pass

    def calculateSubgridError(self, q):
        pass

class ShockCapturing(ShockCapturing_base):
    def __init__(self,
                 coefficients,
                 nd,
                 shockCapturingFactor=0.25,
                 lag=True,
                 nStepsToDelay=None):
        ShockCapturing_base.__init__(self,
                         coefficients,
                         nd,
                         shockCapturingFactor=shockCapturingFactor,
                         lag=lag)
        self.nStepsToDelay = nStepsToDelay
        self.nSteps = 0
        if self.lag:
            logEvent("TADR.ShockCapturing: lagging requested but must delay the first step; switching lagging off and delaying")
            self.nStepsToDelay = 1
            self.lag = False

    def initializeElementQuadrature(self, mesh, t, cq):
        self.mesh = mesh
        self.numDiff = []
        self.numDiff_last = []
        for ci in range(self.nc):
            self.numDiff.append(cq[('numDiff', ci, ci)])
            self.numDiff_last.append(cq[('numDiff', ci, ci)])
            

    def updateShockCapturingHistory(self):
        self.nSteps += 1
        if self.lag:
            for ci in range(self.nc):
                self.numDiff_last[ci][:] = self.numDiff[ci]
        if self.lag == False and self.nStepsToDelay is not None and self.nSteps > self.nStepsToDelay:
            logEvent("TADR.ShockCapturing: switched to lagged shock capturing")
            self.lag = True
            self.numDiff_last = []
            for ci in range(self.nc):
                self.numDiff_last.append(self.numDiff[ci].copy())
        logEvent("TADR: max numDiff %e" % (globalMax(self.numDiff_last[0].max()),))

class NumericalFlux(Advection_DiagonalUpwind_Diffusion_IIPG_exterior):
    def __init__(self,
                 vt,
                 getPointwiseBoundaryConditions,
                 getAdvectiveFluxBoundaryConditions,
                 getDiffusiveFluxBoundaryConditions):#,
                 #getPeriodicBoundaryConditions=None):
        Advection_DiagonalUpwind_Diffusion_IIPG_exterior.__init__(
            self,
            vt,
            getPointwiseBoundaryConditions,
            getAdvectiveFluxBoundaryConditions,
            getDiffusiveFluxBoundaryConditions)#,
            #getPeriodicBoundaryConditions)#
  
class RKEV(TimeIntegration.SSP):
    """
    Wrapper for SSPRK time integration using EV

    ... more to come ...
    """

    def __init__(self, transport, timeOrder=1, runCFL=0.1, integrateInterpolationPoints=False):
        TimeIntegration.SSP.__init__(self,
                                     transport,
                                     integrateInterpolationPoints=integrateInterpolationPoints)
        self.runCFL = runCFL
        self.dtLast = None
        self.isAdaptive = True
        assert transport.coefficients.STABILIZATION_TYPE>1, "SSP method just works for edge based EV methods; i.e., STABILIZATION_TYPE>1"
        assert hasattr(transport, 'edge_based_cfl'), "No edge based cfl defined"
        # About the cfl
        self.cfl = transport.edge_based_cfl
        # Stuff particular for SSP
        self.timeOrder = timeOrder  # order of approximation
        self.nStages = timeOrder  # number of stages total
        self.lstage = 0  # last stage completed
        # storage vectors
        self.u_dof_last = {}
        self.m_old = {}
        # per component stage values, list with array at each stage
        for ci in range(self.nc):
            self.m_last[ci] = transport.q[('u',ci)].copy()
            self.m_old[ci] = transport.q[('u',ci)].copy()
            self.u_dof_last[ci] = transport.u[ci].dof.copy()

    def choose_dt(self):
        maxCFL = 1.0e-6
        maxCFL = max(maxCFL, globalMax(self.cfl.max()))
        self.dt = self.runCFL/maxCFL
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
                    # save stage at quad points
                    self.m_last[ci][:] = self.transport.q[('u',ci)]
                    # DOFs
                    self.transport.u_dof_old[:] = self.transport.u[ci].dof
            elif self.lstage == 2:
                logEvent("Second stage of SSP33 method", level=4)
                for ci in range(self.nc):
                    # Quad points
                    self.m_last[ci][:] = 1./4*self.transport.q[('u',ci)]
                    self.m_last[ci][:] += 3./4*self.m_old[ci]
                    # DOFs
                    self.transport.u_dof_old[:] = 1./4*self.transport.u[ci].dof
                    self.transport.u_dof_old[:] += 3./4* self.u_dof_last[ci]                
            elif self.lstage == 3:
                logEvent("Third stage of SSP33 method", level=4)
                for ci in range(self.nc):
                    # Quad points
                    self.m_last[ci][:] = 2./3*self.transport.q[('u',ci)]
                    self.m_last[ci][:] += 1./3*self.m_old[ci]
                    # DOFs
                    self.transport.u[0].dof[:] = 2./3*self.transport.u[ci].dof
                    self.transport.u[0].dof[:] += 1./3* self.u_dof_last[ci]
                    # update u_dof_old
                    self.transport.u_dof_old[:] = self.u_dof_last[ci]                    
        elif self.timeOrder == 2:
            if self.lstage == 1:
                logEvent("First stage of SSP22 method", level=4)
                for ci in range(self.nc):
                    # save stage at quad points
                    self.m_last[ci][:] = self.transport.q[('u',ci)]
                    # DOFs
                    self.transport.u_dof_old[:] = self.transport.u[ci].dof
            elif self.lstage == 2:
                logEvent("Second stage of SSP22 method", level=4)
                for ci in range(self.nc):
                    # Quad points
                    self.m_last[ci][:] = 1./2*self.transport.q[('u',ci)]
                    self.m_last[ci][:] += 1./2*self.m_old[ci]
                    # DOFs
                    self.transport.u[0].dof[:]  = 1./2*self.transport.u[ci].dof
                    self.transport.u[0].dof[:] += 1./2*self.u_dof_last[ci]
                    # update u_dof_old
                    self.transport.u_dof_old[:] = self.u_dof_last[ci]                    
        else:
            assert self.timeOrder == 1
            for ci in range(self.nc):
                self.m_last[ci][:] = self.transport.q[('u',ci)]

    def initializeTimeHistory(self, resetFromDOF=True):
        """
        Push necessary information into time history arrays
        """
        for ci in range(self.nc):
            self.m_old[ci][:] = self.transport.q[('u',ci)]
            self.m_last[ci][:] = self.transport.q[('u',ci)]
            self.u_dof_last[ci][:] = self.transport.u[ci].dof[:]

    def updateTimeHistory(self, resetFromDOF=False):
        """
        assumes successful step has been taken
        """
        self.t = self.tLast + self.dt
        for ci in range(self.nc):
            self.m_old[ci][:] = self.m_last[ci][:]
            self.u_dof_last[ci][:] = self.transport.u[ci].dof[:]
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
                    
class Coefficients(TC_base):
    from proteus.ctransportCoefficients import VOFCoefficientsEvaluate
    from proteus.ctransportCoefficients import VolumeAveragedVOFCoefficientsEvaluate
    from proteus.cfemIntegrals import copyExteriorElementBoundaryValuesFromElementBoundaryValues
    
    def __init__(self,
                 #aOfX,
                 alpha_L,
                 alpha_T,
                 Dm,
                 LS_model=None,
                 nd=2,
                 V_model=0,
                 RD_model=None,
                 ME_model=1,
                 VOS_model=None,
                 checkMass=True,
                 epsFact=0.0,
                 useMetrics=0.0,
                 sc_uref=1.0,
                 sc_beta=1.0,
                 movingDomain=False,
                 forceStrongConditions=True,
                 STABILIZATION_TYPE='VMS',        
                 ENTROPY_TYPE='POWER',
                 diagonal_conductivity=True,
                 # 0: quadratic
                 # 1: logarithmic
                 # FOR ENTROPY VISCOSITY
                 cE=1.0,
                 cMax=1.0,
                 uL=0.0,
                 uR=1.0,
                 # FOR ARTIFICIAL COMPRESSION
                 cK=0.0,
                 LUMPED_MASS_MATRIX=False,
                 FCT=True,
                 outputQuantDOFs=False,
                 #NULLSPACE INFO
                 nullSpace='NoNullSpace',
                 initialize=True,
                 physicalDiffusion=0.0):
        self.variableNames = ['u']
        self.LS_modelIndex = LS_model
        self.V_model = V_model
        self.RD_modelIndex = RD_model
        self.modelIndex = ME_model
        self.VOS_model=VOS_model
        self.checkMass = checkMass
        self.epsFact = epsFact
        self.flowModelIndex = V_model
        self.modelIndex = ME_model
        self.RD_modelIndex = RD_model
        self.LS_modelIndex = LS_model
        self.V_model = V_model
        self.RD_modelIndex = RD_model
        self.modelIndex = ME_model
        self.VOS_model=VOS_model
        self.checkMass = checkMass
        self.epsFact = epsFact
        self.useMetrics = useMetrics
        self.sc_uref = sc_uref
        self.sc_beta = sc_beta
        self.movingDomain = movingDomain
        self.forceStrongConditions = forceStrongConditions
        self.nd=nd
        #self.aOfX = aOfX
        self.alpha_L= alpha_L
        self.alpha_T= alpha_T
        self.Dm= Dm        
        self.diagonal_conductivity = diagonal_conductivity

        if self.diagonal_conductivity:
            sdInfo= {(0,0):(np.arange(self.nd+1,dtype='i'),
                            np.arange(self.nd,dtype='i'))}
        else:
            sdInfo = {(0, 0): (np.arange(start=0, stop=nd**2 + 1, step=nd, dtype='int32'),
                   np.array([range(nd) for _ in range(nd)], dtype='int32'))}
        
        self.cE = cE
        self.cMax = cMax
        self.uL = uL
        self.uR = uR
        self.cK = cK
        self.LUMPED_MASS_MATRIX = LUMPED_MASS_MATRIX
        self.FCT = FCT
        self.outputQuantDOFs = outputQuantDOFs
        self.nullSpace = nullSpace
        self.flowCoefficients = None
        self.physicalDiffusion=physicalDiffusion
        self.sparseDiffusionTensors = sdInfo
        if initialize:
            self.initialize()
        

        #must keep synchronized with TADR.h enums
        stabilization_types = {"Galerkin":-1, 
                               "VMS":0, 
                               "TaylorGalerkinEV":1, 
                               "EntropyViscosity":2, 
                               "SmoothnessIndicator":3, 
                               "Kuzmin":4}
        entropy_types = {'POWER':0 , 
                         'LOG':1}
        try:
            if isinstance(STABILIZATION_TYPE, int):
                STABILIZATION_TYPE = [key for key, value in stabilization_types.items() if value == STABILIZATION_TYPE][0]
            
            self.STABILIZATION_TYPE = stabilization_types[STABILIZATION_TYPE]
        except:
            raise ValueError("STABILIZATION_TYPE must be one of "+str(stabilization_types.keys())+" not "+STABILIZATION_TYPE)
        try:
            self.ENTROPY_TYPE = entropy_types[ENTROPY_TYPE]
        except:
            raise ValueError("ENTROPY_TYPE must be one of "+str(entropy_types.keys())+" not "+ENTROPY_TYPE)
        self.cE = cE
        self.cMax = cMax
        self.uL = uL
        self.uR = uR
        self.cK = cK
        self.LUMPED_MASS_MATRIX = LUMPED_MASS_MATRIX
        self.FCT = FCT
        self.outputQuantDOFs = outputQuantDOFs
        self.nullSpace = nullSpace
        self.flowCoefficients = None
        self.physicalDiffusion=physicalDiffusion
        if initialize:
            self.initialize()
   

    def initialize(self):
        nc = 1
        mass = {0: {0: 'linear'}}
        advection = {0: {0: 'linear'}}
        hamiltonian = {}
        diffusion = {0: {0: {0: 'constant'}}}
        potential = {0: {0: 'u'} }
        reaction = {0: {0: 'linear'}}
        sparseDiffusionTensors = {(0,0):(np.arange(self.nd+1,dtype='i'),
                                         np.arange(self.nd,dtype='i'))}

        

        TC_base.__init__(self,
                         nc,
                         mass,
                         advection,
                         diffusion,
                         potential,
                         reaction,
                         hamiltonian,
                         self.variableNames,
                         movingDomain=self.movingDomain,
                         sparseDiffusionTensors=sparseDiffusionTensors)
                

    def initializeMesh(self, mesh):
        self.eps = self.epsFact * mesh.h
        
    # def attachModels(self, modelList):
    #     # self
    #     self.model = modelList[self.modelIndex]
    #     # redistanced level set
    #     if self.RD_modelIndex is not None:
    #         self.rdModel = modelList[self.RD_modelIndex]
    #     # level set
    #     if self.LS_modelIndex is not None:
    #         self.lsModel = modelList[self.LS_modelIndex]
    #         self.q_phi = modelList[self.LS_modelIndex].q[('u', 0)]
    #         self.ebqe_phi = modelList[self.LS_modelIndex].ebqe[('u', 0)]
    #         if ('u', 0) in modelList[self.LS_modelIndex].ebq:
    #             self.ebq_phi = modelList[self.LS_modelIndex].ebq[('u', 0)]
    #     else:
    #         self.ebqe_phi = np.zeros(self.model.ebqe[('u', 0)].shape, 'd') # cek hack, we don't need this
    #     # flow model
    #     if self.V_model is not None:
    #         if ('velocity', 0) in modelList[self.V_model].q:
    #             self.q_v = modelList[self.V_model].q[('velocity', 0)]
    #             self.ebqe_v = modelList[self.V_model].ebqe[('velocity', 0)]
    #         else:
    #             self.q_v = modelList[self.V_model].q[('f', 0)]
    #             self.ebqe_v = modelList[self.V_model].ebqe[('f', 0)]
    #         if ('velocity', 0) in modelList[self.V_model].ebq:
    #             self.ebq_v = modelList[self.V_model].ebq[('velocity', 0)]
    #         else:
    #             if ('f', 0) in modelList[self.V_model].ebq:
    #                 self.ebq_v = modelList[self.V_model].ebq[('f', 0)]
    #     else:
    #         self.q_v = np.ones(self.model.q[('u',0)].shape+(self.model.nSpace_global,),'d')
    #         self.ebqe_v = np.ones(self.model.ebqe[('u',0)].shape+(self.model.nSpace_global,),'d')
    #     # VRANS
    #     if self.V_model is not None:
    #         self.flowCoefficients = modelList[self.V_model].coefficients
    #     else:
    #         self.flowCoefficients = None
    #     # Log velocity fields
    #     logEvent(f"Velocity field (q_v): mean={np.mean(self.q_v)}, min={np.min(self.q_v)}, max={np.max(self.q_v)}")
    #     logEvent(f"Boundary velocity (ebqe_v): mean={np.mean(self.ebqe_v)}, min={np.min(self.ebqe_v)}, max={np.max(self.ebqe_v)}")
    #     if self.ebq_v is not None:
    #         logEvent(f"Edge velocity (ebq_v): mean={np.mean(self.ebq_v)}, min={np.min(self.ebq_v)}, max={np.max(self.ebq_v)}")

    def attachModels(self, modelList):
        import pdb
        pdb.set_trace()
        # self
        self.model = modelList[self.modelIndex]
        # redistanced level set
        if self.RD_modelIndex is not None:
            self.rdModel = modelList[self.RD_modelIndex]
        # level set
        if self.LS_modelIndex is not None:
            self.lsModel = modelList[self.LS_modelIndex]
            self.q_phi = modelList[self.LS_modelIndex].q[('u', 0)]
            self.ebqe_phi = modelList[self.LS_modelIndex].ebqe[('u', 0)]
            if ('u', 0) in modelList[self.LS_modelIndex].ebq:
                self.ebq_phi = modelList[self.LS_modelIndex].ebq[('u', 0)]
        else:
            self.ebqe_phi = np.zeros(self.model.ebqe[('u', 0)].shape, 'd') # cek hack, we don't need this
        # flow model
        if self.V_model is not None:
            if ('grad(u)', 0) in modelList[self.V_model].q:
                self.q_v = modelList[self.V_model].q[('grad(u)', 0)]
                self.ebqe_v = modelList[self.V_model].ebqe[('grad(u)', 0)]
            else:
                self.q_v = modelList[self.V_model].q[('f', 0)]
                self.ebqe_v = modelList[self.V_model].ebqe[('f', 0)]
            if ('velocity', 0) in modelList[self.V_model].ebq:
                self.ebq_v = modelList[self.V_model].ebq[('velocity', 0)]
            else:
                if ('f', 0) in modelList[self.V_model].ebq:
                    self.ebq_v = modelList[self.V_model].ebq[('f', 0)]
        else:
            self.q_v = np.ones(self.model.q[('u',0)].shape+(self.model.nSpace_global,),'d')
            self.ebqe_v = np.ones(self.model.ebqe[('u',0)].shape+(self.model.nSpace_global,),'d')
        # VRANS
        if self.V_model is not None:
            self.flowCoefficients = modelList[self.V_model].coefficients
        else:
            self.flowCoefficients = None
        # Log velocity fields
        logEvent(f"Velocity field (q_v): mean={np.mean(self.q_v)}, min={np.min(self.q_v)}, max={np.max(self.q_v)}")
        logEvent(f"Boundary velocity (ebqe_v): mean={np.mean(self.ebqe_v)}, min={np.min(self.ebqe_v)}, max={np.max(self.ebqe_v)}")
        if self.ebq_v is not None:
            logEvent(f"Edge velocity (ebq_v): mean={np.mean(self.ebq_v)}, min={np.min(self.ebq_v)}, max={np.max(self.ebq_v)}")

        
    def preStep(self, t, firstStep=False):
        # SAVE OLD SOLUTION #
        self.model.u_dof_old[:] = self.model.u[0].dof

        # Restart flags for stages of taylor galerkin
        self.model.stage = 1
        self.model.auxTaylorGalerkinFlag = 1
        # Skip velocity field as a function check
        if hasattr(self.model, "updateVelocityFieldAsFunction"):
            self.model.updateVelocityFieldAsFunction()
        
        # COMPUTE NEW VELOCITY (if given by user) #
#        if self.model.hasVelocityFieldAsFunction:
#            self.model.updateVelocityFieldAsFunction()
         # Explicitly update velocity if grad(u) is available
        if ('grad(u_v)', 0) in self.model.q:
            self.model.q[('velocity', 0)] = self.model.q[('grad(u_v)', 0)]
            #self.model.q[('velocity', 0)] = np.copy(self.model.q[('grad(u_v)', 0)])
            logEvent(f"Velocity updated in preStep: mean={self.model.q[('grad(u_v)', 0)].mean()}, "
                 f"min={self.model.q[('grad(u_v)', 0)].min()}, max={self.model.q[('grad(u_v)', 0)].max()}")
            if not np.any(self.model.q[('grad(u_v)', 0)]):
                logEvent("Warning: grad(u_v) is zero, velocity will also be zero!")
         # Log coefficients velocity fields
        logEvent(f"Coefficients velocity field (q_v): "
                 f"mean={np.mean(self.q_v)}, min={np.min(self.q_v)}, max={np.max(self.q_v)}")
        logEvent(f"Coefficients boundary velocity field (ebqe_v): "
                 f"mean={np.mean(self.ebqe_v)}, min={np.min(self.ebqe_v)}, max={np.max(self.ebqe_v)}")
        # Compute velocity and update q_a_mod
        nElements_global, nQuadraturePoints_element, _ = self.q_v.shape  # Extract dimensions from q_v

        # Update q_a_mod (diffusion tensor) if initialized
        if hasattr(self, 'q_a_mod') and self.q_a_mod is not None:
            logEvent("Updating diffusion tensor q_a_mod in preStep...")
            a_rowptr = self.sdInfo[(0, 0)][0]
            a_colind = self.sdInfo[(0, 0)][1]
            for eN in range(nElements_global):  # Loop over elements
                for k in range(nQuadraturePoints_element):  # Loop over valid quadrature points
                    #if eN >= self.q_v.shape[0] or k >= self.q_v.shape[1]:
                    #    raise IndexError(f"Index out of bounds: eN={eN}, k={k}, q_v shape={self.q_v.shape}")
                    # Access velocity and position
                    velocity = self.q_v[eN, k, :] if self.q_v is not None else np.zeros(self.nd)
                    #logEvent(f"D is: mean={np.mean(velocity)}, min={np.min(velocity)}, max={np.max(velo)}")

                    #logEvent("velocity is ", f"{np.mean(velocity)}")
                    v_mag = np.linalg.norm(velocity) 
                    if v_mag > 1e-8:
                        v_unit = velocity / v_mag
                    else:
                        v_unit = np.zeros_like(velocity)
                    # Compute dispersion tensor
                    dispersion_tensor = (
                        self.Dm * np.eye(self.nd)  # Molecular diffusion
                        + self.alpha_L * np.outer(v_unit, v_unit) * v_mag  # Longitudinal dispersion
                        + self.alpha_T * v_mag * (np.eye(self.nd) - np.outer(v_unit, v_unit))  # Transverse dispersion
                    )
                    # Map the diffusion tensor to the sparsity pattern
                    a_val = np.array([dispersion_tensor[row, col] for row, col in zip(a_rowptr[:-1], a_colind)])
                    if a_val.shape == (a_rowptr[-1],):
                        self.q_a_mod[eN, k, :] = a_val
                    else:
                        raise ValueError(f"Unexpected shape {a_val.shape} for a_val. Expected ({self.a_rowptr[-1]},)")
                    #logEvent(f"D is: mean={np.mean(self.q_a_mod)}, min={np.min(self.q_a_mod)}, max={np.max(self.q_a_mod)}")

        #if self.ebq_v is not None:
        #    logEvent(f"Coefficients edge velocity field (ebq_v): "
        #             f"mean={np.mean(self.ebq_v)}, min={np.min(self.ebq_v)}, max={np.max(self.ebq_v)}")
        if self.checkMass:
            self.m_pre = Norms.scalarDomainIntegral(self.model.q['dV_last'],
                                                    self.model.q[('m', 0)],
                                                    self.model.mesh.nElements_owned)
            logEvent("Phase  0 mass before TADR step = %12.5e" % (self.m_pre,), level=2)
        copyInstructions = {}
        return copyInstructions

    def postStep(self, t, firstStep=False):
        self.model.q['dV_last'][:] = self.model.q['dV']
        if self.checkMass:
            self.m_post = Norms.scalarDomainIntegral(self.model.q['dV'],
                                                     self.model.q[('m', 0)],
                                                     self.model.mesh.nElements_owned)
            logEvent("Phase  0 mass after TADR step = %12.5e" % (self.m_post,), level=2)
        copyInstructions = {}
        return copyInstructions

    def evaluate(self, t, c):
        import pdb
        pdb.set_trace()

        if c[('f', 0)].shape == self.q_v.shape:
            v = self.q_v
            phi = self.q_phi

        elif c[('f', 0)].shape == self.ebqe_v.shape:
            v = self.ebqe_v
            phi = self.ebqe_phi

        elif ((self.ebq_v is not None and self.ebq_phi is not None) and c[('f', 0)].shape == self.ebq_v.shape):
            v = self.ebq_v
            phi = self.ebq_phi

        else:
            v = None
            phi = None
        if v is not None:
            if not np.any(v):
                logEvent("Warning: Velocity (v) is zero in evaluate!")
            c[('m',0)] = c[('u',0)]
            c[('dm',0,0)] = np.ones_like(c[('u',0)])
            c[('f',0)][:] = v*c[('u',0)]
            c[('df',0,0)][:] = v
            #c[('a',0,0)][:] = self.physicalDiffusion 
            nd = self.nd
            for i in range(len(c[('r', 0)].flat)):
                #x_i = c['x'].flat[3*i:3*(i+1)]
                velocity = v.flat[nd*i:nd*(i+1)] 
                #c[('a', 0, 0)].flat[nd*nd*i:nd*nd*(i+1)] = self.aOfX[0](x_i).flat
            ## Add dispersion
                v_mag = np.linalg.norm(velocity)  # Magnitude of the velocity vector

                if v_mag > 1e-8:  # Avoid division by zero
                    v_unit = velocity / v_mag  # Unit vector of velocity
                else:
                    v_unit = np.zeros_like(velocity)

                # Dispersion-diffusion tensor D
                dispersion_tensor = (
                    self.phi * self.D_m * np.eye(nd)  # Molecular diffusion
                    + self.alpha_L * np.outer(v_unit, v_unit) * v_mag  # Longitudinal dispersion
                    + self.alpha_T * v_mag * (np.eye(nd) - np.outer(v_unit, v_unit))  # Transverse dispersion
                )

                # Assign the dispersion tensor to the coefficient matrix
                c[('a', 0, 0)].flat[nd*nd*i:nd*nd*(i+1)] = dispersion_tensor.flat
                    #print(c[('a', 0, 0)])
                    # Store the dispersion tensor in the coefficient dictionary
            #c[('dispersion_tensor', 0)] = dispersion_tensor
            
            # Compute diffusion coefficients at each quadrature point
            #for i in range(len(c['x'])):
            #    x_i = c['x'][i]
            #    diffusion_matrix = self.aOfX[0](x_i)
            #    c[('a', 0, 0)][i] = diffusion_matrix


class LevelModel(OneLevelTransport):
    nCalls = 0

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
                 sd=True,
                 movingDomain=False,
                 bdyNullSpace=False):

        self.auxiliaryCallCalculateResidual = False
        #
        # set the objects describing the method and boundary conditions
        #
        self.bdyNullSpace = bdyNullSpace
        self.movingDomain = movingDomain
        self.tLast_mesh = None
        #
        self.name = name
        self.sd = sd
        self.Hess = False
        self.timeTerm = True  # allow turning off  the  time derivative
        self.testIsTrial = True
        self.phiTrialIsTrial = True
        self.u = uDict
        self.ua = {}  # analytical solutions
        self.phi = phiDict
        self.dphi = {}
        self.matType = matType
        self.reuse_test_trial_quadrature = reuse_trial_and_test_quadrature
        if self.reuse_test_trial_quadrature:
            for ci in range(1, coefficients.nc):
                assert self.u[ci].femSpace.__class__.__name__ == self.u[0].femSpace.__class__.__name__, "to reuse_test_trial_quad all femSpaces must be the same!"
        self.u_dof_old = None
        self.mesh = self.u[0].femSpace.mesh  # assume the same mesh for  all components for now
        self.testSpace = testSpaceDict
        self.dirichletConditions = dofBoundaryConditionsDict
        self.dirichletNodeSetList = None  # explicit Dirichlet  conditions for now, no Dirichlet BC constraints
        self.coefficients = coefficients
        self.coefficients.initializeMesh(self.mesh)
        self.nc = self.coefficients.nc
        self.stabilization = stabilization
        self.shockCapturing = shockCapturing
        self.conservativeFlux = conservativeFluxDict  # no velocity post-processing for now
        self.fluxBoundaryConditions = fluxBoundaryConditionsDict
        self.advectiveFluxBoundaryConditionsSetterDict = advectiveFluxBoundaryConditionsSetterDict
        self.diffusiveFluxBoundaryConditionsSetterDictDict = diffusiveFluxBoundaryConditionsSetterDictDict
        # determine whether  the stabilization term is nonlinear
        self.stabilizationIsNonlinear = False
        # cek come back
        if self.stabilization is not None:
            for ci in range(self.nc):
                if ci in coefficients.mass:
                    for flag in list(coefficients.mass[ci].values()):
                        if flag == 'nonlinear':
                            self.stabilizationIsNonlinear = True
                if ci in coefficients.advection:
                    for flag in list(coefficients.advection[ci].values()):
                        if flag == 'nonlinear':
                            self.stabilizationIsNonlinear = True
                if ci in coefficients.diffusion:
                    for diffusionDict in list(coefficients.diffusion[ci].values()):
                        for flag in list(diffusionDict.values()):
                            if flag != 'constant':
                                self.stabilizationIsNonlinear = True
                if ci in coefficients.potential:
                    for flag in list(coefficients.potential[ci].values()):
                        if flag == 'nonlinear':
                            self.stabilizationIsNonlinear = True
                if ci in coefficients.reaction:
                    for flag in list(coefficients.reaction[ci].values()):
                        if flag == 'nonlinear':
                            self.stabilizationIsNonlinear = True
                if ci in coefficients.hamiltonian:
                    for flag in list(coefficients.hamiltonian[ci].values()):
                        if flag == 'nonlinear':
                            self.stabilizationIsNonlinear = True
        # determine if we need element boundary storage
        self.elementBoundaryIntegrals = {}
        for ci in range(self.nc):
            self.elementBoundaryIntegrals[ci] = ((self.conservativeFlux is not None) or
                                                 (numericalFluxType is not None) or
                                                 (self.fluxBoundaryConditions[ci] == 'outFlow') or
                                                 (self.fluxBoundaryConditions[ci] == 'mixedFlow') or
                                                 (self.fluxBoundaryConditions[ci] == 'setFlow'))
        #
        # calculate some dimensions
        #
        self.nSpace_global = self.u[0].femSpace.nSpace_global  # assume same space dim for all variables
        self.nDOF_trial_element = [u_j.femSpace.max_nDOF_element for u_j in list(self.u.values())]
        self.nDOF_phi_trial_element = [phi_k.femSpace.max_nDOF_element for phi_k in list(self.phi.values())]
        self.n_phi_ip_element = [phi_k.femSpace.referenceFiniteElement.interpolationConditions.nQuadraturePoints for phi_k in list(self.phi.values())]
        self.nDOF_test_element = [femSpace.max_nDOF_element for femSpace in list(self.testSpace.values())]
        self.nFreeDOF_global = [dc.nFreeDOF_global for dc in list(self.dirichletConditions.values())]
        self.nVDOF_element = sum(self.nDOF_trial_element)
        self.nFreeVDOF_global = sum(self.nFreeDOF_global)
        #
        NonlinearEquation.__init__(self, self.nFreeVDOF_global)
        #
        # build the quadrature point dictionaries from the input (this
        # is just for convenience so that the input doesn't have to be
        # complete)
        #
        elementQuadratureDict = {}
        elemQuadIsDict = isinstance(elementQuadrature, dict)
        if elemQuadIsDict:  # set terms manually
            for I in self.coefficients.elementIntegralKeys:
                if I in elementQuadrature:
                    elementQuadratureDict[I] = elementQuadrature[I]
                else:
                    elementQuadratureDict[I] = elementQuadrature['default']
        else:
            for I in self.coefficients.elementIntegralKeys:
                elementQuadratureDict[I] = elementQuadrature
        if self.stabilization is not None:
            for I in self.coefficients.elementIntegralKeys:
                if elemQuadIsDict:
                    if I in elementQuadrature:
                        elementQuadratureDict[('stab',) + I[1:]] = elementQuadrature[I]
                    else:
                        elementQuadratureDict[('stab',) + I[1:]] = elementQuadrature['default']
                else:
                    elementQuadratureDict[('stab',) + I[1:]] = elementQuadrature
        if self.shockCapturing is not None:
            for ci in self.shockCapturing.components:
                if elemQuadIsDict:
                    if ('numDiff', ci, ci) in elementQuadrature:
                        elementQuadratureDict[('numDiff', ci, ci)] = elementQuadrature[('numDiff', ci, ci)]
                    else:
                        elementQuadratureDict[('numDiff', ci, ci)] = elementQuadrature['default']
                else:
                    elementQuadratureDict[('numDiff', ci, ci)] = elementQuadrature
        if massLumping:
            for ci in list(self.coefficients.mass.keys()):
                elementQuadratureDict[('m', ci)] = Quadrature.SimplexLobattoQuadrature(self.nSpace_global, 1)
            for I in self.coefficients.elementIntegralKeys:
                elementQuadratureDict[('stab',) + I[1:]] = Quadrature.SimplexLobattoQuadrature(self.nSpace_global, 1)
        if reactionLumping:
            for ci in list(self.coefficients.mass.keys()):
                elementQuadratureDict[('r', ci)] = Quadrature.SimplexLobattoQuadrature(self.nSpace_global, 1)
            for I in self.coefficients.elementIntegralKeys:
                elementQuadratureDict[('stab',) + I[1:]] = Quadrature.SimplexLobattoQuadrature(self.nSpace_global, 1)
        elementBoundaryQuadratureDict = {}
        if isinstance(elementBoundaryQuadrature, dict):  # set terms manually
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
        # mwf include tag telling me which indices are which quadrature rule?
        (self.elementQuadraturePoints, self.elementQuadratureWeights,
         self.elementQuadratureRuleIndeces) = Quadrature.buildUnion(elementQuadratureDict)
        self.nQuadraturePoints_element = self.elementQuadraturePoints.shape[0]
        self.nQuadraturePoints_global = self.nQuadraturePoints_element * self.mesh.nElements_global
        #
        # Repeat the same thing for the element boundary quadrature
        #
        (self.elementBoundaryQuadraturePoints,
         self.elementBoundaryQuadratureWeights,
         self.elementBoundaryQuadratureRuleIndeces) = Quadrature.buildUnion(elementBoundaryQuadratureDict)
        self.nElementBoundaryQuadraturePoints_elementBoundary = self.elementBoundaryQuadraturePoints.shape[0]
        self.nElementBoundaryQuadraturePoints_global = (self.mesh.nElements_global *
                                                        self.mesh.nElementBoundaries_element *
                                                        self.nElementBoundaryQuadraturePoints_elementBoundary)
        #
        # storage dictionaries
        self.scalars_element = set()
        #
        # simplified allocations for test==trial and also check if space is mixed or not
        #
        self.q = {}
        self.ebq = {}
        self.ebq_global = {}
        self.ebqe = {}
        self.phi_ip = {}
        self.edge_based_cfl = np.zeros(self.u[0].dof.shape)
        # mesh
        self.q['x'] = np.zeros((self.mesh.nElements_global, self.nQuadraturePoints_element, 3), 'd')
        self.ebqe['x'] = np.zeros((self.mesh.nExteriorElementBoundaries_global, self.nElementBoundaryQuadraturePoints_elementBoundary, 3), 'd')
        self.q[('u', 0)] = np.zeros((self.mesh.nElements_global, self.nQuadraturePoints_element), 'd')
        self.q[('dV_u', 0)] = (1.0/self.mesh.nElements_global) * np.ones((self.mesh.nElements_global, self.nQuadraturePoints_element), 'd')
        self.q[('grad(u)', 0)] = np.zeros((self.mesh.nElements_global, self.nQuadraturePoints_element, self.nSpace_global), 'd')
        self.q[('m_last', 0)] = np.zeros((self.mesh.nElements_global, self.nQuadraturePoints_element), 'd')
        self.q[('mt', 0)] = np.zeros((self.mesh.nElements_global, self.nQuadraturePoints_element), 'd')
        self.q['dV'] = np.zeros((self.mesh.nElements_global, self.nQuadraturePoints_element), 'd')
        self.q['dV_last'] = -1000 * np.ones((self.mesh.nElements_global, self.nQuadraturePoints_element), 'd')
        self.q[('m_tmp', 0)] = self.q[('u', 0)].copy()
        self.q[('m', 0)] = self.q[('m_tmp', 0)]
        self.q[('cfl', 0)] = np.zeros((self.mesh.nElements_global, self.nQuadraturePoints_element), 'd')
        self.q[('numDiff', 0, 0)] = np.zeros((self.mesh.nElements_global, self.nQuadraturePoints_element), 'd')
        ###################################################
        self.q[('a',0,0)] = np.zeros((self.mesh.nElements_global,self.nQuadraturePoints_element,self.coefficients.sdInfo[(0,0)][0][-1]),'d')
        nd = self.coefficients.nd      
        self.q[('r',0)] = np.zeros((self.mesh.nElements_global,self.nQuadraturePoints_element),'d')
        # Attach `q` to `coefficients`
        self.coefficients.q_a_mod = self.q[('a',0,0)] 
        ###################################################
        #self.calculateQuadrature()
        self.ebqe[('u', 0)] = np.zeros((self.mesh.nExteriorElementBoundaries_global, self.nElementBoundaryQuadraturePoints_elementBoundary), 'd')
        self.ebqe[('grad(u)', 0)] = np.zeros((self.mesh.nExteriorElementBoundaries_global,
                                                 self.nElementBoundaryQuadraturePoints_elementBoundary, self.nSpace_global), 'd')
        self.ebqe[('advectiveFlux_bc_flag', 0)] = np.zeros(
            (self.mesh.nExteriorElementBoundaries_global, self.nElementBoundaryQuadraturePoints_elementBoundary), 'i')
        self.ebqe[('advectiveFlux_bc', 0)] = np.zeros((self.mesh.nExteriorElementBoundaries_global, self.nElementBoundaryQuadraturePoints_elementBoundary), 'd')
        self.ebqe[('advectiveFlux', 0)] = np.zeros((self.mesh.nExteriorElementBoundaries_global, self.nElementBoundaryQuadraturePoints_elementBoundary), 'd')
        #######################################################
        self.ebqe[('a',0,0)] = np.zeros((self.mesh.nExteriorElementBoundaries_global,self.nElementBoundaryQuadraturePoints_elementBoundary,self.coefficients.sdInfo[(0,0)][0][-1]),'d')
        self.ebqe[('df',0,0)] = np.zeros((self.mesh.nExteriorElementBoundaries_global,self.nElementBoundaryQuadraturePoints_elementBoundary,self.nSpace_global),'d')
        self.ebqe[('r',0)] = np.zeros((self.mesh.nExteriorElementBoundaries_global,self.nElementBoundaryQuadraturePoints_elementBoundary),'d')
        self.ebqe['penalty'] = np.zeros((self.mesh.nExteriorElementBoundaries_global,self.nElementBoundaryQuadraturePoints_elementBoundary),'d')
        self.ebqe[('diffusiveFlux_bc_flag',0,0)] = np.zeros((self.mesh.nExteriorElementBoundaries_global,self.nElementBoundaryQuadraturePoints_elementBoundary),'i')
        self.ebqe[('diffusiveFlux_bc',0,0)] = np.zeros((self.mesh.nExteriorElementBoundaries_global,self.nElementBoundaryQuadraturePoints_elementBoundary),'d')
        
        self.points_elementBoundaryQuadrature = set()
        self.scalars_elementBoundaryQuadrature = set([('u', ci) for ci in range(self.nc)])
        self.vectors_elementBoundaryQuadrature = set()
        self.tensors_elementBoundaryQuadrature = set()
        self.inflowBoundaryBC = {}
        self.inflowBoundaryBC_values = {}
        self.inflowFlux = {}
        for cj in range(self.nc):
            self.inflowBoundaryBC[cj] = np.zeros((self.mesh.nExteriorElementBoundaries_global,), 'i')
            self.inflowBoundaryBC_values[cj] = np.zeros((self.mesh.nExteriorElementBoundaries_global, self.nDOF_trial_element[cj]), 'd')
            self.inflowFlux[cj] = np.zeros((self.mesh.nExteriorElementBoundaries_global, self.nElementBoundaryQuadraturePoints_elementBoundary), 'd')
        self.internalNodes = set(range(self.mesh.nNodes_global))

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
            
        del self.internalNodes
        self.internalNodes = None
        

        logEvent("Updating local to global mappings", 2)
        self.updateLocal2Global()
        logEvent("Building time integration object", 2)
        # mwf for interpolating subgrid error for gradients etc
        if self.stabilization and self.stabilization.usesGradientStabilization:
            self.timeIntegration = TimeIntegrationClass(self, integrateInterpolationPoints=True)
        else:
            self.timeIntegration = TimeIntegrationClass(self)

        if options is not None:
            self.timeIntegration.setFromOptions(options)
        logEvent(memory("TimeIntegration", "OneLevelTransport"), level=4)
        logEvent("Calculating numerical quadrature formulas", 2)
        self.calculateQuadrature()
        ############################################

        nd = self.coefficients.nd
        a_rowptr = self.coefficients.sdInfo[(0, 0)][0]
        a_colind = self.coefficients.sdInfo[(0, 0)][1]

        # for eN in range(self.mesh.nElements_global):
        #     for k in range(self.nQuadraturePoints_element):
        #         x = self.q['x'][eN, k, :]
        #         a_full = self.coefficients.aOfX[0](x)
        #         a_val = np.array([a_full[row, col] for row, col in zip(a_rowptr[:-1], a_colind)])

        #         if a_val.shape == (a_rowptr[-1],):
        #             self.q[('a', 0, 0)][eN, k, :] = a_val
        #         else:
        #             raise ValueError(f"Unexpected shape {a_val.shape} for a_val. Expected ({a_rowptr[-1]},)")
        # self.ebqe[('a',0,0)] = np.zeros((,,self.coefficients.sdInfo[(0,0)][0][-1]),'d')
        
        # #########Dealing with boundary diffusion coefficient       
        
        # for ebNE in range(self.mesh.nExteriorElementBoundaries_global):
        #     for kb in range(self.nElementBoundaryQuadraturePoints_elementBoundary):
        #         x = self.ebqe['x'][ebNE, kb, :]
        #         a_full = self.coefficients.aOfX[0](x)
        #         a_val = np.array([a_full[row, col] for row, col in zip(a_rowptr[:-1], a_colind)])
        #         if a_val.shape == (a_rowptr[-1],):
        #             self.ebqe[('a', 0, 0)][ebNE, kb, :] = a_val
        #         else:
        #             raise ValueError(f"Unexpected shape {a_val.shape} for a_val. Expected ({a_rowptr[-1]},)")




        self.setupFieldStrides()

        comm = Comm.get()
        self.comm = comm
        if comm.size() > 1:
            assert numericalFluxType is not None and numericalFluxType.useWeakDirichletConditions, "You must use a numerical flux to apply weak boundary conditions for parallel runs"

        logEvent(memory("stride+offset", "OneLevelTransport"), level=4)
        if numericalFluxType is not None:
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
        # set penalty terms
        # cek todo move into numerical flux initialization
        if 'penalty' in self.ebq_global:
            for ebN in range(self.mesh.nElementBoundaries_global):
                for k in range(self.nElementBoundaryQuadraturePoints_elementBoundary):
                    self.ebq_global['penalty'][ebN, k] = self.numericalFlux.penalty_constant/(self.mesh.elementBoundaryDiametersArray[ebN]**self.numericalFlux.penalty_power)
        # penalty term
        # cek move  to Numerical flux initialization
        if 'penalty' in self.ebqe:
            for ebNE in range(self.mesh.nExteriorElementBoundaries_global):
                ebN = self.mesh.exteriorElementBoundariesArray[ebNE]
                for k in range(self.nElementBoundaryQuadraturePoints_elementBoundary):
                    self.ebqe['penalty'][ebNE, k] = self.numericalFlux.penalty_constant/self.mesh.elementBoundaryDiametersArray[ebN]**self.numericalFlux.penalty_power
        logEvent(memory("numericalFlux", "OneLevelTransport"), level=4)

        self.elementEffectiveDiametersArray = self.mesh.elementInnerDiametersArray
        # use post processing tools to get conservative fluxes, None by default
        from proteus import PostProcessingTools
        self.velocityPostProcessor = PostProcessingTools.VelocityPostProcessingChooser(self)
        logEvent(memory("velocity postprocessor", "OneLevelTransport"), level=4)
        # helper for writing out data storage
        from proteus import Archiver
        self.elementQuadratureDictionaryWriter = Archiver.XdmfWriter()
        self.elementBoundaryQuadratureDictionaryWriter = Archiver.XdmfWriter()
        self.exteriorElementBoundaryQuadratureDictionaryWriter = Archiver.XdmfWriter()
        # TODO get rid of this
        for ci, fbcObject in list(self.fluxBoundaryConditionsObjectsDict.items()):
            self.ebqe[('advectiveFlux_bc_flag', ci)] = np.zeros(self.ebqe[('advectiveFlux_bc', ci)].shape, 'i')
            for t, g in list(fbcObject.advectiveFluxBoundaryConditionsDict.items()):
                if ci in self.coefficients.advection:
                    self.ebqe[('advectiveFlux_bc', ci)][t[0], t[1]] = g(self.ebqe[('x')][t[0], t[1]], self.timeIntegration.t)
                    self.ebqe[('advectiveFlux_bc_flag', ci)][t[0], t[1]] = 1

            # Initialize the diffusive flux boundary condition flags
            #self.ebqe[('diffusiveFlux_bc_flag', ci)] = np.zeros(self.ebqe[('diffusiveFlux_bc', ci)].shape, 'i')
    
        for ck,diffusiveFluxBoundaryConditionsDict in fbcObject.diffusiveFluxBoundaryConditionsDictDict.items():
            self.ebqe[('diffusiveFlux_bc_flag',ck,ci)] = np.zeros(self.ebqe[('diffusiveFlux_bc',ck,ci)].shape,'i')
            for t,g in diffusiveFluxBoundaryConditionsDict.items():
                self.ebqe[('diffusiveFlux_bc',ck,ci)][t[0],t[1]] = g(self.ebqe[('x')][t[0],t[1]],self.timeIntegration.t)
                self.ebqe[('diffusiveFlux_bc_flag',ck,ci)][t[0],t[1]] = 1

        
        self.numericalFlux.setDirichletValues(self.ebqe)
        if hasattr(self.numericalFlux, 'setDirichletValues'):
            self.numericalFlux.setDirichletValues(self.ebqe)
        if not hasattr(self.numericalFlux, 'isDOFBoundary'):
            self.numericalFlux.isDOFBoundary = {0: np.zeros(self.ebqe[('u', 0)].shape, 'i')}
        if not hasattr(self.numericalFlux, 'ebqe'):
            self.numericalFlux.ebqe = {('u', 0): np.zeros(self.ebqe[('u', 0)].shape, 'd')}

        #TODO how to handle redistancing calls for calculateCoefficients,calculateElementResidual etc
        self.globalResidualDummy = None
        compKernelFlag = 0
        self.adr = cTADR_base(self.nSpace_global,
                             self.nQuadraturePoints_element,
                             self.u[0].femSpace.elementMaps.localFunctionSpace.dim,
                             self.u[0].femSpace.referenceFiniteElement.localFunctionSpace.dim,
                             self.testSpace[0].referenceFiniteElement.localFunctionSpace.dim,
                             self.nElementBoundaryQuadraturePoints_elementBoundary,
                             compKernelFlag)

        self.forceStrongConditions = True #False
        if self.forceStrongConditions:
            self.dirichletConditionsForceDOF = DOFBoundaryConditions(self.u[0].femSpace, dofBoundaryConditionsSetterDict[0], weakDirichletConditions=False)

        if self.movingDomain:
            self.MOVING_DOMAIN = 1.0
        else:
            self.MOVING_DOMAIN = 0.0
        if self.mesh.nodeVelocityArray is None:
            self.mesh.nodeVelocityArray = np.zeros(self.mesh.nodeArray.shape, 'd')

        # Stuff added by mql.
        # Some ASSERTS to restrict the combination of the methods
        if self.coefficients.STABILIZATION_TYPE > 1:
            assert self.timeIntegration.isSSP == True, "If STABILIZATION_TYPE>1, use RKEV timeIntegration within TADR model"
            cond = 'levelNonlinearSolver' in dir(options) and (options.levelNonlinearSolver ==
                                                               ExplicitLumpedMassMatrix or options.levelNonlinearSolver == ExplicitConsistentMassMatrixForVOF)
            assert cond, "If STABILIZATION_TYPE>1, use levelNonlinearSolver=ExplicitLumpedMassMatrix or ExplicitConsistentMassMatrixForVOF"
        if 'levelNonlinearSolver' in dir(options) and options.levelNonlinearSolver == ExplicitLumpedMassMatrix:
            assert self.coefficients.LUMPED_MASS_MATRIX, "If levelNonlinearSolver=ExplicitLumpedMassMatrix, use LUMPED_MASS_MATRIX=True"
        if self.coefficients.LUMPED_MASS_MATRIX == True:
            cond = 'levelNonlinearSolver' in dir(options) and options.levelNonlinearSolver == ExplicitLumpedMassMatrix
            assert cond, "Use levelNonlinearSolver=ExplicitLumpedMassMatrix when the mass matrix is lumped"
        if self.coefficients.FCT == True:
            cond = self.coefficients.STABILIZATION_TYPE > 1, "Use FCT just with STABILIZATION_TYPE>1; i.e., edge based stabilization"
        if self.coefficients.STABILIZATION_TYPE==1:
            cond = 'levelNonlinearSolver' in dir(options) and  options.levelNonlinearSolver == TwoStageNewton
            assert cond, "If STABILIZATION_TYPE==1, use levelNonlinearSolver=TwoStageNewton"
        if self.coefficients.STABILIZATION_TYPE==1:
            self.useTwoStageNewton = True
            assert isinstance(self.timeIntegration, TimeIntegration.BackwardEuler_cfl), "If STABILIZATION_TYPE=1, use BackwardEuler_cfl"
            assert options.levelNonlinearSolver == TwoStageNewton, "If STABILIZATION_TYPE=1, use levelNonlinearSolver=TwoStageNewton"
        assert self.coefficients.ENTROPY_TYPE in [0,1], "Set ENTROPY_TYPE={0,1}"
        assert self.coefficients.STABILIZATION_TYPE in [-1,0,1,2,3,4], "Set STABILIZATION_TYPE={-1,0,1,2,3,4}"
        if self.coefficients.STABILIZATION_TYPE==4:
            assert self.coefficients.FCT==True, "If STABILIZATION_TYPE=4, use FCT=True"
            
        # mql. Allow the user to provide functions to define the velocity field
        self.hasVelocityFieldAsFunction = False
        if ('velocityFieldAsFunction') in dir(options):
            self.velocityFieldAsFunction = options.velocityFieldAsFunction
            self.hasVelocityFieldAsFunction = True

        # For edge based methods
        self.ML = None  # lumped mass matrix
        self.MC_global = None  # consistent mass matrix
        self.uLow = None
        self.dt_times_dC_minus_dL = None
        self.dLow = None
        self.min_u_bc = None
        self.max_u_bc = None
        self.quantDOFs = np.zeros(self.u[0].dof.shape, 'd')

        # For Taylor Galerkin methods
        self.stage = 1
        self.auxTaylorGalerkinFlag = 1        
        self.uTilde_dof = np.zeros(self.u[0].dof.shape)
        self.degree_polynomial = 1
        try:
            self.degree_polynomial = self.u[0].femSpace.order
        except:
            pass

    # def calculateQuadrature(self):
    #     self.coefficients.initializeElementQuadrature(self.timeIntegration.t, self.q)
                
    def FCTStep(self):
        rowptr, colind, MassMatrix = self.MC_global.getCSRrepresentation()
        rowptr, colind, MassMatrix = self.MC_global.getCSRrepresentation()
        limited_solution = np.zeros(self.u[0].dof.shape)

        argsDict = cArgumentsDict.ArgumentsDict()
        argsDict["dt"] = self.timeIntegration.dt
        argsDict["NNZ"] = self.nnz
        argsDict["numDOFs"] = len(rowptr) - 1
        argsDict["lumped_mass_matrix"] = self.ML
        argsDict["soln"] = self.u_dof_old
        argsDict["solH"] = self.timeIntegration.u
        argsDict["uLow"] = self.uLow
        argsDict["dLow"] = self.dLow
        argsDict["limited_solution"] = limited_solution
        argsDict["csrRowIndeces_DofLoops"] = rowptr
        argsDict["csrColumnOffsets_DofLoops"] = colind
        argsDict["MassMatrix"] = MassMatrix
        argsDict["dt_times_dH_minus_dL"] = self.dt_times_dC_minus_dL
        argsDict["min_u_bc"] = self.min_u_bc
        argsDict["max_u_bc"] = self.max_u_bc
        argsDict["LUMPED_MASS_MATRIX"] = self.coefficients.LUMPED_MASS_MATRIX
        argsDict["STABILIZATION_TYPE"] = self.coefficients.STABILIZATION_TYPE
        self.adr.FCTStep(argsDict)
        #self.timeIntegration.u[:] = limited_solution
        fromFreeToGlobal=0 #direction copying
        cfemIntegrals.copyBetweenFreeUnknownsAndGlobalUnknowns(fromFreeToGlobal,
                                                               self.offset[0],
                                                               self.stride[0],
                                                               self.dirichletConditions[0].global2freeGlobal_global_dofs,
                                                               self.dirichletConditions[0].global2freeGlobal_free_dofs,
                                                               self.timeIntegration.u,
                                                               limited_solution)
    def updateVelocityFieldAsFunction(self):
        import pdb
        X = {0: self.q[('x')][:, :, 0],
             1: self.q[('x')][:, :, 1],
             2: self.q[('x')][:, :, 2]}
        #pdb.set_trace()
        t = self.timeIntegration.t
        self.coefficients.q_v[..., 0] = self.velocityFieldAsFunction[0](X, t)
        if (self.nSpace_global == 2):
            self.coefficients.q_v[..., 1] = self.velocityFieldAsFunction[1](X, t)
        if (self.nSpace_global == 3):
            self.coefficients.q_v[..., 2] = self.velocityFieldAsFunction[2](X, t)

        # BOUNDARY
        ebqe_X = {0: self.ebqe['x'][:, :, 0],
                  1: self.ebqe['x'][:, :, 1],
                  2: self.ebqe['x'][:, :, 2]}
        self.coefficients.ebqe_v[..., 0] = self.velocityFieldAsFunction[0](ebqe_X, t)
        if (self.nSpace_global == 2):
            self.coefficients.ebqe_v[..., 1] = self.velocityFieldAsFunction[1](ebqe_X, t)
        if (self.nSpace_global == 3):
            self.coefficients.ebqe_v[..., 2] = self.velocityFieldAsFunction[2](ebqe_X, t)
            
    def calculateCoefficients(self):
        pass

    def calculateElementResidual(self):
        if self.globalResidualDummy is not None:
            self.getResidual(self.u[0].dof, self.globalResidualDummy)

    def getMassMatrix(self):
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
        self.u[0].femSpace.elementMaps.getJacobianValues(self.elementQuadraturePoints,
                                                         self.q['J'],
                                                         self.q['inverse(J)'],
                                                         self.q['det(J)'])
        self.q['abs(det(J))'] = np.abs(self.q['det(J)'])
        # SHAPE FUNCTIONS
        self.q[('w',0)] = np.zeros((self.mesh.nElements_global,
                                    self.nQuadraturePoints_element,
                                    self.nDOF_test_element[0]),
                                   'd')
        self.q[('w*dV_m',0)] = self.q[('w',0)].copy()
        self.u[0].femSpace.getBasisValues(self.elementQuadraturePoints, self.q[('w',0)])
        cfemIntegrals.calculateWeightedShape(self.elementQuadratureWeights[('u',0)],
                                             self.q['abs(det(J))'],
                                             self.q[('w',0)],
                                             self.q[('w*dV_m',0)])
        # assume a linear mass term
        dm = np.ones(self.q[('u', 0)].shape, 'd')
        elementMassMatrix = np.zeros((self.mesh.nElements_global,
                                      self.nDOF_test_element[0],
                                      self.nDOF_trial_element[0]), 'd')
        cfemIntegrals.updateMassJacobian_weak_lowmem(dm,
                                                     self.q[('w', 0)],
                                                     self.q[('w*dV_m', 0)],
                                                     elementMassMatrix)
        self.MC_a = self.nzval.copy()
        self.MC_global = SparseMat(self.nFreeDOF_global[0],
                                   self.nFreeDOF_global[0],
                                   self.nnz,
                                   self.MC_a,
                                   self.colind,
                                   self.rowptr)
        cfemIntegrals.zeroJacobian_CSR(self.nnz, self.MC_global)
        cfemIntegrals.updateGlobalJacobianFromElementJacobian_CSR(self.l2g[0]['nFreeDOF'],
                                                                  self.l2g[0]['freeLocal'],
                                                                  self.l2g[0]['nFreeDOF'],
                                                                  self.l2g[0]['freeLocal'],
                                                                  self.csrRowIndeces[(0, 0)],
                                                                  self.csrColumnOffsets[(0, 0)],
                                                                  elementMassMatrix,
                                                                  self.MC_global)

        
        self.ML = np.zeros((self.nFreeDOF_global[0],), 'd')
        for i in range(self.nFreeDOF_global[0]):
            self.ML[i] = self.MC_a[self.rowptr[i]:self.rowptr[i + 1]].sum()
        np.testing.assert_almost_equal(self.ML.sum(),
                                       self.mesh.volume,
                                       err_msg="Trace of lumped mass matrix should be the domain volume", verbose=True)
        
    def initVectors(self):
        self.u_dof_old = np.copy(self.u[0].dof)
        rowptr, colind, MC = self.MC_global.getCSRrepresentation()
        # This is dummy. I just care about the csr structure of the sparse matrix
        self.dt_times_dC_minus_dL = np.zeros(MC.shape, 'd')
        self.uLow = np.zeros(self.u[0].dof.shape, 'd')
        self.dLow = np.zeros(MC.shape, 'd')
        
    def getResidual(self, u, r):
        import copy
        """
        Calculate the element residuals and add in to the global residual
        """

        if self.MC_global is None:
            self.getMassMatrix()
            self.initVectors()

        # Reset some vectors for FCT
        self.min_u_bc = np.zeros(self.u[0].dof.shape, 'd') + 1E10
        self.max_u_bc = np.zeros(self.u[0].dof.shape, 'd') - 1E10
        self.dt_times_dC_minus_dL.fill(0.0)
        self.uLow.fill(0.0)
        self.dLow.fill(0.0)
        
        r.fill(0.0)
        # Load the unknowns into the finite element dof
        self.timeIntegration.calculateCoefs()
        self.timeIntegration.calculateU(u)
        self.setUnknowns(self.timeIntegration.u)
        # cek can put in logic to skip if BC's don't depend on t or u
        # cek todo: faster implementation of boundary conditions
        # Dirichlet boundary conditions
        # if hasattr(self.numericalFlux,'setDirichletValues'):
        if (self.stage!=2):
            self.numericalFlux.setDirichletValues(self.ebqe)
        # flux boundary conditions
        for t, g in list(self.fluxBoundaryConditionsObjectsDict[0].advectiveFluxBoundaryConditionsDict.items()):
            #print(f"Setting advective flux bc at boundary {t[0]}, point {t[1]}")
            self.ebqe[('advectiveFlux_bc', 0)][t[0], t[1]] = g(self.ebqe[('x')][t[0], t[1]], self.timeIntegration.t)
            self.ebqe[('advectiveFlux_bc_flag', 0)][t[0], t[1]] = 1
            #print(f"  advective flux bc value: {self.ebqe[('advectiveFlux_bc', 0)][t[0], t[1]]}")


        # Flux boundary conditions for diffusive terms
        for ck, diffusiveFluxBoundaryConditionsDict in self.fluxBoundaryConditionsObjectsDict[0].diffusiveFluxBoundaryConditionsDictDict.items():
            #self.ebqe[('diffusiveFlux_bc_flag', ck, 0)] = np.zeros(self.ebqe[('diffusiveFlux_bc', ck, 0)].shape, 'i')
            for t, g in diffusiveFluxBoundaryConditionsDict.items():
                #logEvent(f"Setting diffusive flux bc at boundary {t[0]}, point {t[1]}")
                self.ebqe[('diffusiveFlux_bc', ck, 0)][t[0], t[1]] = g(self.ebqe[('x')][t[0], t[1]], self.timeIntegration.t)
                self.ebqe[('diffusiveFlux_bc_flag', ck, 0)][t[0], t[1]] = 1
                #logEvent(f"  diffusive flux bc value: {self.ebqe[('diffusiveFlux_bc', ck, 0)][t[0], t[1]]}")


        if self.forceStrongConditions:
            for dofN, g in list(self.dirichletConditionsForceDOF.DOFBoundaryConditionsDict.items()):
                self.u[0].dof[dofN] = g(self.dirichletConditionsForceDOF.DOFBoundaryPointDict[dofN], self.timeIntegration.t)

        if (self.stage==2 and self.auxTaylorGalerkinFlag==1):
            self.uTilde_dof[:] = self.u[0].dof
            self.auxTaylorGalerkinFlag=0

        argsDict = cArgumentsDict.ArgumentsDict()
        argsDict["dt"] = self.timeIntegration.dt
        argsDict["mesh_trial_ref"] = self.u[0].femSpace.elementMaps.psi
        argsDict["mesh_grad_trial_ref"] = self.u[0].femSpace.elementMaps.grad_psi
        argsDict["mesh_dof"] = self.mesh.nodeArray
        argsDict["mesh_velocity_dof"] = self.mesh.nodeVelocityArray
        argsDict["MOVING_DOMAIN"] = self.MOVING_DOMAIN
        argsDict["mesh_l2g"] = self.mesh.elementNodesArray
        argsDict["dV_ref"] = self.elementQuadratureWeights[('u', 0)]
        argsDict["u_trial_ref"] = self.u[0].femSpace.psi
        argsDict["u_grad_trial_ref"] = self.u[0].femSpace.grad_psi
        argsDict["u_test_ref"] = self.u[0].femSpace.psi
        argsDict["u_grad_test_ref"] = self.u[0].femSpace.grad_psi
        argsDict["mesh_trial_trace_ref"] = self.u[0].femSpace.elementMaps.psi_trace
        argsDict["mesh_grad_trial_trace_ref"] = self.u[0].femSpace.elementMaps.grad_psi_trace
        argsDict["dS_ref"] = self.elementBoundaryQuadratureWeights[('u', 0)]
        argsDict["u_trial_trace_ref"] = self.u[0].femSpace.psi_trace
        argsDict["u_grad_trial_trace_ref"] = self.u[0].femSpace.grad_psi_trace
        argsDict["u_test_trace_ref"] = self.u[0].femSpace.psi_trace
        argsDict["u_grad_test_trace_ref"] = self.u[0].femSpace.grad_psi_trace
        argsDict["normal_ref"] = self.u[0].femSpace.elementMaps.boundaryNormals
        argsDict["boundaryJac_ref"] = self.u[0].femSpace.elementMaps.boundaryJacobians
        argsDict["nElements_global"] = self.mesh.nElements_global
        argsDict["useMetrics"] = self.coefficients.useMetrics
        argsDict["alphaBDF"] = self.timeIntegration.alpha_bdf
        argsDict["lag_shockCapturing"] = self.shockCapturing.lag
        argsDict["shockCapturingDiffusion"] = float(self.shockCapturing.shockCapturingFactor)
        argsDict["sc_uref"] = self.coefficients.sc_uref
        argsDict["sc_alpha"] = self.coefficients.sc_beta
        argsDict["u_l2g"] = self.u[0].femSpace.dofMap.l2g
        argsDict["r_l2g"] = self.l2g[0]['freeGlobal']
        argsDict["elementDiameter"] = self.mesh.elementDiametersArray
        argsDict["degree_polynomial"] = float(self.degree_polynomial)
        argsDict["u_dof"] = self.u[0].dof
        argsDict["u_dof_old"] = self.u_dof_old
        argsDict["velocity"] = self.coefficients.q_v
        argsDict["q_m"] = self.timeIntegration.m_tmp[0]
        argsDict["q_u"] = self.q[('u', 0)]
        ###########################################
        #argsDict["q_a"] = self.q[('a',0,0)]
        argsDict["q_a"] = self.coefficients.q_a_mod #q[('a',0,0)]
        argsDict["q_r"] = self.q[('r',0)]

        argsDict["ebq_a"] = self.ebqe[('a',0,0)]
        argsDict["ebq_r"] = self.ebqe[('r',0)]     

        ###########################################
        argsDict["q_m_betaBDF"] = self.timeIntegration.beta_bdf[0]
        argsDict["q_dV"] = self.q['dV']
        argsDict["q_dV_last"] = self.q['dV_last']
        argsDict["cfl"] = self.q[('cfl', 0)]
        argsDict["edge_based_cfl"] = self.edge_based_cfl
        argsDict["q_numDiff_u"] = self.shockCapturing.numDiff[0]
        argsDict["q_numDiff_u_last"] = self.shockCapturing.numDiff_last[0]
        argsDict["offset_u"] = self.offset[0]
        argsDict["stride_u"] = self.stride[0]
        argsDict["globalResidual"] = r
        argsDict["nExteriorElementBoundaries_global"] = self.mesh.nExteriorElementBoundaries_global
        argsDict["exteriorElementBoundariesArray"] = self.mesh.exteriorElementBoundariesArray
        argsDict["elementBoundaryElementsArray"] = self.mesh.elementBoundaryElementsArray
        argsDict["elementBoundariesArray"] = self.mesh.elementBoundariesArray
        argsDict["elementBoundaryLocalElementBoundariesArray"] = self.mesh.elementBoundaryLocalElementBoundariesArray
        argsDict["ebqe_velocity_ext"] = self.coefficients.ebqe_v
        argsDict["isDOFBoundary_u"] = self.numericalFlux.isDOFBoundary[0]
        argsDict["ebqe_bc_u_ext"] = self.numericalFlux.ebqe[('u', 0)]
        argsDict["isFluxBoundary_u"] = self.ebqe[('advectiveFlux_bc_flag', 0)]
        argsDict["ebqe_bc_advectiveFlux_u_ext"] = self.ebqe[('advectiveFlux_bc',0)]
        argsDict["ebqe_bc_flux_u_ext"] = self.ebqe[('advectiveFlux_bc', 0)]
        argsDict["isDiffusiveFluxBoundary_u"] = self.ebqe[('diffusiveFlux_bc_flag', 0, 0)]
        argsDict["ebqe_bc_diffusiveFlux_u_ext"] = self.ebqe[('diffusiveFlux_bc', 0, 0)]
        


        #argsDict["ebqe_bc_flux_u_ext"] = self.ebqe[('diffusiveFlux_bc',0,0)]
        argsDict["epsFact"] = self.coefficients.epsFact
        argsDict["ebqe_u"] = self.ebqe[('u', 0)]
        argsDict["ebqe_flux"] = self.ebqe[('advectiveFlux', 0)]
        argsDict["stage"] = self.stage
        argsDict["uTilde_dof"] = self.uTilde_dof
        argsDict["cE"] = self.coefficients.cE
        argsDict["cMax"] = self.coefficients.cMax
        argsDict["cK"] = self.coefficients.cK
        argsDict["uL"] = self.coefficients.uL
        argsDict["uR"] = self.coefficients.uR
        argsDict["numDOFs"] = len(self.rowptr) - 1
        argsDict["NNZ"] = self.nnz
        argsDict["csrRowIndeces_DofLoops"] = self.rowptr
        argsDict["csrColumnOffsets_DofLoops"] = self.colind
        argsDict["csrRowIndeces_CellLoops"] = self.csrRowIndeces[(0, 0)]
        argsDict["csrColumnOffsets_CellLoops"] = self.csrColumnOffsets[(0, 0)]
        argsDict["csrColumnOffsets_eb_CellLoops"] = self.csrColumnOffsets_eb[(0, 0)]
        argsDict["ML"] = self.ML
        argsDict["LUMPED_MASS_MATRIX"] = self.coefficients.LUMPED_MASS_MATRIX
        argsDict["STABILIZATION_TYPE"] = self.coefficients.STABILIZATION_TYPE
        argsDict["ENTROPY_TYPE"] = self.coefficients.ENTROPY_TYPE
        argsDict["uLow"] = self.uLow
        argsDict["dLow"] = self.dLow
        argsDict["dt_times_dH_minus_dL"] = self.dt_times_dC_minus_dL
        argsDict["min_u_bc"] = self.min_u_bc
        argsDict["max_u_bc"] = self.max_u_bc
        argsDict["quantDOFs"] = self.quantDOFs
        argsDict["physicalDiffusion"] = self.coefficients.physicalDiffusion
        #argsDict["D"] = self.coefficients.DTypes
        argsDict["isDiffusiveFluxBoundary_u"] = self.ebqe[('diffusiveFlux_bc_flag',0,0)]
        argsDict["isAdvectiveFluxBoundary_u"] = self.ebqe[('advectiveFlux_bc_flag',0)]
        argsDict["ebqe_bc_advectiveFlux_u_ext"] = self.ebqe[('advectiveFlux_bc',0)]
        argsDict["ebqe_penalty_ext"] = self.ebqe['penalty']
        argsDict["eb_adjoint_sigma"] = self.numericalFlux.boundaryAdjoint_sigma
        #print("sigma",self.numericalFlux.boundaryAdjoint_sigma)
        ###################################################################################
        sdInfo = self.coefficients.sdInfo
    
        argsDict["a_rowptr"] = sdInfo[(0, 0)][0]
        argsDict["a_colind"] = sdInfo[(0, 0)][1]
        
        argsDict["alpha_L"] = self.coefficients.alpha_L  # Longitudinal dispersion coefficient
        argsDict["alpha_T"] = self.coefficients.alpha_T  # Transverse dispersion coefficient
        argsDict["Dm"] = self.coefficients.Dm           # Molecular diffusion coefficient



        #argsDict["a_rowptr"] = self.coefficients.sdInfo[(0,0)][0]
        #argsDict["a_colind"] = self.coefficients.sdInfo[(0,0)][1]
        self.adr.calculateResidual(argsDict)

        


        

        if self.forceStrongConditions:
            for dofN, g in list(self.dirichletConditionsForceDOF.DOFBoundaryConditionsDict.items()):
                r[dofN] = 0

        if (self.auxiliaryCallCalculateResidual == False):
            edge_based_cflMax = globalMax(self.edge_based_cfl.max()) * self.timeIntegration.dt
            cell_based_cflMax = globalMax(self.q[('cfl', 0)].max()) * self.timeIntegration.dt
            logEvent("...   Current dt = " + str(self.timeIntegration.dt), level=4)
            logEvent("...   Maximum Cell Based CFL = " + str(cell_based_cflMax), level=2)
            logEvent("...   Maximum Edge Based CFL = " + str(edge_based_cflMax), level=2)

        if self.stabilization:
            self.stabilization.accumulateSubgridMassHistory(self.q)
        logEvent("Global residual", level=9, data=r)
        if self.globalResidualDummy is None:
            self.globalResidualDummy = np.zeros(r.shape, 'd')

        #Debugging statements
        # print("Mesh details:")
        # print(f"Number of elements: {self.mesh.nElements_global}")
        # print(f"Number of exterior element boundaries: {self.mesh.nExteriorElementBoundaries_global}")

        # #For each boundary condition setting
        # for ebNE in range(self.mesh.nExteriorElementBoundaries_global):
        #     for kb in range(self.nElementBoundaryQuadraturePoints_elementBoundary):
        #         if self.ebqe['x'][ebNE, kb, 0] == 0:
        #             #self.ebqe[('advectiveFlux_bc_flag', 0)][ebNE, kb] = 4.0
        #             #self.ebqe[('u', 0)][ebNE, kb] += 1.0
        #             print(f"Boundary {ebNE}, point {kb}:")
        #             print(f"  Coordinates: {self.ebqe['x'][ebNE, kb, :]}")
        #             print(f"  u: {self.ebqe[('u', 0)][ebNE, kb]}")
        #             print(f"  grad(u): {self.ebqe[('grad(u)', 0)][ebNE, kb, :]}")
        #             print(f"  advective flux bc flag: {self.ebqe[('advectiveFlux_bc_flag', 0)][ebNE, kb]}")
        #             print(f"  advective flux bc: {self.ebqe[('advectiveFlux_bc', 0)][ebNE, kb]}")


    def getJacobian(self, jacobian):
        cfemIntegrals.zeroJacobian_CSR(self.nNonzerosInJacobian,
                                       jacobian)
        argsDict = cArgumentsDict.ArgumentsDict()
        argsDict["mesh_trial_ref"] = self.u[0].femSpace.elementMaps.psi
        argsDict["mesh_grad_trial_ref"] = self.u[0].femSpace.elementMaps.grad_psi
        argsDict["mesh_dof"] = self.mesh.nodeArray
        argsDict["mesh_velocity_dof"] = self.mesh.nodeVelocityArray
        argsDict["MOVING_DOMAIN"] = self.MOVING_DOMAIN
        argsDict["mesh_l2g"] = self.mesh.elementNodesArray
        argsDict["dV_ref"] = self.elementQuadratureWeights[('u', 0)]
        argsDict["u_trial_ref"] = self.u[0].femSpace.psi
        argsDict["u_grad_trial_ref"] = self.u[0].femSpace.grad_psi
        argsDict["u_test_ref"] = self.u[0].femSpace.psi
        argsDict["u_grad_test_ref"] = self.u[0].femSpace.grad_psi
        argsDict["mesh_trial_trace_ref"] = self.u[0].femSpace.elementMaps.psi_trace
        argsDict["mesh_grad_trial_trace_ref"] = self.u[0].femSpace.elementMaps.grad_psi_trace
        argsDict["dS_ref"] = self.elementBoundaryQuadratureWeights[('u', 0)]
        argsDict["u_trial_trace_ref"] = self.u[0].femSpace.psi_trace
        argsDict["u_grad_trial_trace_ref"] = self.u[0].femSpace.grad_psi_trace
        argsDict["u_test_trace_ref"] = self.u[0].femSpace.psi_trace
        argsDict["u_grad_test_trace_ref"] = self.u[0].femSpace.grad_psi_trace
        argsDict["normal_ref"] = self.u[0].femSpace.elementMaps.boundaryNormals
        argsDict["boundaryJac_ref"] = self.u[0].femSpace.elementMaps.boundaryJacobians
        argsDict["nElements_global"] = self.mesh.nElements_global
        argsDict["useMetrics"] = self.coefficients.useMetrics
        argsDict["alphaBDF"] = self.timeIntegration.alpha_bdf
        argsDict["lag_shockCapturing"] = self.shockCapturing.lag
        argsDict["shockCapturingDiffusion"] = float(self.shockCapturing.shockCapturingFactor)
        argsDict["u_l2g"] = self.u[0].femSpace.dofMap.l2g
        argsDict["r_l2g"] = self.l2g[0]['freeGlobal']
        argsDict["elementDiameter"] = self.mesh.elementDiametersArray
        argsDict["u_dof"] = self.u[0].dof
        argsDict["velocity"] = self.coefficients.q_v
        argsDict["q_m_betaBDF"] = self.timeIntegration.beta_bdf[0]
        argsDict["cfl"] = self.q[('cfl', 0)]
        argsDict["q_numDiff_u_last"] = self.shockCapturing.numDiff_last[0]
        argsDict["csrRowIndeces_u_u"] = self.csrRowIndeces[(0, 0)]
        argsDict["csrColumnOffsets_u_u"] = self.csrColumnOffsets[(0, 0)]
        argsDict["globalJacobian"] = jacobian.getCSRrepresentation()[2]
        argsDict["nExteriorElementBoundaries_global"] = self.mesh.nExteriorElementBoundaries_global
        argsDict["exteriorElementBoundariesArray"] = self.mesh.exteriorElementBoundariesArray
        argsDict["elementBoundaryElementsArray"] = self.mesh.elementBoundaryElementsArray
        argsDict["elementBoundaryLocalElementBoundariesArray"] = self.mesh.elementBoundaryLocalElementBoundariesArray
        argsDict["ebqe_velocity_ext"] = self.coefficients.ebqe_v
        argsDict["isDOFBoundary_u"] = self.numericalFlux.isDOFBoundary[0]
        argsDict["ebqe_bc_u_ext"] = self.numericalFlux.ebqe[('u', 0)]
        argsDict["isFluxBoundary_u"] = self.ebqe[('advectiveFlux_bc_flag', 0)]
        argsDict["ebqe_bc_flux_u_ext"] = self.ebqe[('advectiveFlux_bc', 0)]

        argsDict["isDiffusiveFluxBoundary_u"] = self.ebqe[('diffusiveFlux_bc_flag', 0, 0)]
        argsDict["ebqe_bc_diffusiveFlux_u_ext"] = self.ebqe[('diffusiveFlux_bc', 0, 0)]
        

        argsDict["csrColumnOffsets_eb_u_u"] = self.csrColumnOffsets_eb[(0, 0)]
        argsDict["STABILIZATION_TYPE"] = self.coefficients.STABILIZATION_TYPE
        argsDict["physicalDiffusion"] = self.coefficients.physicalDiffusion   
        argsDict["ebq_a"] = self.ebqe[('a',0,0)]
        #argsDict["D"] = self.coefficients.DTypes

        sdInfo = self.coefficients.sdInfo
    
        argsDict["a_rowptr"] = sdInfo[(0, 0)][0]
        argsDict["a_colind"] = sdInfo[(0, 0)][1]
        argsDict["q_a"] = self.coefficients.q_a_mod
        #argsDict["q_a"] = self.q[('a',0,0)]
        argsDict["eb_adjoint_sigma"] = self.numericalFlux.boundaryAdjoint_sigma
        argsDict["ebqe_penalty_ext"] = self.ebqe['penalty']

        #argsDict["a_rowptr"] = self.coefficients.sdInfo[(0,0)][0]
        #argsDict["a_colind"] = self.coefficients.sdInfo[(0,0)][1]
        argsDict["alpha_L"] = self.coefficients.alpha_L  # Longitudinal dispersion coefficient
        argsDict["alpha_T"] = self.coefficients.alpha_T  # Transverse dispersion coefficient
        argsDict["Dm"] = self.coefficients.Dm    

        self.adr.calculateJacobian(argsDict)

        # Load the Dirichlet conditions directly into residual
        if self.forceStrongConditions:
            scaling = 1.0  # probably want to add some scaling to match non-dirichlet diagonals in linear system
            for dofN in list(self.dirichletConditionsForceDOF.DOFBoundaryConditionsDict.keys()):
                global_dofN = dofN
                for i in range(self.rowptr[global_dofN],
                               self.rowptr[global_dofN + 1]):
                    if (self.colind[i] == global_dofN):
                        self.nzval[i] = scaling
                    else:
                        self.nzval[i] = 0.0
        logEvent("Jacobian ", level=10, data=jacobian)
        return jacobian

    def calculateElementQuadrature(self):
        """
        Calculate the physical location and weights of the quadrature rules
        and the shape information at the quadrature points.

        This function should be called only when the mesh changes.
        """
        self.u[0].femSpace.elementMaps.getValues(self.elementQuadraturePoints,
                                                 self.q['x'])
        self.u[0].femSpace.elementMaps.getBasisValuesRef(
            self.elementQuadraturePoints)
        self.u[0].femSpace.elementMaps.getBasisGradientValuesRef(
            self.elementQuadraturePoints)
        self.u[0].femSpace.getBasisValuesRef(self.elementQuadraturePoints)
        self.u[0].femSpace.getBasisGradientValuesRef(self.elementQuadraturePoints)
        #self.coefficients.initializeElementQuadrature(self.timeIntegration.t,
        #                                              self.q)
        if self.stabilization is not None:
            self.stabilization.initializeElementQuadrature(
                self.mesh, self.timeIntegration.t, self.q)
            self.stabilization.initializeTimeIntegration(self.timeIntegration)
        if self.shockCapturing is not None:
            self.shockCapturing.initializeElementQuadrature(
                self.mesh, self.timeIntegration.t, self.q)

    def calculateElementBoundaryQuadrature(self):
        pass

    def calculateExteriorElementBoundaryQuadrature(self):
        """
        Calculate the physical location and weights of the quadrature rules
        and the shape information at the quadrature points on global element boundaries.

        This function should be called only when the mesh changes.
        """
        #
        # get physical locations of element boundary quadrature points
        #
        # assume all components live on the same mesh
        self.u[0].femSpace.elementMaps.getBasisValuesTraceRef(
            self.elementBoundaryQuadraturePoints)
        self.u[0].femSpace.elementMaps.getBasisGradientValuesTraceRef(
            self.elementBoundaryQuadraturePoints)
        self.u[0].femSpace.getBasisValuesTraceRef(
            self.elementBoundaryQuadraturePoints)
        self.u[0].femSpace.getBasisGradientValuesTraceRef(
            self.elementBoundaryQuadraturePoints)
        self.u[0].femSpace.elementMaps.getValuesGlobalExteriorTrace(
            self.elementBoundaryQuadraturePoints, self.ebqe['x'])
        self.fluxBoundaryConditionsObjectsDict = dict([(cj, FluxBoundaryConditions(self.mesh,
                                          self.nElementBoundaryQuadraturePoints_elementBoundary,
                                          self.ebqe[('x')],
                                          self.advectiveFluxBoundaryConditionsSetterDict[cj],
                                          self.diffusiveFluxBoundaryConditionsSetterDictDict[cj]))
              for cj in list(self.advectiveFluxBoundaryConditionsSetterDict.keys())])
        self.coefficients.initializeGlobalExteriorElementBoundaryQuadrature(
            self.timeIntegration.t, self.ebqe)

    def estimate_mt(self):
        pass

    def calculateSolutionAtQuadrature(self):
        pass

    def calculateAuxiliaryQuantitiesAfterStep(self):
        pass

    def updateAfterMeshMotion(self):
        pass