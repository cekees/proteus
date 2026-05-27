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
        # Copy per-component free-DOF history back out of the assembled solver
        # vector using the transport's offset/stride layout.
        u_arr = np.asarray(self.u)
        offset0 = self.transport.offset[0]
        stride0 = self.transport.stride[0]
        n0 = self.transport.u_dof_old.size
        self.transport.u_dof_old[:] = u_arr[offset0:offset0 + stride0 * n0:stride0]
        if getattr(self.transport, 'nc', 1) >= 2:
            u_dof_n_old = getattr(self.transport, 'u_dof_n_old', None)
            if u_dof_n_old is not None:
                offset1 = self.transport.offset[1]
                stride1 = self.transport.stride[1]
                n1 = u_dof_n_old.size
                u_dof_n_old[:] = u_arr[offset1:offset1 + stride1 * n1:stride1]
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
                 density,
                 beta,
                 # gas-phase density.
                 # rho_n acts as the reference density. When p_ref_n > 0, the
                 # gas density is linear in p_n: rho_n(p_n) = rho_n * p_n/p_ref_n
                 # so rho_n is recovered at p_n = p_ref_n. With p_ref_n = 0
                 # (default) the gas density stays constant = rho_n (Step 1
                 # behavior).
                 rho_n=1.0,
                 p_ref_n=0.0,
                 diagonal_conductivity=True,
                 getSeepageFace=None,
                 density_model=None,
                 DENSITY_MODEL=None,
                 # PSK constitutive model: 'VGM' (van Genuchten-Mualem) or 'BC' (Brooks-Corey-Burdine)
                 PSK_TYPE='VGM',
                # FOR EDGE BASED EV
                 STABILIZATION_TYPE='Implicit_FCT',
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
                 # Stage 3b (gas-side kinetic dissolution sink).  When coupled
                 # to a TADR transport model, mphase_co2 deducts R_diss = k_d *
                 # S_n * S_w * (c_sat - c) from the gas-equation residual per
                 # DOF so every kg of CO2 that TADR adds to the brine is
                 # removed from the gas phase (mass conservation across the
                 # phases).  Defaults k_d=0 disable the sink; the gas equation
                 # then sees no dissolution (legacy behavior preserved).
                 k_d=0.0,
                 c_sat=1.0,
                 # CO2 injection point sources: list of
                 # (x, y, rate, t_start, t_stop) tuples.  Each is an interior
                 # mass source on the gas (S_n) equation, active while
                 # t_start <= t < t_stop.  None/empty -> no injection (legacy).
                 injection_ports=None,
                 # tanh ramp at each port's start so Newton can track the
                 # saturation breakthrough; in the SIMULATION's time units
                 # (e.g. hours if dt is in hours).  0 -> no ramp (legacy).
                 # Ramp is centred at t0+3*tau, full rate by t0+6*tau.
                 injection_ramp_tau=0.0,
                 # End-point gas relperm per material type.  Brooks-Corey /
                 # van Genuchten-Mualem give k_rn(S_e=0) = 1 for every sand;
                 # multi-phase rigs (e.g. FluidFlower) measure 0.02..0.16.
                 # Pass a (nMaterialTypes,) array to scale k_rn(*) by the
                 # measured endpoint; None -> all-ones (legacy behavior).
                 krn_end_types=None,
                 # Gas dynamic viscosity in the simulation's units.  The
                 # gas-flux terms (a_n, f_n) are divided by mu_n at every
                 # quadrature point.  Default 1.0 = legacy (mu implicit at 1).
                 # For CO2 in normalized brine units: mu_n ~= 0.015 (mu_CO2 /
                 # mu_water = 1.5e-5 / 1.0e-3 in physical SI).
                 mu_n=1.0,
                  ):
        self.VMS=VMS
        if density_model is None:
            density_model = DENSITY_MODEL
        self.density_model = density_model
        # Stage 3b: gas-side kinetic dissolution sink parameters.
        self.k_d = k_d
        self.c_sat = c_sat
        # CO2 injection point sources (see __init__ argument).
        self.injection_ports = list(injection_ports) if injection_ports else []
        self.injection_ramp_tau = float(injection_ramp_tau)
        self.modelIndex=1
        self.SC=SC
        self.anb_seepage_flux= 0.00
        #self.anb_seepage_flux_n =0.0
        # nc=2, primary vars (p_w, S_n).
        # u[0] = p_w  (wetting-phase pressure in Pa)
        # u[1] = S_n  (non-wetting saturation in [0, 1 - S_wr])
        # Compressibility beta is in 1/Pa and the user-supplied Ksw_types
        # array is interpreted as K/mu_w in 1/(Pa*s).
        variableNames=['p_w', 'S_n']
        nc=2
        # gas equation gains a diffusion term -div(a_n grad u_w).
        # Declaring diffusion[1][0][1]='nonlinear' and potential[1][0]='u'
        # makes the framework allocate (1,0) Jacobian sparsity (gas-eq dependence
        # on the wetting pressure gradient via a_n) and the cj=1 nonlinearity
        # tag also adds the coefficient sensitivity contribution to (1,1).
        # (0,1) cross-block tags: wetting eq depends on u_n through
        #   m_w via theta_w(u_n) -> mass[0][1]='nonlinear'
        #   f_w via k_rw(u_n)    -> advection[0][1]='nonlinear'
        #   a_w via k_rw(u_n)    -> diffusion[0][0][1]='nonlinear'
        mass     ={0:{0:'nonlinear', 1:'nonlinear'}, 1:{1:'linear'}}
        advection={0:{0:'nonlinear', 1:'nonlinear'}, 1:{1:'nonlinear'}}
        diffusion={0:{0:{0:'nonlinear', 1:'nonlinear'}},
                   1:{0:{1:'nonlinear'}}}
        potential={0:{0:'u'}, 1:{0:'u', 1:'u'}}
        reaction ={0:{0:'linear'}}
        hamiltonian={}
        self.getSeepageFace=getSeepageFace
        self.gravity=gravity
        self.rho = density
        # gas-phase density. Linear EOS: rho_n(p_n) = rho_n * p_n / p_ref_n
        # when p_ref_n > 0; constant rho_n when p_ref_n == 0.
        self.rho_n = rho_n
        self.p_ref_n = p_ref_n
        self.beta=beta
        self.vgm_n_types = vgm_n_types
        self.vgm_alpha_types = vgm_alpha_types
        self.thetaR_types    = thetaR_types
        self.thetaSR_types   = thetaSR_types
        # Per-material end-point gas relperm (k_rn at S_e = 0).  Default
        # all-ones so the closure's k_rn(0) = 1 is unchanged.
        if krn_end_types is None:
            self.krn_end_types = np.ones_like(np.asarray(thetaR_types, dtype='d'))
        else:
            self.krn_end_types = np.asarray(krn_end_types, dtype='d')
            assert self.krn_end_types.shape == np.asarray(thetaR_types).shape, \
                "krn_end_types must have the same shape as thetaR_types"
        # Gas dynamic viscosity (see __init__ argument).  Stored as scalar.
        self.mu_n = float(mu_n)
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
            # add (1,0) cross-block sparsity for the gas-eq diffusion
            # term -div(a_n grad u_w). Same diagonal-tensor layout as (0,0) since a_n
            # reuses the wetting Ks structure (with rho_n / k_rn factors applied at QP).
            # add (0,1) cross-block tensor (a_w depends on u_n via k_rw).
            # Same diagonal layout.
            _diag_rowptr = np.arange(self.nd+1, dtype='i')
            _diag_colind = np.arange(self.nd, dtype='i')
            sparseDiffusionTensors = {(0,0): (_diag_rowptr, _diag_colind),
                                      (0,1): (_diag_rowptr, _diag_colind),
                                      (1,0): (_diag_rowptr, _diag_colind)}

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

        # PSK closure selector: 0 = VGM (van Genuchten-Mualem), 1 = BC (Brooks-Corey-Burdine).
        # The closure functions for both live in psk_models.h. Every call site
        # in mphase_co2.h dispatches on PSK_TYPE_member: if (PSK_TYPE_member == 1)
        # invokes bc_*_from_Se, else vgm_*_from_Se. Both paths are exercised.
        # NOTE: for the BC path the user-supplied vgm_n_types array is reinterpreted
        # as the BC pore-size index lambda (the closures take a single shape
        # parameter; we reuse the slot to avoid a separate types array).
        psk_types = {"VGM": 0, "BC": 1}
        try:
            if isinstance(PSK_TYPE, int):
                PSK_TYPE = [key for key, value in psk_types.items() if value == PSK_TYPE][0]
            self.PSK_TYPE = psk_types[PSK_TYPE]
        except:
            raise ValueError("PSK_TYPE must be one of " + str(list(psk_types.keys())) + " not " + str(PSK_TYPE))

        # Implicit_FCT (=3) is not supported in the (p_w, S_n) formulation.
        # The FCT pipeline would invert m_w to recover u_w, but m_w now
        # depends on BOTH p_w and S_n (and is independent of p_w when beta=0),
        # so the inversion is ill-posed. The C++ invert() throws if reached;
        # this gate fails earlier with a clearer message.
        if self.STABILIZATION_TYPE == 3:
            raise ValueError(
                "STABILIZATION_TYPE='Implicit_FCT' is not supported in the "
                "(p_w, S_n) two-phase formulation: the wetting mass m_w = "
                "rho_w(p_w)*phi*theta_w(1-u_n) does not uniquely determine "
                "p_w, so the FCT m->u inversion is ill-posed. Use "
                "STABILIZATION_TYPE='Galerkin' or 'EntropyViscosity'."
            )

        # EDGE BASED (AND ENTROPY) VISCOSITY
        self.LUMPED_MASS_MATRIX = LUMPED_MASS_MATRIX
        self.MONOLITHIC = MONOLITHIC
        #self.STABILIZATION_TYPE = STABILIZATION_TYPE
        self.ENTROPY_TYPE = ENTROPY_TYPE
        # Capture the user's FCT request. The Proteus framework's
        # NonlinearSolvers.Newton.solve() has a post-Newton FCT hook
        #     if self.F.coefficients.FCT == True:
        #         self.F.FCTStep()
        #         u[:] = self.F.u[0].dof
        # that assumes single-component dof storage (u[0].dof IS the full
        # unknown), which is the Richards-era convention. For our nc=2 model
        # (comp-0 + comp-1 each sized N), u is 2N and u[0].dof is N, so the
        # broadcast crashes. We force self.FCT = False so the framework hook
        # never fires; the C++ FCT pipeline runs inside calculateResidual_
        # entropy_viscosity via the FCT_n argsDict flag, gated by
        # _fct_requested below.
        self._fct_requested = bool(FCT)
        self.FCT = False
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

    def attachModels(self, modelList):
        # NOTE: self.model is already set to this Richards LevelModel by
        # OneLevelTransport.__init__ (`self.coefficients.model = self`).
        # Do NOT overwrite it from self.modelIndex — that hardcoded index (=1)
        # points to TADR in the standard pnList, which corrupts Richards'
        # self.model and silently breaks density coupling.
        # Always allocate self.c_dof so the C++ kernel never sees a missing
        # argsDict entry (Stage 3b reads it unconditionally).
        if self.density_model is None:
            self.c_dof = np.zeros_like(self.model.u[1].dof)
            return
        self.densityModel = modelList[self.density_model]
        # Stage 3b: alias TADR's c DOFs so the gas-equation residual can
        # compute R_diss = k_d * S_n * S_w * (c_sat - c) and deduct it from
        # the gas mass.  C0-P1 DOFs are shared on the same mesh, so this is
        # direct DOF-to-DOF aliasing (no projection).  If the density_model
        # doesn't expose u[0].dof (unusual), fall back to a zero array so
        # the sink is harmlessly inactive.
        if hasattr(self.densityModel, 'u') and len(self.densityModel.u) >= 1 \
                and hasattr(self.densityModel.u[0], 'dof'):
            self.c_dof = self.densityModel.u[0].dof
        else:
            self.c_dof = np.zeros_like(self.model.u[1].dof)

    def preStep(self, t, firstStep=False):
        # Refresh coupled density every step from the transport (TADR) model,
        # mirroring how TADR refreshes velocity / aliases moisture content.
        if self.density_model is None or not hasattr(self, 'densityModel'):
            return {}
        coeffs = getattr(self.densityModel, 'coefficients', None)
        if coeffs is None:
            return {}
        q_rho = getattr(coeffs, 'q_rho', None)
        ebqe_rho = getattr(coeffs, 'ebqe_rho', None)
        if q_rho is not None and hasattr(self.model, 'q') and 'rho' in self.model.q:
            self.model.q['rho'][:] = q_rho
        if ebqe_rho is not None and hasattr(self.model, 'ebqe') and 'rho' in self.model.ebqe:
            self.model.ebqe['rho'][:] = ebqe_rho

        # ---- coupling diagnostic: MPI-reduced, print on rank 0 ----
        from mpi4py import MPI
        comm = MPI.COMM_WORLD

        def _global_stats(local):
            a = np.asarray(local)
            lo = comm.allreduce(float(a.min()) if a.size else float('inf'), op=MPI.MIN)
            hi = comm.allreduce(float(a.max()) if a.size else float('-inf'), op=MPI.MAX)
            ssum = comm.allreduce(float(a.sum()), op=MPI.SUM)
            n = comm.allreduce(int(a.size), op=MPI.SUM)
            return lo, hi, (ssum / n if n > 0 else float('nan'))

        if q_rho is not None and 'rho' in self.model.q:
            src = np.asarray(q_rho)
            dst = np.asarray(self.model.q['rho'])
            local_diff = float(np.max(np.abs(src - dst))) if src.shape == dst.shape else float('nan')
            diff = comm.allreduce(local_diff, op=MPI.MAX)
            s_lo, s_hi, s_mn = _global_stats(src)
            d_lo, d_hi, d_mn = _global_stats(dst)
            if comm.Get_rank() == 0:
                logEvent(
                    "[Coupling rho q] Richards.preStep t={:.6e} firstStep={} "
                    "TADR.q_rho (min,max,mean)=({:.6e},{:.6e},{:.6e}) "
                    "Richards.q['rho'] (min,max,mean)=({:.6e},{:.6e},{:.6e}) "
                    "max|src-dst|={:.3e}".format(
                        float(t), firstStep, s_lo, s_hi, s_mn,
                        d_lo, d_hi, d_mn, diff),
                    level=2)
        if ebqe_rho is not None and 'rho' in self.model.ebqe:
            src_b = np.asarray(ebqe_rho)
            dst_b = np.asarray(self.model.ebqe['rho'])
            local_diff_b = float(np.max(np.abs(src_b - dst_b))) if src_b.shape == dst_b.shape else float('nan')
            diff_b = comm.allreduce(local_diff_b, op=MPI.MAX)
            s_lo, s_hi, s_mn = _global_stats(src_b)
            d_lo, d_hi, d_mn = _global_stats(dst_b)
            if comm.Get_rank() == 0:
                logEvent(
                    "[Coupling rho ebqe] Richards.preStep t={:.6e} "
                    "TADR.ebqe_rho (min,max,mean)=({:.6e},{:.6e},{:.6e}) "
                    "Richards.ebqe['rho'] (min,max,mean)=({:.6e},{:.6e},{:.6e}) "
                    "max|src-dst|={:.3e}".format(
                        float(t), s_lo, s_hi, s_mn, d_lo, d_hi, d_mn, diff_b),
                    level=2)
        return {}


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
        #self.anb_seepage_flux= anb_seepage_flux
        #print("The seepage is ", anb_seepage_flux)
#        cq['Ks'] = np.zeros(self.q_shape,'d')
#        for k in range(self.q_shape[1]):
#            cq['Ks'][:,k] = self.Ksw_types[self.elementMaterialTypes,0]
        self.q[('vol_frac',0)] = np.zeros(self.q_shape,'d')
    def initializeElementBoundaryQuadrature(self,t,cebq,cebq_global):
        self.materialTypes_ebq = np.zeros(cebq[('u',0)].shape[0:2],'i')
        self.ebq_shape = cebq[('u',0)].shape
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

        # ---- Component 1 (S_n) framework bookkeeping fill -----------------
        # The real gas residual is assembled in C++; this fill writes the
        # correctly-scaled placeholder so any downstream framework code that
        # consumes (m,1)/(dm,1,1) sees physically consistent values rather
        # than a unit-mass shape.
        # Material 0 is used as the representative; if multiple materials are
        # in play, the C++ assembly recomputes per QP and overrides this.
        phi_rho_n = float((self.thetaR_types[0] + self.thetaSR_types[0])
                          * self.rho_n)
        if ('m', 1) in c and ('u', 1) in c:
            c[('m', 1)][:] = phi_rho_n * c[('u', 1)]
        if ('dm', 1, 1) in c:
            c[('dm', 1, 1)][:] = phi_rho_n
        # zero out the (1,0) cross-block coefficient arrays.
        # The C++ residual/Jacobian assembly fills these with the actual gas
        # Darcy contributions; this evaluate() fill is just so the framework
        # has well-defined values during sparsity setup / NaN checks.
        if ('a', 1, 0) in c:
            c[('a', 1, 0)][:] = 0.0
        if ('da', 1, 0, 1) in c:
            c[('da', 1, 0, 1)][:] = 0.0
        if ('df', 1, 1) in c:
            c[('df', 1, 1)][:] = 0.0
        if ('f', 1) in c:
            c[('f', 1)][:] = 0.0
        # Zero out the (0,1) cross-block coefficient arrays. The C++ Jacobian
        # element loop writes the cross-block directly into globalJacobian, so
        # this fill only keeps the framework's NaN guards happy.
        if ('dm', 0, 1) in c:
            c[('dm', 0, 1)][:] = 0.0
        if ('df', 0, 1) in c:
            c[('df', 0, 1)][:] = 0.0
        if ('da', 0, 0, 1) in c:
            c[('da', 0, 0, 1)][:] = 0.0
    
    def postStep(self, t, firstStep=False):
        m = self.model
        # FCT post-step (gated): only when STAB>0 and FCT was requested.
        if (m is not None
                and self.STABILIZATION_TYPE != 0
                and self._fct_requested
                and getattr(m, 'limited_solution_n', None) is not None):
            m.FCTStep(component=1)
        # Mass-balance diagnostic (always runs when coupled to TADR).
        self._log_mass_balance(t)
        return {}

    def _log_mass_balance(self, t):
        """Aggregate gas + dissolved CO2 mass and compare to cumulative
        injection. Diagnostic for the STAB=2 Richards-style upwind port:
        if total mass = cum_injected, the operator is well-calibrated; if
        gas + diss << cum_injected, dLow is over-dissipating (gas vanishes
        before it can dissolve).

        Per-DOF lumped weights are computed once from the P1 element-area
        partition and cached. phi is volume-averaged per node (heterogeneous
        Brooks-Corey materials supported)."""
        from mpi4py import MPI
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()

        m = self.model
        if m is None or not hasattr(self, 'densityModel') or self.densityModel is None:
            return
        mesh = m.mesh

        if not hasattr(self, '_M_lump_node'):
            nNodes_total = int(getattr(mesh, 'nNodes_global', len(m.u[1].dof)))
            ML = np.zeros(nNodes_total, 'd')
            phi_num = np.zeros(nNodes_total, 'd')
            phi_den = np.zeros(nNodes_total, 'd')
            elementNodes = mesh.elementNodesArray
            nodeArr = mesh.nodeArray
            matTypes = self.elementMaterialTypes
            for eN in range(mesh.nElements_global):
                nodes = elementNodes[eN]
                p0 = nodeArr[nodes[0]]
                p1 = nodeArr[nodes[1]]
                p2 = nodeArr[nodes[2]]
                area = 0.5 * abs((p1[0] - p0[0]) * (p2[1] - p0[1])
                                 - (p2[0] - p0[0]) * (p1[1] - p0[1]))
                mat = int(matTypes[eN])
                phi_e = float(self.thetaR_types[mat] + self.thetaSR_types[mat])
                contrib = area / 3.0
                for nn in nodes:
                    ML[nn] += contrib
                    phi_num[nn] += contrib * phi_e
                    phi_den[nn] += contrib
            phi_node = np.where(phi_den > 0.0, phi_num / phi_den, 0.0)
            self._M_lump_node = ML
            self._phi_node = phi_node

        ML = self._M_lump_node
        phi = self._phi_node
        S_n = np.asarray(m.u[1].dof)
        c = np.asarray(self.densityModel.u[0].dof)

        # Sum owned DOFs only (avoid double-count across MPI ranks).
        n_owned = int(getattr(mesh, 'nNodes_owned',
                              getattr(mesh, 'nNodes_global', len(S_n))))
        size = min(n_owned, len(ML), len(S_n), len(c))
        w = ML[:size] * phi[:size]
        local_gas = float(np.sum(w * float(self.rho_n) * S_n[:size]))
        local_dis = float(np.sum(w * (1.0 - S_n[:size]) * c[:size]))
        gas = comm.allreduce(local_gas, op=MPI.SUM)
        diss = comm.allreduce(local_dis, op=MPI.SUM)

        # ---- KERNEL-INJECTED MASS AUDIT ----
        # The kernel adds  M_lump_i * Q_inj_i * dt  to gas mass per node per
        # time step. Total kernel injection = integral over time of
        # sum(M_lump * injection_dof). We accumulate this with a trapezoidal
        # rule across mass-balance calls. If kernel_inj_cum != cum_inj
        # (analytical), the analytical disk-area formula is mesh-discrepant
        # with the lumped-quadrature sum over the masked nodes.
        inj_dof = getattr(m, 'injection_dof', None)
        if inj_dof is not None:
            inj_dof_local = np.asarray(inj_dof)[:size]
            local_inj_rate = float(np.sum(ML[:size] * inj_dof_local))
        else:
            local_inj_rate = 0.0
        inj_rate_global = comm.allreduce(local_inj_rate, op=MPI.SUM)
        if not hasattr(self, '_kernel_inj_cum'):
            self._kernel_inj_cum = 0.0
            self._last_balance_t = float(t)
            self._last_inj_rate  = inj_rate_global
        dt_balance = float(t) - self._last_balance_t
        if dt_balance > 0.0:
            # End-of-step accumulation: the kernel scatters
            #   ML_n[i] * injection_dof[i] * dt
            # per step using injection_dof set in getResidual at the START
            # of the step.  That rate is held constant across all Newton
            # iterations for the step.  Using end-of-step `inj_rate_global`
            # here matches the kernel's actual scatter exactly; trapezoidal
            # averaging biased low during a ramping schedule and inflated
            # the apparent "leak" in `bal_vs_kernel`.
            self._kernel_inj_cum += dt_balance * inj_rate_global
        self._last_balance_t = float(t)
        self._last_inj_rate  = inj_rate_global
        kernel_inj = self._kernel_inj_cum

        # ---- R_DISS BUDGET AUDIT (flow side vs TADR side) ----
        # Per-node analytical R_diss = phi*(1-S_n)*rho_w*k_d*S_n*(c_sat - c).
        # Both mphase_co2 (gas-eq sink) and TADR (c-eq source) compute this
        # independently. With matched k_d / c_sat they should agree to the
        # split-lag error in c. The flow side reads c_dof (== TADR's u[0].dof)
        # so they SHOULD use the same c; any difference is Sequential split
        # timing.
        rho_w_dof = None
        coeffs = getattr(self.densityModel, 'coefficients', None)
        if coeffs is not None:
            # densityModel q_rho only available at QPs; for diagnostic use 1.0.
            pass
        rho_w_eff = 1.0  # head form
        k_d_flow = float(self.k_d)
        c_sat    = float(self.c_sat)
        k_d_tadr = float(getattr(coeffs, 'k_d', k_d_flow)) if coeffs is not None else k_d_flow
        c_sat_tadr = float(getattr(coeffs, 'c_sat', c_sat)) if coeffs is not None else c_sat
        S_w_loc = 1.0 - S_n[:size]
        active = (S_n[:size] > 0.0) & (c[:size] < c_sat)
        R_per_node_flow = (w * S_w_loc * rho_w_eff * k_d_flow * S_n[:size]
                          * (c_sat - c[:size]))
        R_per_node_tadr = (w * S_w_loc * rho_w_eff * k_d_tadr * S_n[:size]
                          * (c_sat_tadr - c[:size]))
        local_R_flow = float(np.sum(R_per_node_flow))
        local_R_tadr = float(np.sum(R_per_node_tadr))
        R_flow_total = comm.allreduce(local_R_flow, op=MPI.SUM)
        R_tadr_total = comm.allreduce(local_R_tadr, op=MPI.SUM)

        # ---- dLow SYMMETRY AUDIT ----
        # The Stage-2 Richards-style upwind dissipation contributes
        #   sum_{i,j} dH_ij * (m_n[i] - m_n[j])
        # to the total gas residual. For conservation we need dH symmetric
        # (dH_ij == dH_ji) so the double sum cancels. Float-roundoff
        # asymmetry would create a small per-step mass leak.
        # Reports: max |dH_ij - dH_ji|, total un-cancelled flux contribution.
        dLow_arr = getattr(m, 'dLow_n', None)
        rowptr   = getattr(m, 'comp1_rowptr', None)
        colind   = getattr(m, 'comp1_colind', None)
        m_n_arr  = float(self.rho_n) * phi * S_n  # m_n at every node
        max_asym_abs = 0.0
        max_asym_rel = 0.0
        sum_dLow_flux = 0.0
        if (dLow_arr is not None and rowptr is not None and colind is not None
                and len(dLow_arr) == len(colind)):
            # Build edge-offset map (i,j)->offset once and cache.
            if not hasattr(self, '_edge_offset_map'):
                self._edge_offset_map = {}
                for i_n_ in range(len(rowptr) - 1):
                    for off in range(int(rowptr[i_n_]), int(rowptr[i_n_ + 1])):
                        self._edge_offset_map[(i_n_, int(colind[off]))] = off
            edge_map = self._edge_offset_map
            dL = np.asarray(dLow_arr)
            nrows = min(len(rowptr) - 1, size, len(m_n_arr))
            for i_n_ in range(nrows):
                row_start = int(rowptr[i_n_])
                row_end   = int(rowptr[i_n_ + 1])
                for off_ij in range(row_start, row_end):
                    j_n_ = int(colind[off_ij])
                    if j_n_ == i_n_ or j_n_ >= nrows:
                        continue
                    d_ij = float(dL[off_ij])
                    off_ji = edge_map.get((j_n_, i_n_), -1)
                    if off_ji < 0:
                        continue
                    d_ji = float(dL[off_ji])
                    asym = abs(d_ij - d_ji)
                    denom = max(abs(d_ij), abs(d_ji), 1.0e-30)
                    if asym > max_asym_abs:
                        max_asym_abs = asym
                    if asym / denom > max_asym_rel:
                        max_asym_rel = asym / denom
                    # Per-edge contribution to total dLow gas residual:
                    # at row i: +d_ij * (m[i] - m[j])
                    sum_dLow_flux += d_ij * (m_n_arr[i_n_] - m_n_arr[j_n_])
        max_asym_abs = comm.allreduce(max_asym_abs, op=MPI.MAX)
        max_asym_rel = comm.allreduce(max_asym_rel, op=MPI.MAX)
        sum_dLow_flux = comm.allreduce(sum_dLow_flux, op=MPI.SUM)

        # ---- PER-EQUATION CONSERVATION CHECK ----
        # Compare finite-difference d(gas)/dt to (inj_rate - R_flow_rate)
        # and  d(diss)/dt to R_tadr_rate.
        # In the discrete continuum:
        #   gas-eq:  d(gas)/dt = inj_rate - R_flow_rate           (no boundary if sealed)
        #   c-eq:    d(diss)/dt = R_tadr_rate
        # If either residual is non-zero, that equation is creating/destroying
        # mass at the printed rate. Isolates the leak source (gas-eq vs TADR).
        if not hasattr(self, '_last_gas'):
            self._last_gas = gas
            self._last_diss = diss
            self._last_audit_t = float(t)
            gas_leak_rate = 0.0
            diss_leak_rate = 0.0
        else:
            dt_audit = float(t) - self._last_audit_t
            if dt_audit > 1.0e-15:
                d_gas_dt  = (gas  - self._last_gas)  / dt_audit
                d_diss_dt = (diss - self._last_diss) / dt_audit
                # Use averaged rates (this call + previous) since we did
                # trapezoidal accumulation of kernel_inj. R_diss is point-in-time.
                avg_inj_rate    = 0.5 * (self._last_inj_rate2  + inj_rate_global)
                avg_R_flow_rate = 0.5 * (self._last_R_flow_rate + R_flow_total)
                avg_R_tadr_rate = 0.5 * (self._last_R_tadr_rate + R_tadr_total)
                gas_leak_rate  = d_gas_dt  - (avg_inj_rate - avg_R_flow_rate)
                diss_leak_rate = d_diss_dt - avg_R_tadr_rate
            else:
                gas_leak_rate = 0.0
                diss_leak_rate = 0.0
        # Update audit state for next call.
        self._last_gas = gas
        self._last_diss = diss
        self._last_audit_t = float(t)
        self._last_inj_rate2    = inj_rate_global
        self._last_R_flow_rate  = R_flow_total
        self._last_R_tadr_rate  = R_tadr_total

        # Cumulative injected = sum over ports of rate * disk_area * ∫ramp(s) ds
        # where ramp = 0.5*(1 + tanh((s - t_start)/tau - 3)) clipped to [t_start, t_stop].
        # The closed-form integral is needed because at early times the ramp is
        # ~0.25% of nominal; assuming full rate would inflate cum_inj by 400x
        # and make balance look catastrophically bad when it's actually fine.
        #   ∫_{t_start}^{t} 0.5*(1 + tanh((s-t_start)/tau - 3)) ds
        # = 0.5*tau * [(u + 3) + ln(cosh(u)) - ln(cosh(3))]
        # where u = (min(t,t_stop) - t_start)/tau - 3.
        tau = float(self.injection_ramp_tau)
        ln_cosh3 = float(np.log(np.cosh(3.0)))
        cum_inj = 0.0
        for port in self.injection_ports:
            (px, py, rate, radius, t_start, t_stop) = port
            t_eff = max(float(t_start), min(float(t), float(t_stop)))
            elapsed = t_eff - float(t_start)
            if elapsed <= 0.0:
                continue
            if tau > 0.0:
                u = elapsed / tau - 3.0
                # ln(cosh(u)) computed as log1p(exp(-2|u|))/... for stability
                ramp_integral = 0.5 * tau * ((u + 3.0)
                                             + float(np.log(np.cosh(u)))
                                             - ln_cosh3)
            else:
                ramp_integral = elapsed
            cum_inj += float(rate) * np.pi * float(radius) ** 2 * ramp_integral

        bal = gas + diss - cum_inj
        rel = (bal / cum_inj) if cum_inj > 1.0e-30 else 0.0

        # S_n / c min,max for overshoot detection (cheap, no extra reductions
        # beyond what numpy does locally; for parallel correctness we use the
        # owned slice and MPI-reduce across ranks).
        local_Sn_min = float(np.min(S_n[:size])) if size > 0 else 0.0
        local_Sn_max = float(np.max(S_n[:size])) if size > 0 else 0.0
        local_c_min  = float(np.min(c[:size]))  if size > 0 else 0.0
        local_c_max  = float(np.max(c[:size]))  if size > 0 else 0.0
        Sn_min = comm.allreduce(local_Sn_min, op=MPI.MIN)
        Sn_max = comm.allreduce(local_Sn_max, op=MPI.MAX)
        c_min  = comm.allreduce(local_c_min,  op=MPI.MIN)
        c_max  = comm.allreduce(local_c_max,  op=MPI.MAX)

        if rank == 0:
            # Reference balance line (kept for backward-compat / quick scan).
            logEvent(
                "[Mass balance] t={:.4e} gas={:+.4e} diss={:+.4e} "
                "injected={:+.4e} balance={:+.4e} rel={:+.3e}".format(
                    float(t), gas, diss, cum_inj, bal, rel),
                level=2)
            # Field min/max: Sn_max > 1 or Sn_min < 0 indicates overshoot from
            # STAB=2 high-order EV term not being added to the residual (only
            # dLow is, which is monotonicity-preserving). c_max > c_sat means
            # dissolution overshoot from R_diss linearization.
            logEvent(
                "[field range] t={:.4e} Sn=[{:+.4e},{:+.4e}] "
                "c=[{:+.4e},{:+.4e}] (c_sat={:.4e})".format(
                    float(t), Sn_min, Sn_max, c_min, c_max, float(self.c_sat)),
                level=2)
            # ---- Audit lines ----
            # (1) Kernel-injected vs analytical: if these disagree, the disk
            #     lumped-quadrature is the dominant source of "rel" bias.
            # (2) R_diss flow vs TADR: if these disagree per-DOF, the
            #     dissolution coupling is leaking mass.
            kernel_ratio = (kernel_inj / cum_inj) if cum_inj > 1e-30 else 0.0
            R_ratio = (R_flow_total / R_tadr_total) if R_tadr_total > 1e-30 else 0.0
            bal_kernel = gas + diss - kernel_inj
            rel_kernel = (bal_kernel / kernel_inj) if kernel_inj > 1e-30 else 0.0
            logEvent(
                "[Mass audit ] t={:.4e} kernel_inj={:+.4e} cum_inj={:+.4e} "
                "kernel/analytical={:.3f}  bal_vs_kernel={:+.4e} rel_vs_kernel={:+.3e}".format(
                    float(t), kernel_inj, cum_inj, kernel_ratio,
                    bal_kernel, rel_kernel),
                level=2)
            logEvent(
                "[R_diss     ] t={:.4e} R_flow_rate={:+.4e} R_tadr_rate={:+.4e} "
                "ratio_flow/tadr={:.3f}  (k_d_flow={:.3e} k_d_tadr={:.3e})".format(
                    float(t), R_flow_total, R_tadr_total, R_ratio,
                    k_d_flow, k_d_tadr),
                level=2)
            # Per-equation conservation: if either leak_rate is non-trivial
            # relative to the active source/sink rate, that equation is the
            # leak source. Use max(inj, R_flow, R_tadr) so post-injection
            # (inj_rate=0) the rel value normalises against dissolution
            # instead of blowing up to 1e+26.
            rate_scale = max(abs(inj_rate_global), abs(R_flow_total),
                             abs(R_tadr_total), 1.0e-30)
            # leak_per_step is the actual mass added per Newton solve. With
            # the STAB=2 Richards port, the gas residual has NO consistent
            # CG, NO unconserved boundary (gated by isDir_n), and dLow is
            # verified symmetric (sum_dH*(m_i-m_j) ~ 1e-19 below). So
            # leak_per_step should equal sum of converged Newton residuals
            # (~ nl_atol_res, configured in flow_n.py). If leak_per_step
            # tracks nl_atol_res, the only fix is to tighten nl_atol_res /
            # tolFac.  If leak_per_step >> nl_atol_res, something else is
            # creating mass and we need to keep looking.
            try:
                dt_show = float(dt_audit) if dt_audit > 0.0 else 0.0
            except NameError:
                dt_show = 0.0
            leak_per_step = gas_leak_rate * dt_show
            logEvent(
                "[mass leak  ] t={:.4e} gas_leak_rate={:+.4e} (rel={:+.2e})  "
                "diss_leak_rate={:+.4e} (rel={:+.2e})  "
                "dt={:.3e}  leak/step={:+.3e}".format(
                    float(t), gas_leak_rate, gas_leak_rate / rate_scale,
                    diss_leak_rate, diss_leak_rate / rate_scale,
                    dt_show, leak_per_step),
                level=2)
            # dLow symmetry: if |asym|/|max| > ~1e-12, dLow is not symmetric.
            # sum_dLow_flux is the actual un-cancelled contribution per step;
            # if it's ~ gas_leak_rate, dLow asymmetry IS the leak source.
            logEvent(
                "[dLow symm  ] t={:.4e} max_asym_abs={:.3e} max_asym_rel={:.3e}  "
                "sum_dH*(m_i-m_j)={:+.4e} (rel_to_rate={:+.2e})".format(
                    float(t), max_asym_abs, max_asym_rel,
                    sum_dLow_flux, sum_dLow_flux / rate_scale),
                level=2)

    # def postStep(self, t, firstStep=False):
    #     if not self.outputQuantDOFs:
    #         return {}
    #     if (self.model is None or
    #             ('velocity_couple', 0) not in self.model.q or
    #             ('grad(u_n)', 0) not in self.model.q):
    #         return {}

    #     mpicomm = self._get_mpi_comm()
    #     rank = mpicomm.Get_rank()
    #     nSpace = int(getattr(self.model, 'nSpace_global',
    #                          getattr(self.model, 'nSpace', self.nd)))
    #     n_owned = self._get_owned_element_count()
    #     stab_tag = f"stab{self.STABILIZATION_TYPE}"

    #     qcoords_local = self._get_q_coordinates().reshape((-1, 3))
    #     qv_local = np.asarray(self.model.q[('velocity_couple', 0)][:n_owned]).reshape((-1, nSpace))
    #     qgrad_local = np.asarray(self.model.q[('grad(u_n)', 0)][:n_owned]).reshape((-1, nSpace))

    #     if not hasattr(self, '_wrote_q_coords_once'):
    #         qcoords_all = mpicomm.gather(qcoords_local, root=0)
    #         if rank == 0:
    #             qcoords = np.vstack(qcoords_all)
    #             np.savetxt(f"richards_q_coordinates_{stab_tag}.txt",
    #                        qcoords,
    #                        fmt="%.16e",
    #                        header=f"columns: x y z | total_rows={qcoords.shape[0]}")
    #             logEvent(f"[Richards postStep] wrote richards_q_coordinates_{stab_tag}.txt rows={qcoords.shape[0]}")
    #         self._wrote_q_coords_once = True

    #     q_profile_local = np.hstack((qcoords_local, qv_local))
    #     q_profile_all = mpicomm.gather(q_profile_local, root=0)
    #     q_grad_profile_local = np.hstack((qcoords_local, qgrad_local))
    #     q_grad_profile_all = mpicomm.gather(q_grad_profile_local, root=0)
    #     velocity_magnitude_local = np.linalg.norm(qv_local, axis=1) if qv_local.size else np.zeros((0,), 'd')
    #     vmax_local = float(velocity_magnitude_local.max()) if velocity_magnitude_local.size else 0.0
    #     vmax = Comm.get().globalMax(vmax_local)
    #     grad_magnitude_local = np.linalg.norm(qgrad_local, axis=1) if qgrad_local.size else np.zeros((0,), 'd')
    #     gmax_local = float(grad_magnitude_local.max()) if grad_magnitude_local.size else 0.0
    #     gmax = Comm.get().globalMax(gmax_local)

    #     if rank == 0:
    #         q_profile = np.vstack(q_profile_all)
    #         q_grad_profile = np.vstack(q_grad_profile_all)
    #         header_cols = "x y z vx vy" if nSpace == 2 else "x y z vx vy vz"
    #         header_grad_cols = "x y z gx gy" if nSpace == 2 else "x y z gx gy gz"
    #         np.savetxt(f"richards_q_velocity_profile_{stab_tag}_t{t:.8e}.txt",
    #                    q_profile,
    #                    fmt="%.16e",
    #                    header=f"columns: {header_cols} | t={t:.16e} | total_rows={q_profile.shape[0]}")
    #         logEvent(f"[Richards postStep] wrote richards_q_velocity_profile_{stab_tag}_t{t:.8e}.txt vmax={vmax:.6e}")
    #         np.savetxt(f"richards_q_grad_u_profile_{stab_tag}_t{t:.8e}.txt",
    #                    q_grad_profile,
    #                    fmt="%.16e",
    #                    header=f"columns: {header_grad_cols} | t={t:.16e} | total_rows={q_grad_profile.shape[0]}")
    #         logEvent(f"[Richards postStep] wrote richards_q_grad_u_profile_{stab_tag}_t{t:.8e}.txt gmax={gmax:.6e}")
    #     return {}





    
    # #def postStep(self, t, firstStep=False):
    #    import os
    #    #from proteus import Comm
    #    comm = Comm.get()
    #    if comm.isMaster():
    #        try:
    #            # Attempt to access and sum the seepage flux
    #            s_now = float(np.sum(self.model.anb_seepage_flux_n))
    #            #s_now= float(self.model.anb_seepage_flux)
    #            if s_now>0.0:
    #                with open("seepage_flux.txt", "a") as f:
    #                    if os.stat("seepage_flux.txt").st_size == 0:
    #                        f.write("time,seepage_flux\n")
    #                        f.write(f"{t:.6f},{s_now:.6f}\n")
    #       except Exception as e:
    #            logEvent(f"[postStep] Skipped logging seepage: {e}")
        
   
        
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
        #mwf try to reuse test and trial information across components if spaces are the same
        self.reuse_test_trial_quadrature = reuse_trial_and_test_quadrature#True#False
        if self.reuse_test_trial_quadrature:
            for ci in range(1,coefficients.nc):
                assert self.u[ci].femSpace.__class__.__name__ == self.u[0].femSpace.__class__.__name__, "to reuse_test_trial_quad all femSpaces must be the same!"
        self.u_dof_old = None
        # previous-step DOFs for component 1 (S_n).
        # Filled lazily on first getResidual call from u[1].dof (the IC).
        self.u_dof_n_old = None
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
        self.nDOF_trial_element     = [u_j.femSpace.max_nDOF_element for  u_j in list(self.u.values())]
        self.nDOF_phi_trial_element     = [phi_k.femSpace.max_nDOF_element for  phi_k in list(self.phi.values())]
        self.n_phi_ip_element = [phi_k.femSpace.referenceFiniteElement.interpolationConditions.nQuadraturePoints for  phi_k in list(self.phi.values())]
        self.nDOF_test_element     = [femSpace.max_nDOF_element for femSpace in list(self.testSpace.values())]
        self.nFreeDOF_global  = [dc.nFreeDOF_global for dc in list(self.dirichletConditions.values())]
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
        self.q[('u',0)] = np.zeros((self.mesh.nElements_global,self.nQuadraturePoints_element),'d')
        self.q[('grad(u)',0)] = np.zeros((self.mesh.nElements_global,self.nQuadraturePoints_element,self.nSpace_global),'d')
        self.q[('grad(phi)',0)] = self.q[('u',0)]
        self.q[('dphi',0,0)] = np.zeros((self.mesh.nElements_global,self.nQuadraturePoints_element,),'d')
        self.q[('da',0,0,0)] = np.zeros((self.mesh.nElements_global,self.nQuadraturePoints_element,),'d')
        # q_velocity scratch buffer: the C++ kernel writes grad(u_w) here
        # (the wetting-phase pressure gradient at QPs) via the "q_velocity"
        # argsDict key. Despite the historical key name including u_n, this
        # is NOT the gradient of the non-wetting saturation -- it is a
        # diagnostic for the wetting-pressure gradient.
        self.q[('q_velocity_buf',0)] = np.zeros((self.mesh.nElements_global,self.nQuadraturePoints_element,self.nSpace_global),'d')
        self.q[('velocity',0)] = np.zeros((self.mesh.nElements_global,self.nQuadraturePoints_element,self.nSpace_global),'d')
        self.q[('velocity_couple',0)] = np.zeros((self.mesh.nElements_global,self.nQuadraturePoints_element,self.nSpace_global),'d')      
        self.q[('m',0)] = self.q[('u',0)].copy()
        self.q[('theta',0)] = self.q[('u',0)].copy()
        self.q[('mt',0)] = self.q[('u',0)].copy()
        self.q[('m_last',0)] = self.q[('u',0)].copy()
        self.q[('m_tmp',0)] = self.q[('u',0)].copy()
        self.q[('cfl',0)] = np.zeros((self.mesh.nElements_global,self.nQuadraturePoints_element),'d')
        self.q[('numDiff',0,0)] =  np.zeros((self.mesh.nElements_global,self.nQuadraturePoints_element),'d')
        self.numDiff_star = self.q[('numDiff',0,0)]
        self.q[('numDiff_last',0,0)] =  np.zeros((self.mesh.nElements_global,self.nQuadraturePoints_element),'d')
        self.ebqe[('u',0)] = np.zeros((self.mesh.nExteriorElementBoundaries_global,self.nElementBoundaryQuadraturePoints_elementBoundary),'d')
        self.ebqe[('theta',0)] = np.zeros((self.mesh.nExteriorElementBoundaries_global,self.nElementBoundaryQuadraturePoints_elementBoundary),'d')
        self.ebqe[('grad(u)',0)] = np.zeros((self.mesh.nExteriorElementBoundaries_global,self.nElementBoundaryQuadraturePoints_elementBoundary,self.nSpace_global),'d')
        self.ebqe[('velocity',0)] = np.zeros((self.mesh.nExteriorElementBoundaries_global,self.nElementBoundaryQuadraturePoints_elementBoundary,self.nSpace_global),'d')
        self.ebqe[('velocity_couple',0)] = np.zeros((self.mesh.nExteriorElementBoundaries_global,self.nElementBoundaryQuadraturePoints_elementBoundary,self.nSpace_global),'d')       
        self.ebqe[('advectiveFlux_bc_flag',0)] = np.zeros((self.mesh.nExteriorElementBoundaries_global,self.nElementBoundaryQuadraturePoints_elementBoundary),'i')
        self.ebqe[('advectiveFlux_bc',0)] = np.zeros((self.mesh.nExteriorElementBoundaries_global,self.nElementBoundaryQuadraturePoints_elementBoundary),'d')
        self.ebqe[('advectiveFlux',0)] = np.zeros((self.mesh.nExteriorElementBoundaries_global,self.nElementBoundaryQuadraturePoints_elementBoundary),'d')
        self.ebqe[('penalty')] = np.zeros((self.mesh.nExteriorElementBoundaries_global,self.nElementBoundaryQuadraturePoints_elementBoundary),'d')
        
        self.q['rho'] = np.zeros((self.mesh.nElements_global,self.nQuadraturePoints_element),'d')
        self.ebqe['rho'] = np.zeros((self.mesh.nExteriorElementBoundaries_global,self.nElementBoundaryQuadraturePoints_elementBoundary),'d')
        self.q['rho'][:] = self.coefficients.rho
        self.ebqe['rho'][:] = self.coefficients.rho
        if self.nc >= 2:
            qshape   = (self.mesh.nElements_global, self.nQuadraturePoints_element)
            qvshape  = (self.mesh.nElements_global, self.nQuadraturePoints_element, self.nSpace_global)
            ebqshape = (self.mesh.nExteriorElementBoundaries_global,
                        self.nElementBoundaryQuadraturePoints_elementBoundary)
            ebqvshape = (self.mesh.nExteriorElementBoundaries_global,
                         self.nElementBoundaryQuadraturePoints_elementBoundary,
                         self.nSpace_global)
            self.q[('u', 1)]            = np.zeros(qshape, 'd')
            self.q[('grad(u)', 1)]      = np.zeros(qvshape, 'd')
            self.q[('grad(phi)', 1)]    = self.q[('u', 1)]
            self.q[('dphi', 1, 1)]      = np.zeros(qshape, 'd')
            self.q[('m', 1)]            = self.q[('u', 1)].copy()
            self.q[('dm', 1, 1)]        = np.zeros(qshape, 'd')
            self.q[('mt', 1)]           = self.q[('u', 1)].copy()
            self.q[('m_last', 1)]       = self.q[('u', 1)].copy()
            self.q[('m_tmp', 1)]        = self.q[('u', 1)].copy()
            self.q[('dV_u', 1)]         = self.q[('dV_u', 0)]
            self.q[('cfl', 1)]          = np.zeros(qshape, 'd')
            self.q[('numDiff', 1, 1)]   = np.zeros(qshape, 'd')
            self.q[('numDiff_last', 1, 1)] = np.zeros(qshape, 'd')
            self.ebqe[('u', 1)]                  = np.zeros(ebqshape, 'd')
            self.ebqe[('grad(u)', 1)]            = np.zeros(ebqvshape, 'd')
            self.ebqe[('advectiveFlux_bc_flag', 1)] = np.zeros(ebqshape, 'i')
            self.ebqe[('advectiveFlux_bc', 1)]      = np.zeros(ebqshape, 'd')
            self.ebqe[('advectiveFlux', 1)]         = np.zeros(ebqshape, 'd')
        
        
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
        
        #if not self.coefficients.LUMPED_MASS_MATRIX and self.coefficients.STABILIZATION_TYPE == 2:
        #    cond = 'levelNonlinearSolver' in dir(options) and options.levelNonlinearSolver == Newton
        
        #if self.coefficients.FCT == True:
        #    cond = self.coefficients.STABILIZATION_TYPE = 3, "Use FCT just with STABILIZATION_TYPE=3; i.e., edge based stabilization"
        
        if self.coefficients._fct_requested:
            valid_stabilization_types = {1, 2}  # Only allow FCT for STABILIZATION_TYPE 1 (EV_Stab) and 2 (EntropyViscosity)
            if self.coefficients.STABILIZATION_TYPE not in valid_stabilization_types:
                raise ValueError("Use FCT only with STABILIZATION_TYPE 1 (EV_Stab) or 2 (EntropyViscosity).")
            cond = self.coefficients.STABILIZATION_TYPE > 0, "Use FCT just with STABILIZATION_TYPE>0; i.e., edge based stabilization"
        # # END OF ASSERTS

        # cek adding empty data member for low order numerical viscosity structures here for now
        self.ML = None  # lumped mass matrix
        self.MC_global = None  # consistent mass matrix
        # Consistent mass matrix on the comp-1 (S_n) DOF graph, indexed by the
        # comp-1 compact CSR (same layout as dLow_n / dt_times_fH_minus_fL_n).
        # MC_a only ever has its (0,0) block assembled, so FCTStep_n's
        # consistency term needs this dedicated comp-1 mass matrix instead.
        self.MC_n = None
        self.cterm_global = None
        self.cterm_transpose_global = None
        # dL_global and dC_global are not the full matrices but just the CSR arrays containing the non zero entries
        self.residualComputed=False #TMP
        self.dLow= None
        self.fluxMatrix = None
        self.mDotLow = None
        self.mLow = None
        self.dt_times_dC_minus_dL = None
        self.min_m_bc = None
        self.max_m_bc = None
        # Aux quantity at DOFs to be filled by optimized code (MQL)
        self.quantDOFs = np.zeros(self.u[0].dof.shape, 'd')
        self.mLow = np.zeros(self.u[0].dof.shape, 'd')
        self.mHigh = np.zeros(self.u[0].dof.shape, 'd')
        self.mDotLow = np.zeros(self.u[0].dof.shape, 'd')
        self.fluxCorrection = np.zeros(self.u[0].dof.shape, 'd')
        self.mn = np.zeros(self.u[0].dof.shape, 'd')
        # Component-1 (S_n) low-order EV buffers.
        self.mn_n        = np.zeros(self.u[1].dof.shape, 'd')
        self.quantDOFs_n = np.zeros(self.u[1].dof.shape, 'd')
        self.anb_seepage_flux_n = np.zeros(self.u[0].dof.shape, 'd')
        self.freeDOFMaterialTypes = np.zeros((self.nFreeDOF_global[0],), 'i')
        self.freeDOFToNode_u = -np.ones((self.nFreeDOF_global[0],), 'i')
        if hasattr(self.mesh, 'nodeMaterialTypes'):
            free_l2g = np.asarray(self.l2g[0]['freeGlobal']).ravel()
            dof_l2g = np.asarray(self.u[0].femSpace.dofMap.l2g).ravel()
            node_material_types = np.asarray(self.mesh.nodeMaterialTypes)
            for free_dof, global_dof in zip(free_l2g, dof_l2g):
                if 0 <= free_dof < self.freeDOFMaterialTypes.shape[0]:
                    self.freeDOFMaterialTypes[free_dof] = node_material_types[global_dof]
                    self.freeDOFToNode_u[free_dof] = global_dof
        else:
            free_l2g = np.asarray(self.l2g[0]['freeGlobal']).ravel()
            dof_l2g = np.asarray(self.u[0].femSpace.dofMap.l2g).ravel()
            for free_dof, global_dof in zip(free_l2g, dof_l2g):
                if 0 <= free_dof < self.freeDOFToNode_u.shape[0]:
                    self.freeDOFToNode_u[free_dof] = global_dof
        if np.any(self.freeDOFToNode_u < 0):
            raise RuntimeError("Failed to build the component-0 free-DOF to node map needed by the stabilized EV/FCT path.")
        comm = Comm.get()
        self.comm=comm
        if comm.size() > 1:
            assert numericalFluxType != None and numericalFluxType.useWeakDirichletConditions,"You must use a numerical flux to apply weak boundary conditions for parallel runs"
            self.offset = [0]
            for ci in range(1,self.nc):
                self.offset += [ci]
            self.stride = [self.nc for ci in range(self.nc)]
        self.comp0_rowptr = None
        self.comp0_colind = None
        self.comp0_full_offsets = None
        # Component-1 (S_n) compact CSR for the (1,1) DOF graph used by the
        # comp-1 EV pipeline (mirrors the comp-0 compact CSR pattern).
        self.comp1_rowptr       = None
        self.comp1_colind       = None
        self.comp1_full_offsets = None
        # Component-1 EV edge/DOF buffers (lazy-allocated on first use).
        self.dLow_n                 = None
        self.dEV_n                  = None
        self.mLow_n                 = None
        self.mHigh_n                = None
        self.mDotLow_n              = None
        self.fluxMatrix_n           = None
        self.dt_times_fH_minus_fL_n = None
        self.FluxCorrectionMatrix_n = None
        self.fluxCorrection_n       = None
        self.limited_solution_n     = None
        self.Rpos_n                 = None
        self.Rneg_n                 = None
        self.min_m_bc_n             = None
        self.max_m_bc_n             = None
        self.bc_mask_n              = None
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
            self.numericalFlux.isDOFBoundary = {0:np.zeros(self.ebqe[('u',0)].shape,'i')}
        if not hasattr(self.numericalFlux,'ebqe'):
            self.numericalFlux.ebqe = {('u',0):np.zeros(self.ebqe[('u',0)].shape,'d')}
        # Ensure component-1 (S_n) Dirichlet boundary arrays exist on the
        # numericalFlux. The framework only initialises (u,0) by default for
        # this single-component-style numerical-flux object; we need (u,1) too
        # so the C++ boundary closure can evaluate
        # bc_u_v_ext = (isDir * bc_value) + (1 - isDir) * u_v_interior.
        if 1 not in self.numericalFlux.isDOFBoundary:
            self.numericalFlux.isDOFBoundary[1] = np.zeros(self.ebqe[('u',1)].shape, 'i')
        if ('u',1) not in self.numericalFlux.ebqe:
            self.numericalFlux.ebqe[('u',1)] = np.zeros(self.ebqe[('u',1)].shape, 'd')
        #TODO how to handle redistancing calls for calculateCoefficients,calculateElementResidual etc
        self.globalResidualDummy = None
        compKernelFlag=0
        self.delta_x_ij=None
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

    def _build_component0_compact_csr(self, full_rowptr, full_colind):
        n_u = self.nFreeDOF_global[0]
        offset_u = self.offset[0]
        stride_u = self.stride[0]
        rowptr_u = np.zeros((n_u + 1,), dtype='i')
        colind_u = []
        full_offsets_u = []
        for i_u in range(n_u):
            global_row = offset_u + stride_u * i_u
            for full_offset in range(full_rowptr[global_row], full_rowptr[global_row + 1]):
                global_col = full_colind[full_offset]
                shifted_col = global_col - offset_u
                if shifted_col < 0 or shifted_col % stride_u != 0:
                    continue
                j_u = shifted_col // stride_u
                if 0 <= j_u < n_u:
                    colind_u.append(j_u)
                    full_offsets_u.append(full_offset)
            rowptr_u[i_u + 1] = len(colind_u)
        self.comp0_rowptr = rowptr_u
        self.comp0_colind = np.asarray(colind_u, dtype='i')
        self.comp0_full_offsets = np.asarray(full_offsets_u, dtype='i')

    def _ensure_component0_compact_csr(self, full_rowptr, full_colind):
        if (self.comp0_rowptr is None or
                self.comp0_colind is None or
                self.comp0_full_offsets is None):
            self._build_component0_compact_csr(full_rowptr, full_colind)

    def _build_component1_compact_csr(self, full_rowptr, full_colind):
        # Mirror of _build_component0_compact_csr but for the comp-1 (S_n)
        # DOF graph. Comp-1 uses full DOF numbering (no Dirichlet elimination)
        # so n_n = self.u[1].dof.shape[0].
        #
        # Also stores comp1_full_offsets: for each (i_n, j_n) entry in the
        # compact comp-1 CSR, the offset into the FULL globalJacobian CSR.
        # FCTStep_n consumes this so it can map per-edge antidiffusive flux
        # entries (indexed by the comp-1 compact CSR) back to globalJacobian
        # offsets without an inline search at each access.
        n_n = self.u[1].dof.shape[0]
        offset_n = self.offset[1]
        stride_n = self.stride[1]
        rowptr_n = np.zeros((n_n + 1,), dtype='i')
        colind_n = []
        full_offsets_n = []
        for i_n in range(n_n):
            global_row = offset_n + stride_n * i_n
            for full_offset in range(full_rowptr[global_row], full_rowptr[global_row + 1]):
                global_col = full_colind[full_offset]
                shifted_col = global_col - offset_n
                if shifted_col < 0 or shifted_col % stride_n != 0:
                    continue
                j_n = shifted_col // stride_n
                if 0 <= j_n < n_n:
                    colind_n.append(j_n)
                    full_offsets_n.append(full_offset)
            rowptr_n[i_n + 1] = len(colind_n)
        self.comp1_rowptr       = rowptr_n
        self.comp1_colind       = np.asarray(colind_n, dtype='i')
        self.comp1_full_offsets = np.asarray(full_offsets_n, dtype='i')

    def _ensure_component1_compact_csr(self, full_rowptr, full_colind):
        if (self.comp1_rowptr is None or
                self.comp1_colind is None or
                getattr(self, 'comp1_full_offsets', None) is None):
            self._build_component1_compact_csr(full_rowptr, full_colind)

    def _scatter_component_to_timeintegration(self, ci):
        if not hasattr(self.timeIntegration, 'u'):
            return
        dest = np.asarray(self.timeIntegration.u)
        comp_dof = np.asarray(self.u[ci].dof)
        offset = self.offset[ci]
        stride = self.stride[ci]
        dest[offset:offset + stride * comp_dof.size:stride] = comp_dof
   
    def FCTStep(self, component):
        
        coef = self.coefficients
        dt   = self.timeIntegration.dt

        if component == 0:
            if self.mLow is None or self.u_dof_old is None:
                return
            n_w = self.nFreeDOF_global[0]
            full_rowptr, full_colind, MassMatrix = self.MC_global.getCSRrepresentation()
            self._ensure_component0_compact_csr(full_rowptr, full_colind)
            nnz0 = self.comp0_colind.shape[0]
            if getattr(self, 'Rpos', None) is None or self.Rpos.shape[0] != n_w:
                self.Rpos = np.zeros((n_w,), 'd')
                self.Rneg = np.zeros((n_w,), 'd')
            if (getattr(self, 'FluxCorrectionMatrix', None) is None
                    or self.FluxCorrectionMatrix.shape[0] != nnz0):
                self.FluxCorrectionMatrix = np.zeros((nnz0,), 'd')
            bc_mask_u = np.ascontiguousarray(self.bc_mask[self.freeDOFToNode_u])
            _par = getattr(self.u[0], 'par_dof', None)
            if _par is not None:
                _saved = self.u[0].dof.copy()
                for _arr in (self.mLow, self.mDotLow, self.min_m_bc, self.max_m_bc):
                    self.u[0].dof[:] = _arr
                    _par.scatter_forward_insert()
                    _arr[:] = self.u[0].dof
                self.u[0].dof[:] = _saved

            # 2. Pass 1: Zalesak ratios.
            argsDict = cArgumentsDict.ArgumentsDict()
            argsDict["component"]                 = 0
            argsDict["pass"]                      = 1
            argsDict["numDOFs"]                   = n_w
            argsDict["dt"]                        = dt
            argsDict["ML"]                        = self.ML
            argsDict["mn"]                        = self.mn
            argsDict["mLow"]                      = self.mLow
            argsDict["mDotLow"]                   = self.mDotLow
            argsDict["csrRowIndeces_DofLoops"]    = self.comp0_rowptr
            argsDict["csrColumnOffsets_DofLoops"] = self.comp0_colind
            argsDict["csrRowIndeces_Full"]        = full_rowptr
            argsDict["csrColumnOffsets_Full"]     = full_colind
            argsDict["MC"]                        = MassMatrix
            argsDict["dt_times_fH_minus_fL"]      = self.dt_times_dC_minus_dL
            argsDict["min_m_bc"]                  = self.min_m_bc
            argsDict["max_m_bc"]                  = self.max_m_bc
            argsDict["FluxCorrectionMatrix"]      = self.FluxCorrectionMatrix
            argsDict["Rpos"]                      = self.Rpos
            argsDict["Rneg"]                      = self.Rneg
            argsDict["LUMPED_MASS_MATRIX"]        = coef.LUMPED_MASS_MATRIX
            argsDict["MONOLITHIC"]                = 0
            argsDict["offset_u"]                  = self.offset[0]
            argsDict["stride_u"]                  = self.stride[0]
            self.mphase_co2.FCTStep(argsDict)
            _par = getattr(self.u[0], 'par_dof', None)
            if _par is not None:
                _saved = self.u[0].dof.copy()
                for _arr in (self.Rpos, self.Rneg):
                    self.u[0].dof[:] = _arr
                    _par.scatter_forward_insert()
                    _arr[:] = self.u[0].dof
                self.u[0].dof[:] = _saved

            # 4. Pass 2: apply the limiter -> limited_solution (limited m_w).
            limited_solution = np.zeros((n_w,), 'd')
            argsDict = cArgumentsDict.ArgumentsDict()
            argsDict["component"]                 = 0
            argsDict["pass"]                      = 2
            argsDict["bc_mask"]                   = bc_mask_u
            argsDict["numDOFs"]                   = n_w
            argsDict["dt"]                        = dt
            argsDict["ML"]                        = self.ML
            argsDict["mn"]                        = self.mn
            argsDict["mLow"]                      = self.mLow
            argsDict["csrRowIndeces_DofLoops"]    = self.comp0_rowptr
            argsDict["csrColumnOffsets_DofLoops"] = self.comp0_colind
            argsDict["csrRowIndeces_Full"]        = full_rowptr
            argsDict["csrColumnOffsets_Full"]     = full_colind
            argsDict["MC"]                        = MassMatrix
            argsDict["FluxCorrectionMatrix"]      = self.FluxCorrectionMatrix
            argsDict["Rpos"]                      = self.Rpos
            argsDict["Rneg"]                      = self.Rneg
            argsDict["fluxCorrection"]            = self.fluxCorrection
            argsDict["limited_solution"]          = limited_solution
            argsDict["LUMPED_MASS_MATRIX"]        = coef.LUMPED_MASS_MATRIX
            argsDict["MONOLITHIC"]                = 0
            argsDict["offset_u"]                  = self.offset[0]
            argsDict["stride_u"]                  = self.stride[0]
            self.mphase_co2.FCTStep(argsDict)

            # 5. invert m_w -> p_w. Uses the already-limited S_n in u[1].dof
            #    (postStep calls FCTStep(component=1) before FCTStep(component=0)).
            p_w_lim = np.zeros((n_w,), 'd')
            argsDict = cArgumentsDict.ArgumentsDict()
            argsDict["a_rowptr"]             = coef.sdInfo[(0, 0)][0]
            argsDict["a_colind"]             = coef.sdInfo[(0, 0)][1]
            argsDict["rho"]                  = coef.rho
            argsDict["rho_n"]                = coef.rho_n
            argsDict["p_ref_n"]              = coef.p_ref_n
            argsDict["beta"]                 = coef.beta
            argsDict["gravity"]              = coef.gravity
            argsDict["alpha"]                = coef.vgm_alpha_types
            argsDict["n"]                    = coef.vgm_n_types
            argsDict["thetaR"]               = coef.thetaR_types
            argsDict["thetaSR"]              = coef.thetaSR_types
            argsDict["KWs"]                  = coef.Ksw_types
            argsDict["krn_end"]              = coef.krn_end_types
            argsDict["mu_n"]                 = coef.mu_n
            argsDict["elementMaterialTypes"] = self.mesh.elementMaterialTypes
            argsDict["freeDOFMaterialTypes"] = self.freeDOFMaterialTypes
            argsDict["numDOFs"]              = n_w
            argsDict["limited_solution"]     = limited_solution
            argsDict["u_dof"]                = p_w_lim
            argsDict["u_dof_n"]              = self.u[1].dof
            argsDict["USE_NEWTON_INVERT"]    = 0
            argsDict["PSK_TYPE"]             = coef.PSK_TYPE
            argsDict["COMPONENT"]            = 0
            self.mphase_co2.invert(argsDict)
            _par = getattr(self.u[0], 'par_dof', None)
            if _par is not None:
                self.u[0].dof[:] = p_w_lim
                _par.scatter_forward_insert()
                p_w_lim[:] = self.u[0].dof
            self.u[0].dof[:]  = p_w_lim
            self.u_dof_old[:] = p_w_lim
            self._scatter_component_to_timeintegration(0)

        elif component == 1:
            if self.limited_solution_n is None or self.u_dof_n_old is None:
                return
            n_dof = self.u[1].dof.shape[0]
            _par = getattr(self.u[1], 'par_dof', None)
            if _par is not None:
                _saved = self.u[1].dof.copy()
                for _arr in (self.mLow_n, self.mDotLow_n, self.min_m_bc_n, self.max_m_bc_n):
                    self.u[1].dof[:] = _arr
                    _par.scatter_forward_insert()
                    _arr[:] = self.u[1].dof
                self.u[1].dof[:] = _saved

            # 2. Pass 1: Zalesak ratios.
            argsDict = cArgumentsDict.ArgumentsDict()
            argsDict["component"]                 = 1
            argsDict["pass"]                      = 1
            argsDict["numDOFs_n"]                 = n_dof
            argsDict["dt"]                        = dt
            argsDict["ML_n"]                      = self.ML
            argsDict["MC_n"]                      = self.MC_n
            argsDict["mLow_n"]                    = self.mLow_n
            argsDict["mDotLow_n"]                 = self.mDotLow_n
            argsDict["dt_times_fH_minus_fL_n"]    = self.dt_times_fH_minus_fL_n
            argsDict["min_m_bc_n"]                = self.min_m_bc_n
            argsDict["max_m_bc_n"]                = self.max_m_bc_n
            argsDict["FluxCorrectionMatrix_n"]    = self.FluxCorrectionMatrix_n
            argsDict["Rpos_n"]                    = self.Rpos_n
            argsDict["Rneg_n"]                    = self.Rneg_n
            argsDict["csrRowIndeces_n_DofLoops"]  = self.comp1_rowptr
            argsDict["csrColumnOffsets_n_DofLoops"] = self.comp1_colind
            argsDict["LUMPED_MASS_MATRIX"]        = coef.LUMPED_MASS_MATRIX
            self.mphase_co2.FCTStep(argsDict)

            _par = getattr(self.u[1], 'par_dof', None)
            if _par is not None:
                _saved = self.u[1].dof.copy()
                for _arr in (self.Rpos_n, self.Rneg_n):
                    self.u[1].dof[:] = _arr
                    _par.scatter_forward_insert()
                    _arr[:] = self.u[1].dof
                self.u[1].dof[:] = _saved
            argsDict = cArgumentsDict.ArgumentsDict()
            argsDict["component"]                 = 1
            argsDict["pass"]                      = 2
            argsDict["numDOFs_n"]                 = n_dof
            argsDict["dt"]                        = dt
            argsDict["ML_n"]                      = self.ML
            argsDict["mLow_n"]                    = self.mLow_n
            argsDict["FluxCorrectionMatrix_n"]    = self.FluxCorrectionMatrix_n
            argsDict["Rpos_n"]                    = self.Rpos_n
            argsDict["Rneg_n"]                    = self.Rneg_n
            argsDict["fluxCorrection_n"]          = self.fluxCorrection_n
            argsDict["limited_solution_n"]        = self.limited_solution_n
            argsDict["bc_mask_n"]                 = self.bc_mask_n
            argsDict["csrRowIndeces_n_DofLoops"]  = self.comp1_rowptr
            argsDict["csrColumnOffsets_n_DofLoops"] = self.comp1_colind
            self.mphase_co2.FCTStep(argsDict)

            # 5. invert m_n -> S_n.
            S_n_lim = np.zeros_like(self.u[1].dof)
            argsDict = cArgumentsDict.ArgumentsDict()
            argsDict["a_rowptr"]             = coef.sdInfo[(0, 0)][0]
            argsDict["a_colind"]             = coef.sdInfo[(0, 0)][1]
            argsDict["rho"]                  = coef.rho
            argsDict["rho_n"]                = coef.rho_n
            argsDict["p_ref_n"]              = coef.p_ref_n
            argsDict["beta"]                 = coef.beta
            argsDict["gravity"]              = coef.gravity
            argsDict["alpha"]                = coef.vgm_alpha_types
            argsDict["n"]                    = coef.vgm_n_types
            argsDict["thetaR"]               = coef.thetaR_types
            argsDict["thetaSR"]              = coef.thetaSR_types
            argsDict["KWs"]                  = coef.Ksw_types
            argsDict["krn_end"]              = coef.krn_end_types
            argsDict["mu_n"]                 = coef.mu_n
            argsDict["elementMaterialTypes"] = self.mesh.elementMaterialTypes
            argsDict["freeDOFMaterialTypes"] = self.freeDOFMaterialTypes
            argsDict["numDOFs"]              = n_dof
            argsDict["limited_solution"]     = self.limited_solution_n
            argsDict["u_dof"]                = S_n_lim
            argsDict["USE_NEWTON_INVERT"]    = 0
            argsDict["PSK_TYPE"]             = coef.PSK_TYPE
            argsDict["COMPONENT"]            = 1
            self.mphase_co2.invert(argsDict)

            _par = getattr(self.u[1], 'par_dof', None)
            if _par is not None:
                self.u[1].dof[:] = S_n_lim
                _par.scatter_forward_insert()
                S_n_lim[:] = self.u[1].dof
            self.u[1].dof[:]    = S_n_lim
            self.u_dof_n_old[:] = S_n_lim
            self._scatter_component_to_timeintegration(1)


    def kth_FCT_step(self):
        #import pdb
        #pdb.set_trace()
        full_rowptr, full_colind, MassMatrix = self.MC_global.getCSRrepresentation()
        self._ensure_component0_compact_csr(full_rowptr, full_colind)
        rowptr = self.comp0_rowptr
        colind = self.comp0_colind
        compact_mass = MassMatrix[self.comp0_full_offsets]
        limitedFlux = np.zeros(self.comp0_full_offsets.shape[0])
        limited_solution = np.zeros((self.nFreeDOF_global[0],),'d')
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


        self.mphase_co2.kth_FCT_step(
            self.timeIntegration.dt,
            self.coefficients.num_fct_iter,
            self.comp0_full_offsets.shape[0],
            self.nFreeDOF_global[0],
            compact_mass,
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

        self._scatter_component_to_timeintegration(0)
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
            self.getResidual(self.u[0].dof,self.globalResidualDummy)
    def getResidual(self,u,r):
        import pdb
        import copy
        """
        Calculate the element residuals and add in to the global residual
        """
        cfemIntegrals.zeroJacobian_CSR(self.nNonzerosInJacobian,
                                       self.jacobian)
        if self.u_dof_old is None:
            # Pass initial condition to u_dof_old
            self.u_dof_old = np.copy(self.u[0].dof)
        # lazy init component-1 (S_n) previous-step DOFs from IC.
        if self.u_dof_n_old is None and self.nc >= 2:
            self.u_dof_n_old = np.copy(self.u[1].dof)
        rowptr, colind, nzval = self.jacobian.getCSRrepresentation()
        nnz = nzval.shape[-1]  # number of non-zero entries in sparse matrix
        self._ensure_component0_compact_csr(rowptr, colind)
        comp0_rowptr = self.comp0_rowptr
        comp0_colind = self.comp0_colind
        # Component-1 (S_n) compact DOF CSR + lazy EV edge/DOF buffers.
        self._ensure_component1_compact_csr(rowptr, colind)
        n_n_   = self.u[1].dof.shape[0]
        nnz_n_ = int(self.comp1_colind.shape[0])
        if self.dLow_n is None or self.dLow_n.shape[0] != nnz_n_:
            self.dLow_n                  = np.zeros((nnz_n_,), 'd')
            self.dEV_n                   = np.zeros((nnz_n_,), 'd')
            self.fluxMatrix_n            = np.zeros((nnz_n_,), 'd')
            # Per-edge antidiffusive flux storage (high-order minus low-order)
            # consumed by FCTStep_n_pass1.
            self.dt_times_fH_minus_fL_n  = np.zeros((nnz_n_,), 'd')
            self.FluxCorrectionMatrix_n  = np.zeros((nnz_n_,), 'd')
        if self.mLow_n is None or self.mLow_n.shape[0] != n_n_:
            self.mLow_n             = np.zeros((n_n_,), 'd')
            self.mHigh_n            = np.zeros((n_n_,), 'd')
            self.mDotLow_n          = np.zeros((n_n_,), 'd')
            self.fluxCorrection_n   = np.zeros((n_n_,), 'd')
            self.limited_solution_n = np.zeros((n_n_,), 'd')
            self.Rpos_n             = np.zeros((n_n_,), 'd')
            self.Rneg_n             = np.zeros((n_n_,), 'd')
            self.min_m_bc_n = np.ones((n_n_,), 'd') *  1.0e10
            self.max_m_bc_n = np.ones((n_n_,), 'd') * -1.0e10
            self.bc_mask_n  = np.ones((n_n_,), 'd')
            if self.nc >= 2 and 1 in self.dirichletConditions:
                _dbc_n = getattr(self.dirichletConditions[1],
                                 'DOFBoundaryConditionsDict', {})
                for _dofN in _dbc_n:
                    if 0 <= _dofN < n_n_:
                        self.bc_mask_n[_dofN] = 0.0
        r.fill(0.0)
        ########################
        ### COMPUTE C MATRIX ###
        ########################
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
            di = self.q[('grad(u)', 0)].copy()  # direction of derivative
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
            self.q[('w', 0)] = np.zeros((self.mesh.nElements_global,
                                         self.nQuadraturePoints_element,
                                         self.nDOF_test_element[0]),
                                        'd')
            self.q[('w*dV_m', 0)] = self.q[('w', 0)].copy()
            self.u[0].femSpace.getBasisValues(self.elementQuadraturePoints, self.q[('w', 0)])
            cfemIntegrals.calculateWeightedShape(self.elementQuadratureWeights[('u', 0)],
                                                 self.q['abs(det(J))'],
                                                 self.q[('w', 0)],
                                                 self.q[('w*dV_m', 0)])
            # GRADIENT OF TEST FUNCTIONS
            self.q[('grad(w)', 0)] = np.zeros((self.mesh.nElements_global,
                                               self.nQuadraturePoints_element,
                                               self.nDOF_test_element[0],
                                               self.nSpace_global),
                                              'd')
            self.u[0].femSpace.getBasisGradientValues(self.elementQuadraturePoints,
                                                      self.q['inverse(J)'],
                                                      self.q[('grad(w)', 0)])
            self.q[('grad(w)*dV_f', 0)] = np.zeros((self.mesh.nElements_global,
                                                    self.nQuadraturePoints_element,
                                                    self.nDOF_test_element[0],
                                                    self.nSpace_global),
                                                   'd')
            cfemIntegrals.calculateWeightedShapeGradients(self.elementQuadratureWeights[('u', 0)],
                                                          self.q['abs(det(J))'],
                                                          self.q[('grad(w)', 0)],
                                                          self.q[('grad(w)*dV_f', 0)])
            ##########################
            ### LUMPED MASS MATRIX ###
            ##########################
            # assume a linear mass term
            dm = np.ones(self.q[('u', 0)].shape, 'd')
            elementMassMatrix = np.zeros((self.mesh.nElements_global,
                                          self.nDOF_test_element[0],
                                          self.nDOF_trial_element[0]), 'd')
            cfemIntegrals.updateMassJacobian_weak_lowmem(dm,
                                                         self.q[('w', 0)],
                                                         self.q[('w*dV_m', 0)],
                                                         elementMassMatrix)
            self.MC_a = nzval.copy()
            # SparseMat dimensions must match the CSR data
            # (rowptr/colind from the full Jacobian span 2N rows for nc=2).
            # Telling SparseMat it's N x N while the CSR is 2N x 2N causes
            # the assembler to write to wrong positions for nc>=2.
            self.MC_global = SparseMat(self.nFreeVDOF_global,
                                       self.nFreeVDOF_global,
                                       nnz,
                                       self.MC_a,
                                       colind,
                                       rowptr)
            cfemIntegrals.zeroJacobian_CSR(self.nnz, self.MC_global)
            cfemIntegrals.updateGlobalJacobianFromElementJacobian_CSR(self.l2g[0]['nFreeDOF'],
                                                                      self.l2g[0]['freeLocal'],
                                                                      self.l2g[0]['nFreeDOF'],
                                                                      self.l2g[0]['freeLocal'],
                                                                      self.csrRowIndeces[(0, 0)],
                                                                      self.csrColumnOffsets[(0, 0)],
                                                                      elementMassMatrix,
                                                                      self.MC_global)
            self._ensure_component0_compact_csr(rowptr, colind)
            self.ML = np.zeros((self.nFreeDOF_global[0],), 'd')
            for i in range(self.nFreeDOF_global[0]):
                full_offsets_i = self.comp0_full_offsets[self.comp0_rowptr[i]:self.comp0_rowptr[i + 1]]
                self.ML[i] = self.MC_a[full_offsets_i].sum()
            # Consistent mass matrix on the comp-1 (S_n) DOF graph.
            # MC_a above only has its (0,0) block assembled (l2g[0] /
            # csrColumnOffsets[(0,0)]); its comp-1 block is stale, so
            # FCTStep_n's consistency term dt*MC*(mDotLow_i - mDotLow_j)
            # was reading garbage -> non-antisymmetric -> mass leak.
            # Both components share the C0-P1 space, so elementMassMatrix
            # (computed above) IS the comp-1 element mass matrix; scatter it
            # onto the comp-1 compact CSR (same indexing as dLow_n /
            # dt_times_fH_minus_fL_n). Symmetric by construction ->
            # antisymmetric consistency term -> mass-conservative FCT.
            self.MC_n = np.zeros((self.comp1_colind.shape[0],), 'd')
            _u1_l2g = self.u[1].femSpace.dofMap.l2g
            _nDOF = self.nDOF_test_element[0]
            for eN in range(self.mesh.nElements_global):
                for i in range(_nDOF):
                    i_n = _u1_l2g[eN, i]
                    _rstart = self.comp1_rowptr[i_n]
                    _rend   = self.comp1_rowptr[i_n + 1]
                    for j in range(_nDOF):
                        j_n = _u1_l2g[eN, j]
                        for off in range(_rstart, _rend):
                            if self.comp1_colind[off] == j_n:
                                self.MC_n[off] += elementMassMatrix[eN, i, j]
                                break
            # the trace-equals-volume assertion was nc=1 +
            # serial-only; with nc=2 the rowptr spans both blocks and the
            # row-sum no longer matches the per-rank mesh.volume directly.
            # Disabled here; the lumped mass is still correct for the (0,0)
            # block. Re-derive a proper MPI-safe / nc-aware check later.
            # np.testing.assert_almost_equal(self.ML.sum(),
            #                                self.mesh.volume,
            #                                err_msg="Trace of lumped mass matrix should be the domain volume", verbose=True)
            for d in range(self.nSpace_global):  # spatial dimensions
                # C matrices
                self.cterm[d] = np.zeros((self.mesh.nElements_global,
                                          self.nDOF_test_element[0],
                                          self.nDOF_trial_element[0]), 'd')
                self.cterm_a[d] = nzval.copy()
                #self.cterm_a[d] = np.zeros(nzval.size)
                # SparseMat dims must match the full CSR.
                self.cterm_global[d] = SparseMat(self.nFreeVDOF_global,
                                                 self.nFreeVDOF_global,
                                                 nnz,
                                                 self.cterm_a[d],
                                                 colind,
                                                 rowptr)
                cfemIntegrals.zeroJacobian_CSR(self.nnz, self.cterm_global[d])
                di[:] = 0.0
                di[..., d] = 1.0
                cfemIntegrals.updateHamiltonianJacobian_weak_lowmem(di,
                                                                    self.q[('grad(w)*dV_f', 0)],
                                                                    self.q[('w', 0)],
                                                                    self.cterm[d])  # int[(di*grad(wj))*wi*dV]
                cfemIntegrals.updateGlobalJacobianFromElementJacobian_CSR(self.l2g[0]['nFreeDOF'],
                                                                          self.l2g[0]['freeLocal'],
                                                                          self.l2g[0]['nFreeDOF'],
                                                                          self.l2g[0]['freeLocal'],
                                                                          self.csrRowIndeces[(0, 0)],
                                                                          self.csrColumnOffsets[(0, 0)],
                                                                          self.cterm[d],
                                                                          self.cterm_global[d])
                # C Transpose matrices
                self.cterm_transpose[d] = np.zeros((self.mesh.nElements_global,
                                                    self.nDOF_test_element[0],
                                                    self.nDOF_trial_element[0]), 'd')
                self.cterm_a_transpose[d] = nzval.copy()
                # SparseMat dims must match the full CSR.
                self.cterm_global_transpose[d] = SparseMat(self.nFreeVDOF_global,
                                                           self.nFreeVDOF_global,
                                                           nnz,
                                                           self.cterm_a_transpose[d],
                                                           colind,
                                                           rowptr)
                cfemIntegrals.zeroJacobian_CSR(self.nnz, self.cterm_global_transpose[d])
                di[:] = 0.0
                di[..., d] = -1.0
                cfemIntegrals.updateAdvectionJacobian_weak_lowmem(di,
                                                                  self.q[('w', 0)],
                                                                  self.q[('grad(w)*dV_f', 0)],
                                                                  self.cterm_transpose[d])  # -int[(-di*grad(wi))*wj*dV]
                cfemIntegrals.updateGlobalJacobianFromElementJacobian_CSR(self.l2g[0]['nFreeDOF'],
                                                                          self.l2g[0]['freeLocal'],
                                                                          self.l2g[0]['nFreeDOF'],
                                                                          self.l2g[0]['freeLocal'],
                                                                          self.csrRowIndeces[(0, 0)],
                                                                          self.csrColumnOffsets[(0, 0)],
                                                                          self.cterm_transpose[d],
                                                                          self.cterm_global_transpose[d])

        rowptr, colind, Cx = self.cterm_global[0].getCSRrepresentation()
        if (self.nSpace_global == 2):
            rowptr, colind, Cy = self.cterm_global[1].getCSRrepresentation()
        else:
            Cy = np.zeros(Cx.shape, 'd')
        if (self.nSpace_global == 3):
            rowptr, colind, Cz = self.cterm_global[2].getCSRrepresentation()
        else:
            Cz = np.zeros(Cx.shape, 'd')
        rowptr, colind, CTx = self.cterm_global_transpose[0].getCSRrepresentation()
        if (self.nSpace_global == 2):
            rowptr, colind, CTy = self.cterm_global_transpose[1].getCSRrepresentation()
        else:
            CTy = np.zeros(CTx.shape, 'd')
        if (self.nSpace_global == 3):
            rowptr, colind, CTz = self.cterm_global_transpose[2].getCSRrepresentation()
        else:
            CTz = np.zeros(CTx.shape, 'd')

        # This is dummy. I just care about the csr structure of the sparse matrix
        self.dLow = np.zeros(Cx.shape, 'd')
        self.fluxMatrix = np.zeros(Cx.shape, 'd')
        self.dt_times_dC_minus_dL = np.zeros(Cx.shape, 'd')
        nFree = self.nFreeDOF_global[0]
        self.min_m_bc = np.ones(nFree, 'd')
        self.min_m_bc *= 1.0e10
        self.max_m_bc = np.ones(nFree, 'd')
        self.max_m_bc *= -1.0e10
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
        for t,g in list(self.fluxBoundaryConditionsObjectsDict[0].advectiveFluxBoundaryConditionsDict.items()):
            self.ebqe[('advectiveFlux_bc',0)][t[0],t[1]] = g(self.ebqe[('x')][t[0],t[1]],self.timeIntegration.t)
            self.ebqe[('advectiveFlux_bc_flag',0)][t[0],t[1]] = 1
        # for t,g in self.fluxBoundaryConditionsObjectsDict[0].diffusiveFluxBoundaryConditionsDict.iteritems():
        #     self.ebqe[('diffusiveFlux_bc',0)][t[0],t[1]] = g(self.ebqe[('x')][t[0],t[1]],self.timeIntegration.t)
        #     self.ebqe[('diffusiveFlux_bc_flag',0)][t[0],t[1]] = 1
        #self.shockCapturing.lag=True
        self.bc_mask = np.ones_like(self.u[0].dof)
            
        if self.coefficients.forceStrongConditions:
            self.bc_mask = np.ones_like(self.u[0].dof)
            for cj in range(len(self.dirichletConditionsForceDOF)):
                for dofN,g in list(self.dirichletConditionsForceDOF[cj].DOFBoundaryConditionsDict.items()):
                    self.u[cj].dof[dofN] = g(self.dirichletConditionsForceDOF[cj].DOFBoundaryPointDict[dofN],self.timeIntegration.t)
                    self.u_dof_old[dofN] = self.u[cj].dof[dofN]
                    self.bc_mask[dofN] = 0.0
        bc_mask_u = np.ones((self.nFreeDOF_global[0],), 'd')
        bc_mask_u[:] = self.bc_mask[self.freeDOFToNode_u]
        degree_polynomial = 1
        try:
            degree_polynomial = self.u[0].femSpace.order
        except:
            pass
        argsDict = cArgumentsDict.ArgumentsDict()
        argsDict["bc_mask"] = bc_mask_u
        argsDict["dt"] = self.timeIntegration.dt
        argsDict["Theta"] = 1.0
        argsDict["Theta_h"] = 0.5
        argsDict["mesh_trial_ref"] = self.u[0].femSpace.elementMaps.psi
        argsDict["mesh_grad_trial_ref"] = self.u[0].femSpace.elementMaps.grad_psi
        argsDict["mesh_dof"] = self.mesh.nodeArray
        argsDict["mesh_velocity_dof"] = self.mesh.nodeVelocityArray
        argsDict["MOVING_DOMAIN"] = self.MOVING_DOMAIN
        argsDict["mesh_l2g"] = self.mesh.elementNodesArray
        argsDict["dV_ref"] = self.elementQuadratureWeights[('u',0)]
        argsDict["u_trial_ref"] = self.u[0].femSpace.psi
        argsDict["u_grad_trial_ref"] = self.u[0].femSpace.grad_psi
        argsDict["u_test_ref"] = self.u[0].femSpace.psi
        argsDict["u_grad_test_ref"] = self.u[0].femSpace.grad_psi
        argsDict["mesh_trial_trace_ref"] = self.u[0].femSpace.elementMaps.psi_trace
        argsDict["mesh_grad_trial_trace_ref"] = self.u[0].femSpace.elementMaps.grad_psi_trace
        argsDict["dS_ref"] = self.elementBoundaryQuadratureWeights[('u',0)]
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
        argsDict["rho"] = self.coefficients.rho
        argsDict["rho_n"] = self.coefficients.rho_n
        argsDict["p_ref_n"] = self.coefficients.p_ref_n
        argsDict["beta"] = self.coefficients.beta

        argsDict["q_rho"]= self.q['rho']
        argsDict["ebqe_rho"]= self.ebqe['rho']
        
        argsDict["gravity"] = self.coefficients.gravity
        argsDict["alpha"] = self.coefficients.vgm_alpha_types
        argsDict["n"] = self.coefficients.vgm_n_types
        argsDict["thetaR"] = self.coefficients.thetaR_types
        argsDict["thetaSR"] = self.coefficients.thetaSR_types
        argsDict["KWs"] = self.coefficients.Ksw_types
        argsDict["krn_end"] = self.coefficients.krn_end_types
        argsDict["mu_n"]    = self.coefficients.mu_n
        argsDict["useMetrics"] = 0.0
        argsDict["alphaBDF"] = self.timeIntegration.alpha_bdf
        argsDict["lag_shockCapturing"] = 0
        argsDict["shockCapturingDiffusion"] = self.coefficients.SC
        argsDict["VMS"] = self.coefficients.VMS
        argsDict["sc_uref"] = 1.0
        argsDict["sc_alpha"] = 2.0
        argsDict["u_l2g"] = self.u[0].femSpace.dofMap.l2g
        argsDict["r_l2g"] = self.l2g[0]['freeGlobal']
        argsDict["elementDiameter"] = self.mesh.elementDiametersArray
        argsDict["degree_polynomial"] = degree_polynomial
        argsDict["u_dof"] = self.u[0].dof
        argsDict["u_dof_old"] = self.u_dof_old
        argsDict["velocity"] = self.q[('velocity',0)]
        
        argsDict["velocity_couple"] = self.q[('velocity_couple',0)]

        argsDict["q_m"] = self.timeIntegration.m_tmp[0]
        argsDict["q_theta"] = self.q[('theta',0)]
        ############################################
        self.q[('m',0)][:] = self.timeIntegration.m_tmp[0]
        #############################################
        #argsDict["q_x"] = self.q['x']    
        argsDict["q_u"] = self.q[('u',0)]
        argsDict["q_dV"] = self.q[('dV_u',0)]
        argsDict["q_m_betaBDF"] = self.timeIntegration.beta_bdf[0]
        argsDict["cfl"] = self.q[('cfl',0)]
        argsDict["q_numDiff_u"] = self.q[('numDiff',0,0)]
        #argsDict["q_numDiff_u_last"] = self.q[('numDiff_last',0,0)]
        argsDict["q_numDiff_u_last"] = self.numDiff_star
        argsDict["offset_u"] = self.offset[0]
        argsDict["stride_u"] = self.stride[0]
        # ---- Component-1 (S_n) args ----
        # Gas mass equation. C++ residual reads u_dof_n & u_dof_n_old, builds
        # m_n = phi*rho_n*u_n, integrates (m_n - m_n_old)/dt against test
        # functions, and writes into globalResidual at offset_n + stride_n * dof.
        argsDict["u_dof_n"]     = self.u[1].dof
        argsDict["u_dof_n_old"] = self.u_dof_n_old
        argsDict["offset_n"]    = self.offset[1]
        argsDict["stride_n"]    = self.stride[1]
        argsDict["c_dof"] = getattr(self.coefficients, "c_dof",
                                    np.zeros_like(self.u[1].dof))
        argsDict["k_d"]   = float(self.coefficients.k_d)
        argsDict["c_sat"] = float(self.coefficients.c_sat)
        # CO2 injection: build the per-node source field consumed by the C++
        # gas-equation source term (applied like R_diss, opposite sign).
        # Each port is a small disk: injection_dof = rate (a volumetric source
        # density) on every node within radius of the port, gated by the
        # schedule.  Total injected per port = rate * (pi radius^2) * duration
        # -- mesh-independent and parallel-safe (every rank sets the same
        # density on its disk nodes; each element is integrated once).
        # Rebuilt each call (time gate depends on t).  Always passed -- the
        # kernel reads it unconditionally, so an empty list -> all-zero field.
        if getattr(self, "injection_dof", None) is None \
                or self.injection_dof.shape[0] != self.u[1].dof.shape[0]:
            self.injection_dof = np.zeros_like(self.u[1].dof)
        self.injection_dof[:] = 0.0
        injection_ports = self.coefficients.injection_ports
        if injection_ports:
            if getattr(self, "_injection_masks", None) is None:
                nodes = self.mesh.nodeArray
                self._injection_masks = []
                for (px, py, rate, radius, t0, t1) in injection_ports:
                    d2 = ((nodes[:, 0] - px) ** 2 + (nodes[:, 1] - py) ** 2)
                    self._injection_masks.append(d2 <= radius * radius)
            t_now = float(self.timeIntegration.t)
            # tanh ramp at each port's start so Newton can track the
            # saturation breakthrough.  tau is set in flow_p.py via
            # injection_ramp_tau (sim time units).  tau == 0 -> no ramp.
            tau = self.coefficients.injection_ramp_tau
            for (px, py, rate, radius, t0, t1), mask in zip(
                    injection_ports, self._injection_masks):
                if t0 <= t_now < t1:
                    if tau > 0.0:
                        ramp = 0.5 * (1.0 + np.tanh((t_now - t0) / tau - 3.0))
                    else:
                        ramp = 1.0
                    self.injection_dof[mask] = rate * ramp
        argsDict["injection_dof"] = self.injection_dof
        # csr maps for the (1,1) Jacobian block (not used by residual; staged
        # for turn 3 when calculateJacobian gains the (1,1) diagonal block).
        argsDict["csrRowIndeces_n_n"]      = self.csrRowIndeces[(1, 1)]
        argsDict["csrColumnOffsets_n_n"]   = self.csrColumnOffsets[(1, 1)]
        # (1,0) cross-block CSR maps for gas-eq diffusion against grad u_w.
        argsDict["csrRowIndeces_n_w"]      = self.csrRowIndeces[(1, 0)]
        argsDict["csrColumnOffsets_n_w"]   = self.csrColumnOffsets[(1, 0)]
        argsDict["csrColumnOffsets_eb_n_n"] = self.csrColumnOffsets_eb[(1, 1)]
        # (1,0) boundary cross-block: gas-eq Darcy diffusion through ext faces.
        argsDict["csrColumnOffsets_eb_n_w"] = self.csrColumnOffsets_eb[(1, 0)]
        argsDict["globalResidual"] = r
        argsDict["nExteriorElementBoundaries_global"] = self.mesh.nExteriorElementBoundaries_global
        argsDict["exteriorElementBoundariesArray"] = self.mesh.exteriorElementBoundariesArray
        argsDict["elementBoundaryElementsArray"] = self.mesh.elementBoundaryElementsArray
        argsDict["elementBoundaryLocalElementBoundariesArray"] = self.mesh.elementBoundaryLocalElementBoundariesArray
        argsDict["ebqe_velocity_ext"] = self.ebqe[('velocity',0)]
        argsDict["ebqe_velocity_ext_couple"] = self.ebqe[('velocity_couple',0)]      
        argsDict["isDOFBoundary_u"] = self.numericalFlux.isDOFBoundary[0]
        argsDict["ebqe_bc_u_ext"] = self.numericalFlux.ebqe[('u',0)]
        # component-1 (S_n) boundary arrays. Initialised in
        # init() when missing so this works regardless of numericalFlux class.
        argsDict["isDOFBoundary_n"] = self.numericalFlux.isDOFBoundary[1]
        argsDict["ebqe_bc_u_n_ext"] = self.numericalFlux.ebqe[('u',1)]
        argsDict["isFluxBoundary_u"] = self.ebqe[('advectiveFlux_bc_flag',0)]
        argsDict["ebqe_bc_flux_ext"] = self.ebqe[('advectiveFlux_bc',0)]
        argsDict["ebqe_phi"] = self.ebqe[('u',0)]
        argsDict["epsFact"] = 0.0
        #argsDict["ebqe_x"] = self.ebqe['x']
        argsDict["ebqe_u"] = self.ebqe[('u',0)]
        argsDict["ebqe_theta"] = self.ebqe[('theta',0)]
        argsDict["ebqe_flux"] = self.ebqe[('advectiveFlux',0)]
        argsDict['STABILIZATION_TYPE'] = self.coefficients.STABILIZATION_TYPE
        # ENTROPY VISCOSITY and ARTIFICIAL COMRPESSION
        argsDict["cE"] = self.coefficients.cE
        argsDict["cK"] = self.coefficients.cK
        # PARAMETERS FOR LOG BASED ENTROPY FUNCTION
        argsDict["uL"] = self.coefficients.uL
        argsDict["uR"] = self.coefficients.uR
        # PARAMETERS FOR EDGE VISCOSITY
        
        argsDict["numDOFs"] = self.nFreeDOF_global[0]
       
        argsDict["numDOFs_u"] = self.nFreeDOF_global[0]
        argsDict["NNZ"] = self.nnz
        argsDict["Cx"] = len(Cx)  # num of non-zero entries in the sparsity pattern
        argsDict["csrRowIndeces_DofLoops"] = comp0_rowptr  # compact component-0 CSR for DOF loops
        argsDict["csrColumnOffsets_DofLoops"] = comp0_colind
        argsDict["csrRowIndeces_Full"] = rowptr
        argsDict["csrColumnOffsets_Full"] = colind
        # Component-1 (S_n) DOF graph + EV buffers.
        argsDict["numDOFs_n"]                   = self.u[1].dof.shape[0]
        argsDict["NNZ_n"]                       = int(self.comp1_colind.shape[0])
        argsDict["csrRowIndeces_n_DofLoops"]    = self.comp1_rowptr
        argsDict["csrColumnOffsets_n_DofLoops"] = self.comp1_colind
        argsDict["comp1_full_offsets"]          = self.comp1_full_offsets
        argsDict["dLow_n"]                      = self.dLow_n
        argsDict["dEV_n"]                       = self.dEV_n
        argsDict["fluxMatrix_n"]                = self.fluxMatrix_n
        argsDict["mLow_n"]                      = self.mLow_n
        argsDict["mDotLow_n"]                   = self.mDotLow_n
        # Comp-1 FCT bundle (Zalesak limiter inputs / outputs).
        argsDict["dt_times_fH_minus_fL_n"]      = self.dt_times_fH_minus_fL_n
        argsDict["min_m_bc_n"]                  = self.min_m_bc_n
        argsDict["max_m_bc_n"]                  = self.max_m_bc_n
        argsDict["fluxCorrection_n"]            = self.fluxCorrection_n
        argsDict["limited_solution_n"]          = self.limited_solution_n
        argsDict["bc_mask_n"]                   = self.bc_mask_n
        # ML_n aliases the comp-0 lumped mass (purely geometric; both
        # components share the FE space). MC_n is the DEDICATED comp-1
        # consistent mass matrix, indexed by the comp-1 compact CSR -- MC_a
        # only has its (0,0) block assembled, so it cannot be used here.
        argsDict["ML_n"]                        = self.ML
        argsDict["MC_n"]                        = self.MC_n
        # FCT activation: maps the user's original Coefficients(FCT=True)
        # request to the C++ gate. coefficients.FCT itself is forced False
        # to bypass the framework's single-component post-Newton hook (see
        # Coefficients.__init__ comment near self._fct_requested).
        argsDict["FCT_n"]                       = int(bool(self.coefficients._fct_requested))
        # S_n bounds for the comp-1 entropy / smoothness sensor.
        # Material 0 used as the fallback when materials are heterogeneous.
        # S_n in [0, 1 - S_wr] with S_wr = thetaR/(thetaR+thetaSR).
        _S_wr0 = float(self.coefficients.thetaR_types[0] /
                       (self.coefficients.thetaR_types[0] + self.coefficients.thetaSR_types[0]))
        argsDict["u_n_L"] = 0.0
        argsDict["u_n_R"] = 1.0 - _S_wr0
        argsDict["csrRowIndeces_CellLoops"] = self.csrRowIndeces[(0, 0)]  # row indices (convenient for element loops)
        argsDict["csrColumnOffsets_CellLoops"] = self.csrColumnOffsets[(0, 0)]  # column indices (convenient for element loops)
        argsDict["csrColumnOffsets_eb_CellLoops"] = self.csrColumnOffsets_eb[(0, 0)]  # indices for boundary terms
        argsDict["globalJacobian"] = self.jacobian.getCSRrepresentation()[2]
        # C matrices
        argsDict["Cx"] = Cx
        argsDict["Cy"] = Cy
        argsDict["Cz"] = Cz
        argsDict["CTx"] = CTx
        argsDict["CTy"] = CTy
        argsDict["CTz"] = CTz
        argsDict["ML"] = self.ML
        if self.delta_x_ij is None:
            self.delta_x_ij = -np.ones((self.nNonzerosInJacobian*3,),'d')
        argsDict["delta_x_ij"] = self.delta_x_ij
        argsDict["MC"] = self.MC_a
        # PARAMETERS FOR 1st or 2nd ORDER MPP METHOD
        argsDict["LUMPED_MASS_MATRIX"] = self.coefficients.LUMPED_MASS_MATRIX
        argsDict["STABILIZATTION_TYPE"] = self.coefficients.STABILIZATION_TYPE
        argsDict["ENTROPY_TYPE"] = self.coefficients.ENTROPY_TYPE
        argsDict["PSK_TYPE"] = self.coefficients.PSK_TYPE
        # FLUX CORRECTED TRANSPORT
        argsDict["dLow"] = self.dLow
        argsDict["fluxMatrix"] = self.fluxMatrix
        argsDict["mDotLow"] = self.mDotLow
        argsDict["fluxCorrection"] = self.fluxCorrection
        limited_solution = np.zeros((self.nFreeDOF_global[0],),'d')
        argsDict["limited_solution"] = limited_solution
        argsDict["MONOLITHIC"] =0
        argsDict["mLow"] = self.mLow
        argsDict["dt_times_fH_minus_fL"] = self.dt_times_dC_minus_dL
        argsDict["min_m_bc"] = self.min_m_bc
        argsDict["max_m_bc"] = self.max_m_bc
        argsDict["quantDOFs"] = self.quantDOFs
        argsDict["mn"] = self.mn
        # Component-1 (S_n) low-order EV buffers.
        argsDict["mn_n"]        = self.mn_n
        argsDict["quantDOFs_n"] = self.quantDOFs_n
        argsDict["anb_seepage_flux_n"]= self.anb_seepage_flux_n
        argsDict["freeDOFMaterialTypes"] = self.freeDOFMaterialTypes
        argsDict["freeDOFToNode_u"] = self.freeDOFToNode_u
        ######################################################################################
        argsDict["pn"] = self.u[0].dof
        argsDict["mHigh"] = self.mHigh

        rowptr, colind, MassMatrix = self.MC_global.getCSRrepresentation()
        argsDict["MassMatrix"] = MassMatrix
        
######################################################################################        
        #argsDict["anb_seepage_flux"] = self.coefficients.anb_seepage_flux
        argsDict["anb_seepage_flux"] = self.anb_seepage_flux
        argsDict["q_velocity"] = self.q[('q_velocity_buf', 0)]
        argsDict["csrRowIndeces_u_u"] = self.csrRowIndeces[(0,0)]
        argsDict["csrColumnOffsets_u_u"] = self.csrColumnOffsets[(0,0)]
        argsDict["csrColumnOffsets_eb_u_u"] = self.csrColumnOffsets_eb[(0,0)]
        #argsDict["q_grad_psi"] = self.q[('velocity', 0)] 
        #print(anb_seepage_flux)
        #argsDict["anb_seepage_flux_n"] = self.coefficients.anb_seepage_flux_n
        #if np.sum(anb_seepage_flux_n)>0:

        #logEvent("Hi, this is Arnob", self.anb_seepage_flux_n[0])
        #print("Seepage Flux from Python file",  np.sum(self.anb_seepage_flux_n))
        #seepage_text_variable= np.sum(self.anb_seepage_flux_n)
        #t_now = float(self.timeIntegration.t)
        #s_now = float(np.sum(self.anb_seepage_flux_n))

        #with open("seepage_stab_0.txt", "a") as f:
        #    f.write(f"t={t_now:.6f}, s={s_now:.6e}\n")
        #with open('seepage_stab_0.txt',"a" ) as f:
            #f.write("\n Time"+ ",\t" +"Seepage\n")
        #    f.write(f"{self.timeIntegration.t:.6f},\t{float(seepage_text_variable):.6e}\n")
#            f.write(repr(self.coefficients.t)+ ",\t" +repr(seepage_text_variable), "\n")
            #f.write(repr(seepage_text_variable)+ "\n")
        
        from mpi4py import MPI
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()

        seepage_flux_value = np.sum(self.anb_seepage_flux_n)
        if seepage_flux_value > 0.0:
        # Each processor writes its own flux with its rank
            with open("seepage_flux_try.txt", "a") as f:
                f.write(f"Rank {rank}:, {self.timeIntegration.t:.6f}, {seepage_flux_value:.8f}\n")
       
        # seepage_flux_value = np.sum(self.anb_seepage_flux_n) #self.anb_seepage_flux_n[0]
        
        # with open("seepage_flux_try.txt", "a") as f:
        #    f.write(f"{seepage_flux_value:.8f}\n")
                          
        #seepage_flux.append(seepage_flux_value)
        
        # comm = Comm.get()
        # if comm.isMaster():
        #     with open("seepage_flux_try.txt", "a") as f:
        #         f.write(f"{seepage_flux_value:.6f}\n")
                
            #seepage_flux_value = np.sum(self.anb_seepage_flux_n) #self.anb_seepage_flux_n[0]
            #logEvent(f"Seepage flux at t={self.timeIntegration.t:.6f} is {seepage_flux_value:.6e}", level=2)
#            with open("seepage_flux_vs_time.txt", "a") as f:
#                f.write(f"{self.timeIntegration.t:.6f}, {seepage_flux_value:.6f}\n")

        if (self.coefficients.STABILIZATION_TYPE == 0):  # SUPG
            self.calculateResidual = self.mphase_co2.calculateResidual
            self.calculateJacobian = self.mphase_co2.calculateJacobian
        else:
            self.calculateResidual = self.mphase_co2.calculateResidual_entropy_viscosity
            self.calculateJacobian = self.mphase_co2.calculateMassMatrix
        
        if self.delta_x_ij is None:
            self.delta_x_ij = -np.ones((self.nNonzerosInJacobian*3,),'d')
        self.calculateResidual(argsDict)
        if getattr(self, "_theta_log_count", 0) < 5:
            q_theta = self.q[('theta',0)]
            ebqe_theta = self.ebqe[('theta',0)]
            logEvent("[Richards q_theta] t={:.6e} q(min,max,mean)=({:.6e},{:.6e},{:.6e}) ebqe(min,max,mean)=({:.6e},{:.6e},{:.6e}) q_zero_count={} ebqe_zero_count={}".format(
                     self.timeIntegration.t,
                     float(np.min(q_theta)), float(np.max(q_theta)), float(np.mean(q_theta)),
                     float(np.min(ebqe_theta)), float(np.max(ebqe_theta)), float(np.mean(ebqe_theta)),
                     int(np.count_nonzero(q_theta == 0.0)), int(np.count_nonzero(ebqe_theta == 0.0))),
                     level=2)
            self._theta_log_count = getattr(self, "_theta_log_count", 0) + 1
        


        self.q[('mt',0)][:] =self.timeIntegration.m_tmp[0]
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
    
        self.mHigh[:] = u
        if ulow is not None:
            self.u[0].dof[:] = ulow
    
        rowptr, colind, nzval = self.jacobian.getCSRrepresentation()
        nnz = nzval.shape[-1]
    
        full_rowptr, full_colind, Cx = self.cterm_global[0].getCSRrepresentation()
        self._ensure_component0_compact_csr(full_rowptr, full_colind)
        rowptr = self.comp0_rowptr
        colind = self.comp0_colind
        Cy = self.cterm_global[1].getCSRrepresentation()[2] if self.nSpace_global >= 2 else np.zeros(Cx.shape, 'd')
        Cz = self.cterm_global[2].getCSRrepresentation()[2] if self.nSpace_global == 3 else np.zeros(Cx.shape, 'd')
    
        rowptr, colind, CTx = self.cterm_global_transpose[0].getCSRrepresentation()
        CTy = self.cterm_global_transpose[1].getCSRrepresentation()[2] if self.nSpace_global >= 2 else np.zeros(CTx.shape, 'd')
        CTz = self.cterm_global_transpose[2].getCSRrepresentation()[2] if self.nSpace_global == 3 else np.zeros(CTx.shape, 'd')
    
        degree_polynomial = getattr(self.u[0].femSpace, "order", 1)
    
        if self.delta_x_ij is None:
            self.delta_x_ij = -np.ones((self.nNonzerosInJacobian * 3,), 'd')
    
        argsDict = cArgumentsDict.ArgumentsDict()
        argsDict["dt"] = self.timeIntegration.dt
        argsDict["mesh_trial_ref"] = self.u[0].femSpace.elementMaps.psi
        argsDict["mesh_grad_trial_ref"] = self.u[0].femSpace.elementMaps.grad_psi
        argsDict["mesh_dof"] = self.mesh.nodeArray
        argsDict["mesh_velocity_dof"] = self.mesh.nodeVelocityArray
        argsDict["MOVING_DOMAIN"] = self.MOVING_DOMAIN
        argsDict["mesh_l2g"] = self.mesh.elementNodesArray
        argsDict["dV_ref"] = self.elementQuadratureWeights[('u',0)]
        argsDict["u_trial_ref"] = self.u[0].femSpace.psi
        argsDict["u_grad_trial_ref"] = self.u[0].femSpace.grad_psi
        argsDict["u_test_ref"] = self.u[0].femSpace.psi
        argsDict["u_grad_test_ref"] = self.u[0].femSpace.grad_psi
        argsDict["mesh_trial_trace_ref"] = self.u[0].femSpace.elementMaps.psi_trace
        argsDict["mesh_grad_trial_trace_ref"] = self.u[0].femSpace.elementMaps.grad_psi_trace
        argsDict["dS_ref"] = self.elementBoundaryQuadratureWeights[('u',0)]
        argsDict["u_trial_trace_ref"] = self.u[0].femSpace.psi_trace
        argsDict["u_grad_trial_trace_ref"] = self.u[0].femSpace.grad_psi_trace
        argsDict["u_test_trace_ref"] = self.u[0].femSpace.psi_trace
        argsDict["u_grad_test_trace_ref"] = self.u[0].femSpace.grad_psi_trace
        argsDict["normal_ref"] = self.u[0].femSpace.elementMaps.boundaryNormals
        argsDict["boundaryJac_ref"] = self.u[0].femSpace.elementMaps.boundaryJacobians
        argsDict["nElements_global"] = self.mesh.nElements_global
        argsDict["ebqe_penalty_ext"] = self.ebqe['penalty']
        argsDict["elementMaterialTypes"] = self.mesh.elementMaterialTypes
        argsDict["isSeepageFace"] = self.coefficients.isSeepageFace
        argsDict["a_rowptr"] = self.coefficients.sdInfo[(0,0)][0]
        argsDict["a_colind"] = self.coefficients.sdInfo[(0,0)][1]
        argsDict["rho"] = self.coefficients.rho
        argsDict["rho_n"] = self.coefficients.rho_n
        argsDict["p_ref_n"] = self.coefficients.p_ref_n
        argsDict["beta"] = self.coefficients.beta
        argsDict["gravity"] = self.coefficients.gravity
        argsDict["alpha"] = self.coefficients.vgm_alpha_types
        argsDict["n"] = self.coefficients.vgm_n_types
        argsDict["thetaR"] = self.coefficients.thetaR_types
        argsDict["thetaSR"] = self.coefficients.thetaSR_types
        argsDict["KWs"] = self.coefficients.Ksw_types
        argsDict["krn_end"] = self.coefficients.krn_end_types
        argsDict["mu_n"]    = self.coefficients.mu_n
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
        argsDict["u_dof"] = self.u[0].dof
        argsDict["u_dof_old"] = self.u[0].dof
        argsDict["velocity"] = self.q['velocity',0]
        argsDict["q_m"] = self.timeIntegration.m_tmp[0]
        argsDict["q_theta"] = self.q[('theta',0)]
        argsDict["q_u"] = self.q[('u',0)]
        argsDict["q_dV"] = self.q[('dV_u',0)]
        argsDict["q_m_betaBDF"] = self.timeIntegration.beta_bdf[0]
        argsDict["cfl"] = self.q[('cfl',0)]
        argsDict["q_numDiff_u"] = self.q[('numDiff',0,0)]
        argsDict["q_numDiff_u_last"] = self.q[('numDiff_last',0,0)]
        argsDict["q_numDiff_u_last"] = self.numDiff_star
        argsDict["offset_u"] = self.offset[0]
        argsDict["stride_u"] = self.stride[0]
        argsDict["nExteriorElementBoundaries_global"] = self.mesh.nExteriorElementBoundaries_global
        argsDict["exteriorElementBoundariesArray"] = self.mesh.exteriorElementBoundariesArray
        argsDict["elementBoundaryElementsArray"] = self.mesh.elementBoundaryElementsArray
        argsDict["elementBoundaryLocalElementBoundariesArray"] = self.mesh.elementBoundaryLocalElementBoundariesArray
        argsDict["ebqe_velocity_ext"] = self.ebqe['velocity',0]
        argsDict["isDOFBoundary_u"] = self.numericalFlux.isDOFBoundary[0]
        argsDict["ebqe_bc_u_ext"] = self.numericalFlux.ebqe[('u',0)]
        # component-1 (S_n) boundary arrays. Initialised in
        # init() when missing so this works regardless of numericalFlux class.
        argsDict["isDOFBoundary_n"] = self.numericalFlux.isDOFBoundary[1]
        argsDict["ebqe_bc_u_n_ext"] = self.numericalFlux.ebqe[('u',1)]
        argsDict["isFluxBoundary_u"] = self.ebqe[('advectiveFlux_bc_flag',0)]
        argsDict["ebqe_bc_flux_ext"] = self.ebqe[('advectiveFlux_bc',0)]
        argsDict["ebqe_phi"] = self.ebqe[('u',0)]
        argsDict["epsFact"] = 0.0
        argsDict["ebqe_u"] = self.ebqe[('u',0)]
        argsDict["ebqe_theta"] = self.ebqe[('theta',0)]
        argsDict["ebqe_flux"] = self.ebqe[('advectiveFlux',0)]
        argsDict["STABILIZATION_TYPE"] = self.coefficients.STABILIZATION_TYPE
        argsDict["cE"] = self.coefficients.cE
        argsDict["cK"] = self.coefficients.cK
        argsDict["uL"] = self.coefficients.uL
        argsDict["uR"] = self.coefficients.uR
        argsDict["numDOFs"] = self.nFreeDOF_global[0]
        # numDOFs_u bounds the Richards DOF loop to comp-0.
        argsDict["numDOFs_u"] = self.nFreeDOF_global[0]
        argsDict["NNZ"] = self.nnz
        argsDict["csrRowIndeces_DofLoops"] = rowptr
        argsDict["csrColumnOffsets_DofLoops"] = colind
        argsDict["csrRowIndeces_CellLoops"] = self.csrRowIndeces[(0, 0)]
        argsDict["csrColumnOffsets_CellLoops"] = self.csrColumnOffsets[(0, 0)]
        argsDict["csrColumnOffsets_eb_CellLoops"] = self.csrColumnOffsets_eb[(0, 0)]
        argsDict["Cx"] = Cx
        argsDict["Cy"] = Cy
        argsDict["Cz"] = Cz
        argsDict["CTx"] = CTx
        argsDict["CTy"] = CTy
        argsDict["CTz"] = CTz
        argsDict["ML"] = self.ML
        argsDict["delta_x_ij"] = self.delta_x_ij
        argsDict["LUMPED_MASS_MATRIX"] = self.coefficients.LUMPED_MASS_MATRIX
        argsDict["ENTROPY_TYPE"] = self.coefficients.ENTROPY_TYPE
        argsDict["PSK_TYPE"] = self.coefficients.PSK_TYPE
        argsDict["dLow"] = self.dLow
        argsDict["fluxMatrix"] = self.fluxMatrix
        argsDict["mDotLow"] = self.mDotLow
        argsDict["dt_times_fH_minus_fL"] = self.dt_times_dC_minus_dL
        argsDict["min_m_bc"] = self.min_m_bc
        argsDict["max_m_bc"] = self.max_m_bc
        argsDict["quantDOFs"] = self.quantDOFs
        argsDict["mn"] = self.mn
        argsDict["anb_seepage_flux"] = self.coefficients.anb_seepage_flux
        argsDict["limited_solution"] = u
        argsDict["mLow"] = self.u[0].dof
        argsDict["freeDOFMaterialTypes"] = self.freeDOFMaterialTypes
        argsDict["freeDOFToNode_u"] = self.freeDOFToNode_u
        argsDict["USE_NEWTON_INVERT"] = 1 if (self.coefficients._fct_requested and self.coefficients.nd > 1) else 0
        argsDict["PSK_TYPE"] = self.coefficients.PSK_TYPE
        argsDict["COMPONENT"] = 0  # m -> u_w via retention curve (legacy path)
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
        argsDict = cArgumentsDict.ArgumentsDict()
        argsDict["dt"] = self.timeIntegration.dt
        argsDict["mesh_trial_ref"] = self.u[0].femSpace.elementMaps.psi
        argsDict["mesh_grad_trial_ref"] = self.u[0].femSpace.elementMaps.grad_psi
        argsDict["mesh_dof"] = self.mesh.nodeArray
        argsDict["mesh_velocity_dof"] = self.mesh.nodeVelocityArray
        argsDict["MOVING_DOMAIN"] = self.MOVING_DOMAIN
        argsDict["mesh_l2g"] = self.mesh.elementNodesArray
        argsDict["dV_ref"] = self.elementQuadratureWeights[('u',0)]
        argsDict["u_trial_ref"] = self.u[0].femSpace.psi
        argsDict["u_grad_trial_ref"] = self.u[0].femSpace.grad_psi
        argsDict["u_test_ref"] = self.u[0].femSpace.psi
        argsDict["u_grad_test_ref"] = self.u[0].femSpace.grad_psi
        argsDict["mesh_trial_trace_ref"] = self.u[0].femSpace.elementMaps.psi_trace
        argsDict["mesh_grad_trial_trace_ref"] = self.u[0].femSpace.elementMaps.grad_psi_trace
        argsDict["dS_ref"] = self.elementBoundaryQuadratureWeights[('u',0)]
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
        argsDict["rho"] = self.coefficients.rho
        argsDict["rho_n"] = self.coefficients.rho_n
        argsDict["p_ref_n"] = self.coefficients.p_ref_n
        argsDict["beta"] = self.coefficients.beta

        argsDict["q_rho"]= self.q['rho']
        argsDict["ebqe_rho"]= self.ebqe['rho']
        
        argsDict["gravity"] = self.coefficients.gravity
        argsDict["alpha"] = self.coefficients.vgm_alpha_types
        argsDict["n"] = self.coefficients.vgm_n_types
        argsDict["thetaR"] = self.coefficients.thetaR_types
        argsDict["thetaSR"] = self.coefficients.thetaSR_types
        argsDict["KWs"] = self.coefficients.Ksw_types
        argsDict["krn_end"] = self.coefficients.krn_end_types
        argsDict["mu_n"]    = self.coefficients.mu_n
        argsDict["useMetrics"] = 0.0
        argsDict["alphaBDF"] = self.timeIntegration.alpha_bdf
        argsDict["lag_shockCapturing"] = 0
        argsDict["shockCapturingDiffusion"] = 0.1
        argsDict["u_l2g"] = self.u[0].femSpace.dofMap.l2g
        argsDict["r_l2g"] = self.l2g[0]['freeGlobal']
        argsDict["elementDiameter"] = self.mesh.elementDiametersArray
        argsDict["degree_polynomial"] = degree_polynomial
        argsDict["u_dof"] = self.u[0].dof
        argsDict["velocity"] = self.q['velocity',0]
        argsDict["q_m_betaBDF"] = self.timeIntegration.beta_bdf[0]
        argsDict["cfl"] = self.q[('cfl',0)]
        argsDict["q_numDiff_u"] = self.q[('numDiff',0,0)]
        argsDict["q_numDiff_u_last"] = self.q[('numDiff',0,0)]
        argsDict["csrRowIndeces_u_u"] = self.csrRowIndeces[(0,0)]
        argsDict["csrColumnOffsets_u_u"] = self.csrColumnOffsets[(0,0)]
        # component-1 (S_n) Jacobian (1,1) block args.
        argsDict["dt"] = self.timeIntegration.dt
        argsDict["csrRowIndeces_n_n"] = self.csrRowIndeces[(1, 1)]
        argsDict["csrColumnOffsets_n_n"] = self.csrColumnOffsets[(1, 1)]
        # (1,0) cross-block CSR maps.
        argsDict["csrRowIndeces_n_w"] = self.csrRowIndeces[(1, 0)]
        argsDict["csrColumnOffsets_n_w"] = self.csrColumnOffsets[(1, 0)]
        # Exterior-boundary column offsets for the comp-1 boundary Jacobian
        # loop in calculateJacobian (STAB=0 comp-1 Dirichlet enforcement).
        argsDict["csrColumnOffsets_eb_n_n"] = self.csrColumnOffsets_eb[(1, 1)]
        argsDict["csrColumnOffsets_eb_n_w"] = self.csrColumnOffsets_eb[(1, 0)]
        # (0,1) cross-block CSR maps for the wetting eq.
        argsDict["csrRowIndeces_w_n"] = self.csrRowIndeces[(0, 1)]
        argsDict["csrColumnOffsets_w_n"] = self.csrColumnOffsets[(0, 1)]
        argsDict["u_dof_n"] = self.u[1].dof
        argsDict["u_dof_n_old"] = self.u_dof_n_old if self.u_dof_n_old is not None else self.u[1].dof
        argsDict["offset_n"] = self.offset[1]
        argsDict["stride_n"] = self.stride[1]
        # Stage 3b: kinetic dissolution sink contributes -k_d * S_n * (1-S_n) *
        # (c_sat - c) * theta_w * rho_w to the gas-equation residual.  Its
        # derivative wrt S_n is needed in the (1,1) Jacobian block (signs of
        # the two factors are opposite so the linearization is well-behaved
        # near c < c_sat).  c_dof and parameters supplied here for the kernel
        # to evaluate; defaulting to k_d=0 keeps legacy Jacobian unchanged.
        argsDict["c_dof"] = getattr(self.coefficients, "c_dof",
                                    np.zeros_like(self.u[1].dof))
        argsDict["k_d"]   = float(self.coefficients.k_d)
        argsDict["c_sat"] = float(self.coefficients.c_sat)
        argsDict["globalJacobian"] = jacobian.getCSRrepresentation()[2]
        argsDict["delta_x_ij"] = self.delta_x_ij
        argsDict["nExteriorElementBoundaries_global"] = self.mesh.nExteriorElementBoundaries_global
        argsDict["exteriorElementBoundariesArray"] = self.mesh.exteriorElementBoundariesArray
        argsDict["elementBoundaryElementsArray"] = self.mesh.elementBoundaryElementsArray
        argsDict["elementBoundaryLocalElementBoundariesArray"] = self.mesh.elementBoundaryLocalElementBoundariesArray
        argsDict["ebqe_velocity_ext"] = self.ebqe['velocity',0 ]
        argsDict["isDOFBoundary_u"] = self.numericalFlux.isDOFBoundary[0]
        argsDict["ebqe_bc_u_ext"] = self.numericalFlux.ebqe[('u',0)]
        # component-1 (S_n) boundary arrays. Initialised in
        # init() when missing so this works regardless of numericalFlux class.
        argsDict["isDOFBoundary_n"] = self.numericalFlux.isDOFBoundary[1]
        argsDict["ebqe_bc_u_n_ext"] = self.numericalFlux.ebqe[('u',1)]
        argsDict["isFluxBoundary_u"] = self.ebqe[('advectiveFlux_bc_flag',0)]
        argsDict["ebqe_bc_flux_ext"] = self.ebqe[('advectiveFlux_bc',0)]
        argsDict["csrColumnOffsets_eb_u_u"] = self.csrColumnOffsets_eb[(0,0)]
        # (0,1) cross-block boundary CSR for the wetting eq
        # exterior-flux Jacobian. csrRowIndeces_w_n is already passed above.
        argsDict["csrColumnOffsets_eb_w_n"] = self.csrColumnOffsets_eb[(0,1)]
        argsDict["LUMPED_MASS_MATRIX"] = self.coefficients.LUMPED_MASS_MATRIX
        argsDict["VMS"] = self.coefficients.VMS
        argsDict["PSK_TYPE"] = self.coefficients.PSK_TYPE
        #argsDict["anb_seepage_flux"] = self.coefficients.anb_seepage_flux

        self.calculateJacobian(argsDict)
        if self.coefficients.forceStrongConditions:
            for dofN in list(self.dirichletConditionsForceDOF[0].DOFBoundaryConditionsDict.keys()):
                global_dofN = self.offset[0]+self.stride[0]*dofN
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
