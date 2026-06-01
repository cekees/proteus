"""
TADR transport problem for the FluidFlower domain (start_again case).

Solves the dissolved-CO2 concentration c.  Two-way coupled with the
mphase_co2 flow model -- V_model=0 reads the brine Darcy velocity from
mphase_co2, and rho_f/rho_s feed the density back; the Stage-3b kinetic
dissolution (k_d, c_sat) is the gas -> dissolved-CO2 transfer.

c starts at 0 everywhere; dissolved CO2 is produced entirely by the
dissolution coupling -- exactly as in coupling_density_2layer.  The two
injection wells live on the gas side (flow_p.py's two ports); dissolution
then produces c at both automatically.
"""
from __future__ import absolute_import

import numpy as np

from proteus import *
from proteus.default_p import *
from proteus import Profiling
from proteus.mprans import TADR

try:
    from .domain import domain
except ImportError:
    from domain import domain


logEvent = Profiling.logEvent

nd = 2

# --- Module-level knobs read by transport_n.py -----------------------------
physicalDiffusion         = 0.0
shockCapturingFactor_tadr = 0.0
lag_shockCapturing_tadr   = True
checkMass                 = False
parallel                  = True

# --- Dispersion / molecular diffusion --------------------------------------
# (m, h) units: D_CO2 in water ~ 2e-9 m^2/s = 7.2e-6 m^2/h at 20 C.
Dm      = 7.2e-6
alpha_L = 0.0
alpha_T = 0.0


def a(x):
    return np.array([[Dm, 0.0], [0.0, Dm]])


aOfX = {0: a}


# Dump the full TADR velocity field on the first VEL_SNAPSHOT_FIRST_N
# reporting steps (one file CCS_vel_step{n}_t{t}.txt each).  These are the
# very first times, when the injection ramps up and |v| (hence the CFL limit)
# is at its sharpest.  Bump this to capture more early steps.
VEL_SNAPSHOT_FIRST_N = 8


class MyCoefficients(TADR.Coefficients):
    def postStep(self, t, firstStep=False):
        copyInstructions = super().postStep(t, firstStep)

        # Local-equilibrium dissolution flash (k_d -> inf limit).  Runs here,
        # after BOTH the flow (S_n) and transport (c) solves have converged
        # (Sequential_MinModelStep advances flow, then transport), as a
        # once-per-step nodal gas<->brine CO2 exchange in C++.  flowCoefficients
        # is the mphase_co2 Coefficients (set in TADR.attachModels via V_model);
        # it mutates the flow S_n and this model's c in place and repairs both
        # models' time history.  No-op unless dissolution_mode='flash'.
        flow_coeffs = getattr(self, 'flowCoefficients', None)
        if flow_coeffs is not None and hasattr(flow_coeffs, 'apply_dissolution_flash'):
            flow_coeffs.apply_dissolution_flash(t)

        from mpi4py import MPI
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()

        # q_v is the brine Darcy velocity TADR advects with, refreshed from
        # mphase_co2's ('velocity_couple',0) in preStep.  Lives at element
        # quadrature points; restrict to owned elements to avoid ghost dupes.
        n_own = self.model.mesh.nElements_owned
        qv = np.asarray(self.q_v)[:n_own]                              # (nE,nQ,nd)
        qx = np.asarray(self.model.q['x'])[:n_own]                     # (nE,nQ,3)
        h_e = np.asarray(self.model.mesh.elementDiametersArray)[:n_own]  # (nE,)
        vmag = np.sqrt(np.sum(qv * qv, axis=-1))                       # (nE,nQ)

        # --- per-step CFL diagnostic: where/how hard velocity bites dt -------
        if vmag.size:
            vmax_e = vmag.max(axis=1)                  # per-element peak |v|
            local_vmax = float(vmax_e.max())
            fe = int(np.argmax(vmax_e)); fq = int(np.argmax(vmag[fe]))
            local_x, local_y = float(qx[fe, fq, 0]), float(qx[fe, fq, 1])
            local_dt = float((h_e / np.maximum(vmax_e, 1.0e-30)).min())
            local_sum, local_cnt = float(vmag.sum()), int(vmag.size)
        else:
            local_vmax, local_x, local_y = -1.0, float('nan'), float('nan')
            local_dt, local_sum, local_cnt = float('inf'), 0.0, 0

        g_vmax = comm.allreduce(local_vmax, op=MPI.MAX)
        g_dt = comm.allreduce(local_dt, op=MPI.MIN)
        g_sum = comm.allreduce(local_sum, op=MPI.SUM)
        g_cnt = comm.allreduce(local_cnt, op=MPI.SUM)
        winners = comm.gather((local_vmax, local_x, local_y), root=0)
        if rank == 0:
            gx, gy = max(winners, key=lambda w: w[0])[1:]
            g_mean = g_sum / g_cnt if g_cnt else float('nan')
            logEvent(
                "[TADR CFL] t={:.6e} max|v|={:.6e} at (x,y)=({:.4f},{:.4f}) "
                "mean|v|={:.6e} dt_cfl=min(h/|v|)={:.6e}".format(
                    float(t), g_vmax, gx, gy, g_mean, g_dt), level=1)

        # --- full-field snapshot dump on the first N reporting steps -------
        # Quadrature coords are static, so write x,y ONCE (CCS_vel_coords.txt);
        # each step writes only vx,vy.  Row ordering matches the coords file
        # (same owned-element order, same rank order in comm.gather).
        if not hasattr(self, "_vel_snap_n"):
            self._vel_snap_n = 0
        if self._vel_snap_n < VEL_SNAPSHOT_FIRST_N:
            n = self._vel_snap_n
            self._vel_snap_n += 1
            if n == 0:
                xy = (np.column_stack([qx[..., 0].ravel(), qx[..., 1].ravel()])
                      if vmag.size else np.empty((0, 2)))
                xy_all = comm.gather(xy, root=0)
                if rank == 0:
                    C = np.vstack(xy_all)
                    np.savetxt("CCS_vel_coords.txt", C, fmt="%.10e",
                               header="cols: x y  (matches row order of CCS_vel_step*.txt)")
                    logEvent("[TADR CFL] wrote CCS_vel_coords.txt rows={}".format(C.shape[0]), level=1)
            v = (np.column_stack([qv[..., 0].ravel(), qv[..., 1].ravel()])
                 if vmag.size else np.empty((0, 2)))
            v_all = comm.gather(v, root=0)
            if rank == 0:
                V = np.vstack(v_all)
                fname = "CCS_vel_step{:03d}_t{:.6e}.txt".format(n, float(t))
                np.savetxt(fname, V, fmt="%.10e",
                           header="step={} t_actual={:.10e} cols: vx vy".format(n, float(t)))
                logEvent("[TADR CFL] wrote {} rows={}".format(fname, V.shape[0]), level=1)

        return copyInstructions


LevelModelType = TADR.LevelModel
coefficients = MyCoefficients(
    aOfX,
    alpha_L=alpha_L,
    alpha_T=alpha_T,
    Dm=Dm,
    porosity=1.0,                  # aliased to phi*S_w via V_model
    V_model=0,                     # mphase_co2 sits at index 0
    specified_velocity=False,
    checkMass=False,
    FCT=False,
    LUMPED_MASS_MATRIX=True,
    STABILIZATION_TYPE=5,
    diagonal_conductivity=True,
    ENTROPY_TYPE="POWER",
    cE=0.1,
    cK=1.0,
    physicalDiffusion=0.0,
    rho_f=1.0,
    rho_s=1.010,
    forceStrongConditions=True,
    # Dissolution is handled by the local-equilibrium flash (see flow_p.py /
    # mphase_co2.apply_dissolution_flash), so the in-residual kinetic TADR
    # source MUST be off here to avoid double counting.
    k_d= 0.0, #0.01, #0.0008 , #1.e-3,  (flash mode: kinetic source disabled)
    c_sat=1.0,
)
coefficients.variableNames = ["c"]


# --- Initial condition -----------------------------------------------------
# c = 0 everywhere; dissolved CO2 is produced entirely by the Stage-3b
# dissolution coupling (gas dissolving into the brine), exactly as in
# coupling_density_2layer.  The two injection wells come from the two gas
# ports in flow_p.py -- dissolution produces c at both automatically.
c_sat = 1.0


class IC_c:
    def uOfXT(self, x, t):
        return 0.0


initialConditions = {0: IC_c()}


# --- Boundary conditions ---------------------------------------------------
# Closed system on c: no Dirichlet anywhere, zero flux on every wall.
def getDBC_c(x, flag):
    return None


dirichletConditions = {0: getDBC_c}


def getZeroAdvFlux(x, flag):
    return lambda x, t: 0.0


def getZeroDiffFlux(x, flag):
    return lambda x, t: 0.0


fluxBoundaryConditions = {0: "setFlow"}
advectiveFluxBoundaryConditions = {0: getZeroAdvFlux}
diffusiveFluxBoundaryConditions = {0: {0: getZeroDiffFlux}}
