"""
mphase_co2 flow problem for the FluidFlower domain (start_again case).

UNIT SYSTEM: head form, (length = m, time = hour, rho_w = 1, g = 1).
Pressure is interpreted as pressure HEAD in metres of water; the solver
sees q_w = -k_rw * KsTypes * (grad h - z_hat), the textbook Richards form.

domain.py builds the FluidFlower PSLG and keeps the 16 original facies region
flags (1..16).  Each flag maps to one of the six FluidFlower sands (ESF, C, D,
E, F, G) or the silicone-sealed fault.  Per-sand Brooks-Corey parameters are
from the Nordbotten Final Benchmark Description (Tables 3-6).  Conversions:
  KsTypes [m/h] = K [Darcy] * 9.869233e-13 * rho_w_phys * g_phys / mu_w * 3600
  alphaBC [1/m] = 1 / (p_d [Pa] / (rho_w_phys * g_phys))     (p_d in m head)
  thetaR, thetaSR, lambdaBC  already dimensionless -- used as-is

Initial conditions (Nordbotten 2.1, 2.6): domain 100% water-filled (S_n = 0),
hydrostatic head h(y) = (H_WT - y) with water table H_WT = 1.50 m.
Boundaries: bottom, left and right closed (no-flow); top open (Dirichlet h
at the water-table head, S_n = 0).  CO2 enters through two timed disk sources
at the injection ports (0.90, 0.30) and (1.70, 0.70) -- see injection_ports.
"""
from proteus import Domain
from proteus.default_p import *
from proteus.mphase_co2 import mphase_co2

import numpy as np

try:
    from .domain import domain, vertices, triangleOptions
except ImportError:
    from domain import domain, vertices, triangleOptions


nd = 2
nLevels = 1

# domain extent (FluidFlower bounding box), from the txt vertices
Lx = max(v[0] for v in vertices)
Lz = max(v[1] for v in vertices)

# Free water table elevation above the bottom (Nordbotten 2.1) [m].
H_WT = 1.50


# --- Brooks-Corey closure: six FluidFlower sands + silicone fault ----------
# Source data: Nordbotten Final Benchmark Description -- Table 3 (K),
# Table 4 (phi), Table 5 (S_wi), Table 6 (entry pressure p_d).  lambda = 2.0
# is the Flemisch et al. (2024) participant consensus.
nMediaTypes = 16

# Physical reference constants used only for converting Darcy -> m/h and
# Pa -> m of head.  They do NOT appear in the solver inputs.
RHO_W_PHYS = 1002.0          # brine density [kg/m^3] (Nordbotten 2.5)
G_PHYS     = 9.81            # gravity [m/s^2]
MU_W_PHYS  = 1.0e-3          # brine viscosity [Pa.s]
DARCY      = 9.869233e-13    # 1 Darcy in m^2
SEC_PER_HR = 3600.0

# K_w [m/h] per Darcy = k [m^2/Darcy] * rho * g / mu * 3600 s/h
DARCY_TO_KW_HOUR = DARCY * RHO_W_PHYS * G_PHYS / MU_W_PHYS * SEC_PER_HR
# p_d [Pa] -> p_d in m of head: divide by rho_w * g
PA_TO_M_HEAD = 1.0 / (RHO_W_PHYS * G_PHYS)

# key : (K [Darcy], lambdaBC, p_d [Pa], thetaR, thetaSR, krn_end)
# Nordbotten Table 6 reports p_d = 0 (below experimental detection) for E/F/G.
# Brooks-Corey is mathematically undefined at p_d = 0 (alphaBC = 1/0).  Per
# Nordbotten Section 2.3 "open modelling choices", we assign a small finite
# p_d for E/F/G as a numerical regularisation.  10 Pa (~1 mm of head) is the
# CSIRO/Flemisch participant convention -- well below the experimental
# detection threshold of ~10 mbar (1000 Pa) and below sand D's measured 98.1
# Pa.  This is a numerical choice, not a measured value, and is documented
# as such in any benchmark write-up.  FAULT (flag 15) is the silicone-sealed
# fault: a near-impermeable barrier (K = 1e-7 * K_F, high entry pressure).
# krn_end is the end-point gas relative permeability from Nordbotten Table 5
# (k_rel,gas at S_w = S_wi).  Brooks-Corey gives k_rn(S_e=0) = 1 for every
# sand; without this scaling the gas plume migrates 5-50x too fast.
K_F_DARCY = 4259.0
SAND = {
    # Brooks-Corey parameters from Nordbotten Tables 3-6 (un-boosted lab values).
    # ESF p_d = 1471.5 Pa is the measured capillary entry pressure.  Calibration
    # bumps to 5x and 10x produced the same plume shape (gas reaches cap, pool
    # stays at dissolution-injection equilibrium, no lateral spread) -- so
    # cap strength is not the limiting mechanism for lateral pool growth.
    # ESF krn_end was 0.09 (Nordbotten Table 5, low-confidence per benchmark).
    # Lowered to 0.01 to reduce gas mobility through the seal: with krn=0.09
    # the gas/water mobility ratio in ESF is ~9, so gas zips through once the
    # entry pressure is crossed and the pancake-under-seal never forms.  0.01
    # is well within the experimental uncertainty noted in the FBD.
    "ESF":   (   44.0,           2.0, 1471.5,           0.1392, 0.2958, 0.01),
    "C":     (  473.0,           2.0,  294.3,           0.0609, 0.3741, 0.05),
    "D":     ( 1110.0,           2.0,   98.1,           0.0528, 0.3872, 0.02),
    "E":     ( 2005.0,           2.0,   10.0,           0.0540, 0.3960, 0.10),
    "F":     ( 4259.0,           2.0,   10.0,           0.0528, 0.3872, 0.11),
    "G":     ( 9580.0,           2.0,   10.0,           0.0450, 0.4050, 0.16),
    # FAULT (flag 15): silicone-sealed fault; keep ESF params so it's at
    # least as restrictive as the seal facies.  krn_end lowered to match ESF.
    "FAULT": (   44.0,           2.0, 1471.5,           0.1392, 0.2958, 0.01),
}

# region flag (1..16) -> sand key  (plot_fluid_flower_sand.py: region_to_sand)
# Flags 6 and 12 host the two injection-port regions (markers 1.62,0.75 and
# 0.81,0.25).  They are mapped to "F" (very coarse upper, the main reservoir
# facies in Nordbotten et al. 2022 Fig. 4) so the injection sites match the
# paper geometry and the FluidFlower_sand visualization.
FLAG_TO_SAND = {
     1: "G",   2: "ESF",  3: "C",      4: "D",
     5: "E",   6: "F",    7: "D",      8: "E",
     9: "C",  10: "D",   11: "ESF",   12: "F",
    13: "G",  14: "G",   15: "FAULT", 16: "G",
}

KsTypes       = np.zeros((nMediaTypes + 1, nd), 'd')
lambdaBCtypes = np.zeros(nMediaTypes + 1, 'd')
alphaBCtypes  = np.zeros(nMediaTypes + 1, 'd')
thetaRtypes   = np.zeros(nMediaTypes + 1, 'd')
thetaSRtypes  = np.zeros(nMediaTypes + 1, 'd')
krnEndTypes   = np.zeros(nMediaTypes + 1, 'd')

# Fill region flags 1..16; index 0 is a fallback mphase_co2 reads
# unconditionally -- default it to sand G (the bulk reservoir sand).
for flag in range(nMediaTypes + 1):
    K_D, lam, p_d, thetaR, thetaSR, krn_end = SAND[FLAG_TO_SAND.get(flag, "G")]
    K_w_hour            = K_D * DARCY_TO_KW_HOUR        # [m/h]
    p_d_head            = p_d * PA_TO_M_HEAD            # [m of head]
    KsTypes[flag, :]    = [K_w_hour, K_w_hour]
    lambdaBCtypes[flag] = lam
    alphaBCtypes[flag]  = 1.0 / p_d_head                # [1/m]
    thetaRtypes[flag]   = thetaR
    thetaSRtypes[flag]  = thetaSR
    krnEndTypes[flag]   = krn_end


# --- Physical constants (head-form scaling) -------------------------------
# rho_w = 1, g = 1 by construction -> "pressure" is head in metres.
rho_f   = 1.0
g_mag   = 1.0
gravity = np.array([0.0, -g_mag, 0.0])


# --- CO2 injection disk sources -------------------------------------------
# (x, y, rate, radius, t_start, t_stop): each port is a small disk source on
# the gas (S_n) equation, active while t_start <= t < t_stop.  rate has units
# of (rho_w-mass per unit pore volume per hour), so:
#   total injected per port [rho_w * m^2, per m of out-of-plane depth]
#     = rate * (pi * radius^2) * (t_stop - t_start)
# To relax the advective CFL we WIDEN the source disk and DROP the rate so the
# injected mass per port is unchanged.  Holding mass = rate * pi * r^2 * dt
# fixed gives  rate(r) = RATE_REF * (R_REF / r)^2.  Widening r lowers the peak
# Darcy velocity at the port ~ (R_REF/r)^2, which is what enlarges the stable
# dt.  Change INJ_RADIUS only; INJ_RATE follows automatically.
R_REF      = 0.04                              # reference (paper) radius [m]
RATE_REF   = 8.76e-3                           # rate at R_REF -> paper mass
INJ_RADIUS = 0.04                              # widened source-disk radius [m]
INJ_RATE   = RATE_REF * (R_REF / INJ_RADIUS) ** 2   # mass-preserving rate
INJ_STOP   = 5.0                               # 5 h injection end (in hours)

INJ_RAMP_TAU = 10.0 / 3600.0
injection_ports = [
    (0.90, 0.30, INJ_RATE, INJ_RADIUS, 0.0,             INJ_STOP),
    (1.70, 0.70, INJ_RATE, INJ_RADIUS, 0.45 * INJ_STOP, INJ_STOP),
]


LevelModelType = mphase_co2.LevelModel
coefficients = mphase_co2.Coefficients(
    nd,
    KsTypes,
    lambdaBCtypes,
    alphaBCtypes,
    thetaRtypes,
    thetaSRtypes,
    gravity=gravity,
    density=rho_f,
    rho_n=0.0018,            # CO2/water density ratio (1.8 / 1002) at p_n=0 (atmospheric)
    beta=0.0001, #4.5e-6,
    # Compressible CO2: exponential EOS rho_n(p_n)=rho_n*exp(p_n/p_ref_n).
    # p_ref_n = atmospheric in head ~ P_atm/(rho_w*g) = 101325/(1002*9.81) ~ 10.3 m,
    # so beta_n = 1/p_ref_n ~ 0.1 /m (CO2 ~ ideal gas near atmospheric).  This lets a
    # saturating cell absorb injected gas via pressure rise instead of S_n overshoot.
    # Set p_ref_n=0.0 to revert to incompressible (constant rho_n).
    p_ref_n=10.3,  # atmospheric in head (physical CO2 EOS); was 5. (debug leftover),
                   # which doubled both the EOS amplification and the Newton stiffness
                   # at the F->ESF breakthrough front.
    PSK_TYPE='BC',
    diagonal_conductivity=True,
    density_model=1,             
    STABILIZATION_TYPE=2,
    ENTROPY_TYPE=1,
    LUMPED_MASS_MATRIX=False,
    FCT=False,
    num_fct_iter=0,
    cE=1.0, uL=0.0, uR=1.0, cK=1.0,
    outputQuantDOFs=False,
    # Finite-rate implicit dissolution: k_d is the dissolution RATE [1/h] of the
    # local-driving-force relaxation toward equilibrium (tau_diss ~ 1/(k_d*S_n)).
    # k_d -> inf recovers the instantaneous flash (dissolves in place, no rise);
    # smaller k_d lets the free-gas plume rise before it dissolves.  TUNE HERE:
    # raise k_d for more dissolution / less rise, lower it for more rise.
    k_d=0.1, #0.01,
    c_sat=1.0,
    dissolution_mode='flash',
    X_sat=0.0015,
    injection_ports=injection_ports,
    injection_ramp_tau=INJ_RAMP_TAU,
    krn_end_types=krnEndTypes,
    mu_n=0.015 , #0.015, #0.015,
    reconstruct_velocity_rt0=False,
)


# --- Initial conditions ----------------------------------------------------
class IC_pw:
    """Hydrostatic head referenced to the free water table at y = H_WT:
        h(x,0) = H_WT - y   [m of head]
    Note: H_WT = 1.50 m sits ~0.20 m above the porous-medium top, so the top
    of the sand has h ~ 0.20 m > 0 (water column above the rig)."""

    def uOfXT(self, x, t):
        return H_WT - x[1]


class IC_Sn:
    """Domain initially 100% water-filled (Nordbotten 2.6): S_n = 0
    everywhere.  CO2 enters only through the timed injection disk sources."""

    def uOfXT(self, x, t):
        return 0.0


initialConditions = {0: IC_pw(), 1: IC_Sn()}


# --- Boundary conditions (Nordbotten 2.1, sealed-top variant) -------------
# Bottom, left, right: closed (no-flow on both p_w and S_n).
# Top: Dirichlet on p_w only (hydrostatic head to the water table) -- this
#      pins the pressure datum.  S_n is no-flow at the top so gas POOLS at
#      the surface rather than being vented (FluidFlower-style sealed cell).
#      Mass-balance diagnostic showed the original S_n=0 Dirichlet was
#      removing ~95% of injected CO2 through the top.
# Boundaries are picked out by coordinate (domain.py tags edges
# marker_1..16, not named sides); the top is the y ~ Lz edge.
TOL = 1.0e-6


def _on_top(x):
    return x[1] > Lz - TOL


def getDBC_pw(x, flag):
    # Open top for pressure: hydrostatic head to the water table.
    if _on_top(x):
        return lambda x, t: H_WT - x[1]
    return None


def getDBC_Sn(x, flag):
    # No Dirichlet anywhere for gas saturation -- sealed system on S_n.
    return None


dirichletConditions = {0: getDBC_pw, 1: getDBC_Sn}


fluxBoundaryConditions = {0: 'noFlow', 1: 'noFlow'}


def getNoFlow_pw(x, flag):
    # Bottom, left, right: no-flow on p_w.  Top: Dirichlet governs (return None).
    if _on_top(x):
        return None
    return lambda x, t: 0.0


def getNoFlow_Sn(x, flag):
    # Closed gas boundary on EVERY face including the top.
    return lambda x, t: 0.0


advectiveFluxBoundaryConditions = {0: getNoFlow_pw, 1: getNoFlow_Sn}
diffusiveFluxBoundaryConditions = {0: {0: getNoFlow_pw, 1: getNoFlow_Sn},
                                   1: {0: getNoFlow_pw, 1: getNoFlow_Sn}}
