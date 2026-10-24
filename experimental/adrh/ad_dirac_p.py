"""Transient linear advection-diffusion-reaction with the Dirac-IC Gaussian, as a plain Proteus model: the
Python-path reference for the ADRH engine's time integration (ad_dirac_regression.py).

    u_t + div(b u - kappa grad u) + c u = 0 on the unit square or cube (AD_ND = 2 or 3),
    exact solution AnalyticalSolutions.LinearADR_Decay_DiracIC(n=nd, b, a=kappa, c, tStart, u0, x0=xc),
    initial condition the exact solution at T = 0 (physical time tStart), boundary data from it as AD_BC says.

Coefficients: proteus's LinearVADR_ConstantCoefficients (M = 1, A = kappa I, B = b, C = c). Galerkin, no
stabilization. Parameters come from the environment (AD_*), set by the driver.
"""
import os
import numpy as np
from proteus import *
from proteus.default_p import *
from proteus.TransportCoefficients import LinearVADR_ConstantCoefficients
from proteus.AnalyticalSolutions import LinearADR_Decay_DiracIC

nd = int(os.environ.get('AD_ND', '3'))
SPACE = os.environ.get('AD_SPACE', 'p1')
x0 = (0.0,) * nd                                  # the domain's origin: proteus reads p.x0 for its regular mesh
L = (1.0,) * nd
domain = None
b = np.array([float(v) for v in os.environ.get('AD_B', '0.5,0.25,0.125').split(',')])[:nd]
kappa = float(os.environ.get('AD_KAPPA', '0.02'))
c = float(os.environ.get('AD_C', '0.5'))
tStart = float(os.environ.get('AD_TSTART', '0.1'))
u0 = float(os.environ.get('AD_U0', '1.0'))
xc = np.array([float(v) for v in os.environ.get('AD_XC', '0.3,0.4,0.5').split(',')])[:nd]   # the pulse's centre at time 0
T = float(os.environ.get('AD_T', '0.3'))

solution = LinearADR_Decay_DiracIC(n=nd, b=b, a=kappa, c=c, tStart=tStart, u0=u0, x0=xc)
coefficients = LinearVADR_ConstantCoefficients(nc=1, M=[1.0], A=[kappa * np.eye(nd)], B=[b], C=[c])
coefficients.variableNames = ['u0']
initialConditions = {0: solution}
eps = 1.0e-8

BC = os.environ.get('AD_BC', 'strong')
assert BC in ('strong', 'mixed', 'inflow', 'robin', 'iipg', 'nipg', 'sipg'), BC
robin_alpha, robin_beta = 1.0, 2.0
not_dirichlet = {'mixed': [nd], 'robin': [nd], 'inflow': [0, nd]}.get(BC, [])   # 0 = the plane x = 0, nd = x = 1


def getDBC(x, flag):
    distances = [x[i] - x0[i] for i in range(nd)] + [x0[i] + L[i] - x[i] for i in range(nd)]
    distances = [d for i, d in enumerate(distances) if i not in not_dirichlet]
    if min(distances) < eps:
        return lambda x, t: solution.uOfXT(x, t)


def _qn(x, t, sign):
    """-kappa grad u . n on a face x = const with n = sign * e_x."""
    return sign * solution.diffusiveFluxOfXT(x, t)[0]


def getDiffusiveFlux(x, flag):
    if BC in ('mixed', 'inflow') and abs(x[0] - (x0[0] + L[0])) < eps:
        return lambda x, t: _qn(x, t, 1.0)
    if BC == 'inflow' and abs(x[0] - x0[0]) < eps:
        return lambda x, t: _qn(x, t, -1.0)


def getAdvectiveFlux(x, flag):
    # (b . n) u on the face x = 0, n = -e_x
    if BC == 'inflow' and abs(x[0] - x0[0]) < eps:
        return lambda x, t: -b[0] * solution.uOfXT(x, t)


def getRobin(x, flag):
    if BC == 'robin' and abs(x[0] - (x0[0] + L[0])) < eps:
        return lambda x, t: (robin_alpha, robin_beta, robin_alpha * solution.uOfXT(x, t) + robin_beta * _qn(x, t, 1.0))


dirichletConditions = {0: getDBC}
advectiveFluxBoundaryConditions = {0: getAdvectiveFlux} if BC == 'inflow' else {}
diffusiveFluxBoundaryConditions = {0: {0: getDiffusiveFlux}} if BC in ('mixed', 'inflow') else {0: {}}
fluxBoundaryConditions = {0: 'outFlow' if BC != 'strong' else 'noFlow'}
analyticalSolution = {0: solution}
