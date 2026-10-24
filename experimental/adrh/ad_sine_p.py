"""Steady advection-diffusion with the manufactured sine solution, as a plain Proteus model: the
Python-path reference for the ADRH engine regression (ad_regression.py).

    div(b u - kappa grad u) + d(x) = 0 on the unit square or cube (AD_ND = 2 or 3),
    u = sin(omega.x + omega0) on the boundary, imposed as AD_BC says.

Galerkin: no stabilization, so it discretizes exactly what the engine does.
Parameters come from the environment (AD_*) so the driver can vary them.
"""
import os
import numpy as np
from proteus import *
from proteus.default_p import *
from proteus.TransportCoefficients import TC_base

nd = int(os.environ.get('AD_ND', '3'))
SPACE = os.environ.get('AD_SPACE', 'p1')          # p1, p2 (simplices) or q1 (cubes) Lagrange, see ad_sine_n.py
x0 = (0.0,) * nd
L = (1.0,) * nd
domain = None
initialConditions = None
b = np.array([float(v) for v in os.environ.get('AD_B', '1.0,0.5,0.25').split(',')])[:nd]
kappa = float(os.environ.get('AD_KAPPA', '0.2'))
omega = np.array([float(v) for v in os.environ.get('AD_OMEGA', '3.0,2.0,4.0').split(',')])[:nd]
omega0 = float(os.environ.get('AD_OMEGA0', '0.3'))


def exact(x):
    return np.sin(np.dot(omega, x[:nd]) + omega0)


class AdvectionDiffusionSine(TC_base):
    def __init__(self):
        TC_base.__init__(self, 1,
                         mass={}, advection={0: {0: 'linear'}}, diffusion={0: {0: {0: 'constant'}}},
                         potential={0: {0: 'u'}}, reaction={0: {0: 'constant'}}, hamiltonian={},
                         sparseDiffusionTensors={})

    def evaluate(self, t, c):
        u, x = c[('u', 0)], c['x']
        c[('f', 0)][...] = u[..., None] * b
        c[('df', 0, 0)][...] = b
        c[('a', 0, 0)][...] = 0.0
        for i in range(nd):
            c[('a', 0, 0)][..., i * nd + i] = kappa
        c[('da', 0, 0, 0)][...] = 0.0
        c[('phi', 0)][...] = u
        c[('dphi', 0, 0)][...] = 1.0
        arg = x[..., :nd] @ omega + omega0
        c[('r', 0)][...] = -np.dot(b, omega) * np.cos(arg) - kappa * np.dot(omega, omega) * np.sin(arg)
        c[('dr', 0, 0)][...] = 0.0


coefficients = AdvectionDiffusionSine()
coefficients.variableNames = ['u0']
eps = 1.0e-8


# AD_BC selects the boundary treatment, matching ad_regression.py:
#   strong (default)  Dirichlet data on the whole boundary, imposed strongly (numericalFluxType = None)
#   mixed             as strong except the face x = 1, where the diffusive flux -kappa grad u . n is prescribed and
#                     advection leaves by the interior value (StrongDirichletFactory({0: 'outFlow'}); AD_MIXED=1 is this)
#   inflow            as mixed, and on the face x = 0 (inflow, b . n < 0) the advective flux (b . n) u and the
#                     diffusive flux are both prescribed
#   robin             as mixed, but the face x = 1 carries alpha u + beta q.n = g (RobinFactory over the same base)
#   iipg, nipg, sipg  Dirichlet data on the whole boundary, imposed weakly by
#                     Advection_DiagonalUpwind_Diffusion_{IIPG,NIPG,SIPG}_exterior (the Dirichlet dofs stay unknowns)
BC = os.environ.get('AD_BC', 'mixed' if os.environ.get('AD_MIXED', '0') == '1' else 'strong')
assert BC in ('strong', 'mixed', 'inflow', 'robin', 'iipg', 'nipg', 'sipg'), BC
robin_alpha, robin_beta = 1.0, 2.0
# indices into the distance list of getDBC: 0 = the plane x = 0, nd = the plane x = 1
not_dirichlet = {'mixed': [nd], 'robin': [nd], 'inflow': [0, nd]}.get(BC, [])


def getDBC(x, flag):
    distances = [x[i] - x0[i] for i in range(nd)] + [x0[i] + L[i] - x[i] for i in range(nd)]
    distances = [d for i, d in enumerate(distances) if i not in not_dirichlet]
    if min(distances) < eps:
        return lambda x, t: exact(x)


def _qn(x, sign):
    """-kappa grad u . n on a face x = const with n = sign * e_x."""
    return -kappa * sign * omega[0] * np.cos(np.dot(omega, x[:nd]) + omega0)


def getDiffusiveFlux(x, flag):
    if BC in ('mixed', 'inflow') and abs(x[0] - (x0[0] + L[0])) < eps:
        return lambda x, t: _qn(x, 1.0)
    if BC == 'inflow' and abs(x[0] - x0[0]) < eps:
        return lambda x, t: _qn(x, -1.0)


def getAdvectiveFlux(x, flag):
    # (b . n) u on the inflow face x = 0, n = -e_x
    if BC == 'inflow' and abs(x[0] - x0[0]) < eps:
        return lambda x, t: -b[0] * exact(x)


def getRobin(x, flag):
    # alpha u + beta q.n = g on the face x = 1, with g from the exact solution
    if BC == 'robin' and abs(x[0] - (x0[0] + L[0])) < eps:
        return lambda x, t: (robin_alpha, robin_beta, robin_alpha * exact(x) + robin_beta * _qn(x, 1.0))


dirichletConditions = {0: getDBC}
advectiveFluxBoundaryConditions = {0: getAdvectiveFlux} if BC == 'inflow' else {}
diffusiveFluxBoundaryConditions = {0: {0: getDiffusiveFlux}} if BC in ('mixed', 'inflow') else {0: {}}
fluxBoundaryConditions = {0: 'outFlow' if BC != 'strong' else 'noFlow'}
analyticalSolution = {}
