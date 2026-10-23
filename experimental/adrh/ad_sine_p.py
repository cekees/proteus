"""Steady advection-diffusion with the manufactured sine solution, as a plain Proteus model: the
Python-path reference for the ADRH engine regression (ad_regression.py).

    div(b u - kappa grad u) + d(x) = 0 on the unit cube, u = sin(omega.x + omega0) on all of the boundary.

Galerkin: no stabilization, strong Dirichlet conditions, so it discretizes exactly what the engine does.
Parameters come from the environment (AD_*) so the driver can vary them.
"""
import os
import numpy as np
from proteus import *
from proteus.default_p import *
from proteus.TransportCoefficients import TC_base

nd = 3
x0 = (0.0, 0.0, 0.0)
L = (1.0, 1.0, 1.0)
domain = None
initialConditions = None
b = np.array([float(v) for v in os.environ.get('AD_B', '1.0,0.5,0.25').split(',')])
kappa = float(os.environ.get('AD_KAPPA', '0.2'))
omega = np.array([float(v) for v in os.environ.get('AD_OMEGA', '3.0,2.0,4.0').split(',')])
omega0 = float(os.environ.get('AD_OMEGA0', '0.3'))


def exact(x):
    return np.sin(np.dot(omega, x[:3]) + omega0)


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
        arg = x @ omega + omega0
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
#   iipg, nipg, sipg  Dirichlet data on the whole boundary, imposed weakly by
#                     Advection_DiagonalUpwind_Diffusion_{IIPG,NIPG,SIPG}_exterior (the Dirichlet dofs stay unknowns)
BC = os.environ.get('AD_BC', 'mixed' if os.environ.get('AD_MIXED', '0') == '1' else 'strong')
assert BC in ('strong', 'mixed', 'inflow', 'iipg', 'nipg', 'sipg'), BC
mixed = BC in ('mixed', 'inflow')
not_dirichlet = {'mixed': [3], 'inflow': [0, 3]}.get(BC, [])     # indices into distances below: 0 = x=0, 3 = x=1


def getDBC(x, flag):
    distances = [x[i] - x0[i] for i in range(3)] + [x0[i] + L[i] - x[i] for i in range(3)]
    distances = [d for i, d in enumerate(distances) if i not in not_dirichlet]
    if min(distances) < eps:
        return lambda x, t: exact(x)


def getDiffusiveFlux(x, flag):
    # -kappa grad u . n with n = +-e_x on the faces x = 1 and x = 0
    if mixed and abs(x[0] - (x0[0] + L[0])) < eps:
        return lambda x, t: -kappa * omega[0] * np.cos(np.dot(omega, x[:3]) + omega0)
    if BC == 'inflow' and abs(x[0] - x0[0]) < eps:
        return lambda x, t: kappa * omega[0] * np.cos(np.dot(omega, x[:3]) + omega0)


def getAdvectiveFlux(x, flag):
    # (b . n) u on the inflow face x = 0, n = -e_x
    if BC == 'inflow' and abs(x[0] - x0[0]) < eps:
        return lambda x, t: -b[0] * exact(x)


dirichletConditions = {0: getDBC}
advectiveFluxBoundaryConditions = {0: getAdvectiveFlux} if BC == 'inflow' else {}
diffusiveFluxBoundaryConditions = {0: {0: getDiffusiveFlux}} if mixed else {0: {}}
fluxBoundaryConditions = {0: 'outFlow' if BC != 'strong' else 'noFlow'}
analyticalSolution = {}
