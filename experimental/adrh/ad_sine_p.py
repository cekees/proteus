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


def getDBC(x, flag):
    if min(min(x[i] - x0[i], x0[i] + L[i] - x[i]) for i in range(3)) < eps:
        return lambda x, t: exact(x)


dirichletConditions = {0: getDBC}
advectiveFluxBoundaryConditions = {}
diffusiveFluxBoundaryConditions = {0: {}}
fluxBoundaryConditions = {0: 'noFlow'}
analyticalSolution = {}
