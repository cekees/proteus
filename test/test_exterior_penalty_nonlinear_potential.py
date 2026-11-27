"""The exterior interior-penalty fluxes impose u = g weakly: the boundary flux is -a grad(phi(u)).n plus a penalty on
u - g.  The penalty was scaled by a alone, too weak by the factor dphi/du for a steep potential, so
Diffusion_SIPG_exterior lost its accuracy: for phi = u^2 + 10u (dphi/du ~ 10) P1 converged at L2 rates 2.59, 1.11,
1.53 (nn = 11 to 81), u reached 15% of its amplitude on the boundary and P2 diverged, while the same problem with
strong Dirichlet conditions converges at rates 2 and 3.  The penalty is now scaled by max(1, dphi/du(g))
(NF_base.exteriorPenalty): never weaker than before, so a degenerate potential (dphi/du(g) = 0, as for u^r) keeps it.

The problem (the nonlinear example of notebooks/Introduction.ipynb): on the unit square, a = I,

    div(-grad(phi(u))) + r(x) = 0,   phi(u) = u^2 + 10u,   u = 0 on the boundary,
    r(x) = -16 pi sin(4 pi x) sin(4 pi y).

It is linear in phi (Kirchhoff form), so phi = sin(4 pi x) sin(4 pi y)/(2 pi) and u = -5 + sqrt(25 + phi).
"""
import types

import numpy as np
import pytest

from proteus import runtime

runtime.start(log=None)

from proteus import (AnalyticalSolutions, Domain, FemTools, Norms, NumericalFlux, NumericalSolution,  # noqa: E402
                     Quadrature, TransportCoefficients, default_n, default_p, default_s, defaults)

import fd_jacobian  # noqa: E402


def r(x):
    return -16.0*np.pi*np.sin(4*np.pi*x[..., 0])*np.sin(4*np.pi*x[..., 1])


class KirchhoffPoisson(TransportCoefficients.TC_base):
    """div(-grad(phi(u))) + r = 0 with phi = c2 u^2 + c1 u."""

    def __init__(self, c2=1.0, c1=10.0):
        TransportCoefficients.TC_base.__init__(self, nc=1, variableNames=['u'],
                                               diffusion={0: {0: {0: 'constant'}}},
                                               potential={0: {0: 'nonlinear'}},
                                               reaction={0: {0: 'constant'}})
        self.c2, self.c1 = c2, c1

    def evaluate(self, t, c):
        c[('a', 0, 0)][..., 0] = 1.0
        c[('a', 0, 0)][..., 3] = 1.0
        c[('r', 0)][:] = r(c['x'][:])
        c[('phi', 0)][:] = self.c2*c[('u', 0)]**2 + self.c1*c[('u', 0)]
        c[('dphi', 0, 0)][:] = 2*self.c2*c[('u', 0)] + self.c1


class Exact(AnalyticalSolutions.SteadyState):
    def __init__(self, c2=1.0, c1=10.0):
        AnalyticalSolutions.SteadyState.__init__(self)
        self.c2, self.c1 = c2, c1

    def uOfX(self, x):
        phi = np.sin(4*np.pi*x[0])*np.sin(4*np.pi*x[1])/(2*np.pi)
        if self.c2 == 0.0:
            return phi/self.c1
        return (-self.c1 + np.sqrt(self.c1**2 + 4*self.c2*phi))/(2*self.c2)


def _module(name, defaults_module):
    """A fresh p or n module with the defaults, so the shared default_p/default_n are never modified."""
    m = types.ModuleType(name)
    m.__dict__.update({k: v for k, v in vars(defaults_module).items() if not k.startswith('__')})
    return m


def solve(nn, flux, space=FemTools.C0_AffineLinearOnSimplexWithNodalBasis, c2=1.0, c1=10.0):
    """Solve on an nn x nn node grid; return (model, L2 error, max |u| on the boundary)."""
    p = _module("kirchhoff_p", default_p)
    n = _module("kirchhoff_n", default_n)
    p.name = "kirchhoff"
    p.nd = 2
    p.domain = Domain.RectangularDomain(L=[1.0, 1.0], x=[0.0, 0.0])
    p.coefficients = KirchhoffPoisson(c2, c1)

    def getDBC(x, flag):
        if flag in [p.domain.boundaryTags[k] for k in ('left', 'top', 'right', 'bottom')]:
            return lambda x, t: 0.0

    p.dirichletConditions = {0: getDBC}
    n.femSpaces = {0: space}
    n.elementQuadrature = Quadrature.SimplexGaussQuadrature(2, 5)
    n.elementBoundaryQuadrature = Quadrature.SimplexGaussQuadrature(1, 5)
    n.nn = nn
    n.numericalFluxType = flux
    n.tolFac = 0.0
    n.nl_atol_res = 1.0e-12
    n.maxNonlinearIts = 25
    so = defaults.System_base(name="kirchhoff", pnList=[(p, n)], sList=[default_s])
    so.tnList = [0.0, 1.0]
    ns = NumericalSolution.NS_base(so, [p], [n], so.sList, runtime.RunOptions())
    assert not ns.calculateSolution("kirchhoff")
    model = ns.modelList[0].levelModelList[-1]
    error = Norms.L2errorSFEMvsAF(Exact(c2, c1), model.q['x'], model.q[('dV_u', 0)], model.q[('u', 0)], T=1.0)
    X = getattr(model.u[0].femSpace.dofMap, 'lagrangeNodesArray', model.mesh.nodeArray)  # P2 has edge dofs
    on_boundary = (np.isclose(X[:, 0], 0) | np.isclose(X[:, 0], 1) | np.isclose(X[:, 1], 0) | np.isclose(X[:, 1], 1))
    return model, error, np.abs(model.u[0].dof[on_boundary]).max()


SIPG = NumericalFlux.Diffusion_SIPG_exterior


def test_sipg_converges_at_second_order_for_a_nonlinear_potential(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    runs = [solve(nn, SIPG) for nn in (11, 21, 41)]
    errors = [e for _, e, _ in runs]
    rates = np.log2(np.array(errors[:-1])/np.array(errors[1:]))
    assert rates[-1] > 1.8, (errors, rates)
    # the boundary values are at the level of the discretization error, not 15% of the 0.0159 amplitude
    assert runs[-1][2] < 0.1*0.0159, runs[-1][2]


def test_sipg_matches_strong_dirichlet_conditions(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    _, weak, _ = solve(41, SIPG)
    _, strong, _ = solve(41, None)
    assert weak < 1.5*strong, (weak, strong)


def test_sipg_p2_converges_at_third_order_for_a_nonlinear_potential(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    errors = [solve(nn, SIPG, space=FemTools.C0_AffineQuadraticOnSimplexWithNodalBasis)[1] for nn in (11, 21, 41)]
    rates = np.log2(np.array(errors[:-1])/np.array(errors[1:]))
    assert rates[-1] > 2.8, (errors, rates)


@pytest.mark.parametrize("c2,c1", [(1.0, 10.0), (0.0, 10.0)])
def test_scaled_penalty_keeps_the_jacobian_exact(tmp_path, monkeypatch, c2, c1):
    """J d equals the central difference of the residual.  The residual is quadratic in u, so the difference is exact
    up to round-off: the penalty's factor is evaluated at g, so it adds nothing to the Jacobian.  One call per solve:
    getJacobian reuses the coefficients of the last residual evaluation, so a second call would see the previous call's
    perturbation."""
    monkeypatch.chdir(tmp_path)
    model, _, _ = solve(11, SIPG, c2=c2, c1=c1)
    error = fd_jacobian.levelmodel_error(model, 1e-3)
    assert error < 1e-10, error
