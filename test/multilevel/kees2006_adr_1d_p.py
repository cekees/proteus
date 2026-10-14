"""The degenerate steady advection-diffusion test problem of Kees et al. (2006).

Kees, Farthing, Howington, Jenkins & Kelley, *Nonlinear multilevel iterative
methods for multiscale models of air/water flow in porous media*, CMWR XVI,
section 2.1, equation (2):

.. math::

    \\nabla \\cdot \\left[ b u^q - a \\nabla (u^r) \\right] = 0
    \\quad\\text{on } (0,1), \\quad u(0) = 1,\\ u(1) = 0

For ``r > 1`` this has roughly the same degeneracy as Richards' equation and
two-phase flow: :math:`d\\phi/du \\to 0` as :math:`u \\to 0`, so the linearized
operator loses ellipticity and the problem develops a thin boundary layer.
That is precisely the regime the paper studies, and the reason this -- rather
than a plain Poisson problem -- is the right baseline for the multilevel
solvers.

Both cases the paper solves analytically are already implemented in
``proteus.AnalyticalSolutions``:

* ``q = r = 1``   -- linear, paper eq. (3), ``LinearAD_SteadyState``
* ``q = 1, r = 2`` -- degenerate, paper eq. (4), ``NonlinearAD_SteadyState``

Select between them with the ``r`` context option.
"""
import numpy as np
from proteus import AnalyticalSolutions, Context, Domain
from proteus.default_p import *
from proteus.TransportCoefficients import NonlinearVADR_pqrst

opts = Context.Options([
    ("a", 5.0e-1, "diffusion coefficient"),
    ("b", 1.0, "advection coefficient"),
    ("r", 2, "exponent on the potential u^r; r=1 is linear, r=2 is degenerate"),
    ("nn", 21, "nodes in the coarsest mesh"),
])

name = "kees2006_adr_1d_r%d" % opts.r
nd = 1
L = (1.0, 1.0, 1.0)
domain = Domain.RectangularDomain(L=L[:1], x=(0.0,), name=name)

a, b, r = opts.a, opts.b, opts.r
q = 1  # the paper fixes q=1 for both analytical cases

# A is the diffusion tensor and B the advection vector, so in 1D they are
# a 1x1 and a length-1 array respectively -- not scalars.
coefficients = NonlinearVADR_pqrst(nc=1,
                                   M={0: 0.0},          # steady state
                                   A={0: np.array([[a]])},
                                   B={0: np.array([b])},
                                   C={0: 0.0},          # no reaction
                                   p={0: 1.0},
                                   q={0: float(q)},
                                   r={0: float(r)},
                                   s={0: 1.0},
                                   t={0: 0.0})

if r == 1:
    _exact = AnalyticalSolutions.LinearAD_SteadyState(b=b, a=a)
else:
    _exact = AnalyticalSolutions.NonlinearAD_SteadyState(b=b, q=q, a=a, r=r)
analyticalSolution = {0: _exact}


def getDBC(x, flag):
    if x[0] <= 1.0e-8:
        return lambda x, t: 1.0
    if x[0] >= 1.0 - 1.0e-8:
        return lambda x, t: 0.0


dirichletConditions = {0: getDBC}
fluxBoundaryConditions = {0: 'noFlow'}
advectiveFluxBoundaryConditions = {}
diffusiveFluxBoundaryConditions = {0: {}}


class ConstantIC(object):
    """A constant initial iterate, as in the paper's non-nested-iteration case."""
    def uOfXT(self, x, t):
        return 1.0 - x[0]


initialConditions = {0: ConstantIC()}
T = 1.0
sd = True  # in 1D the sparse and dense diffusion layouts coincide (nd*nd == 1)
