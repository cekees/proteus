"""Problem setups shared by the MPI helpers in this directory; not a test module."""
from proteus import TransportCoefficients


def make_uncoupled_components(p, n, nc, space):
    """Turn the 2D Poisson problem into ``nc`` uncoupled copies of itself.

    The physics module builds its per-component dictionaries from a module-level
    ``nc`` at import, so they are rebuilt here. Every component solves the same
    equation, so a correct multi-component transfer must give each the answer
    the single-component problem gives -- which is what makes this a sharp test
    of how components are interleaved.
    """
    comps = range(nc)
    p.nc = nc
    p.analyticalSolution = dict((i, p.u5Ex()) for i in comps)
    p.dirichletConditions = dict((i, p.getDBC5) for i in comps)
    p.aOfX = dict((i, p.a5) for i in comps)
    p.fOfX = dict((i, p.f5) for i in comps)
    p.advectiveFluxBoundaryConditions = dict((i, p.getAdvFluxBC5) for i in comps)
    p.diffusiveFluxBoundaryConditions = dict((i, {i: p.getDiffFluxBC5}) for i in comps)
    p.analyticalSolutionVelocity = dict(
        (i, p.velEx(p.analyticalSolution[i], p.aOfX[i])) for i in comps)
    p.coefficients = TransportCoefficients.PoissonEquationCoefficients(
        p.aOfX, p.fOfX, nc, p.nd)
    p.coefficients.variableNames = ['u%d' % i for i in comps]
    p.fluxBoundaryConditions = dict((i, 'setFlow') for i in comps)
    n.femSpaces = dict((i, space) for i in comps)
