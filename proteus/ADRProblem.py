"""Build and run Proteus problems from an advection-diffusion-reaction form.

The input is the plain-data problem that ``ymf.symbolic.adr_problem``
produces from a YMF specification: for each scalar equation, the mass,
advective flux, diffusion tensors, reaction and Hamiltonian as numpy code
strings, with their derivatives and Proteus's dependence flags; plus the
geometry, Dirichlet data, initial conditions and the exact solution.

That is the strong form, already sorted into the slots of Proteus's

    dm_i/dt + div(f_i - sum_k a_ik grad(phi_k)) + r_i + H_i = 0

Two transformations then make it a discrete problem, and both are kept
separate here because each can map one input to several outputs:

- strong -> continuous weak: :class:`ADRCoefficients` gives the equations
  to Proteus's transport machinery, which integrates the flux terms by
  parts; the boundary treatment (strong Dirichlet, outflow fluxes) is set
  by :func:`physics`.
- continuous weak -> discrete weak: :func:`numerics` chooses the finite
  element spaces per unknown, quadrature, stabilization and time stepping.

Nothing here imports sympy: the symbolic work happened in the front end,
and this module only compiles and evaluates the code strings it was given.
"""

import numpy

from proteus import (FemTools, LinearAlgebraTools, LinearSolvers, NonlinearSolvers,
                     NumericalFlux, Quadrature, StepControl, TimeIntegration, defaults)
from proteus.TransportCoefficients import TC_base

__all__ = ["ADRCoefficients", "physics", "numerics", "run", "l2_errors", "ELEMENTS"]


def _compile(code):
    return compile(code, "<adr>", "eval")


#: names the code strings may use, beyond coordinates, time and the unknowns
_GLOBALS = {"__builtins__": {"abs": abs}, "numpy": numpy}


class _Evaluator(object):
    """Compiled code strings, evaluated against a quadrature dictionary."""

    def __init__(self, components, dim):
        self.components = list(components)
        self.dim = dim

    def namespace(self, t, c, with_gradients):
        x = c['x']
        ns = {"t": t, "x": x[..., 0]}
        if self.dim > 1:
            ns["y"] = x[..., 1]
        if self.dim > 2:
            ns["z"] = x[..., 2]
        for j, name in enumerate(self.components):
            if ('u', j) in c:
                ns[name] = c[('u', j)]
            if with_gradients and ('grad(u)', j) in c:
                g = c[('grad(u)', j)]
                for k in range(self.dim):
                    ns["grad_%s_%d" % (name, k)] = g[..., k]
        return ns


class ADRCoefficients(TC_base):
    """A TC_base whose coefficients are given as numpy code strings.

    ``adr`` is the ``"adr"`` part of a ymf ADR problem: ``components`` (the
    scalar unknowns, in order) and one entry per equation.
    """

    def __init__(self, adr):
        self.adr = adr
        self.nd = adr["dim"]
        names = list(adr["components"])
        index = {name: i for i, name in enumerate(names)}
        self._eval = _Evaluator(names, self.nd)
        self._needs_gradients = False

        mass, advection, diffusion, potential, reaction, hamiltonian = {}, {}, {}, {}, {}, {}
        sdInfo = {}
        self._terms = []   # (key, compiled code, slice) to fill on evaluate
        for i, eq in enumerate(adr["equations"]):
            def flags(block):
                return {index[c]: f for c, f in block["flags"].items()}

            if "mass" in eq:
                b = eq["mass"]
                mass[i] = flags(b)
                self._add(('m', i), b["m"])
                for c, code in b["dm"].items():
                    self._add(('dm', i, index[c]), code)
            if "advection" in eq:
                b = eq["advection"]
                advection[i] = flags(b)
                for k, code in enumerate(b["f"]):
                    self._add(('f', i), code, k)
                for c, codes in b["df"].items():
                    for k, code in enumerate(codes):
                        self._add(('df', i, index[c]), code, k)
            if "diffusion" in eq:
                diffusion[i] = {}
                for phi, b in eq["diffusion"].items():
                    ck = index[phi]
                    diffusion[i][ck] = {index[c]: f for c, f in b["flags"].items()}
                    potential[ck] = {ck: 'u'}
                    sdInfo[(i, ck)] = (numpy.array(b["rowptr"], 'i'),
                                       numpy.array(b["colind"], 'i'))
                    for n, code in enumerate(b["a"]):
                        self._add(('a', i, ck), code, n)
                    for c, codes in b["da"].items():
                        for n, code in enumerate(codes):
                            self._add(('da', i, ck, index[c]), code, n)
            if "reaction" in eq:
                b = eq["reaction"]
                reaction[i] = flags(b)
                self._add(('r', i), b["r"])
                for c, code in b["dr"].items():
                    self._add(('dr', i, index[c]), code)
            if "hamiltonian" in eq:
                b = eq["hamiltonian"]
                hamiltonian[i] = flags(b)
                self._needs_gradients = True
                self._add(('H', i), b["H"])
                for c, codes in b["dH"].items():
                    for k, code in enumerate(codes):
                        self._add(('dH', i, index[c]), code, k)

        TC_base.__init__(self, nc=len(names), mass=mass, advection=advection,
                         diffusion=diffusion, potential=potential, reaction=reaction,
                         hamiltonian=hamiltonian, variableNames=names,
                         sparseDiffusionTensors=sdInfo)
        vectors = [i for i, n in enumerate(names) if n.endswith("_0")]
        if vectors:
            stem = names[vectors[0]][:-2]
            self.vectorComponents = [index["%s_%d" % (stem, k)] for k in range(self.nd)]
            self.vectorName = stem

    def _add(self, key, code, last=None):
        self._terms.append((key, _compile(code), last))

    def evaluate(self, t, c):
        ns = self._eval.namespace(t, c, self._needs_gradients)
        for key, code, last in self._terms:
            if key not in c:
                continue
            value = eval(code, _GLOBALS, ns)
            if last is None:
                c[key][...] = value
            else:
                c[key][..., last] = value


class _Field(object):
    """uOfXT for one component, from a code string (x is one point or many)."""

    def __init__(self, code, dim):
        self.code = _compile(code)
        self.dim = dim

    def uOfXT(self, x, t):
        x = numpy.asarray(x)
        ns = {"t": t, "x": x[..., 0]}
        if self.dim > 1:
            ns["y"] = x[..., 1]
        if self.dim > 2:
            ns["z"] = x[..., 2]
        value = eval(self.code, _GLOBALS, ns)
        return value + 0.0 * x[..., 0] if numpy.ndim(x) > 1 else value

    def uOfX(self, x):
        return self.uOfXT(x, 0.0)


def _dirichlet(entries, dim, tol):
    """getDBC(x, flag) for one component.

    Proteus calls it for every node, interior ones included, with flag 0
    for the interior; it must return None there, and for boundary nodes
    no entry claims.
    """
    tests = [(None if e["where"] is None else _compile(e["where"]), _Field(e["value"], dim))
             for e in entries]

    def getDBC(x, flag):
        if flag == 0:
            return None
        ns = {"tol": tol, "x": x[0]}
        if dim > 1:
            ns["y"] = x[1]
        if dim > 2:
            ns["z"] = x[2]
        for where, field in tests:
            if where is None or eval(where, _GLOBALS, ns):
                return lambda x, t, field=field: field.uOfXT(x, t)
        return None
    return getDBC


def _periodic(axes, lower, upper, tol, dirichlet):
    """getPDBC(x, flag): the key point that identifies periodic copies of a node.

    A node on either face of a periodic axis maps to the same key -- its
    coordinates with that axis snapped to the lower face -- and Proteus
    gives nodes with equal keys one degree of freedom.

    Proteus lets a periodic match override a Dirichlet value at the same
    node, which would silently drop, say, a wall velocity at a corner or a
    pressure datum on a periodic face. ``dirichlet`` is this component's
    getDBC; where it claims a node, the node is not periodic.
    """
    def getPDBC(x, flag):
        if dirichlet(x, flag if flag else 1) is not None:
            return None
        on = [a for a in axes
              if abs(x[a] - lower[a]) <= tol or abs(x[a] - upper[a]) <= tol]
        if not on:
            return None
        key = numpy.array([round(float(v), 8) for v in x[:3]])
        for a in on:
            key[a] = lower[a]
        return key
    return getPDBC


def _zero_on_periodic_faces(axes, lower, upper, tol):
    def getAFBC(x, flag):
        if any(abs(x[a] - lower[a]) <= tol or abs(x[a] - upper[a]) <= tol for a in axes):
            return lambda x, t: 0.0
        return None
    return getAFBC


def physics(problem, name):
    """The continuous problem: geometry, coefficients, boundary and initial data."""
    adr = problem["adr"]
    dim = adr["dim"]
    names = adr["components"]
    lower, upper = problem["geometry"]["lower"], problem["geometry"]["upper"]
    p = defaults.Physics_base(nd=dim, name=name)
    # The old-style box route (domain None, L and x0) is the one that hands
    # the numerics' nnx/nny/nnz to the mesh generator.
    p.domain = None
    p.L = tuple(u - l for u, l in zip(upper, lower)) + (1.0,) * (3 - dim)
    p.x0 = tuple(lower) + (0.0,) * (3 - dim)
    p.coefficients = ADRCoefficients(adr)

    tol = 1e-8 * max(p.L[:dim])
    by_component = {}
    for e in problem["dirichlet"]:
        by_component.setdefault(e["component"], []).append(e)
    p.dirichletConditions = {i: _dirichlet(by_component.get(n, []), dim, tol)
                             for i, n in enumerate(names)}
    periodic = {e["component"]: e["axes"] for e in problem.get("periodic", [])}
    if periodic:
        p.periodicDirichletConditions = {
            i: (_periodic(periodic[n], lower, upper, tol, p.dirichletConditions[i])
                if n in periodic else (lambda x, flag: None))
            for i, n in enumerate(names)}
    # On a periodic face the boundary flux of each equation cancels against
    # the opposite face's, so it is set to zero there, on both faces.
    # Computing it from the trace instead (as on every other boundary) is
    # what Proteus does by default, and when the flux couples components --
    # the pressure in the momentum flux -- the two faces do not cancel and
    # the solution is garbage. Elsewhere the flux comes from the trace.
    p.advectiveFluxBoundaryConditions = {
        i: _zero_on_periodic_faces(periodic.get(n, []), lower, upper, tol)
        for i, n in enumerate(names)}
    p.diffusiveFluxBoundaryConditions = {i: {} for i in range(len(names))}
    for i, eq in enumerate(adr["equations"]):
        for phi in eq.get("diffusion", {}):
            p.diffusiveFluxBoundaryConditions[i][names.index(phi)] = (lambda x, flag: None)
    # Advective fluxes through the boundary are computed from the solution's
    # trace -- on Dirichlet boundaries, the boundary data -- so inflow and
    # outflow are both handled; 'noFlow' would drop the term.
    p.fluxBoundaryConditions = {i: ('outFlow' if "advection" in eq else 'noFlow')
                                for i, eq in enumerate(adr["equations"])}

    exact = {names.index(n): _Field(code, dim) for n, code in problem.get("exact", {}).items()}
    initial = {names.index(n): _Field(code, dim) for n, code in problem.get("initial", {}).items()}
    # No initial condition means a steady problem's starting guess: zero,
    # never the exact solution, which would hand Newton the answer.
    for i in range(len(names)):
        if i not in initial:
            initial[i] = _Field("0.0", dim)
    p.initialConditions = initial
    p.analyticalSolution = exact
    return p


#: (family, order) -> Proteus finite element space on simplices
ELEMENTS = {
    ("CG", 1): FemTools.C0_AffineLinearOnSimplexWithNodalBasis,
    ("CG", 2): FemTools.C0_AffineQuadraticOnSimplexWithNodalBasis,
    ("DG", 0): FemTools.DG_Constants,
    ("DG", 1): FemTools.DG_AffineLinearOnSimplexWithNodalBasis,
    ("DG", 2): FemTools.DG_AffineQuadraticOnSimplexWithNodalBasis,
}


def numerics(problem, spaces, cells, time=None, quadrature_order=None):
    """The discrete problem.

    ``spaces`` maps each scalar component to a (family, order) pair;
    ``cells`` is the number of mesh cells along each axis; ``time`` is None
    for a steady problem or a dict with ``dt`` and ``t_end``.
    """
    adr = problem["adr"]
    dim = adr["dim"]
    names = adr["components"]
    n = defaults.Numerics_base()
    n.femSpaces = {}
    orders = []
    for i, name in enumerate(names):
        family, order = spaces[name]
        if (family, order) not in ELEMENTS:
            raise ValueError("no Proteus space for %s%d on simplices" % (family, order))
        n.femSpaces[i] = ELEMENTS[(family, order)]
        orders.append(order)
    q = quadrature_order or min(2 * max(orders) + 2, 5 if dim == 3 else 8)
    n.elementQuadrature = Quadrature.SimplexGaussQuadrature(dim, q)
    n.elementBoundaryQuadrature = Quadrature.SimplexGaussQuadrature(dim - 1, q)
    n.nnx = n.nny = cells + 1
    if dim == 3:
        n.nnz = cells + 1
    n.nLevels = 1
    if any(f == "DG" for f, _ in spaces.values()):
        n.numericalFluxType = NumericalFlux.Advection_DiagonalUpwind_Diffusion_IIPG_exterior \
            if all(f == "CG" for f, _ in spaces.values()) else \
            NumericalFlux.Advection_DiagonalUpwind_Diffusion_IIPG
    if time is None:
        n.timeIntegration = TimeIntegration.NoIntegration
    else:
        n.timeIntegration = TimeIntegration.BackwardEuler
        n.stepController = StepControl.FixedStep
        n.DT = float(time["dt"])
    if problem.get("periodic"):
        n.periodicDirichletConditions = None   # set from the physics in run()
        # The parallel-periodic path numbers every component's periodic DOFs
        # with component 0's space, which breaks mixed (e.g. Taylor-Hood)
        # spaces; the serial path handles each component's own space.
        n.parallelPeriodic = False
    n.multilevelNonlinearSolver = NonlinearSolvers.Newton
    n.levelNonlinearSolver = NonlinearSolvers.Newton
    n.fullNewtonFlag = True
    n.maxNonlinearIts = 25
    n.maxLineSearches = 0
    n.tolFac = 0.0
    n.nl_atol_res = 1.0e-10
    n.matrix = LinearAlgebraTools.SparseMatrix
    n.multilevelLinearSolver = LinearSolvers.LU
    n.levelLinearSolver = LinearSolvers.LU
    return n


def l2_errors(model, problem, t):
    """L2 error of each component with an exact solution, by quadrature."""
    names = problem["adr"]["components"]
    dim = problem["adr"]["dim"]
    q = model.q
    errors = {}
    for name, code in problem.get("exact", {}).items():
        i = names.index(name)
        exact = _Field(code, dim).uOfXT(q['x'], t)
        errors[name] = float(numpy.sqrt(numpy.sum((q[('u', i)] - exact) ** 2 * q[('dV_u', i)])))
    return errors


def run(problem, spaces, cells, time=None, name="adr", opts=None):
    """Solve; return (NS_base, the finest level model, L2 errors at the end)."""
    from proteus import NumericalSolution, default_s
    if opts is None:
        from proteus.iproteus import opts
    p = physics(problem, name)
    n = numerics(problem, spaces, cells, time)
    if problem.get("periodic"):
        n.periodicDirichletConditions = p.periodicDirichletConditions
    so = defaults.System_base(name=name, pnList=[(p, n)], sList=[default_s])
    if time is None:
        so.tnList = [0.0, 1.0]
    else:
        outputs = int(time.get("outputs", 1))
        so.tnList = [float(time["t_end"]) * k / outputs for k in range(outputs + 1)]
        p.T = float(time["t_end"])
        # The system-level controller sets each step's size; the model's
        # own DT alone does not, and the run would take one step per output.
        from proteus import SplitOperator
        so.systemStepControllerType = SplitOperator.Sequential_FixedStep
        so.dt_system_fixed = float(time["dt"])
    tnList = so.tnList
    ns = NumericalSolution.NS_base(so, [p], [n], so.sList, opts)
    ns.calculateSolution(name)
    model = ns.modelList[0].levelModelList[-1]
    return ns, model, l2_errors(model, problem, tnList[-1])
