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

__all__ = ["ADRCoefficients", "physics", "numerics", "run", "emit_pn", "describe",
           "l2_errors", "ELEMENTS"]


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
        self._terms = []   # (key, compiled code, slice, source) to fill on evaluate
        for i, eq in enumerate(adr["equations"]):
            def flags(block, constant_key=i):
                # ymf says how a coefficient depends on each unknown, and
                # nothing for a constant; Proteus wants 'constant' keyed by
                # some component, by convention the equation's own.
                deps = {index[c]: f for c, f in block["depends_on"].items()}
                return deps or {constant_key: 'constant'}

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
                    # Proteus's diffusion flags are 'constant' or 'nonlinear'
                    diffusion[i][ck] = ({index[c]: 'nonlinear' for c in b["depends_on"]}
                                        or {ck: 'constant'})
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
                hamiltonian[i] = {index[c]: f for c, f in b["depends_on"].items()}
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
        self._terms.append((key, _compile(code), last, code))

    def evaluate(self, t, c):
        ns = self._eval.namespace(t, c, self._needs_gradients)
        for key, code, last, _ in self._terms:
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


def reorder(problem, first):
    """The same problem with the components (and their equations) reordered.

    Equation i belongs to component i, so the two are permuted together;
    everything else refers to components by name. ``first`` lists the
    components to put first, in order.
    """
    import copy
    adr = problem["adr"]
    names = list(adr["components"])
    order = list(first) + [n for n in names if n not in first]
    out = copy.deepcopy(problem)
    out["adr"]["components"] = order
    out["adr"]["equations"] = [copy.deepcopy(adr["equations"][names.index(n)]) for n in order]
    return out


#: weak-form stabilization methods numerics() can install
STABILIZATIONS = ("none", "supg/pspg")


def velocity_pressure(problem):
    """(velocity components, pressure component) of a velocity-pressure system."""
    vectors = [c for c in problem["unknowns"].values() if len(c) > 1]
    scalars = [c[0] for c in problem["unknowns"].values() if len(c) == 1]
    if len(vectors) != 1 or len(scalars) != 1:
        raise ValueError("SUPG/PSPG is wired for velocity-pressure systems (one vector "
                         "and one scalar unknown), not %s" % list(problem["unknowns"]))
    return vectors[0], scalars[0]


def numerics(problem, spaces, cells, time=None, quadrature_order=None,
             stabilization="none", coefficients=None, levels=1,
             tolerance=None, max_iterations=None):
    """The discrete problem.

    ``spaces`` maps each scalar component to a (family, order) pair;
    ``cells`` is the number of mesh cells along each axis; ``time`` is None
    for a steady problem or a dict with ``dt``. ``stabilization`` is one of
    :data:`STABILIZATIONS`; ``supg/pspg`` installs Proteus's residual-based
    velocity-pressure stabilization (ASGS), which needs the problem in
    pressure-first order (see :func:`reorder`) and its ``coefficients``.
    ``tolerance`` and ``max_iterations`` set Newton's absolute residual
    tolerance (default 1e-10) and iteration limit (default 25).
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
    # nLevels > 1: the box mesh above is the coarsest, refined uniformly
    # levels-1 times; the solution of record is on the finest.
    n.nLevels = int(levels)
    if any(f == "DG" for f, _ in spaces.values()):
        n.numericalFluxType = NumericalFlux.Advection_DiagonalUpwind_Diffusion_IIPG_exterior \
            if all(f == "CG" for f, _ in spaces.values()) else \
            NumericalFlux.Advection_DiagonalUpwind_Diffusion_IIPG
    stabilization = (stabilization or "none").lower()
    if stabilization not in STABILIZATIONS:
        raise ValueError("stabilization %r is not wired up (have %s)"
                         % (stabilization, ", ".join(STABILIZATIONS)))
    if stabilization == "supg/pspg":
        from proteus import SubgridError
        velocity, pressure = velocity_pressure(problem)
        if names[0] != pressure or names[1:1 + dim] != velocity:
            raise ValueError("SUPG/PSPG needs the pressure first, then the velocity "
                             "(reorder the problem); got %s" % names)
        # The stabilization reads the density from the momentum equations'
        # mass coefficient (dm), so they need their ρ ∂v/∂t term even when
        # the solve is steady (which drops it).
        missing = [c for c, eq in zip(names, adr["equations"]) if c in velocity and "mass" not in eq]
        if missing:
            raise ValueError("SUPG/PSPG takes the density from the time derivative "
                             "ρ ∂v/∂t, which the equations for %s lack; include it "
                             "(a steady solve drops it)" % ", ".join(missing))
        n.subgridError = SubgridError.NavierStokesASGS_velocity_pressure(
            coefficients, dim, lag=False)
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
    n.maxNonlinearIts = int(max_iterations or 25)
    n.maxLineSearches = 0
    n.tolFac = 0.0
    n.nl_atol_res = float(tolerance or 1.0e-10)
    n.matrix = LinearAlgebraTools.SparseMatrix
    n.multilevelLinearSolver = LinearSolvers.LU
    n.levelLinearSolver = LinearSolvers.LU
    return n


def _name(obj):
    return obj.__name__ if isinstance(obj, type) else type(obj).__name__


def describe(problem, n):
    """The discrete problem ``n`` (from :func:`numerics`), as plain data.

    Names the Proteus classes each choice resolved to, so that a record of
    the run says exactly what was solved, not only what was asked for.
    """
    names = problem["adr"]["components"]
    out = {
        "spaces": {names[i]: _name(space) for i, space in sorted(n.femSpaces.items())},
        "quadrature": {"element": int(n.elementQuadrature.order),
                       "boundary": int(n.elementBoundaryQuadrature.order)},
        "mesh": {"nodes_per_side": int(n.nnx), "levels": int(n.nLevels)},
        "time_integration": ("none" if n.timeIntegration is TimeIntegration.NoIntegration
                             else _name(n.timeIntegration)),
        "nonlinear_solver": {"type": _name(n.levelNonlinearSolver),
                             "atol": float(n.nl_atol_res),
                             "max_iterations": int(n.maxNonlinearIts)},
        "linear_solver": _name(n.levelLinearSolver),
    }
    if getattr(n, "subgridError", None) is not None:
        out["subgrid_error"] = _name(n.subgridError)
    if getattr(n, "numericalFluxType", None) is not None:
        out["numerical_flux"] = _name(n.numericalFluxType)
    if n.timeIntegration is not TimeIntegration.NoIntegration:
        out["time_step"] = float(n.DT)
    return out


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


def setup(problem, spaces, cells, time=None, name="adr", stabilization="none",
          levels=1, tolerance=None, max_iterations=None):
    """The problem as solved, its physics and numerics, and the system settings.

    Returns ``(problem, p, n, system)``: the problem (reordered pressure
    first for SUPG/PSPG), the Physics_base and Numerics_base, and a dict of
    the System_base settings (``tnList``, and for a time-dependent solve
    ``dt_system_fixed``). :func:`run` solves this; :func:`emit_pn` writes it
    out as p, n and so files.
    """
    if (stabilization or "none").lower() == "supg/pspg":
        velocity, pressure = velocity_pressure(problem)
        problem = reorder(problem, [pressure] + velocity)
    p = physics(problem, name)
    n = numerics(problem, spaces, cells, time, stabilization=stabilization,
                 coefficients=p.coefficients, levels=levels,
                 tolerance=tolerance, max_iterations=max_iterations)
    if problem.get("periodic"):
        n.periodicDirichletConditions = p.periodicDirichletConditions
    if time is None:
        system = {"tnList": [0.0, 1.0]}
    else:
        if not problem.get("time"):
            raise ValueError("a time step was given for a problem with no time interval")
        t0, t1 = (float(v) for v in problem["time"])
        outputs = int(time.get("outputs", 1))
        p.T = t1
        # The system-level controller sets each step's size; the model's
        # own DT alone does not, and the run would take one step per output.
        system = {"tnList": [t0 + (t1 - t0) * k / outputs for k in range(outputs + 1)],
                  "dt_system_fixed": float(time["dt"])}
    return problem, p, n, system


def run(problem, spaces, cells, time=None, name="adr", stabilization="none",
        opts=None, levels=1, tolerance=None, max_iterations=None):
    """Solve; return (NS_base, the finest level model, L2 errors at the end).

    ``time`` is None for a steady solve, or ``{"dt": ...}`` to step through
    the problem's own time interval (``problem["time"]``); ``outputs`` (1 by
    default) sets how many evenly spaced archive frames to write. The
    archive Proteus writes holds the numerical approximation only; what
    produced it is the caller's to record (see :func:`describe`).
    """
    from proteus import NumericalSolution, SplitOperator, default_s
    if opts is None:
        from proteus.iproteus import opts
    problem, p, n, system = setup(problem, spaces, cells, time, name, stabilization,
                                  levels, tolerance, max_iterations)
    so = defaults.System_base(name=name, pnList=[(p, n)], sList=[default_s])
    so.tnList = system["tnList"]
    if "dt_system_fixed" in system:
        so.systemStepControllerType = SplitOperator.Sequential_FixedStep
        so.dt_system_fixed = system["dt_system_fixed"]
    ns = NumericalSolution.NS_base(so, [p], [n], so.sList, opts)
    ns.calculateSolution(name)
    model = ns.modelList[0].levelModelList[-1]
    return ns, model, l2_errors(model, problem, so.tnList[-1])


# --- writing a problem out as p, n and so files ---------------------------------


def _spelled(module, obj):
    """``Module.Name`` for a class, preferring an alias to a ``*_base`` name
    (StepControl.FixedStep, not StepControl.SC_base)."""
    names = sorted(k for k, v in vars(module).items() if v is obj)
    better = [k for k in names if not k.endswith("_base")]
    return "%s.%s" % (module.__name__.rsplit(".", 1)[-1], (better or names or [obj.__name__])[0])


def _coordinates(dim):
    return ["x", "y", "z"][:dim]


def _function(dim, code):
    """A lambda of the coordinates and t, around one code string."""
    return "lambda %s, t: %s" % (", ".join(_coordinates(dim)), code)


def emit_pn(problem, spaces, cells, time=None, name="adr", stabilization="none",
            levels=1, tolerance=None, max_iterations=None, directory=".", header=None):
    """Write the problem as a classic Proteus ``<name>_p.py``, ``_n.py`` and ``_so.py``.

    The files say in Python exactly what :func:`run` builds in memory: the
    settings are read from the objects :func:`setup` returns, and the
    coefficients from the same code strings, so ``parun <name>_so.py``
    solves bitwise the same problem. They are written to be read: one
    function per boundary condition and region, the coefficients one term
    per line. The system's name is ``<name>_pn``, so a run of the files
    does not overwrite the archive of the run they describe.

    ``header`` is a line or two of text for each file's docstring (say, the
    output it reproduces). Returns the three paths.
    """
    import os
    import pprint
    from proteus import SubgridError
    problem, p, n, system = setup(problem, spaces, cells, time, name, stabilization,
                                  levels, tolerance, max_iterations)
    c = p.coefficients
    adr = problem["adr"]
    dim = adr["dim"]
    names = list(adr["components"])
    xyz = _coordinates(dim)
    lower, upper = list(problem["geometry"]["lower"]), list(problem["geometry"]["upper"])
    tol = 1e-8 * max(p.L[:dim])
    about = [header] if header else []

    def doc(what):
        return ['"""%s for %s.' % (what, name), ""] + about + [
            "Written by proteus.ADRProblem.emit_pn; run with parun %s_so.py." % name,
            "Components: %s." % ", ".join("%d = %s" % (i, nm) for i, nm in enumerate(names)),
            '"""']

    # ---- p ----
    P = doc("Physics") + [
        "import numpy",
        "from proteus.default_p import *",
        "from proteus.TransportCoefficients import TC_base",
        "",
        "nd = %d" % dim,
        "name = %r" % name,
        "# the old-style box: domain None, with L and x0, hands nnx/nny/nnz to the mesher",
        "domain = None",
        "L = %r" % (tuple(p.L),),
        "x0 = %r" % (tuple(p.x0),),
    ]
    if time is not None:
        P.append("T = %r" % p.T)
    P += ["", "",
          "class Coefficients(TC_base):",
          '    """dm_i/dt + div(f_i - sum_k a_ik grad(phi_k)) + r_i + H_i = 0"""',
          "",
          "    def __init__(self):",
          "        TC_base.__init__(",
          "            self, nc=%d, variableNames=%r," % (c.nc, list(c.variableNames))]
    for key in ("mass", "advection", "diffusion", "potential", "reaction", "hamiltonian"):
        P.append("            %s=%r," % (key, getattr(c, key)))
    P.append("            sparseDiffusionTensors={%s})" % ", ".join(
        "%r: (numpy.array(%r, 'i'), numpy.array(%r, 'i'))" % (k, list(map(int, r)), list(map(int, cl)))
        for k, (r, cl) in c.sdInfo.items()))
    # What the numerics added to the coefficients' stencil (SUPG/PSPG couples
    # the continuity equation to the pressure) is written here: parun loads
    # the p file without caching it, so the n file's import makes a second
    # Coefficients, and a stencil entry added to that one alone is lost.
    plain = ADRCoefficients(adr).stencil
    for i, row in enumerate(c.stencil):
        added = sorted(set(row) - set(plain[i]))
        if added:
            P += ["        # added by %s: equation %d couples to %s"
                  % (type(n.subgridError).__name__, i, ", ".join(names[k] for k in added)),
                  "        self.stencil[%d].update(%r)" % (i, added)]
    if getattr(c, "vectorComponents", None) is not None:
        P += ["        self.vectorComponents = %r" % list(c.vectorComponents),
              "        self.vectorName = %r" % c.vectorName]
    P += ["", "    def evaluate(self, t, c):"]
    P += ["        %s = c['x'][..., %d]" % (v, k) for k, v in enumerate(xyz)]
    for j, nm in enumerate(names):
        P.append("        %s = c[('u', %d)]" % (nm, j))
    if c._needs_gradients:
        for j, nm in enumerate(names):
            P.append("        if ('grad(u)', %d) in c:" % j)
            P += ["            grad_%s_%d = c[('grad(u)', %d)][..., %d]" % (nm, k, j, k)
                  for k in range(dim)]
    for key, _, last, code in c._terms:
        target = "c[%r][...]" % (key,) if last is None else "c[%r][..., %d]" % (key, last)
        P.append("        if %r in c: %s = %s" % (key, target, code))
    P += ["", "", "coefficients = Coefficients()", "", "",
          "class Field(object):",
          '    """A function of the coordinates and t, as initial or boundary data."""',
          "",
          "    def __init__(self, f):",
          "        self.f = f",
          "",
          "    def uOfXT(self, X, t):",
          "        X = numpy.asarray(X)",
          "        value = self.f(%s, t)" % ", ".join("X[..., %d]" % k for k in range(dim)),
          "        return value + 0.0 * X[..., 0] if numpy.ndim(X) > 1 else value",
          "",
          "    def uOfX(self, X):",
          "        return self.uOfXT(X, 0.0)",
          ""]
    exact = problem.get("exact", {})
    initial = problem.get("initial", {})
    P += ["", "analyticalSolution = {"]
    P += ["    %d: Field(%s),   # %s" % (i, _function(dim, exact[nm]), nm)
          for i, nm in enumerate(names) if nm in exact]
    P += ["}", "# a steady problem with no initial condition starts Newton from zero,",
          "# never from the exact solution", "initialConditions = {"]
    P += ["    %d: Field(%s),   # %s" % (i, _function(dim, initial.get(nm, "0.0")), nm)
          for i, nm in enumerate(names)]
    P += ["}", "", "tol = %r" % tol]

    by_component = {}
    for e in problem["dirichlet"]:
        by_component.setdefault(e["component"], []).append(e)
    for i, nm in enumerate(names):
        entries = by_component.get(nm, [])
        P += ["", "",
              "def getDBC_%s(X, flag):" % nm,
              "    if flag == 0:     # an interior node",
              "        return None"]
        if entries:
            P.append("    %s = %s" % (", ".join(xyz), ", ".join("X[%d]" % k for k in range(dim))))
        for e in entries:
            test = "True" if e["where"] is None else e["where"]
            P += ["    if %s:   # region %s" % (test, e["region"]),
                  "        return lambda X, t: Field(%s).uOfXT(X, t)" % _function(dim, e["value"])]
        P.append("    return None")
    P += ["", "", "dirichletConditions = {%s}" % ", ".join(
        "%d: getDBC_%s" % (i, nm) for i, nm in enumerate(names))]

    periodic = {e["component"]: list(e["axes"]) for e in problem.get("periodic", [])}
    if periodic:
        P += ["", "",
              "# Periodic: a node on either face of a periodic axis is keyed by its",
              "# coordinates with that axis snapped to the lower face; equal keys share",
              "# a degree of freedom. A node a Dirichlet condition claims is not periodic.",
              "LOWER = %r" % (lower,),
              "UPPER = %r" % (upper,),
              "",
              "",
              "def on_faces(X, axes):",
              "    return [a for a in axes",
              "            if abs(X[a] - LOWER[a]) <= tol or abs(X[a] - UPPER[a]) <= tol]",
              "",
              "",
              "def periodic(getDBC, axes):",
              "    def getPDBC(X, flag):",
              "        if getDBC(X, flag if flag else 1) is not None:",
              "            return None",
              "        on = on_faces(X, axes)",
              "        if not on:",
              "            return None",
              "        key = numpy.array([round(float(v), 8) for v in X[:3]])",
              "        for a in on:",
              "            key[a] = LOWER[a]",
              "        return key",
              "    return getPDBC",
              "",
              "",
              "def zero_on_faces(axes):",
              "    # the two faces' fluxes cancel, so each is set to zero",
              "    def getAFBC(X, flag):",
              "        if on_faces(X, axes):",
              "            return lambda X, t: 0.0",
              "        return None",
              "    return getAFBC",
              "",
              "",
              "periodicDirichletConditions = {%s}" % ", ".join(
                  "%d: periodic(getDBC_%s, %r)" % (i, nm, periodic[nm]) if nm in periodic
                  else "%d: lambda X, flag: None" % i for i, nm in enumerate(names)),
              "advectiveFluxBoundaryConditions = {%s}" % ", ".join(
                  "%d: zero_on_faces(%r)" % (i, periodic[nm]) if nm in periodic
                  else "%d: lambda X, flag: None" % i for i, nm in enumerate(names))]
    else:
        P += ["advectiveFluxBoundaryConditions = {%s}" % ", ".join(
            "%d: lambda X, flag: None" % i for i in range(len(names)))]
    P += ["diffusiveFluxBoundaryConditions = {%s}" % ", ".join(
              "%d: {%s}" % (i, ", ".join("%d: lambda X, flag: None" % k for k in sorted(d)))
              for i, d in sorted(p.diffusiveFluxBoundaryConditions.items())),
          "# advective fluxes through the boundary come from the trace",
          "fluxBoundaryConditions = %r" % (p.fluxBoundaryConditions,)]

    # ---- n ----
    from proteus import default_n
    N = doc("Numerics") + [
        "from proteus.default_n import *",
        "from proteus import (FemTools, LinearAlgebraTools, LinearSolvers, NonlinearSolvers,",
        "                     NumericalFlux, Quadrature, StepControl, SubgridError, TimeIntegration)",
        "from %s_p import %s" % (name, ", ".join(["coefficients", "nd"] + (
            ["periodicDirichletConditions"] if periodic else []))),
        "",
        "femSpaces = {"]
    N += ["    %d: %s,   # %s" % (i, _spelled(FemTools, n.femSpaces[i]), names[i])
          for i in sorted(n.femSpaces)]
    N += ["}",
          "elementQuadrature = Quadrature.SimplexGaussQuadrature(nd, %d)" % n.elementQuadrature.order,
          "elementBoundaryQuadrature = Quadrature.SimplexGaussQuadrature(nd - 1, %d)"
          % n.elementBoundaryQuadrature.order]
    N.append("%s = %d   # %d cells per side" % (" = ".join(["nnx", "nny", "nnz"][:dim]), n.nnx, cells))
    N.append("nLevels = %d   # uniform refinements of that mesh, plus one" % n.nLevels)
    if n.numericalFluxType is not default_n.numericalFluxType:
        N.append("numericalFluxType = %s" % _spelled(NumericalFlux, n.numericalFluxType))
    if n.subgridError is not None:
        if not isinstance(n.subgridError, SubgridError.NavierStokesASGS_velocity_pressure):
            raise ValueError("emit_pn does not know how to write %s" % type(n.subgridError).__name__)
        N.append("subgridError = SubgridError.NavierStokesASGS_velocity_pressure("
                 "coefficients, nd, lag=False)")
    if n.timeIntegration is TimeIntegration.NoIntegration:
        N.append("timeIntegration = TimeIntegration.NoIntegration   # steady")
    else:
        N += ["timeIntegration = %s" % _spelled(TimeIntegration, n.timeIntegration),
              "stepController = %s" % _spelled(StepControl, n.stepController),
              "DT = %r" % n.DT]
    if periodic:
        N += ["# the serial periodic path: the parallel one numbers every component's",
              "# periodic DOFs with component 0's space, which breaks mixed spaces",
              "parallelPeriodic = False"]
    N += ["multilevelNonlinearSolver = %s" % _spelled(NonlinearSolvers, n.multilevelNonlinearSolver),
          "levelNonlinearSolver = %s" % _spelled(NonlinearSolvers, n.levelNonlinearSolver),
          "fullNewtonFlag = %r" % n.fullNewtonFlag,
          "maxNonlinearIts = %d" % n.maxNonlinearIts,
          "maxLineSearches = %d" % n.maxLineSearches,
          "tolFac = %r" % n.tolFac,
          "nl_atol_res = %r" % n.nl_atol_res,
          "matrix = %s" % _spelled(LinearAlgebraTools, n.matrix),
          "multilevelLinearSolver = %s" % _spelled(LinearSolvers, n.multilevelLinearSolver),
          "levelLinearSolver = %s" % _spelled(LinearSolvers, n.levelLinearSolver)]

    # ---- so ----
    S = doc("The system") + [
        "from proteus.default_so import *",
        "from proteus import default_s",
        "",
        "name = %r   # not %r: a run of these files keeps clear of the original's archive"
        % (name + "_pn", name),
        "pnList = [(%r, %r)]" % (name + "_p", name + "_n"),
        "sList = [default_s]",
        "tnList = %s" % pprint.pformat(system["tnList"])]
    if "dt_system_fixed" in system:
        S += ["# the system's controller sets each step; the model's DT alone does not",
              "systemStepControllerType = Sequential_FixedStep",
              "dt_system_fixed = %r" % system["dt_system_fixed"]]

    paths = []
    for suffix, lines in (("_p", P), ("_n", N), ("_so", S)):
        path = os.path.join(directory, name + suffix + ".py")
        with open(path, "w") as f:
            f.write("\n".join(lines) + "\n")
        paths.append(path)
    return paths
