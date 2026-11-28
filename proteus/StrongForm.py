"""Strong forms as Python objects, written out as a ymf spec.

The operators of the strong form are Python objects, so a problem is posed in Python the way it is written on paper::

    from proteus.StrongForm import Box, Coefficient, Problem, Unknown, div, grad, pi, sin, sqrt, x, y

    u = Unknown("u")
    a = Coefficient("a", 1.0)
    φ = Coefficient("φ", u**2 + 10*u)                 # a coefficient may name the unknown
    r = Coefficient("r", -16*pi*sin(4*pi*x)*sin(4*pi*y))

    problem = Problem("Nonlinear Poisson", Box((0, 1), (0, 1)), unknowns=[u])
    problem.equation(div(-a*grad(φ)) + r == 0)
    problem.dirichlet(u, 0.0, on=problem.boundary)
    problem.initial(u, 0.0)
    problem.exact(u, -5 + sqrt(25 + sin(4*pi*x)*sin(4*pi*y)/(2*pi)))
    problem.discretization("P1", order=1, cells=[40, 80])

    problem.spec()                  # the ymf spec, as data
    problem.write("np.ymf")         # and as a file, which pro spec, pro weak and pro run read
    problem.run("out")              # pro run: the archive, its outputs keyed by what produced them

Nothing here is Proteus-specific: the objects print themselves in ymf's notation (``∇·(−a ∇φ) + r = 0``), and
the spec they write is the same document a person writes by hand, which ymf reads (``ymf.symbolic``) and any solver
that reads ymf can solve. The Python objects are a way to write that document, not a second description of the
problem: :meth:`Problem.spec` holds every fact once, as the hand-written file does, and nothing here imports sympy.
Solving goes through the spec (:meth:`Problem.run`, :meth:`Problem.adr`), so a problem posed in Python and the same
problem written as YAML are solved by the same code and give the same numbers.

Expressions are kept as written, not simplified: ``u**2 + 10*u`` prints as ``u² + 10u``. Python's ``==`` between
two expressions makes an :class:`Equation`. Supported: numbers, the coordinates ``x, y, z``, time ``t``, ``pi``,
``+ - * / **``, the functions of the notation (``sin cos tan atan exp log sqrt sinh cosh tanh abs``), ``grad``,
``div``, ``lap``, ``dt``, ``dot``, ``outer`` and vectors (``vector(a, b)``).
"""

import numbers
import os

__all__ = ["Expr", "Equation", "Unknown", "Coefficient", "Box", "Region", "Problem",
           "x", "y", "z", "t", "pi", "vector",
           "grad", "div", "lap", "dt", "dot", "outer",
           "sin", "cos", "tan", "atan", "exp", "log", "sqrt", "sinh", "cosh", "tanh", "absolute"]

_SUPERSCRIPTS = {2: "²", 3: "³", 4: "⁴"}

# precedence of what an expression prints as: a sum binds loosest, then a product (and a quotient), then a power;
# an atom (a name, a number, a call, a parenthesized expression) is never wrapped
_SUM, _PRODUCT, _POWER, _ATOM = 1, 2, 3, 4


def _number_text(value):
    if isinstance(value, bool):
        raise TypeError("a boolean is not a number of the notation")
    if isinstance(value, numbers.Integral) or (isinstance(value, float) and value.is_integer() and abs(value) < 1e15):
        return "%d" % value
    return repr(float(value))


def _wrap(expr, level):
    text = expr.notation()
    return "(%s)" % text if expr.precedence < level else text


class Expr(object):
    """An expression of the strong form; Python's operators build more of them."""

    precedence = _ATOM

    # --- notation -----------------------------------------------------------------------------------------------
    def notation(self):
        """The expression in ymf's notation, as a spec writes it."""
        raise NotImplementedError

    def __str__(self):
        return self.notation()

    def __repr__(self):
        return "<%s %s>" % (type(self).__name__, self.notation())

    def names(self):
        """Every named object (unknown, coefficient) the expression reads."""
        out = []
        for child in self.children():
            _extend(out, child.names())
        return out

    def children(self):
        return ()

    # --- arithmetic ---------------------------------------------------------------------------------------------
    def __add__(self, other):
        return _sum(self, _expr(other))

    def __radd__(self, other):
        return _sum(_expr(other), self)

    def __sub__(self, other):
        return _sum(self, -_expr(other))

    def __rsub__(self, other):
        return _sum(_expr(other), -self)

    def __mul__(self, other):
        return _product(self, _expr(other))

    def __rmul__(self, other):
        return _product(_expr(other), self)

    def __truediv__(self, other):
        return Quotient(self, _expr(other))

    def __rtruediv__(self, other):
        return Quotient(_expr(other), self)

    def __pow__(self, other):
        return Power(self, _expr(other))

    def __neg__(self):
        return self.inner if isinstance(self, Negative) else Negative(self)

    def __pos__(self):
        return self

    def __eq__(self, other):
        return Equation(self, _expr(other))

    __hash__ = object.__hash__


def _extend(out, more):
    """Append what ``out`` does not hold yet, by identity: ``==`` between expressions makes an Equation."""
    for n in more:
        if not any(n is m for m in out):
            out.append(n)
    return out


def _expr(value):
    if isinstance(value, Expr):
        return value
    if isinstance(value, numbers.Number):
        return Negative(Number(-value)) if value < 0 else Number(value)
    if isinstance(value, (tuple, list)):
        return Vector(*value)
    raise TypeError("%r is not an expression of the strong form" % (value,))


class Number(Expr):
    def __init__(self, value):
        self.value = value
        if value < 0:
            raise ValueError("Number holds a magnitude; a negative number is -Number")

    def notation(self):
        return _number_text(self.value)


class Name(Expr):
    """A name the notation knows: a coordinate, time, π, an unknown or a coefficient."""

    def __init__(self, name):
        self.name = name

    def notation(self):
        return self.name


class Unknown(Name):
    """An unknown of the problem. ``rank`` 1 is a vector, whose components are ``v_0, v_1, ...``."""

    def __init__(self, name, rank=0, units=None):
        Name.__init__(self, name)
        self.rank = rank
        self.units = units

    def names(self):
        return [self]


class Coefficient(Name):
    """A coefficient: a number, or an expression in the coordinates, time, other coefficients and the unknowns.

    It appears in the equations by name; its value is written once, under the spec's ``coefficients``.
    """

    def __init__(self, name, value, units=None):
        Name.__init__(self, name)
        self.value = value if isinstance(value, numbers.Number) else _expr(value)
        self.units = units

    def names(self):
        out = [self]
        if isinstance(self.value, Expr):
            out.extend(n for n in self.value.names() if n is not self)
        return out

    def entry(self):
        if isinstance(self.value, Expr):
            entry = {"formula": self.value.notation()}
        else:
            entry = float(self.value) if self.units is None else {"value": float(self.value)}
        if self.units is not None:
            entry["units"] = self.units
        return entry


x, y, z, t = Name("x"), Name("y"), Name("z"), Name("t")
pi = Name("π")


class Negative(Expr):
    precedence = _PRODUCT

    def __init__(self, inner):
        self.inner = inner

    def children(self):
        return (self.inner,)

    def notation(self):
        return "−" + _wrap(self.inner, _PRODUCT)


class Sum(Expr):
    precedence = _SUM

    def __init__(self, terms):
        self.terms = terms

    def children(self):
        return tuple(self.terms)

    def notation(self):
        text = _wrap(self.terms[0], _SUM)
        for term in self.terms[1:]:
            if isinstance(term, Negative):
                text += " − " + _wrap(term.inner, _PRODUCT)
            else:
                text += " + " + _wrap(term, _SUM + 1)
        return text


def _sum(a, b):
    return Sum((a.terms if isinstance(a, Sum) else [a]) + (b.terms if isinstance(b, Sum) else [b]))


class Product(Expr):
    precedence = _PRODUCT

    def __init__(self, factors):
        self.factors = factors

    def children(self):
        return tuple(self.factors)

    def notation(self):
        text = ""
        previous = None
        for factor in self.factors:
            # a quotient inside a product is wrapped: "1/2 π" would read either way
            piece = "(%s)" % factor.notation() if isinstance(factor, Quotient) else _wrap(factor, _PRODUCT + 1)
            if previous is None:
                text = piece
            elif isinstance(previous, Number) and not piece[0].isdigit():
                text += piece                    # 10u, 16π, 2(a + b)
            else:
                text += " " + piece              # a ∇φ, π x, sin(πx) sin(πy)
            previous = factor
        return text


def _product(a, b):
    # a sign comes out to the front: (-a)*grad(φ) is −a ∇φ
    if isinstance(a, Negative):
        return -_product(a.inner, b)
    if isinstance(b, Negative):
        return -_product(a, b.inner)
    return Product((a.factors if isinstance(a, Product) else [a]) + (b.factors if isinstance(b, Product) else [b]))


class Quotient(Expr):
    precedence = _PRODUCT

    def __init__(self, numerator, denominator):
        self.numerator, self.denominator = numerator, denominator

    def children(self):
        return (self.numerator, self.denominator)

    def notation(self):
        return "%s/%s" % (_wrap(self.numerator, _PRODUCT), _wrap(self.denominator, _POWER))


class Power(Expr):
    precedence = _POWER

    def __init__(self, base, exponent):
        self.base, self.exponent = base, exponent

    def children(self):
        return (self.base, self.exponent)

    def notation(self):
        base = _wrap(self.base, _ATOM)
        e = self.exponent
        if isinstance(e, Number) and isinstance(e.value, numbers.Integral) and e.value in _SUPERSCRIPTS:
            return base + _SUPERSCRIPTS[e.value]
        return "%s^%s" % (base, _wrap(e, _ATOM))


class Call(Expr):
    """A function of the notation applied to its arguments."""

    def __init__(self, function, *args):
        self.function, self.args = function, [_expr(a) for a in args]

    def children(self):
        return tuple(self.args)

    def notation(self):
        return "%s(%s)" % (self.function, ", ".join(a.notation() for a in self.args))


class Vector(Expr):
    def __init__(self, *entries):
        self.entries = [_expr(e) for e in entries]

    def children(self):
        return tuple(self.entries)

    def notation(self):
        return "(%s)" % ", ".join(e.notation() for e in self.entries)


class Operator(Expr):
    """A differential operator applied to the operand that follows it: ∇u, ∇·F, Δu."""

    precedence = _POWER

    def __init__(self, symbol, operand):
        self.symbol, self.operand = symbol, _expr(operand)

    def children(self):
        return (self.operand,)

    def notation(self):
        return self.symbol + _wrap(self.operand, _ATOM)


class TimeDerivative(Expr):
    def __init__(self, operand):
        self.operand = _expr(operand)

    def children(self):
        return (self.operand,)

    def notation(self):
        if isinstance(self.operand, Name):
            return "∂%s/∂t" % self.operand.notation()
        return "dt(%s)" % self.operand.notation()


class Binary(Expr):
    """A dot (·) or outer (⊗) product of vectors."""

    precedence = _PRODUCT

    def __init__(self, symbol, a, b):
        self.symbol, self.a, self.b = symbol, _expr(a), _expr(b)

    def children(self):
        return (self.a, self.b)

    def notation(self):
        return "%s%s%s" % (_wrap(self.a, _POWER), self.symbol, _wrap(self.b, _POWER))


def grad(e):
    """∇e"""
    return Operator("∇", e)


def div(e):
    """∇·e"""
    return Operator("∇·", e)


def lap(e):
    """Δe"""
    return Operator("Δ", e)


def dt(e):
    """∂e/∂t"""
    return TimeDerivative(e)


def dot(a, b):
    return Binary("·", a, b)


def outer(a, b):
    return Binary("⊗", a, b)


def vector(*entries):
    return Vector(*entries)


def _function(name):
    def apply(arg):
        return Call(name, arg)
    apply.__name__ = name
    apply.__doc__ = "%s(e) in the notation" % name
    return apply


sin, cos, tan, atan = _function("sin"), _function("cos"), _function("tan"), _function("atan")
exp, log, sqrt = _function("exp"), _function("log"), _function("sqrt")
sinh, cosh, tanh = _function("sinh"), _function("cosh"), _function("tanh")
absolute = _function("abs")


class Equation(object):
    """``lhs == rhs``: one (scalar or vector) equation of the strong form."""

    def __init__(self, lhs, rhs):
        self.lhs, self.rhs = lhs, rhs

    def notation(self):
        return "%s = %s" % (self.lhs.notation(), self.rhs.notation())

    def names(self):
        return _extend([], self.lhs.names() + self.rhs.names())

    def __str__(self):
        return self.notation()

    def __repr__(self):
        return "<Equation %s>" % self.notation()

    def __bool__(self):
        raise TypeError("an Equation is not a truth value: pass it to Problem.equation")


# --- the problem ---------------------------------------------------------------------------------------------------


class Box(object):
    """Ω = [a0, b0] × [a1, b1] (× [a2, b2]), optionally with a time interval ``time=(t0, t1)``."""

    def __init__(self, *intervals, time=None):
        if not 1 <= len(intervals) <= 3:
            raise ValueError("a box has 1, 2 or 3 intervals")
        self.intervals = [tuple(i) for i in intervals]
        self.time = tuple(time) if time is not None else None

    @property
    def dim(self):
        return len(self.intervals)

    def notation(self):
        text = "Ω = " + " × ".join("[%s, %s]" % (_number_text(a), _number_text(b)) for a, b in self.intervals)
        if self.time is not None:
            text += ",  t ∈ (%s, %s]" % tuple(_number_text(v) for v in self.time)
        return text


class Region(object):
    """A named part of the boundary, by a predicate in the notation: ``"∂Ω"``, ``"x = 0 or x = 1"``."""

    def __init__(self, name, where):
        self.name, self.where = name, where

    def entry(self):
        return {"name": self.name, "where": self.where}


def _value(value):
    """A condition's value: a number (``value``) or an expression (``formula``)."""
    if isinstance(value, numbers.Number):
        return {"value": float(value)}
    return {"formula": _expr(value).notation()}


class Problem(object):
    """A problem posed by its strong form, written out as a ymf spec.

    ``unknowns`` are :class:`Unknown` objects, in the order the equations determine them. The coefficients are found
    in the equations, conditions and other coefficients; give them in ``coefficients`` too to fix their order in the
    spec.
    """

    def __init__(self, name, domain, unknowns, coefficients=(), processes=("diffusion",), assumptions=None,
                 formulation=None, provenance="human_specified"):
        self.name = name
        self.domain = domain
        self.unknowns = list(unknowns)
        self._coefficients = list(coefficients)
        self.processes = list(processes)
        self.assumptions = (list(assumptions) if assumptions is not None
                            else ["steady"] if domain.time is None else [])
        self.formulation = formulation
        self.provenance = provenance
        self.equations = []
        self.regions = []
        self.boundary_conditions = []
        self.initial_conditions = []
        self.analytical = []
        self.discretizations = []
        #: the whole boundary, ∂Ω
        self.boundary = self.region("boundary", "∂Ω")

    # --- posing ---------------------------------------------------------------------------------------------------
    def equation(self, equation):
        if not isinstance(equation, Equation):
            raise TypeError("pass an equation, lhs == rhs")
        self.equations.append(equation)
        return equation

    def region(self, name, where):
        if any(r.name == name for r in self.regions):
            raise ValueError("a boundary region named %r exists" % name)
        region = Region(name, where)
        self.regions.append(region)
        return region

    def dirichlet(self, unknown, value, on):
        self._condition("dirichlet", unknown, value, on)

    def neumann(self, unknown, value, on):
        """The outward diffusive flux, q·n with q = −a∇φ."""
        self._condition("neumann", unknown, value, on)

    def _condition(self, kind, unknown, value, on):
        self._check_unknown(unknown)
        if not any(on is r for r in self.regions):
            raise ValueError("%r is not a region of this problem: make it with Problem.region" % (on,))
        self.boundary_conditions.append(dict({"region": on.name, "variable": unknown.name, "type": kind},
                                             **_value(value)))

    def initial(self, unknown, value):
        """The initial condition, which for a steady problem is Newton's starting guess."""
        self._check_unknown(unknown)
        if isinstance(value, numbers.Number):
            self.initial_conditions.append({"field": unknown.name, "type": "constant", "value": float(value)})
        else:
            self.initial_conditions.append({"field": unknown.name, "type": "function",
                                            "formula": _expr(value).notation()})

    def exact(self, unknown, value, name="exact", method="closed_form"):
        """An analytical solution, for the errors of every output."""
        self._check_unknown(unknown)
        lines = "%s = %s" % (unknown.name, _expr(value).notation())
        for entry in self.analytical:
            if entry["name"] == name:
                entry["solution"]["formula"] += "\n" + lines
                return
        self.analytical.append({"name": name, "provenance": self.provenance, "method": method,
                                "solution": {"formula": lines}})

    def discretization(self, name, order=1, family="CG", cells=None, h=None, levels=None,
                       nonlinear_solver="newton", linear_solver="superlu", tolerance=1.0e-10, **extra):
        """A finite element discretization: the same ``family`` and ``order`` for every unknown, a box's ``cells``
        per side (a list is a convergence study), and the solver. ``extra`` keys are copied into the entry."""
        entry = {"name": name, "provenance": self.provenance,
                 "finite_element": {"fields": {"family": family, "order": int(order)}}}
        mesh = {}
        for key, value in (("cells", cells), ("h", h), ("levels", levels)):
            if value is not None:
                mesh[key] = list(value) if isinstance(value, (list, tuple)) else value
        if mesh:
            entry["mesh"] = mesh
        entry["solver"] = {"type": "nonlinear", "nonlinear_solver": nonlinear_solver,
                           "linear_solver": linear_solver, "tolerance": float(tolerance)}
        entry.update(extra)
        self.discretizations.append(entry)

    def _check_unknown(self, unknown):
        if not any(unknown is u for u in self.unknowns):
            raise ValueError("%r is not an unknown of this problem" % (unknown,))

    # --- the spec -------------------------------------------------------------------------------------------------
    def coefficients(self):
        """Every coefficient the problem reads, in the order given, then in the order found."""
        found = list(self._coefficients)
        sources = list(self.equations) + list(self._coefficients)
        while sources:
            for n in sources.pop(0).names():
                if isinstance(n, Coefficient) and not any(n is c for c in found):
                    found.append(n)
                    sources.append(n)
        names = [c.name for c in found]
        twins = sorted({n for n in names if names.count(n) > 1})
        if twins:
            raise ValueError("two different coefficients are named %s" % ", ".join(twins))
        return found

    def spec(self, state=None):
        """The ymf spec, as plain data: what :meth:`write` writes."""
        if not self.equations:
            raise ValueError("the problem has no equations")
        unknowns = [u.name if u.rank == 0 and u.units is None
                    else dict({"name": u.name}, **({"rank": u.rank} if u.rank else {}),
                              **({"units": u.units} if u.units else {}))
                    for u in self.unknowns]
        strong = {"provenance": self.provenance,
                  "unknowns": unknowns,
                  "equation_formulation": self.formulation or ", ".join(self.processes),
                  "equations": ["%s  in Ω" % e.notation() for e in self.equations],
                  "domain": self.domain.notation(),
                  "boundary_regions": [r.entry() for r in self.regions
                                       if any(bc["region"] == r.name for bc in self.boundary_conditions)
                                       or r is not self.boundary]}
        if self.boundary_conditions:
            strong["boundary_conditions"] = [dict(bc) for bc in self.boundary_conditions]
        if self.initial_conditions:
            strong["initial_conditions"] = [dict(ic) for ic in self.initial_conditions]
        coefficients = self.coefficients()
        if coefficients:
            strong["coefficients"] = {c.name: c.entry() for c in coefficients}
        model = {"name": self.name,
                 "physical_model": {"provenance": self.provenance, "processes": list(self.processes),
                                    "assumptions": list(self.assumptions)},
                 "strong_form": strong}
        paths = {}
        if self.analytical:
            paths["analytical"] = [dict(a, solution=dict(a["solution"])) for a in self.analytical]
        paths["discretizations"] = [dict(d) for d in self.discretizations]
        doc = {}
        if state is None:
            state = "discretized" if self.discretizations and all(
                "mesh" in d for d in self.discretizations) else "well-posed"
        if state:
            doc["ymf_state"] = state
        doc["model"] = model
        doc["solution_paths"] = paths
        return doc

    def yaml(self, state=None):
        """The spec as YAML text, Unicode as written."""
        import yaml
        return yaml.safe_dump(self.spec(state), sort_keys=False, allow_unicode=True, width=120)

    def write(self, path, state=None):
        """Write the spec to ``path``; return the path."""
        with open(path, "w", encoding="utf-8") as f:
            f.write(self.yaml(state))
        return path

    # --- solving, through the spec --------------------------------------------------------------------------------
    def adr(self):
        """ymf's plain-data ADR problem of the spec (needs ``ymf[symbolic]``): what :mod:`proteus.ADRProblem`
        solves."""
        from ymf.symbolic import adr_problem
        return adr_problem(self.spec())

    def run(self, outdir, stem=None, discretization=None, check=False):
        """``pro run`` on the spec, written to ``<outdir>/<stem>.ymf``: the archive's path, which holds every output
        (keyed by a hash of what produced it) and its errors against the analytical solution. A rerun computes
        only what the archive lacks; ``check`` reruns those too and compares them bitwise."""
        from proteus.cli import main
        os.makedirs(outdir, exist_ok=True)
        stem = stem or "".join(ch if ch.isalnum() else "_" for ch in self.name.lower()).strip("_")
        path = self.write(os.path.join(outdir, stem + ".ymf"))
        argv = ["run", path, "--outdir", outdir]
        for name in ([discretization] if isinstance(discretization, str) else discretization or []):
            argv += ["--discretization", name]
        if check:
            argv.append("--check")
        status = main(argv)
        if status:
            raise RuntimeError("pro run %s exited %s" % (path, status))
        return os.path.join(outdir, stem + ".archive.ymf")
