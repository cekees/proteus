"""proteus.StrongForm (a strong form posed with Python operators, written as a ymf spec) and the nonlinear potential
of proteus.ADRProblem (div(-a grad(phi(u))), ymf's `potential` block).

The printing tests need nothing but proteus. The tests that read the spec back need ymf[symbolic]; those that read a
nonlinear potential need a ymf whose classifier knows one, and skip on an older ymf.
"""
import os

import numpy
import pytest

from proteus import ADRProblem
from proteus.StrongForm import (Box, Coefficient, Equation, Problem, Unknown, div, dt, grad, lap, pi, sin, sqrt,
                                vector, x, y)


def kirchhoff(cells=(40, 80)):
    """The nonlinear example of the Introduction notebook: div(-a grad(phi)) + r = 0, phi = u^2 + 10u."""
    u = Unknown("u")
    a = Coefficient("a", 1.0)
    phi = Coefficient("φ", u**2 + 10*u)
    r = Coefficient("r", -16*pi*sin(4*pi*x)*sin(4*pi*y))
    problem = Problem("Nonlinear Poisson", Box((0, 1), (0, 1)), unknowns=[u])
    problem.equation(div(-a*grad(phi)) + r == 0)
    problem.dirichlet(u, 0.0, on=problem.boundary)
    problem.initial(u, 0.0)
    problem.exact(u, -5 + sqrt(25 + sin(4*pi*x)*sin(4*pi*y)/(2*pi)))
    problem.discretization("P1", order=1, cells=list(cells), tolerance=1e-12)
    return problem


KIRCHHOFF_YAML = """\
ymf_state: discretized
model:
  name: "Nonlinear Poisson"
  physical_model: {provenance: human_specified, processes: [diffusion], assumptions: [steady]}
  strong_form:
    provenance: human_specified
    unknowns: [u]
    equation_formulation: "diffusion"
    equations:
      - "∇·(−a∇φ) + r = 0  in Ω"
    domain: "Ω = [0, 1] × [0, 1]"
    boundary_regions:
      - {name: boundary, where: "∂Ω"}
    boundary_conditions:
      - {region: boundary, variable: u, type: dirichlet, value: 0.0}
    initial_conditions:
      - {field: u, type: constant, value: 0.0}
    coefficients:
      a: 1.0
      φ: {formula: "u² + 10u"}
      r: {formula: "−16π sin(4πx) sin(4πy)"}
solution_paths:
  analytical:
    - name: exact
      provenance: human_specified
      method: closed_form
      solution:
        formula: "u = −5 + sqrt(25 + sin(4πx) sin(4πy)/(2π))"
  discretizations:
    - name: P1
      provenance: human_specified
      finite_element: {fields: {family: CG, order: 1}}
      mesh: {cells: [40, 80]}
      solver: {type: nonlinear, nonlinear_solver: newton, linear_solver: superlu, tolerance: 1.0e-12}
"""


# --- printing: the objects write ymf's notation ------------------------------------------------------------------


def test_expressions_print_as_a_spec_writes_them():
    u = Unknown("u")
    a = Coefficient("a", 1.0)
    phi = Coefficient("φ", u**2 + 10*u)
    assert (u**2 + 10*u).notation() == "u² + 10u"
    assert (div(-a*grad(phi)) + 1 == 0).notation() == "∇·(−a ∇φ) + 1 = 0"
    assert (-16*pi*sin(4*pi*x)).notation() == "−16π sin(4π x)"
    assert (-5 + sqrt(25 + sin(x)/(2*pi))).notation() == "−5 + sqrt(25 + sin(x)/(2π))"
    assert (dt(u) - lap(u) == x*y).notation() == "∂u/∂t − Δu = x y"
    assert (div(vector(1, 0)*u)).notation() == "∇·((1, 0) u)"
    assert ((x + y)**3).notation() == "(x + y)³"
    assert (u**0.5).notation() == "u^0.5"
    assert (1/(2*pi)*x).notation() == "(1/(2π)) x"
    assert (u - (x - y)).notation() == "u − (x − y)"


def test_an_equation_is_not_a_truth_value():
    u = Unknown("u")
    equation = u == 0
    assert isinstance(equation, Equation)
    with pytest.raises(TypeError):
        bool(equation)


def test_the_spec_holds_each_coefficient_once_by_name():
    spec = kirchhoff().spec()
    strong = spec["model"]["strong_form"]
    assert strong["equations"] == ["∇·(−a ∇φ) + r = 0  in Ω"]
    assert strong["coefficients"] == {"a": 1.0, "φ": {"formula": "u² + 10u"},
                                      "r": {"formula": "−16π sin(4π x) sin(4π y)"}}
    assert strong["boundary_conditions"] == [{"region": "boundary", "variable": "u", "type": "dirichlet",
                                              "value": 0.0}]
    assert spec["ymf_state"] == "discretized"
    assert spec["solution_paths"]["analytical"][0]["solution"]["formula"] == \
        "u = −5 + sqrt(25 + sin(4π x) sin(4π y)/(2π))"


def test_a_condition_on_a_region_of_another_problem_is_refused():
    u = Unknown("u")
    p, q = Problem("p", Box((0, 1)), [u]), Problem("q", Box((0, 1)), [u])
    with pytest.raises(ValueError, match="not a region of this problem"):
        p.dirichlet(u, 0.0, on=q.boundary)
    with pytest.raises(ValueError, match="not an unknown"):
        p.dirichlet(Unknown("v"), 0.0, on=p.boundary)


def test_two_coefficients_with_one_name_are_refused():
    u = Unknown("u")
    p = Problem("p", Box((0, 1)), [u])
    p.equation(div(-Coefficient("a", 1.0)*grad(u)) + Coefficient("a", 2.0) == 0)
    with pytest.raises(ValueError, match="named a"):
        p.spec()


# --- read back by ymf -----------------------------------------------------------------------------------------------


def _symbolic():
    pytest.importorskip("sympy")
    pytest.importorskip("strictyaml")
    return pytest.importorskip("ymf.symbolic")


def _knows_nonlinear_potentials(symbolic):
    space = symbolic.Space(1, {"u": 0})
    try:
        symbolic.adr_form(symbolic.parse_equation("∇·(−∇(u²)) = 0", space), space)
    except symbolic.SymbolicError:
        return False
    return True


@pytest.mark.parametrize("written,python", [
    ("u² + 10u", lambda u: u**2 + 10*u),
    ("−16π sin(4πx) sin(4πy)", lambda u: -16*pi*sin(4*pi*x)*sin(4*pi*y)),
    ("−5 + sqrt(25 + sin(4πx) sin(4πy)/(2π))", lambda u: -5 + sqrt(25 + sin(4*pi*x)*sin(4*pi*y)/(2*pi))),
    ("(1/(2π)) x u³", lambda u: 1/(2*pi)*x*u**3),
    ("x − (y − u)/2", lambda u: x - (y - u)/2),
    ("(x + y)^0.5 + 2/3", lambda u: (x + y)**0.5 + 2/3),
])
def test_what_the_objects_print_reads_back_as_written(written, python):
    symbolic = _symbolic()
    import sympy
    space = symbolic.Space(2, {"u": 0})
    printed = python(Unknown("u")).notation()
    difference = symbolic.parse_expression(printed, space) - symbolic.parse_expression(written, space)
    assert sympy.simplify(difference) == 0, (printed, written)


def test_the_python_problem_is_the_yaml_problem(tmp_path):
    symbolic = _symbolic()
    if not _knows_nonlinear_potentials(symbolic):
        pytest.skip("this ymf refuses a nonlinear potential")
    from ymf import closure
    (tmp_path / "written.ymf").write_text(KIRCHHOFF_YAML, encoding="utf-8")
    kirchhoff().write(str(tmp_path / "python.ymf"))
    written = symbolic.adr_problem(closure.plain(closure.load(str(tmp_path / "written.ymf"))[0]))
    python = symbolic.adr_problem(closure.plain(closure.load(str(tmp_path / "python.ymf"))[0]))
    assert python == written
    assert written["adr"]["equations"][0]["diffusion"]["u"]["potential"]["phi"] in ("u**2 + 10.0*u", "u**2 + 10*u")


def test_a_problem_runs_through_pro_run_and_reproduces(tmp_path, monkeypatch):
    """Linear, so any ymf[symbolic] runs it: -Δu = 2π² sin(πx) sin(πy), u = sin(πx) sin(πy)."""
    _symbolic()
    monkeypatch.chdir(tmp_path)
    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        monkeypatch.setenv(var, "1")
    u = Unknown("u")
    problem = Problem("Poisson", Box((0, 1), (0, 1)), unknowns=[u])
    problem.equation(-lap(u) == 2*pi**2*sin(pi*x)*sin(pi*y))
    problem.dirichlet(u, 0.0, on=problem.boundary)
    problem.exact(u, sin(pi*x)*sin(pi*y))
    problem.discretization("P1", order=1, cells=[8, 16])
    archive = problem.run(str(tmp_path / "out"))
    from ymf import closure
    _, outputs, _ = closure.load(archive, check_state=False)
    assert len(outputs) == 2
    assert archive == problem.run(str(tmp_path / "out"), check=True)
    assert os.path.exists(archive)


# --- the nonlinear potential in ADRProblem, from plain data (no sympy) ----------------------------------------------


EXACT = "numpy.sqrt(25 + numpy.sin(4*numpy.pi*x)*numpy.sin(4*numpy.pi*y)/(2*numpy.pi)) - 5"


def kirchhoff_problem(phi="u**2 + 10.0*u", dphi="2*u + 10.0"):
    """The plain data ymf's adr_problem gives for the notebook's problem, written out."""
    return {
        "adr": {"dim": 2, "components": ["u"], "equations": [{
            "component": "u",
            "diffusion": {"u": {"rowptr": [0, 1, 2], "colind": [0, 1], "a": ["1.0", "1.0"], "da": {},
                                "depends_on": {},
                                "potential": {"phi": phi, "dphi": {"u": dphi}, "depends_on": {"u": "nonlinear"}}}},
            "reaction": {"r": "-16*numpy.pi*numpy.sin(4*numpy.pi*x)*numpy.sin(4*numpy.pi*y)", "dr": {},
                         "depends_on": {}}}]},
        "unknowns": {"u": ["u"]},
        "domain": {"lower": [0.0, 0.0], "upper": [1.0, 1.0]},
        "dirichlet": [{"component": "u", "region": "boundary", "where": None, "value": "0.0"}],
        "neumann": [], "robin": [], "periodic": [],
        "initial": {"u": "0.0"}, "exact": {"u": EXACT}, "time": None, "has_mass": False,
    }


def test_a_potential_block_sets_proteus_nonlinear_potential():
    c = ADRProblem.ADRCoefficients(kirchhoff_problem()["adr"])
    assert c.potential == {0: {0: 'nonlinear'}}
    u = numpy.linspace(-1, 1, 6).reshape(2, 3)
    q = {'x': numpy.zeros((2, 3, 3)), ('u', 0): u, ('phi', 0): numpy.zeros((2, 3)),
         ('dphi', 0, 0): numpy.zeros((2, 3)), ('a', 0, 0): numpy.zeros((2, 3, 2)), ('r', 0): numpy.zeros((2, 3))}
    c.evaluate(0.0, q)
    numpy.testing.assert_array_equal(q[('phi', 0)], u**2 + 10.0*u)
    numpy.testing.assert_array_equal(q[('dphi', 0, 0)], 2*u + 10.0)


def test_two_potentials_of_one_component_are_refused():
    adr = {"dim": 1, "components": ["u", "v"], "equations": [
        {"component": c, "diffusion": {"u": {"rowptr": [0, 1], "colind": [0], "a": ["1.0"], "da": {},
                                             "depends_on": {},
                                             "potential": {"phi": phi, "dphi": {"u": "1.0"},
                                                           "depends_on": {"u": "nonlinear"}}}}}
        for c, phi in (("u", "u**3"), ("v", "u**2"))]}
    with pytest.raises(ValueError, match="one potential per component"):
        ADRProblem.ADRCoefficients(adr)


def test_a_potential_of_another_component_is_refused():
    problem = kirchhoff_problem()
    problem["adr"]["equations"][0]["diffusion"]["u"]["potential"]["depends_on"] = {"v": "nonlinear"}
    with pytest.raises(ValueError, match="function of u_k alone"):
        ADRProblem.ADRCoefficients(problem["adr"])


def test_the_nonlinear_potential_converges_at_second_order_with_an_exact_jacobian(tmp_path, monkeypatch):
    """The notebook's 1.x solve with strong Dirichlet conditions gave 8.91e-4, 2.34e-4, 5.92e-5 at 20, 40, 80
    cells; the same problem through ADRProblem gives the same errors. Newton from zero meets 1e-12 in 3
    iterations: the Jacobian carries dphi (an inexact one converges linearly, if at all)."""
    from proteus import runtime
    runtime.start(log=None)
    monkeypatch.chdir(tmp_path)
    errors = []
    for cells in (20, 40):
        _, model, error = ADRProblem.run(kirchhoff_problem(), {"u": ("CG", 1)}, cells, name="k%d" % cells,
                                         tolerance=1e-12)
        errors.append(error["u"])
    numpy.testing.assert_allclose(errors, [8.914e-4, 2.338e-4], rtol=1e-3)
    assert numpy.log2(errors[0]/errors[1]) > 1.9
    ns, _, _ = ADRProblem.run(kirchhoff_problem(), {"u": ("CG", 1)}, 8, name="newton", tolerance=1e-12)
    assert ns.modelList[0].solver.solverList[-1].its <= 4
