"""proteus.ADRProblem: Proteus problems from a plain-data ADR form.

The unit tests build the ADR form by hand -- ADRProblem never needs sympy.
The end-to-end tests go through scripts/ymf_run and need ymf[symbolic].
"""
import os
import subprocess
import sys

import numpy
import pytest

from proteus import ADRProblem

SCRIPT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "scripts", "ymf_run")


def poisson_adr(diffusion_depends_on=None):
    """-div(a grad u) = 1 on the unit square, written out as ymf emits it."""
    return {
        "dim": 2, "components": ["u"],
        "equations": [{
            "component": "u",
            "diffusion": {"u": {"rowptr": [0, 1, 2], "colind": [0, 1], "a": ["1.0", "1.0"],
                                "da": {}, "depends_on": diffusion_depends_on or {}}},
            "reaction": {"r": "-1.0", "dr": {}, "depends_on": {}},
        }],
    }


def test_constant_coefficients_get_proteus_constant_flags():
    c = ADRProblem.ADRCoefficients(poisson_adr())
    assert c.diffusion == {0: {0: {0: 'constant'}}}
    assert c.reaction == {0: {0: 'constant'}}
    assert c.potential == {0: {0: 'u'}}
    assert c.sdInfo[(0, 0)][0].tolist() == [0, 1, 2]


def test_any_dependence_makes_diffusion_nonlinear():
    # Proteus's diffusion flags are only 'constant' or 'nonlinear'
    c = ADRProblem.ADRCoefficients(poisson_adr({"u": "linear"}))
    assert c.diffusion == {0: {0: {0: 'nonlinear'}}}


def test_coefficients_are_evaluated_from_their_code_strings():
    adr = poisson_adr()
    adr["equations"][0]["reaction"]["r"] = "-x*y"
    c = ADRProblem.ADRCoefficients(adr)
    x = numpy.random.default_rng(0).random((3, 4, 3))
    q = {'x': x, ('u', 0): numpy.zeros((3, 4)), ('r', 0): numpy.zeros((3, 4)),
         ('a', 0, 0): numpy.zeros((3, 4, 2))}
    c.evaluate(0.0, q)
    numpy.testing.assert_array_equal(q[('r', 0)], -x[..., 0] * x[..., 1])
    assert (q[('a', 0, 0)] == 1.0).all()


def test_reorder_permutes_components_and_their_equations_together():
    problem = {"adr": {"dim": 2, "components": ["v_0", "v_1", "p"],
                       "equations": [{"component": n} for n in ("v_0", "v_1", "p")]},
               "unknowns": {"v": ["v_0", "v_1"], "p": ["p"]}}
    out = ADRProblem.reorder(problem, ["p", "v_0", "v_1"])
    assert out["adr"]["components"] == ["p", "v_0", "v_1"]
    assert [e["component"] for e in out["adr"]["equations"]] == ["p", "v_0", "v_1"]
    assert problem["adr"]["components"] == ["v_0", "v_1", "p"]      # untouched


def test_supg_pspg_refuses_systems_that_are_not_velocity_pressure():
    with pytest.raises(ValueError, match="velocity-pressure"):
        ADRProblem.velocity_pressure({"unknowns": {"u": ["u"], "w": ["w"]}})


def neumann_poisson():
    """-Laplace(u) = -4 on the unit square, u = x^2 + y^2.

    Dirichlet on three sides; on x = 1 the outward diffusive flux
    q.n = -(grad u).n = -2x. Written as Proteus's ADR residual,
    -div(a grad u) + r = 0, so r = +4.
    """
    return {
        "adr": {
            "dim": 2, "components": ["u"],
            "equations": [{
                "component": "u",
                "diffusion": {"u": {"rowptr": [0, 1, 2], "colind": [0, 1],
                                    "a": ["1.0", "1.0"], "da": {}, "depends_on": {}}},
                "reaction": {"r": "4.0", "dr": {}, "depends_on": {}},
            }],
        },
        "domain": {"lower": [0.0, 0.0], "upper": [1.0, 1.0]},
        "dirichlet": [{"component": "u", "region": "three_sides",
                       "where": "x <= tol or y <= tol or y >= 1.0 - tol",
                       "value": "x**2 + y**2"}],
        "neumann": [{"component": "u", "region": "right",
                     "where": "x >= 1.0 - tol", "value": "-2.0*x"}],
        "exact": {"u": "x**2 + y**2"},
    }


def robin_poisson(alpha="1.0", beta="1.0", g="y**2 - 1.0"):
    """-Laplace(u) = -4 on the unit square, u = x^2 + y^2.

    On x = 1 the outward flux is q.n = -(grad u).n = -2, and u = 1 + y^2, so
    alpha u + beta (q.n) = g holds with alpha = beta = 1 and g = y^2 - 1.
    Dirichlet on the other three sides.
    """
    return {
        "adr": {
            "dim": 2, "components": ["u"],
            "equations": [{
                "component": "u",
                "diffusion": {"u": {"rowptr": [0, 1, 2], "colind": [0, 1],
                                    "a": ["1.0", "1.0"], "da": {}, "depends_on": {}}},
                "reaction": {"r": "4.0", "dr": {}, "depends_on": {}},
            }],
        },
        "domain": {"lower": [0.0, 0.0], "upper": [1.0, 1.0]},
        "dirichlet": [{"component": "u", "region": "three_sides",
                       "where": "x <= tol or y <= tol or y >= 1.0 - tol",
                       "value": "x**2 + y**2"}],
        "robin": [{"component": "u", "region": "right", "where": "x >= 1.0 - tol",
                   "alpha": alpha, "beta": beta, "g": g}],
        "exact": {"u": "x**2 + y**2"},
    }


def _rate(problem, cells=(8, 16), **kw):
    errors = [ADRProblem.run(problem, {"u": ("CG", 1)}, c, **kw)[2]["u"] for c in cells]
    return errors, numpy.log2(errors[0] / errors[1])


def test_a_robin_face_converges_at_the_right_rate():
    errors, rate = _rate(robin_poisson())
    assert errors[0] < 1.0e-2
    assert 1.8 < rate < 2.2, (errors, rate)


def test_robin_coefficients_may_vary_along_the_boundary():
    """alpha and beta are expressions, not just numbers."""
    # alpha = 3, beta = 2  =>  g = 3(1 + y^2) + 2(-2)
    errors, rate = _rate(robin_poisson(alpha="3.0", beta="2.0",
                                       g="3.0*(1.0 + y**2) - 4.0"))
    assert errors[0] < 1.0e-2
    assert 1.8 < rate < 2.2, (errors, rate)


def test_a_wrong_robin_right_hand_side_does_not_converge():
    """The sharp check: g off by one stalls the error instead of refining."""
    errors, rate = _rate(robin_poisson(g="y**2"))
    assert errors[0] > 1.0e-1
    assert rate < 0.5, (errors, rate)


def test_robin_with_no_u_coefficient_is_the_neumann_answer():
    """alpha = 0 reduces to a prescribed flux, down a different code path.

    The Robin class writes the flux itself; a neumann condition is applied by
    OneLevelTransport after the numerical flux runs. Agreeing to the bit is
    what says the Robin flux has the same sign and magnitude.
    """
    robin, _ = _rate(robin_poisson(alpha="0.0", g="-2.0"))
    neumann, _ = _rate(neumann_poisson())
    assert robin == neumann


def test_a_robin_condition_with_no_flux_coefficient_is_refused():
    """beta = 0 is a Dirichlet condition and should be imposed strongly."""
    with pytest.raises(ValueError, match="beta = 0"):
        ADRProblem.run(robin_poisson(beta="0.0"), {"u": ("CG", 1)}, 4)


def test_a_robin_condition_on_a_nonexistent_unknown_is_refused():
    problem = robin_poisson()
    problem["robin"][0]["component"] = "v"
    with pytest.raises(ValueError, match="not an unknown"):
        ADRProblem.physics(problem, "adr")


def hydrograph_heat():
    """du/dt - Laplace(u) = f on the unit square, u = (x^2 + y^2)(1 + t).

    f = (x^2 + y^2) - 4(1 + t), so the ADR reaction (residual form
    m_t + div(-a grad u) + r = 0) is r = -f = 4 + 4t - x^2 - y^2.

    On x = 1 the outward diffusive flux is -(grad u).n = -2x(1 + t) = -2(1+t),
    given as a two-sample hydrograph over t in [0, 1]. Linear interpolation
    is EXACT for it, so the solution still converges to the exact answer --
    which is what makes this a test of the interpolation and not just of the
    plumbing. Evaluating the table at a fixed time (ignoring t) gives -2
    throughout and the solution is wrong.
    """
    return {
        "adr": {
            "dim": 2, "components": ["u"],
            "equations": [{
                "component": "u",
                "mass": {"m": "u", "dm": {"u": "1"}, "depends_on": {"u": "linear"}},
                "diffusion": {"u": {"rowptr": [0, 1, 2], "colind": [0, 1],
                                    "a": ["1.0", "1.0"], "da": {}, "depends_on": {}}},
                "reaction": {"r": "4.0 + 4.0*t - x**2 - y**2", "dr": {},
                             "depends_on": {}},
            }],
        },
        "domain": {"lower": [0.0, 0.0], "upper": [1.0, 1.0]},
        "time": [0.0, 1.0],
        "has_mass": True,
        "dirichlet": [{"component": "u", "region": "three_sides",
                       "where": "x <= tol or y <= tol or y >= 1.0 - tol",
                       "value": "(x**2 + y**2)*(1.0 + t)"}],
        "neumann": [{"component": "u", "region": "inflow",
                     "where": "x >= 1.0 - tol",
                     "series": [[0.0, -2.0], [1.0, -4.0]]}],
        "initial": {"u": "x**2 + y**2"},
        "exact": {"u": "(x**2 + y**2)*(1.0 + t)"},
    }


def stage_hydrograph():
    """du/dt - Laplace(u) = x on the unit square, u = (1 + t)x.

    On x = 1 the value is 1 + t -- uniform along the face, varying in time --
    which is what a stage record is: a gauge reading, not a formula. Given as
    the two samples [[0, 1], [1, 2]], for which linear interpolation is
    exact. The other three sides carry the formula.

    This is the case a `type: hydrograph` could not express, because it was
    wired to the flux path. Supplying a value by table is orthogonal to which
    kind of condition it is.
    """
    return {
        "adr": {
            "dim": 2, "components": ["u"],
            "equations": [{
                "component": "u",
                "mass": {"m": "u", "dm": {"u": "1"}, "depends_on": {"u": "linear"}},
                "diffusion": {"u": {"rowptr": [0, 1, 2], "colind": [0, 1],
                                    "a": ["1.0", "1.0"], "da": {}, "depends_on": {}}},
                "reaction": {"r": "-x", "dr": {}, "depends_on": {}},
            }],
        },
        "domain": {"lower": [0.0, 0.0], "upper": [1.0, 1.0]},
        "time": [0.0, 1.0],
        "has_mass": True,
        "dirichlet": [
            {"component": "u", "region": "gauge", "where": "x >= 1.0 - tol",
             "series": [[0.0, 1.0], [1.0, 2.0]]},
            {"component": "u", "region": "other_sides",
             "where": "x <= tol or y <= tol or y >= 1.0 - tol",
             "value": "(1.0 + t)*x"},
        ],
        "initial": {"u": "x"},
        "exact": {"u": "(1.0 + t)*x"},
    }


def test_a_dirichlet_value_may_be_tabulated():
    """A stage hydrograph: the gauge face is given by table, not formula.

    u = (1 + t)x is linear in x, so P1 represents it exactly and the table is
    exact for 1 + t; what is left is the solver's tolerance, ~1e-8. A frozen
    table gives >1e-2 (next test), six orders away, so this threshold is not
    a near thing.
    """
    errors, _ = _rate(stage_hydrograph(), time={"dt": 0.05})
    assert errors[0] < 1.0e-7, errors


def test_a_frozen_stage_table_does_not_track_the_solution():
    """The sharp check: hold the gauge at its first reading and it is wrong."""
    problem = stage_hydrograph()
    problem["dirichlet"][0]["series"] = [[0.0, 1.0], [1.0, 1.0]]
    errors, _ = _rate(problem, time={"dt": 0.05})
    assert errors[0] > 1.0e-2, errors


def test_the_emitted_p_file_carries_a_tabulated_dirichlet(tmp_path):
    ADRProblem.emit_pn(stage_hydrograph(), {"u": ("CG", 1)}, 8, time={"dt": 0.5},
                       name="stage", directory=str(tmp_path))
    text = (tmp_path / "stage_p.py").read_text()
    assert "def getDBC_u(X, flag):" in text
    assert "numpy.interp(t, T, Q)" in text


def test_a_hydrograph_is_interpolated_in_time():
    """A two-sample table that is exact for the true flux still converges."""
    problem = hydrograph_heat()
    errors = [ADRProblem.run(problem, {"u": ("CG", 1)}, cells,
                             time={"dt": 0.05})[2]["u"]
              for cells in (8, 16)]
    assert errors[0] < 2.0e-2, errors
    rate = numpy.log2(errors[0] / errors[1])
    assert rate > 1.5, (errors, rate)


def test_a_table_holds_its_end_values():
    """Outside the record the first and last samples hold; nothing extrapolates."""
    flux = ADRProblem._value_of({"series": [[0.0, -2.0], [1.0, -4.0]]}, 2)
    assert flux(None, -5.0) == -2.0
    assert flux(None, 0.5) == -3.0
    assert flux(None, 99.0) == -4.0


def test_the_emitted_p_file_carries_the_hydrograph_table(tmp_path):
    ADRProblem.emit_pn(hydrograph_heat(), {"u": ("CG", 1)}, 8, time={"dt": 0.5},
                       name="hyd", directory=str(tmp_path))
    text = (tmp_path / "hyd_p.py").read_text()
    assert "numpy.interp(t, T, Q)" in text
    assert "[0.0, 1.0]" in text and "[-2.0, -4.0]" in text


def test_a_neumann_face_converges_at_the_right_rate():
    """The sharp check: without the flux the solution does not converge at all.

    Dropping the condition leaves x = 1 with the natural zero-flux condition,
    which solves a different problem -- the error stalls near 2.2e-1 and the
    rate goes slightly negative. Getting the sign wrong is worse still. So
    second-order convergence here is evidence the value AND its sign reach
    Proteus intact.
    """
    problem = neumann_poisson()
    errors = [ADRProblem.run(problem, {"u": ("CG", 1)}, cells)[2]["u"]
              for cells in (8, 16)]
    assert errors[0] < 1.0e-2
    rate = numpy.log2(errors[0] / errors[1])
    assert 1.8 < rate < 2.2, (errors, rate)


def test_a_neumann_condition_needs_something_to_flux():
    problem = neumann_poisson()
    del problem["adr"]["equations"][0]["diffusion"]
    with pytest.raises(ValueError, match="no diffusion term"):
        ADRProblem.physics(problem, "adr")


def test_a_neumann_condition_on_a_nonexistent_unknown_is_refused():
    problem = neumann_poisson()
    problem["neumann"][0]["component"] = "v"
    with pytest.raises(ValueError, match="not an unknown"):
        ADRProblem.physics(problem, "adr")


def test_the_emitted_p_file_carries_the_neumann_flux(tmp_path):
    """parun has to solve the same problem the library just did."""
    ADRProblem.emit_pn(neumann_poisson(), {"u": ("CG", 1)}, 8,
                       name="neu", directory=str(tmp_path))
    text = (tmp_path / "neu_p.py").read_text()
    assert "def getDFBC_u(X, flag):" in text
    assert "diffusiveFluxBoundaryConditions = {0: {0: getDFBC_u}}" in text
    # the value and its sign, not just the hook
    assert "-2.0*x" in text or "-2.0 * x" in text


# --- end to end, through ymf_run ---------------------------------------------------


@pytest.fixture
def poisson_spec(tmp_path):
    pytest.importorskip("ymf.symbolic")
    pytest.importorskip("strictyaml")
    path = tmp_path / "poisson.ymf"
    path.write_text('''
model:
  name: "Poisson"
  physical_model: {provenance: human_specified}
  strong_form:
    provenance: human_specified
    unknowns: [u]
    equation_formulation: "Poisson"
    equations: ["-Δu = 2π² sin(πx) sin(πy)  in Ω"]
    domain: "Ω = [0, 1] × [0, 1]"
    boundary_regions: [{name: wall, where: "∂Ω"}]
    boundary_conditions: [{region: wall, variable: u, type: dirichlet, value: 0.0}]
  weak_forms:
    - label: galerkin
      provenance: human_specified
      derivation: "integrate by parts"
      solution_spaces: {trial: H1, test: H1_0}
      bilinear: "∫ ∇u·∇v"
      linear: "∫ f v"
      stabilization_method: none
solution_paths:
  analytical:
    - name: exact
      provenance: human_specified
      method: closed_form
      solution: {formula: "u = sin(πx) sin(πy)"}
  discretizations:
    - name: P1
      provenance: human_specified
      from_weak_form: galerkin
      finite_element: {fields: {family: CG, order: 1}}
      solver: {type: linear}
''', encoding="utf-8")
    return path


def ymf_run(*args):
    return subprocess.run([sys.executable, SCRIPT, *map(str, args)],
                          capture_output=True, text=True)


def rows(stdout):
    return [line.split() for line in stdout.splitlines() if line.strip()[:1].isdigit()]


def test_a_spec_runs_end_to_end_into_its_archive(poisson_spec, tmp_path):
    out = ymf_run(poisson_spec, "--cells", "4", "8", "--outdir", tmp_path)
    assert out.returncode == 0, out.stderr[-2000:]
    table = rows(out.stdout)
    assert [r[0] for r in table] == ["4", "8"] and [r[-1] for r in table] == ["new", "new"]
    assert 1.8 < float(table[1][2]) < 2.2          # P1 in L2
    from ymf import closure
    spec, outputs, _ = closure.load(tmp_path / "poisson.archive.ymf")
    assert spec["model"]["strong_form"]["equations"] == ["-Δu = 2π² sin(πx) sin(πy)  in Ω"]
    assert spec["solution_paths"]["discretizations"][0]["mesh"] == {"cells": [4, 8]}
    assert spec["composition"]["overrides"][0]["set_by"] == "ymf_run --cells 4 8"
    key = [k for k in outputs if "/cells=8/" in k][0]
    output = outputs[key]
    assert output["realization"] == {"cells": 8, "levels": 1}
    assert output["input"]["sha256"].startswith(key.rsplit("/", 1)[1])
    discrete = output["transformations"]["adr_to_discrete"]
    assert discrete["spaces"] == {"u": "C0_AffineLinearOnSimplexWithNodalBasis"}
    assert discrete["mesh"] == {"nodes_per_side": 9, "levels": 1}
    assert set(output["verification"]["l2_errors"]) == {"u"}
    files = sorted(p.name for p in tmp_path.iterdir() if p.suffix in (".h5", ".xmf"))
    assert files == sorted(closure.file_stem("poisson", k) + s
                           for k in outputs for s in (".h5", ".xmf"))


def test_like_make_it_computes_only_what_is_missing(poisson_spec, tmp_path):
    assert ymf_run(poisson_spec, "--cells", "4", "--outdir", tmp_path).returncode == 0
    out = ymf_run(poisson_spec, "--cells", "4", "8", "--outdir", tmp_path)
    assert out.returncode == 0, out.stderr[-2000:]
    assert [r[-1] for r in rows(out.stdout)] == ["archive", "new"]    # "in the archive"
    assert "2 outputs, 1 new" in out.stdout


def test_an_archive_reruns_to_bitwise_identical_results(poisson_spec, tmp_path):
    assert ymf_run(poisson_spec, "--cells", "4", "--outdir", tmp_path).returncode == 0
    archive = tmp_path / "poisson.archive.ymf"
    out = ymf_run(archive, "--check")
    assert out.returncode == 0, out.stdout[-2000:] + out.stderr[-2000:]
    assert "reproduced, bitwise" in out.stdout
    from ymf import closure
    _, outputs, _ = closure.load(archive)
    (output,) = outputs.values()
    assert len(output["reproduced"]) == 1


def test_a_differing_rerun_is_refused_unless_kept(poisson_spec, tmp_path):
    assert ymf_run(poisson_spec, "--cells", "4", "--outdir", tmp_path).returncode == 0
    archive = tmp_path / "poisson.archive.ymf"
    # tamper with the record, as a different machine's result would differ
    from ymf.archive import read_document, write_document
    document = read_document(archive)
    (output,) = document["outputs"].values()
    output["verification"]["l2_errors"]["u"] *= 1.5
    write_document(archive, document)
    out = ymf_run(archive, "--check")
    assert out.returncode == 1 and "NOT REPRODUCED" in out.stdout and "L2(u)" in out.stdout
    out = ymf_run(archive, "--check", "--keep-different")
    assert out.returncode == 0, out.stdout[-2000:]
    keys = list(read_document(archive)["outputs"])
    assert len(keys) == 2 and keys[1] == keys[0] + "~2"


def test_an_edited_spec_does_not_silently_replace_the_archive(poisson_spec, tmp_path):
    assert ymf_run(poisson_spec, "--cells", "4", "--outdir", tmp_path).returncode == 0
    poisson_spec.write_text(poisson_spec.read_text().replace('"Poisson"', '"Poisson, renamed"'))
    out = ymf_run(poisson_spec, "--cells", "4", "--outdir", tmp_path)
    assert out.returncode != 0 and "earlier version of the input" in out.stdout
    out = ymf_run(poisson_spec, "--cells", "4", "--outdir", tmp_path, "--prune")
    assert out.returncode == 0 and "1 outputs, 1 new" in out.stdout


def test_an_inline_multilevel_run_is_self_contained_and_reproducible(poisson_spec, tmp_path):
    out = ymf_run(poisson_spec, "--cells", "2", "--levels", "3", "--inline", "--outdir", tmp_path)
    assert out.returncode == 0, out.stderr[-2000:]
    from ymf import closure
    from ymf.archive import domain_arrays
    archive = tmp_path / "poisson.archive.ymf"
    _, outputs, _ = closure.load(archive)
    ((key, output),) = outputs.items()
    assert "/cells=2/levels=3/" in key and output["storage"] == "inline"
    assert sorted(p.name for p in tmp_path.iterdir()) == sorted(
        ["poisson.ymf", "poisson.archive.ymf", closure.file_stem("poisson", key) + ".xmf"])
    nodes = [a for k, a in domain_arrays(output["approximation"]).items()
             if k.endswith("Geometry")][-1]
    assert nodes.shape == (81, 3)        # the finest level: 2 cells refined twice -> 8x8
    out = ymf_run(archive, "--check")
    assert out.returncode == 0 and "reproduced, bitwise" in out.stdout, out.stdout[-2000:]


def test_emitted_pn_files_reproduce_the_output_bitwise_under_parun(poisson_spec, tmp_path):
    import numpy
    out = ymf_run(poisson_spec, "--cells", "4", "--outdir", tmp_path, "--emit-pn")
    assert out.returncode == 0, out.stderr[-2000:]
    from ymf import closure
    from ymf.archive import domain_arrays, read_ymf
    _, outputs, _ = closure.load(tmp_path / "poisson.archive.ymf")
    ((key, output),) = outputs.items()
    stem = closure.file_stem("poisson", key)
    work = tmp_path / "pn"
    work.mkdir()
    for suffix in ("_p", "_n", "_so"):
        (work / (stem + suffix + ".py")).write_text((tmp_path / (stem + suffix + ".py")).read_text())
    parun = os.path.join(os.path.dirname(sys.executable), "parun")
    run = subprocess.run([parun, stem + "_so.py"], cwd=work, capture_output=True, text=True)
    assert run.returncode == 0, run.stdout[-2000:] + run.stderr[-2000:]
    mine = domain_arrays(read_ymf(work / (stem + "_pn.ymf"))[0], work)
    recorded = domain_arrays(output["approximation"], tmp_path)
    assert set(mine) == set(recorded)
    assert all(numpy.array_equal(mine[k], recorded[k]) for k in recorded)


@pytest.fixture
def kovasznay_spec(tmp_path):
    """Kovasznay flow, Taylor-Hood, velocity given all round: p up to a constant."""
    pytest.importorskip("ymf.symbolic")
    pytest.importorskip("strictyaml")
    path = tmp_path / "kovasznay.ymf"
    path.write_text('''
model:
  name: "Kovasznay"
  physical_model: {provenance: human_specified}
  strong_form:
    provenance: human_specified
    unknowns: [{name: v, rank: 1}, p]
    equation_formulation: "Navier-Stokes"
    equations:
      - "∇·(ρ v⊗v) − μΔv + ∇p = 0  in Ω"
      - "∇·v = 0  in Ω"
    domain: "Ω = [-0.5, 1.0] × [-0.5, 1.5]"
    boundary_regions: [{name: outer, where: "∂Ω"}]
    boundary_conditions:
      - {region: outer, variable: v, type: dirichlet,
         formula: "(1 - exp(λx) cos(2πy), λ/(2π) exp(λx) sin(2πy))"}
    coefficients:
      ρ: {value: 1.0}
      μ: {value: 0.025}
      λ: "ρ/(2μ) - sqrt(ρ²/(4μ²) + 4π²)"
  weak_forms:
    - label: mixed
      provenance: human_specified
      derivation: "integrate by parts"
      solution_spaces: {trial: "H1² × L2", test: "H1_0² × L2"}
      bilinear: "..."
      linear: "..."
      stabilization_method: none
solution_paths:
  analytical:
    - name: exact
      provenance: human_specified
      method: closed_form
      solution:
        formula: |-
          v = (1 - exp(λx) cos(2πy), λ/(2π) exp(λx) sin(2πy))
          p = (1 - exp(2λx))/2
  discretizations:
    - name: taylor_hood
      provenance: human_specified
      from_weak_form: mixed
      finite_element: {velocity: {family: CG, order: 2}, pressure: {family: CG, order: 1}}
      mesh: {cells: [4, 8]}
      solver: {type: nonlinear}
''', encoding="utf-8")
    return path


def test_a_pressure_up_to_a_constant_is_solved_in_the_null_space(kovasznay_spec, tmp_path):
    import numpy
    out = ymf_run(kovasznay_spec, "--outdir", tmp_path, "--emit-pn")
    assert out.returncode == 0, out.stderr[-2000:]
    assert "L2(p)*" in out.stdout
    table = rows(out.stdout)
    assert 2.5 < float(table[1][2]) and 1.5 < float(table[1][6])      # v_0 P2, p P1 rates
    from ymf import closure
    from ymf.archive import domain_arrays, read_ymf
    archive = tmp_path / "kovasznay.archive.ymf"
    _, outputs, _ = closure.load(archive)
    output = outputs[[k for k in outputs if "/cells=4/" in k][0]]
    assert output["verification"]["modulo_constants"] == ["p"]
    discrete = output["transformations"]["adr_to_discrete"]
    assert discrete["null_space"] == {"constant": ["p"]}
    assert discrete["linear_solver"] == "KSP_petsc4py"
    assert output["transformations"]["solve"]["by"]["threads"]["OMP_NUM_THREADS"] == "1"
    check = ymf_run(archive, "--check")
    assert check.returncode == 0 and check.stdout.count("reproduced, bitwise") == 2, check.stdout[-2000:]
    # and the emitted files reproduce it under parun, single-threaded
    stem = closure.file_stem("kovasznay", [k for k in outputs if "/cells=4/" in k][0])
    work = tmp_path / "pn"
    work.mkdir()
    for suffix in ("_p", "_n", "_so"):
        (work / (stem + suffix + ".py")).write_text((tmp_path / (stem + suffix + ".py")).read_text())
    env = dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", VECLIB_MAXIMUM_THREADS="1")
    parun = os.path.join(os.path.dirname(sys.executable), "parun")
    run = subprocess.run([parun, stem + "_so.py"], cwd=work, capture_output=True, text=True, env=env)
    assert run.returncode == 0, run.stdout[-2000:] + run.stderr[-2000:]
    mine = domain_arrays(read_ymf(work / (stem + "_pn.ymf"))[0], work)
    recorded = domain_arrays(output["approximation"], tmp_path)
    assert all(numpy.array_equal(mine[k], recorded[k]) for k in recorded)
