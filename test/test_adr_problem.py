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


# --- end to end, through ymf_run ---------------------------------------------------


@pytest.fixture
def poisson_spec(tmp_path):
    pytest.importorskip("ymf.symbolic")
    pytest.importorskip("strictyaml")
    path = tmp_path / "poisson.ymf"
    path.write_text('''
Problem:
  name: "Poisson"
  physical_model: {provenance: human_specified}
  strong_form:
    provenance: human_specified
    unknowns: [u]
    equation_formulation: "Poisson"
    equations: ["-Δu = 2π² sin(πx) sin(πy)  in Ω"]
    domain: "Ω = [0, 1] × [0, 1]"
    boundary_regions: [{name: wall, geometry: "∂Ω"}]
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


def test_a_spec_runs_end_to_end_and_converges(poisson_spec, tmp_path):
    out = ymf_run(poisson_spec, "--cells", "4", "8", "--outdir", tmp_path)
    assert out.returncode == 0, out.stderr[-2000:]
    rows = [line.split() for line in out.stdout.splitlines() if line.strip()[:1].isdigit()]
    assert [r[0] for r in rows] == ["4", "8"]
    assert 1.8 < float(rows[1][2]) < 2.2          # P1 in L2
    from ymf.archive import read_ymf
    _, extra = read_ymf(tmp_path / "poisson_P1_8.ymf")
    assert extra["spec"]["Problem"]["strong_form"]["equations"] == ["-Δu = 2π² sin(πx) sin(πy)  in Ω"]
    assert extra["run"] == {"discretization": "P1", "cells": 8, "levels": 1,
                            "spaces": {"u": ["CG", 1]}, "time": None,
                            "stabilization": "none", "storage": "hdf5"}
    assert set(extra["results"]["l2_errors"]) == {"u"}


def test_an_archive_reruns_to_bitwise_identical_results(poisson_spec, tmp_path):
    assert ymf_run(poisson_spec, "--cells", "4", "--outdir", tmp_path).returncode == 0
    out = ymf_run(tmp_path / "poisson_P1_4.ymf", "--check", "--outdir", tmp_path)
    assert out.returncode == 0, out.stdout[-2000:] + out.stderr[-2000:]
    assert "idempotent" in out.stdout


def test_an_inline_multilevel_run_is_self_contained_and_reproducible(poisson_spec, tmp_path):
    out = ymf_run(poisson_spec, "--cells", "2", "--levels", "3", "--inline", "--outdir", tmp_path)
    assert out.returncode == 0, out.stderr[-2000:]
    names = sorted(p.name for p in tmp_path.glob("poisson_P1_2x3.*"))
    assert names == ["poisson_P1_2x3.xmf", "poisson_P1_2x3.ymf"]       # no .h5
    from ymf.archive import read_ymf, domain_arrays
    domain, extra = read_ymf(tmp_path / "poisson_P1_2x3.ymf")
    assert (extra["run"]["cells"], extra["run"]["levels"], extra["run"]["storage"]) == (2, 3, "inline")
    nodes = [a for k, a in domain_arrays(domain).items() if k.endswith("Geometry")][-1]
    assert nodes.shape == (81, 3)        # the finest level: 2 cells refined twice -> 8x8
    out = ymf_run(tmp_path / "poisson_P1_2x3.ymf", "--check", "--outdir", tmp_path)
    assert out.returncode == 0 and "idempotent" in out.stdout, out.stdout[-2000:]
